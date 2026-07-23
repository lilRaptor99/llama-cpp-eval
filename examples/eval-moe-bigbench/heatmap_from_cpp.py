#!/usr/bin/env python3
# type: ignore

"""Render MoE expert-routing heatmaps from the C++ `llama-eval-moe-bigbench`
JSON.

Reads `expert_counts.json` produced by `llama-eval-moe-bigbench` and writes:
  - `routing_heatmap.png`                       layer x expert overview
  - `routing_heatmap_by_task.png`               27 tasks x (L*E) cells (log1p)
  - `routing_heatmap_by_task_normalized.png`    27 tasks x (L*E) cells (row-normalized)
  - `accuracy_by_task.png`                      per-task exact-match accuracy
  - `counts_total.json`                         aggregated [L, E] matrix (sum across tasks)
  - `metadata.json`                             pass-through + computed totals

Schema of the input JSON (`expert_counts.json`):
  {
    "model": "<hf model id>",
    "model_arch": {"name": "olmoe", "n_layer": 16, "n_expert": 64, "n_expert_used": 8},
    "config":  {"questions_per_task": 50, "gen_tokens": 128,
                "prompt_format": "zero_shot_direct_qa",
                "match_metric": "normalized_exact_match"},
    "dataset": "maveriq/bigbenchhard",
    "split":   "train",
    "totals":  {"tasks_run": 27, "questions_total": 1350,
                "tokens_total_prefill": ..., "tokens_total_generated": ...,
                "correct": 412, "accuracy": 0.515},
    "tasks": {
       "<task>": {
         "questions": 50,
         "n_tokens_prefill": ...,
         "n_tokens_generated": ...,
         "n_correct": 23,
         "match_rate": 0.46,
         "layer_expert_counts": [[int, int, ...] * 64] * 16  // [L, E] int64
       },
       ...
    }
  }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- I/O

def _load_cpp_json(path: Path) -> tuple[dict, np.ndarray, dict[str, np.ndarray], int, dict]:
    """Parse the C++ `expert_counts.json` file.

    Returns (arch, counts_total, task_counts, total_tokens, metadata).
    """
    with open(path) as f:
        data = json.load(f)

    arch = data.get("model_arch", {})
    L = int(arch.get("n_layer", 16))
    E = int(arch.get("n_expert", 64))

    tasks_raw = data.get("tasks", {})
    if not tasks_raw:
        raise SystemExit(f"[error] no tasks found in {path}")

    task_counts: dict[str, np.ndarray] = {}
    for task_name, body in tasks_raw.items():
        mat = np.asarray(body["layer_expert_counts"], dtype=np.int64)
        if mat.shape != (L, E):
            raise SystemExit(
                f"[error] task '{task_name}' has shape {mat.shape}, expected ({L}, {E})"
            )
        task_counts[task_name] = mat

    counts_total = np.zeros((L, E), dtype=np.int64)
    for mat in task_counts.values():
        counts_total += mat

    total_tokens = sum(
        int(v.get("n_tokens_prefill", 0)) + int(v.get("n_tokens_generated", 0))
        for v in tasks_raw.values()
    )

    metadata = {
        "model": data.get("model"),
        "model_arch": arch,
        "config": data.get("config", {}),
        "dataset": data.get("dataset"),
        "split": data.get("split"),
        "totals": data.get("totals", {}),
        "total_tokens_seen": total_tokens,
        "expected_top_k": int(arch.get("n_expert_used", 8)),
    }
    return arch, counts_total, task_counts, total_tokens, metadata


def _save_metadata(metadata: dict, counts_total: np.ndarray, path: Path) -> None:
    """Persist a Python-style metadata.json + counts_total.json for downstream tools."""
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(path.parent / "counts_total.json", "w") as f:
        json.dump(counts_total.tolist(), f)


# --------------------------------------------------------------- heatmap renderers

def save_overview_heatmap(
    counts: np.ndarray,
    total_tokens: int,
    top_k: int,
    path: Path,
    *,
    colormap: str = "viridis",
    dpi: int = 120,
) -> None:
    """Layer x expert heatmap of per-token activation rate."""
    L, E = counts.shape
    per_token = counts.astype(np.float64) / max(total_tokens, 1)

    fig, ax = plt.subplots(figsize=(max(14, E * 0.32), max(4, L * 0.5)))
    im = ax.imshow(per_token, aspect="auto", cmap=colormap)
    ax.set_xlabel("expert id")
    ax.set_ylabel("layer")
    ax.set_title(
        f"MoE per-token activation rate (top-{top_k} of {E}); "
        f"{total_tokens:,} tokens"
    )
    plt.colorbar(
        im, ax=ax,
        label=f"selections / token (uniform = {top_k / E:.4f})",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_task_heatmap(
    task_counts: dict[str, np.ndarray],
    tasks: list[str],
    path: Path,
    scale: str = "log",
    *,
    colormap: str = "viridis",
    dpi: int = 120,
) -> bool:
    """Render an overall task heatmap + per-layer-pair breakdown panels.

    Reuses the PopQA layout verbatim: overall panel on top, then a row of
    per-layer-pair panels sharing the row-major (layer-major) cell layout.
    """
    if scale not in ("linear", "log"):
        raise ValueError(f"scale must be 'linear' or 'log'; got {scale!r}")

    # Stack [tasks, L, E] + drop any tasks with zero counts.
    rows_3d: list[np.ndarray] = []
    labels:  list[str]       = []
    for t in tasks:
        mat = task_counts.get(t)
        if mat is None or mat.size == 0 or mat.sum() <= 0:
            continue
        rows_3d.append(mat.astype(np.int64, copy=False))
        labels.append(t)
    if not rows_3d:
        return False

    M_3d = np.stack(rows_3d)
    L, E = M_3d.shape[1:]

    M_3d_disp = np.log1p(M_3d) if scale == "log" else M_3d.astype(np.float64)
    cbar_label = "log1p(count)" if scale == "log" else "count"

    return _draw_category_heatmap(
        M_3d_disp, labels, L, E, cbar_label,
        title_prefix=(
            f"MoE expert activations by BIG-Bench Hard task  -  "
            f"{len(labels)} tasks x {L * E} cells (overall + per-layer-pair, scale={scale})"
        ),
        path=path, colormap=colormap, dpi=dpi,
    )


def save_task_heatmap_normalized(
    task_counts: dict[str, np.ndarray],
    tasks: list[str],
    path: Path,
    *,
    colormap: str = "viridis",
    dpi: int = 120,
) -> bool:
    """Row-normalized task heatmap (overall + per-layer-pair breakdown).

    Each task's [L, E] matrix is divided by its row-sum so each row sums to 1
    -- surfaces the per-task expert-mix shape independent of token volume.
    """
    # Stack [tasks, L, E] + drop any tasks with zero counts.
    rows_3d: list[np.ndarray] = []
    labels:  list[str]       = []
    for t in tasks:
        mat = task_counts.get(t)
        if mat is None or mat.size == 0 or mat.sum() <= 0:
            continue
        rows_3d.append(mat.astype(np.int64, copy=False))
        labels.append(t)
    if not rows_3d:
        return False

    M_3d = np.stack(rows_3d)
    L, E = M_3d.shape[1:]
    # Row-normalize across (L, E) per task so each row sums to 1.
    row_sums = M_3d.sum(axis=(1, 2), keepdims=True)
    safe = np.where(row_sums == 0, 1, row_sums)
    M_3d_norm = M_3d.astype(np.float64) / safe

    return _draw_category_heatmap(
        M_3d_norm, labels, L, E,
        cbar_label="fraction of task's selections",
        title_prefix=(
            f"MoE expert activations by BIG-Bench Hard task  -  "
            f"{len(labels)} tasks x {L * E} cells (row-normalized; "
            f"overall + per-layer-pair breakdown)"
        ),
        path=path, colormap=colormap, dpi=dpi,
    )


def save_accuracy_chart(
    task_counts: dict[str, np.ndarray],
    tasks_raw: dict,
    tasks: list[str],
    path: Path,
    *,
    dpi: int = 120,
) -> bool:
    """Horizontal bar chart of match_rate per task, sorted descending.

    `tasks_raw` is the raw dict-of-dicts from the JSON (we need n_correct /
    n_questions / match_rate fields).
    """
    rows = []
    for t in tasks:
        body = tasks_raw.get(t)
        if body is None:
            continue
        n_q = int(body.get("questions", 0))
        if n_q <= 0:
            continue
        rows.append((t, n_q, int(body.get("n_correct", 0)), float(body.get("match_rate", 0.0))))
    if not rows:
        return False

    rows.sort(key=lambda t: t[3], reverse=True)
    labels = [r[0] for r in rows]
    rates  = [r[3] for r in rows]
    n_qs   = [r[1] for r in rows]
    n_ok   = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(labels))))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, rates, color="#3a7ca5", edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("match_rate (whitespace-normalised exact match)")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(
        f"BIG-Bench Hard match_rate by task  -  "
        f"{len(labels)} tasks, overall accuracy = "
        f"{sum(n_ok) / max(1, sum(n_qs)):.3f}"
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    for bar, q, k in zip(bars, n_qs, n_ok):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{k}/{q}",
            va="center", fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


# Reused from eval-moe-popqa/heatmap_from_cpp.py (verbatim). Renders an
# overall panel plus per-layer-pair subplots using a shared grid spec.
def _draw_category_heatmap(
    M_3d_disp: np.ndarray,
    labels: list[str],
    L: int,
    E: int,
    cbar_label: str,
    title_prefix: str,
    path: Path,
    *,
    colormap: str = "viridis",
    dpi: int = 120,
    layers_per_subplot: int = 2,
) -> bool:
    if not labels:
        return False

    M_flat_disp = M_3d_disp.reshape(M_3d_disp.shape[0], -1)

    n_groups = L // layers_per_subplot
    if n_groups == 0 or L % layers_per_subplot != 0:
        raise ValueError(
            f"num_layers={L} not divisible by layers_per_subplot={layers_per_subplot}"
        )
    layer_pairs = [
        (i * layers_per_subplot, (i + 1) * layers_per_subplot - 1)
        for i in range(n_groups)
    ]

    n_cols = 4
    n_subplot_rows = (n_groups + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(32, 8 + 6 * n_subplot_rows))
    height_ratios = [3] + [1.5] * n_subplot_rows
    gs = fig.add_gridspec(
        1 + n_subplot_rows, n_cols,
        height_ratios=height_ratios,
        hspace=0.5, wspace=0.3,
    )

    ax_overall = fig.add_subplot(gs[0, :])
    im_overall = ax_overall.imshow(M_flat_disp, aspect="auto", cmap=colormap)
    ax_overall.set_xlabel(f"expert cell (layer-major: {E} experts x {L} layers)")
    ax_overall.set_ylabel("task")
    ax_overall.set_title(
        f"All layers (overall)  -  {len(labels)} tasks x "
        f"{M_flat_disp.shape[1]} cells"
    )
    plt.colorbar(im_overall, ax=ax_overall, label=cbar_label)

    layer_starts = [layer_idx * E for layer_idx in range(L)]
    ax_overall.set_xticks(layer_starts)
    ax_overall.set_xticklabels(
        [f"L{l}" for l in range(L)], rotation=0, fontsize=8
    )
    ax_overall.set_yticks(range(len(labels)))
    ax_overall.set_yticklabels(labels, fontsize=8)

    for i, (lo, hi) in enumerate(layer_pairs):
        row = 1 + i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])

        M_pair = M_3d_disp[:, lo:hi + 1, :].reshape(M_3d_disp.shape[0], -1)
        im = ax.imshow(M_pair, aspect="auto", cmap=colormap)
        ax.set_title(f"Layers {lo}-{hi}")
        ax.set_xlabel("expert cell")
        ax.set_ylabel("task")
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.05, pad=0.04)

        n_pair_layers = hi - lo + 1
        pair_layer_starts = [j * E for j in range(n_pair_layers)]
        ax.set_xticks(pair_layer_starts)
        ax.set_xticklabels(
            [f"L{lo + j}" for j in range(n_pair_layers)],
            rotation=0, fontsize=7,
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)

    fig.suptitle(title_prefix, fontsize=14)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


# ----------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Path to expert_counts.json produced by llama-eval-moe-bigbench.",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Where to write heatmaps and JSON. Defaults to the input file's directory.",
    )
    parser.add_argument(
        "--heatmap",
        choices=("overview", "tasks", "accuracy", "normalized", "all"),
        default="all",
        help="Which plot(s) to produce.",
    )
    parser.add_argument(
        "--scale", choices=("linear", "log"), default="log",
        help="Color scale for the task heatmap (raw counts or log1p).",
    )
    parser.add_argument("--colormap", type=str, default="viridis",
                        help="Matplotlib colormap name.")
    parser.add_argument("--dpi", type=int, default=120,
                        help="Output PNG DPI.")
    args = parser.parse_args()

    input_path: Path = args.input
    output_dir: Path = args.output_dir if args.output_dir is not None else input_path.parent

    print(f"[load] input = {input_path.resolve()}")
    arch, counts_total, task_counts, total_tokens, metadata = _load_cpp_json(input_path)
    L = int(arch.get("n_layer", counts_total.shape[0]))
    E = int(arch.get("n_expert", counts_total.shape[1]))
    top_k = int(arch.get("n_expert_used", 8))
    # Preserve original `tasks` ordering (sorted by source JSON) for viz.
    with open(input_path) as f:
        _raw = json.load(f)
    tasks = list(_raw.get("tasks", {}).keys())
    print(
        f"[load] arch={arch.get('name')} L={L} E={E}, "
        f"tasks={len(tasks)}, total_tokens={total_tokens:,}, top_k={top_k}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_metadata(metadata, counts_total, output_dir / "metadata.json")
    print(f"[save] {output_dir.resolve() / 'metadata.json'}")
    print(f"[save] {output_dir.resolve() / 'counts_total.json'}")

    if args.heatmap in ("overview", "all"):
        p = output_dir / "routing_heatmap.png"
        save_overview_heatmap(
            counts_total, total_tokens, top_k, p,
            colormap=args.colormap, dpi=args.dpi,
        )
        print(f"[save] {p.resolve()}")

    if args.heatmap in ("tasks", "all"):
        p = output_dir / "routing_heatmap_by_task.png"
        wrote = save_task_heatmap(
            task_counts, tasks, p, args.scale,
            colormap=args.colormap, dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()} (scale={args.scale})")
        else:
            print(f"[skip] no tasks with activations; {p.name} not written")

    if args.heatmap in ("normalized", "all"):
        p = output_dir / "routing_heatmap_by_task_normalized.png"
        wrote = save_task_heatmap_normalized(
            task_counts, tasks, p,
            colormap=args.colormap, dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()} (row-normalized)")
        else:
            print(f"[skip] no tasks with activations; {p.name} not written")

    if args.heatmap in ("accuracy", "all"):
        p = output_dir / "accuracy_by_task.png"
        wrote = save_accuracy_chart(
            task_counts, _raw.get("tasks", {}), tasks, p,
            dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()}")
        else:
            print(f"[skip] no tasks with completions; {p.name} not written")

    print(f"[done] output_dir = {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
