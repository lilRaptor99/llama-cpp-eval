#!/usr/bin/env python3
# type: ignore

"""Render MoE expert-routing heatmaps from the C++ `llama-eval-moe-humaneval`
JSON.

Reads `expert_counts.json` produced by `llama-eval-moe-humaneval` and writes:
  - `routing_heatmap.png`                            layer x expert overview
  - `routing_heatmap_by_task.png`                    164 tasks x (L*E) cells (log1p)
  - `routing_heatmap_by_task_normalized.png`         164 tasks x (L*E) cells (row-normalized)
  - `counts_total.json`                              aggregated [L, E] matrix (sum across tasks)
  - `metadata.json`                                  pass-through + computed totals

This script does **not** plot any accuracy / match-rate chart -- HumanEval
routing capture never scores the generated code (see
`config.match_metric = "none"` in the input JSON).

Schema of the input JSON (`expert_counts.json`):
  {
    "model": "<hf model id>",
    "model_arch": {"name": "olmoe", "n_layer": 16, "n_expert": 64, "n_expert_used": 8},
    "config":  {"questions_per_task": 164, "gen_tokens": 256,
                "prompt_format": "humaneval_zero_shot_continuation",
                "match_metric": "none"},
    "dataset": "openai/openai_humaneval",
    "split":   "test",
    "totals":  {"tasks_run": 164, "questions_total": 164,
                "tokens_total_prefill": ..., "tokens_total_generated": ...,
                "completions_truncated": ...},
    "tasks": {
       "<task_id>": {
         "entry_point": "...",
         "questions": 1,
         "n_tokens_prefill": ...,
         "n_tokens_generated": ...,
         "completion_truncated": false,
         "layer_expert_counts": [[int, int, ...] * 64] * 16,  // [L, E] int64
         "completion": "..."
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
    for task_id, body in tasks_raw.items():
        mat = np.asarray(body["layer_expert_counts"], dtype=np.int64)
        if mat.shape != (L, E):
            raise SystemExit(
                f"[error] task '{task_id}' has shape {mat.shape}, expected ({L}, {E})"
            )
        task_counts[task_id] = mat

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


def save_task_grid_heatmap(
    task_counts: dict[str, np.ndarray],
    tasks: list[str],
    path: Path,
    *,
    scale: str = "log",
    colormap: str = "viridis",
    dpi: int = 100,
) -> bool:
    """Per-problem grid of small L x E heatmaps (log1p or raw counts).

    HumanEval has 164 problems -- too many for one row. We lay them out in
    a `ceil(sqrt(n))` x `ceil(sqrt(n))` grid with a dynamic figure height.
    Each cell is a small L x E heatmap (L ~16, E ~64) so we keep L on the
    vertical axis and E on the horizontal axis within each cell.
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

    n = len(labels)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # Per-cell size: keep cells readable even at 164 problems.
    cell_w = max(1.5, E * 0.18)
    cell_h = max(1.2, L * 0.22)
    fig_w = cols * cell_w
    fig_h = rows * cell_h

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    # Compute a single vmin/vmax so all cells share the same color scale.
    vmin = float(M_3d_disp.min())
    vmax = float(M_3d_disp.max())

    last_im = None
    for idx in range(n):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        im = ax.imshow(M_3d_disp[idx], aspect="auto", cmap=colormap,
                       vmin=vmin, vmax=vmax)
        ax.set_title(labels[idx], fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        last_im = im

    # Hide unused cells.
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(),
                     label=cbar_label, fraction=0.01, pad=0.01, shrink=0.6)

    fig.suptitle(
        f"MoE expert activations by HumanEval task_id (no scoring)  -  "
        f"{n} problems x {L} layers x {E} experts (scale={scale})",
        fontsize=12,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def save_task_grid_heatmap_normalized(
    task_counts: dict[str, np.ndarray],
    tasks: list[str],
    path: Path,
    *,
    colormap: str = "viridis",
    dpi: int = 100,
) -> bool:
    """Row-normalized per-problem grid (each problem's [L, E] sums to 1).

    Surfaces the per-problem expert-mix shape independent of token volume.
    """
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
    row_sums = M_3d.sum(axis=(1, 2), keepdims=True)
    safe = np.where(row_sums == 0, 1, row_sums)
    M_3d_norm = M_3d.astype(np.float64) / safe

    n = len(labels)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    cell_w = max(1.5, E * 0.18)
    cell_h = max(1.2, L * 0.22)
    fig_w = cols * cell_w
    fig_h = rows * cell_h

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    vmin = float(M_3d_norm.min())
    vmax = float(M_3d_norm.max())

    last_im = None
    for idx in range(n):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        im = ax.imshow(M_3d_norm[idx], aspect="auto", cmap=colormap,
                       vmin=vmin, vmax=vmax)
        ax.set_title(labels[idx], fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        last_im = im

    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(),
                     label="fraction of task's selections",
                     fraction=0.01, pad=0.01, shrink=0.6)

    fig.suptitle(
        f"MoE expert activations by HumanEval task_id (row-normalized)  -  "
        f"{n} problems x {L} layers x {E} experts",
        fontsize=12,
    )
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
        help="Path to expert_counts.json produced by llama-eval-moe-humaneval.",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Where to write heatmaps and JSON. Defaults to the input file's directory.",
    )
    parser.add_argument(
        "--heatmap",
        choices=("overview", "tasks", "normalized", "all"),
        default="all",
        help="Which plot(s) to produce. 'tasks' is the log1p per-task grid.",
    )
    parser.add_argument(
        "--scale", choices=("linear", "log"), default="log",
        help="Color scale for the per-task grid (raw counts or log1p).",
    )
    parser.add_argument("--colormap", type=str, default="viridis",
                        help="Matplotlib colormap name.")
    parser.add_argument("--dpi", type=int, default=100,
                        help="Output PNG DPI.")
    args = parser.parse_args()

    input_path: Path = args.input
    output_dir: Path = args.output_dir if args.output_dir is not None else input_path.parent

    print(f"[load] input = {input_path.resolve()}")
    arch, counts_total, task_counts, total_tokens, metadata = _load_cpp_json(input_path)
    L = int(arch.get("n_layer", counts_total.shape[0]))
    E = int(arch.get("n_expert", counts_total.shape[1]))
    top_k = int(arch.get("n_expert_used", 8))
    # Preserve original `tasks` ordering (order from source JSON) for viz.
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
        wrote = save_task_grid_heatmap(
            task_counts, tasks, p,
            scale=args.scale,
            colormap=args.colormap, dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()} (scale={args.scale})")
        else:
            print(f"[skip] no tasks with activations; {p.name} not written")

    if args.heatmap in ("normalized", "all"):
        p = output_dir / "routing_heatmap_by_task_normalized.png"
        wrote = save_task_grid_heatmap_normalized(
            task_counts, tasks, p,
            colormap=args.colormap, dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()} (row-normalized)")
        else:
            print(f"[skip] no tasks with activations; {p.name} not written")

    print(f"[done] output_dir = {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())