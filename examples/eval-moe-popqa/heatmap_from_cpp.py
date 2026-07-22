#!/usr/bin/env python3
# type: ignore

"""Render OLMoE expert-routing heatmaps from the C++ `llama-eval-moe-popqa`
JSON.

Reads `expert_counts.json` produced by `llama-eval-moe-popqa` and writes:
  - `routing_heatmap.png`                  layer x expert overview
  - `routing_heatmap_by_prop.png`          ~16 props x (L*E) cells (log1p)
  - `match_rate_by_prop.png`               per-prop substring-match accuracy
  - `counts_total.json`                    aggregated [L, E] matrix (sum across props)
  - `metadata.json`                        pass-through + computed totals

Schema of the input JSON (`expert_counts.json`):
  {
    "model": "<hf model id>",
    "model_arch": {"name": "olmoe", "n_layer": 16, "n_expert": 64, "n_expert_used": 8},
    "config":  {"questions_per_prop": 50, "gen_tokens": 16, "prompt_format": ..., "match_metric": ...},
    "totals":  {"props_run": 16, "questions_total": 800, "tokens_total_prefill": ..., "tokens_total_generated": ...,
                "correct": 412, "accuracy": 0.515},
    "props": {
       "<prop>": {
         "questions": 50,
         "n_tokens_prefill": 19460,
         "n_tokens_generated": 768,
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


TOP_K = 8


# --------------------------------------------------------------------------- I/O

def _load_cpp_json(path: Path) -> tuple[dict, np.ndarray, dict[str, np.ndarray], int, dict]:
    """Parse the C++ `expert_counts.json` file.

    Returns (arch, counts_total, prop_counts, total_tokens, metadata).
    """
    with open(path) as f:
        data = json.load(f)

    arch = data.get("model_arch", {})
    L = int(arch.get("n_layer", 16))
    E = int(arch.get("n_expert", 64))

    props_raw = data.get("props", {})
    if not props_raw:
        raise SystemExit(f"[error] no props found in {path}")

    prop_counts: dict[str, np.ndarray] = {}
    for prop_name, body in props_raw.items():
        mat = np.asarray(body["layer_expert_counts"], dtype=np.int64)
        if mat.shape != (L, E):
            raise SystemExit(
                f"[error] prop '{prop_name}' has shape {mat.shape}, expected ({L}, {E})"
            )
        prop_counts[prop_name] = mat

    counts_total = np.zeros((L, E), dtype=np.int64)
    for mat in prop_counts.values():
        counts_total += mat

    total_tokens = sum(int(v.get("n_tokens_prefill", 0)) + int(v.get("n_tokens_generated", 0))
                       for v in props_raw.values())

    metadata = {
        "model": data.get("model"),
        "model_arch": arch,
        "config": data.get("config", {}),
        "totals": data.get("totals", {}),
        "total_tokens_seen": total_tokens,
        "expected_top_k": TOP_K,
    }
    return arch, counts_total, prop_counts, total_tokens, metadata


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
        f"OLMoE per-token activation rate (top-{TOP_K} of {E}); "
        f"{total_tokens:,} tokens"
    )
    plt.colorbar(
        im, ax=ax,
        label=f"selections / token (uniform = {TOP_K / E:.4f})",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_prop_heatmap(
    prop_counts: dict[str, np.ndarray],
    props: list[str],
    path: Path,
    scale: str = "log",
    *,
    colormap: str = "viridis",
    dpi: int = 120,
) -> bool:
    """Render an overall prop heatmap + per-layer-pair breakdown panels.

    Reuses the MMLU `_draw_category_heatmap` renderer; the per-prop partition
    is the natural PopQA analog of MMLU's per-subject category aggregation.
    """
    if scale not in ("linear", "log"):
        raise ValueError(f"scale must be 'linear' or 'log'; got {scale!r}")

    # Stack [props, L, E] + drop any props with zero counts.
    rows_3d: list[np.ndarray] = []
    labels:  list[str]       = []
    for p in props:
        mat = prop_counts.get(p)
        if mat is None or mat.size == 0 or mat.sum() <= 0:
            continue
        rows_3d.append(mat.astype(np.int64, copy=False))
        labels.append(p)
    if not rows_3d:
        return False

    M_3d = np.stack(rows_3d)
    L, E = M_3d.shape[1:]

    M_3d_disp = np.log1p(M_3d) if scale == "log" else M_3d.astype(np.float64)
    cbar_label = "log1p(count)" if scale == "log" else "count"

    return _draw_category_heatmap(
        M_3d_disp, labels, L, E, cbar_label,
        title_prefix=(
            f"OLMoE expert activations by PopQA relation type (`prop`)  -  "
            f"{len(labels)} props x {L * E} cells (overall + per-layer-pair, scale={scale})"
        ),
        path=path, colormap=colormap, dpi=dpi,
    )


def save_match_rate_chart(
    prop_counts: dict[str, np.ndarray],
    props_raw: dict,
    props: list[str],
    path: Path,
    *,
    dpi: int = 120,
) -> bool:
    """Horizontal bar chart of match_rate per prop, sorted descending.

    `props_raw` is the raw dict-of-dicts from the JSON (we need n_correct /
    n_questions / match_rate fields).
    """
    rows = []
    for p in props:
        body = props_raw.get(p)
        if body is None:
            continue
        n_q = int(body.get("questions", 0))
        if n_q <= 0:
            continue
        rows.append((p, n_q, int(body.get("n_correct", 0)), float(body.get("match_rate", 0.0))))
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
    ax.set_xlabel("match_rate (substring containment)")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(
        f"PopQA match_rate by relation type (`prop`)  -  "
        f"{len(labels)} props, overall accuracy = "
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


# Reused from MMLU heatmap_from_cpp.py (verbatim). Renders an overall panel
# plus per-layer-pair subplots using a shared grid spec.
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
    ax_overall.set_ylabel("prop (relation type)")
    ax_overall.set_title(
        f"All layers (overall)  -  {len(labels)} props x "
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
        ax.set_ylabel("prop")
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
        help="Path to expert_counts.json produced by llama-eval-moe-popqa.",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Where to write heatmaps and JSON. Defaults to the input file's directory.",
    )
    parser.add_argument(
        "--heatmap",
        choices=("overview", "props", "accuracy", "all"),
        default="all",
        help="Which plot(s) to produce.",
    )
    parser.add_argument(
        "--scale", choices=("linear", "log"), default="log",
        help="Color scale for the prop heatmap (raw counts or log1p).",
    )
    parser.add_argument("--colormap", type=str, default="viridis",
                        help="Matplotlib colormap name.")
    parser.add_argument("--dpi", type=int, default=120,
                        help="Output PNG DPI.")
    args = parser.parse_args()

    input_path: Path = args.input
    output_dir: Path = args.output_dir if args.output_dir is not None else input_path.parent

    print(f"[load] input = {input_path.resolve()}")
    arch, counts_total, prop_counts, total_tokens, metadata = _load_cpp_json(input_path)
    L = int(arch.get("n_layer", counts_total.shape[0]))
    E = int(arch.get("n_expert", counts_total.shape[1]))
    # Preserve original `props` ordering (sorted by source JSON) for viz.
    with open(input_path) as f:
        _raw = json.load(f)
    props = list(_raw.get("props", {}).keys())
    print(
        f"[load] arch={arch.get('name')} L={L} E={E}, "
        f"props={len(props)}, total_tokens={total_tokens:,}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_metadata(metadata, counts_total, output_dir / "metadata.json")
    print(f"[save] {output_dir.resolve() / 'metadata.json'}")
    print(f"[save] {output_dir.resolve() / 'counts_total.json'}")

    if args.heatmap in ("overview", "all"):
        p = output_dir / "routing_heatmap.png"
        save_overview_heatmap(
            counts_total, total_tokens, p,
            colormap=args.colormap, dpi=args.dpi,
        )
        print(f"[save] {p.resolve()}")

    if args.heatmap in ("props", "all"):
        p = output_dir / "routing_heatmap_by_prop.png"
        wrote = save_prop_heatmap(
            prop_counts, props, p, args.scale,
            colormap=args.colormap, dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()} (scale={args.scale})")
        else:
            print(f"[skip] no props with activations; {p.name} not written")

    if args.heatmap in ("accuracy", "all"):
        p = output_dir / "match_rate_by_prop.png"
        wrote = save_match_rate_chart(
            prop_counts, _raw.get("props", {}), props, p,
            dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()}")
        else:
            print(f"[skip] no props with completions; {p.name} not written")

    print(f"[done] output_dir = {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
