#!/usr/bin/env python3
"""Render expert coactivation as a layer-expert node graph.

Nodes:
  - one node per (layer, expert), arranged on a fixed grid
  - x-axis: expert index
  - y-axis: layer index

Edges:
    - intra-layer edges from intra_pair_counts[L, E, E]
    - inter-layer edges only between adjacent layers (L -> L+1)
        using inter_* counts (all 4 output formats supported)
  - line width encodes edge weight (raw count by default)

The script is designed to work with output from:
  examples/eval-moe-coactivation/eval-moe-coactivation.cpp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def parse_figsize(value: str) -> Tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--figsize must be 'W,H'")
    try:
        w = float(parts[0].strip())
        h = float(parts[1].strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--figsize values must be numeric") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("--figsize values must be > 0")
    return w, h


def _decode_sparse_coo(coo: Dict, expected_rank: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    shape = tuple(int(x) for x in coo["shape"])
    if len(shape) != expected_rank:
        raise ValueError(f"sparse shape rank mismatch: got {len(shape)}, expected {expected_rank}")

    indices = coo["indices"]
    values = coo["values"]
    if len(indices) != len(values):
        raise ValueError("sparse COO mismatch: len(indices) != len(values)")

    arr = np.zeros(shape, dtype=np.int64)
    observed = np.zeros(shape, dtype=bool)
    for idx, val in zip(indices, values):
        if len(idx) != expected_rank:
            raise ValueError(f"sparse COO index rank mismatch: got {len(idx)}, expected {expected_rank}")
        t_idx = tuple(int(x) for x in idx)
        arr[t_idx] = int(val)
        observed[t_idx] = True
    return arr, observed


def load_inter_canonical(container: Dict, config: Dict, n_layer: int, n_expert: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load inter-layer counts to canonical [L, L, E, E] and observed mask.

    Returns:
      inter_full: int64 [L, L, E, E]
      observed:   bool  [L, L, E, E]
      mode:       short mode label for diagnostics/title
    """
    inter_k_lag = int(config.get("inter_k_lag", 0))
    sparse_min_count = int(config.get("sparse_min_count", 0))

    full = np.zeros((n_layer, n_layer, n_expert, n_expert), dtype=np.int64)
    observed = np.zeros((n_layer, n_layer, n_expert, n_expert), dtype=bool)

    if inter_k_lag > 0:
        if sparse_min_count > 0:
            key = "inter_klag_counts_sparse"
            if key not in container:
                raise KeyError(f"missing key: {key}")
            klag_arr, klag_obs = _decode_sparse_coo(container[key], expected_rank=4)
            mode = "klag_sparse"
        else:
            key = "inter_klag_counts"
            if key not in container:
                raise KeyError(f"missing key: {key}")
            klag_arr = np.array(container[key], dtype=np.int64)
            if klag_arr.ndim != 4:
                raise ValueError("inter_klag_counts must be rank-4 [L, K, E, E]")
            klag_obs = np.ones_like(klag_arr, dtype=bool)
            mode = "klag_dense"

        if klag_arr.shape[0] != n_layer or klag_arr.shape[2] != n_expert or klag_arr.shape[3] != n_expert:
            raise ValueError(
                "inter_klag shape mismatch with model_arch "
                f"(got {klag_arr.shape}, expected [{n_layer}, K, {n_expert}, {n_expert}])"
            )

        k_max = klag_arr.shape[1]
        for l1 in range(n_layer):
            for k_idx in range(k_max):
                l2 = l1 + k_idx + 1
                if l2 < n_layer:
                    full[l1, l2] = klag_arr[l1, k_idx]
                    observed[l1, l2] = klag_obs[l1, k_idx]
    else:
        if sparse_min_count > 0:
            key = "inter_pair_counts_sparse"
            if key not in container:
                raise KeyError(f"missing key: {key}")
            dense_arr, dense_obs = _decode_sparse_coo(container[key], expected_rank=4)
            mode = "full_sparse"
        else:
            key = "inter_pair_counts"
            if key not in container:
                raise KeyError(f"missing key: {key}")
            dense_arr = np.array(container[key], dtype=np.int64)
            if dense_arr.ndim != 4:
                raise ValueError("inter_pair_counts must be rank-4 [L, L, E, E]")
            dense_obs = np.ones_like(dense_arr, dtype=bool)
            mode = "full_dense"

        if dense_arr.shape != (n_layer, n_layer, n_expert, n_expert):
            raise ValueError(
                "inter_pair shape mismatch with model_arch "
                f"(got {dense_arr.shape}, expected [{n_layer}, {n_layer}, {n_expert}, {n_expert}])"
            )

        full = dense_arr
        observed = dense_obs

    return full, observed, mode


def tick_values(max_index: int) -> List[int]:
    if max_index <= 12:
        step = 1
    elif max_index <= 48:
        step = 4
    elif max_index <= 128:
        step = 8
    else:
        step = 16
    ticks = list(range(0, max_index + 1, step))
    if ticks[-1] != max_index:
        ticks.append(max_index)
    return ticks


def collect_edges(
    intra: np.ndarray,
    inter: np.ndarray,
    inter_observed: np.ndarray,
    layer_min: int,
    layer_max: int,
    include_intra: bool,
    include_inter: bool,
    top_k_per_layer_pair: int,
) -> List[Tuple[int, int, int, int, int]]:
    """Collect pruned edges as (l1, e1, l2, e2, count)."""
    n_layer = intra.shape[0]
    n_expert = intra.shape[1]

    buckets: Dict[Tuple[int, int], List[Tuple[int, int, int, int, int]]] = {}

    if include_intra:
        for l in range(layer_min, layer_max + 1):
            block = intra[l]
            key = (l, l)
            vals: List[Tuple[int, int, int, int, int]] = []
            for e1 in range(n_expert):
                # Skip self loops for readability.
                for e2 in range(e1 + 1, n_expert):
                    c = int(block[e1, e2])
                    if c > 0:
                        vals.append((l, e1, l, e2, c))
            buckets[key] = vals

    if include_inter:
        for l1 in range(layer_min, layer_max):
            l2 = l1 + 1
            if l2 > layer_max:
                continue
            block = inter[l1, l2]
            obs = inter_observed[l1, l2]
            key = (l1, l2)
            vals = []
            for e1 in range(n_expert):
                for e2 in range(n_expert):
                    if not obs[e1, e2]:
                        continue
                    c = int(block[e1, e2])
                    if c > 0:
                        vals.append((l1, e1, l2, e2, c))
            buckets[key] = vals

    edges: List[Tuple[int, int, int, int, int]] = []
    for vals in buckets.values():
        if not vals:
            continue
        vals.sort(key=lambda x: x[4], reverse=True)
        if top_k_per_layer_pair > 0:
            vals = vals[:top_k_per_layer_pair]
        edges.extend(vals)

    return edges


def widths_from_counts(
    counts: np.ndarray,
    min_linewidth: float,
    max_linewidth: float,
    log_width_scale: bool,
) -> np.ndarray:
    if counts.size == 0:
        return np.array([], dtype=np.float64)

    vals = counts.astype(np.float64)
    if log_width_scale:
        vals = np.log1p(vals)

    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax <= vmin:
        return np.full_like(vals, (min_linewidth + max_linewidth) * 0.5, dtype=np.float64)

    t = (vals - vmin) / (vmax - vmin)
    return min_linewidth + t * (max_linewidth - min_linewidth)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot expert coactivation graph from eval-moe-coactivation JSON")
    ap.add_argument("-i", "--input", required=True, help="Path to coactivation JSON")
    ap.add_argument("-o", "--output", default=None, help="Output image path (default: <input-dir>/expert_graph.png)")
    ap.add_argument("--subject", default=None, help="Optional subject key to plot instead of aggregate")
    ap.add_argument("--include-intra", dest="include_intra", action="store_true", default=True)
    ap.add_argument("--no-include-intra", dest="include_intra", action="store_false")
    ap.add_argument("--include-inter", dest="include_inter", action="store_true", default=True)
    ap.add_argument("--no-include-inter", dest="include_inter", action="store_false")
    ap.add_argument("--top-k-per-layer-pair", type=int, default=20, help="Keep top-K edges within each (L1,L2) block")
    ap.add_argument("--layer-min", type=int, default=0, help="Min layer index to render")
    ap.add_argument("--layer-max", type=int, default=None, help="Max layer index to render (default: last layer)")
    ap.add_argument("--figsize", type=parse_figsize, default=(14.0, 8.5), help="Figure size as W,H")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--node-size", type=float, default=150.0)
    ap.add_argument("--edge-alpha", type=float, default=0.20)
    ap.add_argument("--min-linewidth", type=float, default=0.20)
    ap.add_argument("--max-linewidth", type=float, default=3.20)
    ap.add_argument("--log-width-scale", action="store_true", help="Use log1p(count) for width scaling")
    args = ap.parse_args()

    input_path = Path(args.input)
    with input_path.open() as f:
        data = json.load(f)

    n_layer = int(data["model_arch"]["n_layer"])
    n_expert = int(data["model_arch"]["n_expert"])
    config = data.get("config", {})

    if args.layer_max is None:
        layer_max = n_layer - 1
    else:
        layer_max = args.layer_max
    layer_min = args.layer_min

    if layer_min < 0 or layer_max < 0 or layer_min > layer_max or layer_max >= n_layer:
        raise ValueError(f"invalid layer range [{layer_min}, {layer_max}] for n_layer={n_layer}")

    if args.subject is not None:
        subjects = data.get("subjects", {})
        if args.subject not in subjects:
            raise KeyError(f"subject '{args.subject}' not found")
        block = subjects[args.subject]
        source_name = f"subject:{args.subject}"
    else:
        block = data["aggregate"]
        source_name = "aggregate"

    intra = np.array(block["intra_pair_counts"], dtype=np.int64)
    if intra.shape != (n_layer, n_expert, n_expert):
        raise ValueError(
            "intra_pair_counts shape mismatch "
            f"(got {intra.shape}, expected [{n_layer}, {n_expert}, {n_expert}])"
        )

    include_inter = args.include_inter
    inter_mode = "none"
    inter = np.zeros((n_layer, n_layer, n_expert, n_expert), dtype=np.int64)
    inter_observed = np.zeros_like(inter, dtype=bool)

    if include_inter:
        try:
            inter, inter_observed, inter_mode = load_inter_canonical(block, config, n_layer, n_expert)
        except KeyError as exc:
            include_inter = False
            print(f"[warn] inter disabled for source '{source_name}': {exc}")

    edges = collect_edges(
        intra=intra,
        inter=inter,
        inter_observed=inter_observed,
        layer_min=layer_min,
        layer_max=layer_max,
        include_intra=args.include_intra,
        include_inter=include_inter,
        top_k_per_layer_pair=max(0, args.top_k_per_layer_pair),
    )

    counts = np.array([e[4] for e in edges], dtype=np.int64)
    widths = widths_from_counts(
        counts=counts,
        min_linewidth=args.min_linewidth,
        max_linewidth=args.max_linewidth,
        log_width_scale=args.log_width_scale,
    )

    x = np.arange(n_expert, dtype=np.float64)
    y = np.arange(layer_min, layer_max + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=args.figsize, dpi=args.dpi)

    if edges:
        segments = [
            ((float(e1), float(l1)), (float(e2), float(l2)))
            for (l1, e1, l2, e2, _) in edges
        ]
        # Draw strong edges last for clearer visibility.
        order = np.argsort(widths)
        lc = LineCollection(
            [segments[i] for i in order],
            linewidths=widths[order],
            colors="#6b7280",
            alpha=args.edge_alpha,
            zorder=1,
        )
        ax.add_collection(lc)

    ax.scatter(
        xx.ravel(),
        yy.ravel(),
        s=args.node_size,
        facecolors="#f9fafb",
        edgecolors="#475569",
        linewidths=1.0,
        zorder=3,
    )

    ax.set_xlim(-0.6, n_expert - 0.4)
    ax.set_ylim(layer_min - 0.4, layer_max + 0.4)
    ax.invert_yaxis()

    ax.set_xlabel("Expert Index (within layer)")
    ax.set_ylabel("Layer Index")

    xticks = tick_values(n_expert - 1)
    yticks = tick_values(layer_max)
    yticks = [t for t in yticks if layer_min <= t <= layer_max]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_title(str(data.get("model", "")), fontsize=11)

    if counts.size > 0:
        min_c = int(counts.min())
        max_c = int(counts.max())
        info = f"edges={counts.size}, count range=[{min_c}, {max_c}]"
    else:
        info = "edges=0"
    ax.text(
        0.01,
        0.01,
        info,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#374151",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#d1d5db", "boxstyle": "round,pad=0.2"},
    )

    ax.grid(True, which="major", linestyle="--", color="#e5e7eb", linewidth=0.8, zorder=0)

    if args.output is None:
        output_path = input_path.parent / "expert_graph.png"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"[plot] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
