#!/usr/bin/env python3
"""Produce a compact heatmap of selected layers x experts highlighting hot experts.

Reads `counts_total_overview.json` emitted by the overview aggregator and
produces a small PNG suitable for a single column in a double-column paper.

Usage examples:
  python3 examples/eval-moe-overview/highlight_hot_experts.py   --results-dir build/results_old   --model allenai--OLMoE-1B-7B-0125-Instruct-GGUF --quant Q4_K_M   --layers 0-4 --experts 3-18   --output figures/olmoe_q2_hot.png   --dpi 300 --width-inches 3.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import numpy as np


def parse_index_list(s: str, max_value: int | None = None) -> List[int]:
    """Parse a comma-separated list of indices and ranges like "0,2,5-8".
    Special token "all" returns all indices in range(max_value).
    """
    if not s:
        return []
    s = s.strip()
    if s.lower() == "all":
        if max_value is None:
            raise ValueError("max_value must be provided for 'all'")
        return list(range(max_value))
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i = int(a)
            b_i = int(b)
            if b_i < a_i:
                raise ValueError(f"invalid range {part}")
            out.update(range(a_i, b_i + 1))
        else:
            out.add(int(part))
    res = sorted(x for x in out if max_value is None or (0 <= x < max_value))
    return res


def find_counts_json(results_dir: Path, model: str, quant: str) -> Path:
    """Locate counts_total_overview.json for given model and quant.
    Tolerate legacy layout where quant is absent.
    """
    # New layout: <results_dir>/<model>/<quant>/overall/counts_total_overview.json
    p = results_dir / model / quant / "overall" / "counts_total_overview.json"
    if p.exists():
        return p
    # Legacy layout: <results_dir>/<model>/overall/counts_total_overview.json
    p2 = results_dir / model / "overall" / "counts_total_overview.json"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"counts_total_overview.json not found for {model}/{quant}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("build/results"))
    p.add_argument("--model", required=True)
    p.add_argument("--quant", required=True)
    p.add_argument("--layers", default="all",
                   help="Comma/list or ranges of layer indices, e.g. '0,2,5-7' or 'all'")
    p.add_argument("--experts", default="all",
                   help="Comma/list or ranges of expert indices, e.g. '0,1,2-10' or 'all'")
    p.add_argument("--output", type=Path, default=Path("hot_experts.png"))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--width-inches", type=float, default=3.3,
                   help="Target figure width in inches (single-column ~3.3in)")
    p.add_argument("--log", action="store_true", help="Use log10 color scale")
    args = p.parse_args()

    counts_json = find_counts_json(args.results_dir, args.model, args.quant)
    with open(counts_json, "r") as f:
        data = json.load(f)

    counts = np.array(data["counts_total"], dtype=np.float64)
    L, E = counts.shape

    layers = parse_index_list(args.layers, max_value=L)
    experts = parse_index_list(args.experts, max_value=E)
    if not layers:
        layers = list(range(L))
    if not experts:
        experts = list(range(E))

    sub = counts[np.array(layers)[:, None], np.array(experts)[None, :]]

    # Figure sizing: width fixed; height scales with number of layers but kept compact
    width = max(1.5, args.width_inches)
    # Per-layer height to keep figure readable for a single-column figure
    per_layer_h = 0.35
    height = max(1.0, len(layers) * per_layer_h)

    fig, ax = plt.subplots(figsize=(width, height), dpi=args.dpi)
    if args.log:
        # avoid log(0)
        sub_plot = np.log10(sub + 1.0)
        im = ax.imshow(sub_plot, aspect="auto", cmap="Blues", interpolation="nearest")
        # cb_label = "log10(raw count + 1)"
    else:
        im = ax.imshow(sub, aspect="auto", cmap="Blues", interpolation="nearest")
        # cb_label = "raw activation count"

    # Tick labels: layer indices at left, expert indices at bottom (rotated if needed)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(i) for i in layers], fontsize=8)
    ax.set_ylabel("Layer", fontsize=9)

    ax.set_xticks(range(len(experts)))
    ax.set_xticklabels([str(e) for e in experts], fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Expert ID", fontsize=9)

    # No vertical highlight lines by default; keep plot clean for paper figures.

    def _millions_formatter(x, pos):
        if x >= 1e6:
            return f"{x/1e6:.1f}M"
        if x >= 1e3:
            return f"{x/1e3:.1f}k"
        return f"{int(x)}"

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(_millions_formatter))
    # cbar.set_label(cb_label, fontsize=8, labelpad=6)

    fig.tight_layout(rect=(0, 0, 0.98, 1.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {args.output} — layers={layers} experts={experts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
