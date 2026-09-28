#!/usr/bin/env python3
# type: ignore

"""Publication-quality heatmap renderer for the MoE-routing eval suite.

Companion to the existing heatmap generators in `examples/eval-moe-overview/`
(`aggregate_overview.py`, `jaccard_sweep_from_cpp.py`,
`cross_quant_jaccard_sweep_from_cpp.py`). Those render at 120 DPI with
multi-line titles and 30-degree rotated labels sized for screen
inspection. This script renders at high DPI (~300) and deliberately
**strips** every title / axis heading / set_xlabel / set_ylabel call
so the figure drops cleanly into a LaTeX `\\includegraphics` slot
inside a double-column paper.

Two views, both at K = n_expert_used:

  1. Cross-dataset (per `(model, quant)` cell)        — pairwise Jaccard
     of top-K expert sets across the five benchmarks.
  2. Cross-quantization (per model)                   — pairwise Jaccard
     of top-K expert sets across quantizations.

A small `--per-cell` mode also emits one compact heatmap of the raw
`counts_total_overview.json` [L, E] matrix per `(model, quant)`
("routing strip") so the paper can show WHERE the experts are hot
without paying for the full L x E circle-grid plot. The routing strip
is the only view in this module that carries axis labels (`Layer
index` / `Expert index`) — integer-only ticks would be ambiguous
without them. The Jaccard / JS-divergence heatmaps continue to strip
axis labels because their tick labels (benchmark names, quant names)
are self-explanatory.

Outputs (`<output-dir>/`):

  Cross-dataset (per (model, quant) cell):
    `<model_safe>__<quant_safe>__jaccard_dataset.png`
    `<model_safe>__<quant_safe>__jaccard_dataset_K2x.png`
    `<model_safe>__<quant_safe>__jaccard_dataset_K3x.png`
    `<model_safe>__<quant_safe>__js_divergence_dataset.png`

  Cross-quantization (per model, with at least 2 quants):
    `<model_safe>__jaccard_quant.png`
    `<model_safe>__jaccard_quant_K2x.png`
    `<model_safe>__jaccard_quant_K3x.png`
    `<model_safe>__js_divergence_quant.png`

  Per-cell routing strip (optional, `--per-cell`):
    `<model_safe>__<quant_safe>__routing_strip.png`
    (raw activation counts, layer & expert indices on the axes)

The filename embeds the model id (slashes -> double-dashes, the same
mapping `scripts/run-evaluator.sh::model_safe` uses) and the quant
(internal directory name like `Q4_K_M`).

Mathematical contract (identical to `jaccard_sweep_from_cpp.py` /
`cross_quant_jaccard_sweep_from_cpp.py`):

  For each layer l and K value:
    A_i_K[l]  = top-K experts of (counts aggregated across datasets for unit i)
                (descending count, ties -> ascending expert id)
    Jaccard(A_i, A_j) = |A_i ∩ A_j| / |A_i ∪ A_j|
    Edge cases:
      both empty   -> 1.0
      only one has -> 0.0

  Cross-dataset unit i   = single `moe-<dataset>/expert_counts.json`
                           (the [L, E] matrix from
                           `aggregate.marginal_expert_counts`)
  Cross-quant  unit i   = the per-quant aggregated-across-datasets
                           `[L, E]` matrix from
                           `<model>/<quant>/overall/counts_total_overview.json`

The Jaccard pairwise matrix is averaged across all `L` layers (one
float per pair, the mean over layers), matching the convention used
by the existing screen-orientation plots.

## Publishable labels (request)

  Benchmarks (datasets):  BBH, HumanEval, INCLUDE, MMLU, PopQA
  Quants:                 BF16, FP8, Q4_K_M, Q8_0, ...

The user-supplied naming convention is honored in two places:

  1. The dataset directory names on disk are still `moe-<dataset>/`
     (these are written by the C++ binaries and we don't rename them).
     The published benchmark label is mapped from the *directory prefix*
     using `_DATASET_PUBLISH_LABELS` below. Examples:
        moe-mmlu       -> MMLU
        moe-popqa      -> PopQA
        moe-bigbench   -> BBH
        moe-humaneval  -> HumanEval
        moe-include    -> INCLUDE
     Any unknown dataset passes through its top-level name unchanged so
     adding a new benchmark doesn't break this script.

  2. The quant directory name on disk is `Q4_K_M`, `Q2_K`, `BF16`,
     etc. — already in publishable form. We deliberately do NOT rewrite
     it (no lowercase, no spaces). For reference: `Q4_K_M`, `Q5_K_M`,
     `Q6_K`, `Q8_0`, `BF16`, `FP8` are all publishable as-is.

## CLI

  Multi-mode (recommended; one invocation walks the entire results tree):

        python3 paper_heatmaps.py --results-dir build/results
            [--models <id1> [<id2> ...]]
            [--quants <q1> [<q2> ...]]
            [--per-cell]              # also emit <model>__<quant>__routing_strip.png
            [--output-dir <dir>]     # default: <results-dir>/paper_heatmaps
            [--dpi 300]
            [--cmap Blues]
            [--width-in 3.3]         # target figure width in inches
            [--per-cell-fig-height N]# rows = N*per_layer_in for the routing strip
            [--per-cell-fig-rows N]  # alternative: rows = N

  Single-cell (debug only):

        python3 paper_heatmaps.py --cell-dir <path>
            [--model-id <hf-id>] [--quant <tag>]
            [--k-min 1] [--k-max-frac 0.5] [--num-ks 21]
            [--output-dir <dir>]

The script is **idempotent by filename**: a destination PNG that
already exists is skipped (with `[skip] ...; pass --force` on stdout),
so rerunning on a populated results tree is cheap.

This script does NOT import any sibling peer script under
`examples/eval-moe-overview/`. The math helpers are inlined below —
small enough (≈30 lines) that copy-paste is cheaper than introducing
a new shared-library convention that the README elsewhere forbids.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter


# ----------------------------------------------------------------- constants

# Dataset directory on disk -> publishable label.
# Anything not in this map falls through to the un-prefixed directory
# name (e.g. "arc" stays "ARC"), so new benchmarks never break this
# script — they just don't get an alias until we add one.
_DATASET_PUBLISH_LABELS: dict[str, str] = {
    "moe-bigbench":  "BBH",
    "moe-humaneval": "HumanEval",
    "moe-include":   "INCLUDE",
    "moe-mmlu":      "MMLU",
    "moe-popqa":     "PopQA",
}

# Quant directory on disk -> publishable label.
# We treat the directory name as already publishable (uppercase, no
# whitespace). Add an entry here to FORCE a label override (e.g. some
# repos ship `Q4_K_M-f16` and you want `Q4_K_M`); leave it absent to
# pass the directory name through verbatim.
_QUANT_PUBLISH_LABELS: dict[str, str] = {
    "Q2_K":   "Q2_K",
    "Q2_K_S": "Q2_K_S",
    "Q3_K_S": "Q3_K_S",
    "Q3_K_M": "Q3_K_M",
    "Q3_K_L": "Q3_K_L",
    "Q4_0":   "Q4_0",
    "Q4_1":   "Q4_1",
    "Q4_K_S": "Q4_K_S",
    "Q4_K_M": "Q4_K_M",
    "Q5_0":   "Q5_0",
    "Q5_1":   "Q5_1",
    "Q5_K_S": "Q5_K_S",
    "Q5_K_M": "Q5_K_M",
    "Q6_0":   "Q6_0",
    "Q6_K":   "Q6_K",
    "Q6_K_S": "Q6_K_S",
    "Q8_0":   "Q8_0",
    "BF16":   "BF16",
    "F16":    "F16",
    "F32":    "F32",
    "FP8":    "FP8",
    "Q8_K":   "Q8_K",
}

# Publishable colormap (publication convention: sequential blue).
DEFAULT_CMAP = "Blues"

# Separate colormap for the JS-divergence heatmaps. The user flagged
# that `Blues_r` collapses the small divergence values (most cross-
# quant matrices live in `0..0.05`) onto a single dark-blue colour,
# making the differences between dataset / quant pairs visually
# indistinguishable even though they're numerically distinct.
#
# We switch to `viridis_r` because:
#
#   1. It's perceptually uniform — a fixed step in the data maps to
#      a fixed perceptual step in colour, so small value deltas stay
#      visible regardless of where they sit on the [0, 1] range.
#   2. It's multi-hue (dark purple → teal → yellow), so even in the
#      low-value region the hue drifts from purple toward teal as the
#      value climbs — `Blues_r`'s single-hue dark-blue gradient is
#      nearly flat in that region.
#   3. It's the de-facto standard sequential colormap for scientific
#      publication (matplotlib's default since v2.0), so paper
#      reviewers will recognise it.
#   4. It's accessible for the most common forms of colour-vision
#      deficiency (the CIELAB L* varies monotonically along the
#      colormap, so even grayscale printing preserves the ordering).
DEFAULT_JSD_CMAP = "viridis_r"

# Default figure width when not specified. ~3.3in = single column in a
# double-column paper (typical NeurIPS / ACL / IEEE conference style).
DEFAULT_WIDTH_IN = 3.3

# Color-bar shrink factor for the cross-dataset heatmap (5-row square
# matrix). A smaller cbar makes the heatmap dominate the figure area,
# which is what paper reviewers expect (the matrix IS the message).
_CBAR_FRACTION = 0.046
_CBAR_PAD      = 0.04

# text-threshold for in-cell label colour (vs background).
# Thresholds chosen to match the existing screen-orientation plots so
# the same matrix rendered here is visually equivalent.
_TEXT_THRESHOLD = 0.55


# ------------------------------------------------------------------- helpers

def _model_safe(name: str) -> str:
    """Mirror `scripts/run-evaluator.sh::model_safe` (replace / with --)."""
    return name.replace("/", "--")


def _reverse_model_safe(safe: str) -> str:
    """Inverse of `_model_safe`: first `--` -> `/`."""
    if "--" in safe:
        return safe.replace("--", "/", 1)
    return safe


def _publish_dataset_label(ds_dir_name: str) -> str:
    """Map a `moe-<dataset>` directory name to its publishable label."""
    return _DATASET_PUBLISH_LABELS.get(
        ds_dir_name, ds_dir_name[len("moe-"):].upper() if ds_dir_name.startswith("moe-") else ds_dir_name
    )


def _publish_quant_label(quant_dir_name: str) -> str:
    """Map a quant directory name to its publishable label.

    Honours the explicit overrides first; falls back to verbatim for
    anything we don't recognise (e.g. user added `Q4_K_M-f16` to disk).
    """
    if quant_dir_name in _QUANT_PUBLISH_LABELS:
        return _QUANT_PUBLISH_LABELS[quant_dir_name]
    return quant_dir_name


def _top_k_set_for_layer(counts: np.ndarray, layer: int, K: int) -> set[int]:
    """Return the top-K expert IDs in `counts[layer]`, ties -> asc id.

    Mirrors `aggregate_overview.compute_top_k`'s `kind="stable"` ordering.
    K is clamped to [1, E]. Empty counts return empty set.
    """
    row = counts[layer].astype(np.int64)
    E = row.shape[0]
    if K <= 0 or E == 0:
        return set()
    if K >= E:
        order = np.argsort(-row, kind="stable")
    else:
        top_unsorted = np.argpartition(-row, K - 1)[:K]
        order = top_unsorted[np.argsort(-row[top_unsorted], kind="stable")]
    return {int(e) for e in order[:K]}


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _make_k_values(n_expert: int, n_expert_used: int,
                   k_min: int, k_max_frac: float, num_ks: int
                   ) -> list[int]:
    """K-sweep grid: `num_ks` samples in [k_min, ceil(n_expert*k_max_frac)],
    always including n_expert_used, ceil(0.125*E), 2*k_used, 3*k_used.

    Clamped to [1, n_expert] (the 2x/3x K_used values may exceed E).
    """
    k_min_i = max(1, int(k_min))
    k_max_i = max(k_min_i, int(np.ceil(n_expert * k_max_frac)))
    base = np.linspace(k_min_i, k_max_i, num=max(1, int(num_ks)))
    base_ints = sorted({int(round(float(v))) for v in base})
    must_include = {
        int(n_expert_used),
        int(np.ceil(n_expert * 0.125)),
        int(min(2 * n_expert_used, n_expert)),
        int(min(3 * n_expert_used, n_expert)),
    }
    base_ints = sorted(set(base_ints) | must_include)
    return [k for k in base_ints if 1 <= k <= n_expert]


def _layer_averaged_pairwise_jaccard(
        counts_list: list[np.ndarray], K: int, L: int
        ) -> np.ndarray:
    """Return the pairwise Jaccard matrix averaged across L layers.

    Returns M[i, j] = mean over layers of `_jaccard(top-K_i(l), top-K_j(l))`.
    """
    N = len(counts_list)
    if N == 0:
        return np.zeros((0, 0), dtype=np.float64)
    out = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        out[i, i] = 1.0
    # Pre-compute per-(unit, layer) top-K sets.
    sets: list[list[set[int]]] = [
        [_top_k_set_for_layer(c, l, K) for l in range(L)] for c in counts_list
    ]
    for i in range(N):
        for j in range(i + 1, N):
            v = float(np.mean([_jaccard(sets[i][l], sets[j][l]) for l in range(L)]))
            out[i, j] = v
            out[j, i] = v
    return out


def _jensen_shannon_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen-Shannon divergence between two non-negative count vectors.

    The result is normalized with log base 2, so it lies in [0, 1] for
    probability distributions. Empty-vs-empty is treated as 0.0 and
    empty-vs-non-empty as 1.0 for a stable heatmap scale. Identical to
    `jaccard_sweep_from_cpp.py::_jensen_shannon_divergence`.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sum_a = float(a.sum())
    sum_b = float(b.sum())
    if sum_a <= 0.0 and sum_b <= 0.0:
        return 0.0
    if sum_a <= 0.0 or sum_b <= 0.0:
        return 1.0
    p = a / sum_a
    q = b / sum_b
    m = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    kl_pm = float(np.sum(p[mask_p] * np.log2(p[mask_p] / m[mask_p])))
    kl_qm = float(np.sum(q[mask_q] * np.log2(q[mask_q] / m[mask_q])))
    return 0.5 * (kl_pm + kl_qm)


def _layer_averaged_js_divergence_matrix(
        counts_list: list[np.ndarray]) -> np.ndarray:
    """Return the pairwise layer-averaged JS divergence matrix.

    Identical to `jaccard_sweep_from_cpp.py::_layer_averaged_js_divergence_matrix`.
    Returns M[i, j] = mean over layers of `_jensen_shannon_divergence(...)`.
    """
    n_items = len(counts_list)
    if n_items == 0:
        return np.zeros((0, 0), dtype=np.float64)
    n_layers = int(counts_list[0].shape[0])
    out = np.zeros((n_items, n_items), dtype=np.float64)
    for i in range(n_items):
        for j in range(i, n_items):
            vals = [
                _jensen_shannon_divergence(counts_list[i][layer],
                                           counts_list[j][layer])
                for layer in range(n_layers)
            ]
            out[i, j] = out[j, i] = float(np.mean(vals)) if vals else 0.0
    return out


def _millions_formatter(x: float, pos: int) -> str:
    """Tick formatter for routing-strip colorbar: 12.3M / 456.7K / 42.

    Identical convention to `aggregate_overview.py` / `eval-moe-mmlu`
    routing-graph heatmaps so the cbar reads the same way.
    """
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    if x >= 1e3:
        return f"{x/1e3:.1f}K"
    return f"{int(x)}"


def _integer_axis_ticks(N: int, max_ticks: int = 12) -> tuple[list[int], int]:
    """Pick `(tick_positions, tick_step)` for an axis of length `N`.

    Goal: never show fewer than 2 ticks and never more than `max_ticks`
    (paper figures cram a lot of tick labels in a small horizontal
    budget — we cap at ~12 by default so 3-digit labels like ``"112"``
    stay legible at single-column width). Always include 0 and the
    last index so the axis range stays clear.

    The step grows geometrically with `N`:
      - N <= `max_ticks`       -> step = 1 (every index)
      - N <= 2 * max_ticks     -> step = 2
      - N <= 4 * max_ticks     -> step = 4
      - N <= 8 * max_ticks     -> step = 8
      - N > 8 * max_ticks      -> step = `N // max_ticks` (clamped >= 16)
    """
    if N <= 0:
        return [], 1
    if N <= max_ticks:
        step = 1
    elif N <= 2 * max_ticks:
        step = 2
    elif N <= 4 * max_ticks:
        step = 4
    elif N <= 8 * max_ticks:
        step = 8
    else:
        step = max(16, N // max_ticks)
    ticks = list(range(0, N, step))
    if ticks[-1] != N - 1:
        ticks.append(N - 1)
    return ticks, step


def _render_paper_heatmap(
        mat: np.ndarray, labels: list[str], *, width_in: float,
        height_in: float, cmap: str, cell_text: bool = True,
        cell_text_format: str = "{v:.2f}",
        cbar: bool = True, dpi: int = 300,
        ) -> plt.Figure:
    """Render a publication-quality heatmap.

    No `set_title`, no `set_xlabel`, no `set_ylabel`, no axis spines —
    exactly what `\\includegraphics` wants.

    `width_in` and `height_in` are interpreted in inches; combined with
    `dpi` they determine the on-disk PNG resolution.
    `cell_text` toggles the per-cell numeric annotation; for a paper
    figure with N>5 we usually want it OFF.
    """
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    vmin, vmax = 0.0, 1.0
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto",
                   interpolation="nearest")
    # Tick labels (the ONLY labels the paper figure carries).
    N = len(labels)
    # Decide whether the bottom (x) labels need rotation. At 8pt with
    # `width_in = 3.3in`, `N = 5`, and a colorbar stealing ~10% of the
    # width, the per-cell budget is ~0.6in; the widest label
    # ("HumanEval", 9 chars) takes ~0.55in at 8pt, which is borderline.
    # We rotate when N >= 4 OR any label exceeds 5 chars to play it
    # safe at single-column figures. For 2-3 element matrices (cross-
    # quant view) we keep labels horizontal so the figure looks clean.
    widest = max((len(s) for s in labels), default=0)
    rotate_x = (N >= 4) or (widest > 5)
    if N > 0:
        ax.set_xticks(range(N))
        ax.set_xticklabels(
            labels,
            rotation=(30 if rotate_x else 0),
            ha=("right" if rotate_x else "center"),
            fontsize=8,
        )
        ax.set_yticks(range(N))
        ax.set_yticklabels(labels, rotation=0, fontsize=8)
    else:
        # 1x1 corner case: a single tick at index 0.
        ax.set_xticks([0])
        ax.set_xticklabels([labels[0] if labels else ""], fontsize=8)
        ax.set_yticks([0])
        ax.set_yticklabels([labels[0] if labels else ""], fontsize=8)
    # Tick marks off (cleaner at high DPI).
    ax.tick_params(axis="both", which="both", length=0)
    # No frame.
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Per-cell numeric annotations (optional).
    if cell_text:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = float(mat[i, j])
                # Jaccard heatmap convention: light cells -> black text,
                # dark cells -> white text. The diagonal is always 1.0
                # (deep blue), so it MUST be white.
                color = "black" if v < _TEXT_THRESHOLD else "white"
                ax.text(j, i, cell_text_format.format(v=v), ha="center",
                        va="center", fontsize=8, color=color)
    if cbar:
        cb = fig.colorbar(im, ax=ax, fraction=_CBAR_FRACTION, pad=_CBAR_PAD)
        cb.ax.tick_params(labelsize=7, length=0)
        cb.outline.set_visible(False)
    # When x labels are rotated, the bottom margin needs extra room.
    bottom_pad = 0.30
    if rotate_x:
        bottom_pad = 0.32 + 0.02 * (widest - 5)
    fig.tight_layout(pad=bottom_pad)
    return fig


def _render_paper_js_divergence_heatmap(
        mat: np.ndarray, labels: list[str], *, width_in: float,
        height_in: float, dpi: int = 300,
        cmap: str = DEFAULT_JSD_CMAP,
        ) -> plt.Figure:
    """Publication-quality JS-divergence heatmap.

    Range contract: the colorbar always spans **fixed `vmin=0, vmax=1`**
    so the reader can compare JS-divergence values across the paper
    against the canonical metric range, not against whatever happens
    to sit in this particular cell. This is the same convention as
    the screen-orientation JS-divergence plot in
    `jaccard_sweep_from_cpp.py` / `cross_quant_jaccard_sweep_from_cpp.py`.

    Colormap: defaults to `viridis_r` (`DEFAULT_JSD_CMAP`) which is
    perceptually uniform, multi-hue (dark purple → teal → yellow), and
    shows visible variation in the low-value region that `Blues_r`
    collapses onto a near-uniform dark blue. Override via the
    `--jsd-cmap` CLI flag (e.g. `magma_r`, `inferno_r`, `cividis`).

    Text contrast: cell text colour is chosen via the absolute value
    (vs the colormap's mid-point at `v=0.5`), so off-diagonal cells
    that drift slightly above 0.0 still get white text on the dark
    purple background — readable.

    6 colourbar ticks at `0, 0.2, 0.4, 0.6, 0.8, 1.0` with 3-decimal
    formatting (covers the full metric range uniformly).

    No title, no axis heading, no spines — exactly what
    `\\includegraphics` wants.
    """
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)

    # -------- fixed metric range ----------
    # The user explicitly requested 0..1 so JS-divergence values can be
    # compared across figures without re-scaling per-cell.
    vmin, vmax = 0.0, 1.0

    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap,
                   aspect="auto", interpolation="nearest")
    N = len(labels)
    # Same rotation rule as the Jaccard renderer so the two views
    # look consistent in the paper.
    widest = max((len(s) for s in labels), default=0)
    rotate_x = (N >= 4) or (widest > 5)
    if N > 0:
        ax.set_xticks(range(N))
        ax.set_xticklabels(
            labels,
            rotation=(30 if rotate_x else 0),
            ha=("right" if rotate_x else "center"),
            fontsize=8,
        )
        ax.set_yticks(range(N))
        ax.set_yticklabels(labels, rotation=0, fontsize=8)
    else:
        ax.set_xticks([0])
        ax.set_xticklabels([labels[0] if labels else ""], fontsize=8)
        ax.set_yticks([0])
        ax.set_yticklabels([labels[0] if labels else ""], fontsize=8)
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # -------- per-cell text ----------
    # 4dp cell text (e.g. `0.0360`, `0.0002`) — readable across the
    # full metric range. Diagonal cells naturally render as `0.0000`.
    cell_fmt = "{v:.4f}"
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = float(mat[i, j])
            # Threshold at 0.5 (the absolute mid-point of the colormap)
            # — values below 0.5 get white text on the dark half of
            # the colormap; values at or above get black text. The
            # diagonal (v=0) is unambiguously on the white-text side.
            color = "white" if v < 0.5 else "black"
            ax.text(j, i, cell_fmt.format(v=v), ha="center", va="center",
                    fontsize=8, color=color)

    cb = fig.colorbar(im, ax=ax, fraction=_CBAR_FRACTION, pad=_CBAR_PAD)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.outline.set_visible(False)
    # 6 ticks uniformly distributed across [0, 1] (the fixed metric
    # range). The reader sees the position of every value relative to
    # the canonical range at a glance.
    cb.set_ticks(np.linspace(0.0, 1.0, 6))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(
        lambda x, p: f"{x:.3f}"
    ))

    bottom_pad = 0.30
    if rotate_x:
        bottom_pad = 0.32 + 0.02 * (widest - 5)
    fig.tight_layout(pad=bottom_pad)
    return fig


def _render_paper_routing_strip(
        counts: np.ndarray, *, width_in: float, dpi: int = 300,
        cmap: str = "Blues",
        ) -> plt.Figure:
    """Publication-quality routing strip: raw `[L, E]` activation counts.

    The plot is the same data as the screen-orientation routing graph
    (`eval-moe-overview/aggregate_overview.py::routing_heatmap_overview`)
    but with paper-friendly tweaks:

      1. raw counts on the colorbar (no log1p) — the paper figure
         needs to convey the data the same way the per-dataset CSVs do,
         not a transformed version that's harder to reason about,
      2. layer index ticks on the y-axis + ``Layer index`` axis label
         so the integer ticks (`0, 8, 16, ...`) are unambiguous,
      3. expert index ticks on the x-axis + ``Expert index`` axis label
         (the existing setup was missing these labels, so the indices
         were opaque — the user flagged it as ``unclear``),
      4. ``Activation count`` colorbar label,
      5. tick density auto-tuned to the figure width via
         `_integer_axis_ticks(N, max_ticks=...)` so 3-digit labels
         like ``120`` stay legible at single-column width.

    The Jaccard and JS-divergence heatmaps continue to strip axis
    labels (the row/column TICK LABELS carry their own meaning — the
    benchmark names or quant names are self-explanatory). The routing
    strip is the one view where the integers need axis labels.

    The colorbar uses `_millions_formatter` (`12.3M` / `456.7K` /
    `42`) to match the routing-graph heatmaps.
    """
    L, E = counts.shape
    # Width is `width_in`; height grows with L but is capped so the
    # strip stays single-column (≤ 2x width) even for 48-layer Qwen3.
    # We pad height slightly to leave room for the new ``Layer index``
    # y-label and the (larger) tick labels.
    max_h_in = 2.4 * width_in
    per_layer_h_in = min(0.18, max_h_in / max(L, 1))
    strip_h = max(1.0, L * per_layer_h_in)
    fig, ax = plt.subplots(figsize=(width_in, strip_h), dpi=dpi)

    # Raw counts (no log1p). vmin is the matrix min — 0 for almost every
    # real model; vmax is the matrix max so the colorbar reads in the
    # paper the same way the raw counts JSON does.
    im = ax.imshow(counts.astype(np.float64), aspect="auto",
                   cmap=cmap, interpolation="nearest")

    # Tick density: aim for ≤12 ticks so 3-digit labels stay legible in
    # a single-column paper. If the axis is the Y axis (layers), we can
    # afford a few more ticks because each row has plenty of vertical
    # space; cap the X axis tighter because 3-digit numbers also need
    # more horizontal room than 1-digit.
    y_ticks, _ = _integer_axis_ticks(L, max_ticks=14)
    x_ticks, _ = _integer_axis_ticks(E, max_ticks=11)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(t) for t in y_ticks],
                       fontsize=7, rotation=0)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(t) for t in x_ticks],
                       fontsize=7, rotation=0)
    ax.tick_params(axis="both", which="both", length=0, pad=2.0)

    # Axis labels — these are the only heatmap-style figures in this
    # module that still carry them. Justified by the ambiguity of bare
    # integer ticks: ``0, 32, 64, 96`` could mean layer or expert
    # without the heading.
    ax.set_ylabel("Layer index", fontsize=8, labelpad=2)
    ax.set_xlabel("Expert index", fontsize=8, labelpad=4)

    # Spines off keeps the paper convention (no frame) but we keep the
    # tick marks themselves off too — only the labels + colorbar speak.
    for spine in ax.spines.values():
        spine.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=_CBAR_FRACTION, pad=_CBAR_PAD)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(_millions_formatter))
    cb.outline.set_visible(False)
    cb.set_label("Activation count", fontsize=8, labelpad=4)

    fig.tight_layout(pad=0.30)
    return fig


# --------------------------------------------------------------- cell loading

def _load_one_dataset_counts(json_path: Path
                              ) -> tuple[dict[str, Any], np.ndarray, int]:
    """Load a single `moe-<ds>/expert_counts.json` and return
    `(arch, marginal_counts[L,E], n_tokens)`.

    `arch` is a dict with keys `name`, `n_layer`, `n_expert`,
    `n_expert_used`. On missing `aggregate` block we raise ValueError
    (callers should skip with a `[warn]`).
    """
    with open(json_path) as f:
        data = json.load(f)
    arch_in = data.get("model_arch", {})
    L = int(arch_in.get("n_layer", 0))
    E = int(arch_in.get("n_expert", 0))
    k = int(arch_in.get("n_expert_used", 0))
    if L <= 0 or E <= 0 or k <= 0:
        raise ValueError(f"malformed model_arch in {json_path}")
    agg = data.get("aggregate") or {}
    marginal = agg.get("marginal_expert_counts")
    if marginal is None:
        raise ValueError(f"missing aggregate.marginal_expert_counts in {json_path}")
    counts = np.asarray(marginal, dtype=np.int64)
    if counts.shape != (L, E):
        raise ValueError(
            f"aggregate shape {counts.shape} != ({L}, {E}) for {json_path}"
        )
    totals = data.get("totals") or {}
    n_tokens = int(totals.get("tokens_total", 0))
    return (
        {"name": arch_in.get("name", "unknown"), "n_layer": L,
         "n_expert": E, "n_expert_used": k},
        counts,
        n_tokens,
    )


def _load_one_quant_overview(json_path: Path
                              ) -> tuple[dict[str, Any], np.ndarray, int]:
    """Load one `<model>/<quant>/overall/counts_total_overview.json` and
    return `(arch, counts_total[L,E], tokens_total)`.
    """
    with open(json_path) as f:
        data = json.load(f)
    arch_in = data.get("model_arch") or {}
    L = int(arch_in.get("n_layer", 0))
    E = int(arch_in.get("n_expert", 0))
    k = int(arch_in.get("n_expert_used", 0))
    if L <= 0 or E <= 0 or k <= 0:
        raise ValueError(f"malformed model_arch in {json_path}")
    raw = data.get("counts_total")
    if raw is None:
        raise ValueError(f"missing counts_total in {json_path}")
    counts = np.asarray(raw, dtype=np.int64)
    if counts.shape != (L, E):
        raise ValueError(
            f"counts_total shape {counts.shape} != ({L}, {E}) for {json_path}"
        )
    return (
        {"name": arch_in.get("name", "unknown"), "n_layer": L,
         "n_expert": E, "n_expert_used": k},
        counts,
        int(data.get("totals_tokens", 0)),
    )


# ---------------------------------------------------------- cross-dataset view

def _iter_cell_paths(cell_dir: Path) -> list[tuple[str, Path]]:
    """Return `[(dataset_label, expert_counts_json), ...]` for a cell.

    A "cell" is one `(model, quant)` directory containing the five
    `moe-<dataset>/expert_counts.json` files.
    """
    out: list[tuple[str, Path]] = []
    if not cell_dir.is_dir():
        return out
    for sub in sorted(p for p in cell_dir.iterdir() if p.is_dir()):
        ec = sub / "expert_counts.json"
        if not ec.is_file():
            continue
        ds = sub.name  # e.g. "moe-mmlu"
        out.append((ds, ec))
    return out


def _make_figure_size(N: int, width_in: float, cells_per_inch: float = 5.5,
                       min_height_in: float = 1.4) -> tuple[float, float]:
    """Pick `(width_in, height_in)` for a square-ish N x N heatmap.

    We keep width at `width_in` (paper column budget) and grow height
    proportionally to N (so 5 benchmarks stay roughly square, not
    squashed into a 3.3 x 3.3 rectangle that makes labels overlap).
    """
    h = max(min_height_in, N * width_in / cells_per_inch)
    return width_in, h


def _process_one_dataset_cell(
        cell_dir: Path, *, output_dir: Path,
        width_in: float, dpi: int, cmap: str,
        k_min: int, k_max_frac: float, num_ks: int,
        model_id: str | None = None,
        quant: str | None = None,
        force: bool = False,
        per_cell: bool = False,
        jsd_cmap: str = DEFAULT_JSD_CMAP,
        ) -> dict[str, Any]:
    """Process one (model, quant) cell for cross-dataset top-K Jaccard.

    Writes one PNG per (K_in_{K_model, 2*K_model, 3*K_model}) into
    `output_dir`. Filenames embed model + quant (see module docstring).
    Skips when a destination already exists (`--force` to overwrite).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    items = _iter_cell_paths(cell_dir)
    if not items:
        return {"cell_dir": str(cell_dir), "skipped": True,
                "reason": "no expert_counts.json"}

    # Resolve model & quant labels for the filename + the publishable axis.
    cell_dir = cell_dir.resolve()
    safe_model = cell_dir.parent.name
    quant_dir = cell_dir.name
    model_label = model_id or _reverse_model_safe(safe_model)
    quant_label = quant or quant_dir
    publish_quant = _publish_quant_label(quant_dir)

    # Load every dataset and enforce arch consistency.
    per_ds_counts: dict[str, np.ndarray] = {}
    per_ds_tokens: dict[str, int] = {}
    arch: dict[str, Any] | None = None
    skipped: list[str] = []
    for ds, ec in items:
        try:
            a, c, t = _load_one_dataset_counts(ec)
        except (ValueError, json.JSONDecodeError, OSError) as e:
            print(f"[warn] {ec}: {e}; skipping")
            skipped.append(ds)
            continue
        if arch is None:
            arch = a
        else:
            if (a["n_layer"] != arch["n_layer"]
                    or a["n_expert"] != arch["n_expert"]
                    or a["n_expert_used"] != arch["n_expert_used"]):
                print(f"[warn] {ec}: arch mismatch; skipping")
                skipped.append(ds)
                continue
        # Use the publishable dataset label as the key.
        publish_ds = _publish_dataset_label(ds)
        per_ds_counts[publish_ds] = c
        per_ds_tokens[publish_ds] = t

    if arch is None or len(per_ds_counts) < 2:
        # Need at least 2 datasets to compute pairwise Jaccard.
        return {"cell_dir": str(cell_dir), "skipped": True,
                "reason": f"datasets<2 (loaded={len(per_ds_counts)})"}

    L = arch["n_layer"]
    E = arch["n_expert"]
    K_model = arch["n_expert_used"]
    K_values = _make_k_values(E, K_model, k_min, k_max_frac, num_ks)

    # Sort dataset labels with the user-requested order at the front.
    preferred_order = ["BBH", "HumanEval", "INCLUDE", "MMLU", "PopQA"]
    labels = [d for d in preferred_order if d in per_ds_counts]
    labels += [d for d in sorted(per_ds_counts.keys()) if d not in labels]
    counts_list = [per_ds_counts[d] for d in labels]

    # Figure size for a square-ish heatmap (cap the height so 5x5 is OK).
    N = len(labels)
    width_in_use, height_in_use = _make_figure_size(N, width_in)

    # File prefix: model + quant in their publishable spelling, with
    # all filesystem-unfriendly chars replaced by underscores. This
    # embeds the cell identity directly in the filename (per spec).
    fname_base = _safe_filename(f"{model_label}__{publish_quant}")

    rendered: list[tuple[int, str]] = []
    skipped_files: list[str] = []
    for K in K_values:
        # Determine the alias (K_used, K2x, K3x, or "" for the base).
        if K == K_model:
            alias = ""
        elif K == min(2 * K_model, E) and 2 * K_model <= E:
            alias = "K2x"
        elif K == min(3 * K_model, E) and 3 * K_model <= E:
            alias = "K3x"
        else:
            continue  # mid-grid K; only the three are rendered

        if alias == "":
            out_png = output_dir / f"{fname_base}__jaccard_dataset.png"
        elif alias == "K2x":
            out_png = output_dir / f"{fname_base}__jaccard_dataset_K2x.png"
        else:
            out_png = output_dir / f"{fname_base}__jaccard_dataset_K3x.png"

        if out_png.exists() and not force:
            print(f"[skip] {out_png} exists; pass --force to regenerate")
            skipped_files.append(str(out_png))
            continue

        mat = _layer_averaged_pairwise_jaccard(counts_list, K, L)
        # 5x5 paper figure: keep the in-cell text (the matrix is small
        # enough that the labels stay legible at 300 DPI). For 6x6+
        # we drop the labels automatically (the matrix data IS the
        # message, the colourbar tells the scale).
        with_text = (N <= 6)
        fig = _render_paper_heatmap(
            mat, labels, width_in=width_in_use, height_in=height_in_use,
            cmap=cmap, cell_text=with_text, dpi=dpi,
        )
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        rendered.append((K, str(out_png)))
        print(f"[save] {out_png}  (K={K})")

    # ---- js_divergence_dataset.png (cross-dataset JS-divergence,
    # layer-averaged, single heatmap)
    js_png = output_dir / f"{fname_base}__js_divergence_dataset.png"
    if js_png.exists() and not force:
        print(f"[skip] {js_png} exists; pass --force to regenerate")
        skipped_files.append(str(js_png))
    else:
        js_mat = _layer_averaged_js_divergence_matrix(counts_list)
        fig = _render_paper_js_divergence_heatmap(
            js_mat, labels,
            width_in=width_in_use, height_in=height_in_use, dpi=dpi,
            cmap=jsd_cmap,
        )
        fig.savefig(js_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        rendered.append((-1, str(js_png)))
        print(f"[save] {js_png}  (layer-averaged JS divergence)")

    # ---------------- per-cell routing strip (optional)
    per_cell_png: str | None = None
    if per_cell:
        # Use the aggregated counts matrix from
        # `<model>/<quant>/overall/counts_total_overview.json` if it
        # exists; otherwise sum the per-dataset matrices ourselves.
        overview_p = cell_dir / "overall" / "counts_total_overview.json"
        if overview_p.is_file():
            try:
                arch_ov, counts_agg, _tokens = _load_one_quant_overview(overview_p)
            except (ValueError, json.JSONDecodeError, OSError) as e:
                print(f"[warn] {overview_p}: {e}; falling back to per-dataset sum")
                counts_agg = np.zeros((L, E), dtype=np.int64)
                for c in per_ds_counts.values():
                    counts_agg += c
                arch_ov = arch
        else:
            counts_agg = np.zeros((L, E), dtype=np.int64)
            for c in per_ds_counts.values():
                counts_agg += c
            arch_ov = arch
        out_png = output_dir / f"{fname_base}__routing_strip.png"
        if out_png.exists() and not force:
            print(f"[skip] {out_png} exists; pass --force to regenerate")
            skipped_files.append(str(out_png))
        else:
            # Per-cell routing strip: raw counts (no log1p) with
            # layer & expert indices on the axes. Helper handles the
            # adaptive height, the integer tick selection, and the
            # millions-formatter on the colorbar.
            fig = _render_paper_routing_strip(
                counts_agg, width_in=width_in, dpi=dpi, cmap=cmap,
            )
            fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            per_cell_png = str(out_png)
            print(f"[save] {out_png}  (routing strip L={L} E={E})")

    return {"cell_dir": str(cell_dir), "skipped": False,
            "model": model_label, "quant": publish_quant,
            "n_datasets": N, "labels": labels, "k_values": K_values,
            "rendered": rendered, "skipped_files": skipped_files,
            "datasets_skipped_from_load": skipped,
            "per_cell_png": per_cell_png, "arch": arch}


# ------------------------------------------------------ cross-quantization view

def _iter_quant_overviews(model_dir: Path
                           ) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    """Return `[(quant_label, counts_total, arch), ...]` for one model."""
    out: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    if not model_dir.is_dir():
        return out
    for sub in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        p = sub / "overall" / "counts_total_overview.json"
        if not p.is_file():
            continue
        try:
            a, c, _t = _load_one_quant_overview(p)
        except (ValueError, json.JSONDecodeError, OSError) as e:
            print(f"[warn] {p}: {e}; skipping")
            continue
        out.append((sub.name, c, a))
    return out


def _process_one_cross_quant(
        model_dir: Path, *, output_dir: Path,
        width_in: float, dpi: int, cmap: str,
        k_min: int, k_max_frac: float, num_ks: int,
        model_id: str | None = None,
        force: bool = False,
        jsd_cmap: str = DEFAULT_JSD_CMAP,
        ) -> dict[str, Any]:
    """Process one model directory for cross-quant Jaccard heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = model_dir.resolve()
    safe_model = model_dir.name
    model_label = model_id or _reverse_model_safe(safe_model)

    items = _iter_quant_overviews(model_dir)
    if len(items) < 2:
        return {"model_dir": str(model_dir), "skipped": True,
                "reason": f"quants<2 (loaded={len(items)})"}

    per_quant_counts: dict[str, np.ndarray] = {}
    per_quant_tokens: dict[str, int] = {}
    arch: dict[str, Any] | None = None
    for q_label, counts, a in items:
        if arch is None:
            arch = a
        else:
            if (a["n_layer"] != arch["n_layer"]
                    or a["n_expert"] != arch["n_expert"]
                    or a["n_expert_used"] != arch["n_expert_used"]):
                print(f"[warn] {model_dir}/{q_label}: arch mismatch; skipping")
                continue
        publish_q = _publish_quant_label(q_label)
        per_quant_counts[publish_q] = counts
        per_quant_tokens[publish_q] = int(counts.sum())
    if arch is None or len(per_quant_counts) < 2:
        return {"model_dir": str(model_dir), "skipped": True,
                "reason": "quants<2 after arch filtering"}

    L = arch["n_layer"]
    E = arch["n_expert"]
    K_model = arch["n_expert_used"]
    K_values = _make_k_values(E, K_model, k_min, k_max_frac, num_ks)

    # Sort quant labels with the most-aggressive quantization first
    # (typical paper convention: Q2_K, Q4_K_M, BF16, ...).
    quant_priority = [
        "FP8", "BF16", "F16", "F32",
        "Q8_0", "Q8_K", "Q6_K", "Q6_0", "Q6_K_S",
        "Q5_1", "Q5_K_M", "Q5_K_S", "Q5_0",
        "Q4_1", "Q4_K_M", "Q4_K_S", "Q4_0",
        "Q3_K_L", "Q3_K_M", "Q3_K_S",
        "Q2_K", "Q2_K_S",
    ]
    labels = [q for q in quant_priority if q in per_quant_counts]
    labels += [q for q in sorted(per_quant_counts.keys()) if q not in labels]
    counts_list = [per_quant_counts[q] for q in labels]

    N = len(labels)
    width_in_use, height_in_use = _make_figure_size(N, width_in,
                                                     cells_per_inch=4.0)
    fname_base = _safe_filename(f"{model_label}")

    rendered: list[tuple[int, str]] = []
    skipped_files: list[str] = []
    for K in K_values:
        if K == K_model:
            alias = ""
        elif K == min(2 * K_model, E) and 2 * K_model <= E:
            alias = "K2x"
        elif K == min(3 * K_model, E) and 3 * K_model <= E:
            alias = "K3x"
        else:
            continue
        if alias == "":
            out_png = output_dir / f"{fname_base}__jaccard_quant.png"
        elif alias == "K2x":
            out_png = output_dir / f"{fname_base}__jaccard_quant_K2x.png"
        else:
            out_png = output_dir / f"{fname_base}__jaccard_quant_K3x.png"

        if out_png.exists() and not force:
            print(f"[skip] {out_png} exists; pass --force to regenerate")
            skipped_files.append(str(out_png))
            continue
        mat = _layer_averaged_pairwise_jaccard(counts_list, K, L)
        with_text = (N <= 6)
        fig = _render_paper_heatmap(
            mat, labels, width_in=width_in_use, height_in=height_in_use,
            cmap=cmap, cell_text=with_text, dpi=dpi,
        )
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        rendered.append((K, str(out_png)))
        print(f"[save] {out_png}  (K={K})")

    # ---- js_divergence_quant.png (cross-quant JS-divergence,
    # layer-averaged, single heatmap). Same convention as the
    # cross-quant viewport in `cross_quant_jaccard_sweep_from_cpp.py`.
    js_png = output_dir / f"{fname_base}__js_divergence_quant.png"
    if js_png.exists() and not force:
        print(f"[skip] {js_png} exists; pass --force to regenerate")
        skipped_files.append(str(js_png))
    else:
        js_mat = _layer_averaged_js_divergence_matrix(counts_list)
        fig = _render_paper_js_divergence_heatmap(
            js_mat, labels,
            width_in=width_in_use, height_in=height_in_use, dpi=dpi,
            cmap=jsd_cmap,
        )
        fig.savefig(js_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        rendered.append((-1, str(js_png)))
        print(f"[save] {js_png}  (layer-averaged JS divergence)")

    return {"model_dir": str(model_dir), "skipped": False,
            "model": model_label, "n_quants": N, "labels": labels,
            "k_values": K_values, "rendered": rendered,
            "skipped_files": skipped_files, "js_png": str(js_png),
            "arch": arch}


# ----------------------------------------------------------------- filename safe

def _safe_filename(s: str) -> str:
    """Make a string safe for use as a filename (no '/' etc).

    We deliberately keep `.`, `-`, and `_` since the model id and
    quant names are expected to contain them. Everything else that
    isn't an alphanumeric / `_` / `-` / `.` is replaced with `_`.
    Multiple `_` are collapsed.
    """
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    out = re.sub(r"_+", "_", out)
    return out.strip("_")


# ---------------------------------------------------------- iter & dispatch

def _iter_cells_multi(results_dir: Path, only_models: list[str] | None,
                      only_quants: list[str] | None
                      ) -> list[tuple[Path, str | None, str | None]]:
    """Walk `<results_dir>/<model>/<quant>/moe-*/expert_counts.json`.

    Returns a list of `(cell_dir, model_id, quant)` tuples for every
    model × quant combo that has at least one dataset.
    """
    out: list[tuple[Path, str | None, str | None]] = []
    if not results_dir.is_dir():
        return out
    for mdir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        # `m.name` is `org--repo-GGUF` (filesystem-safe form).
        # The HF model id is the inverse mapping (`org/repo-GGUF`).
        model_id = _reverse_model_safe(mdir.name)
        if only_models and model_id not in only_models:
            continue
        # Iterate every <quant> subdirectory.
        for qdir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            quant = qdir.name
            if only_quants and quant not in only_quants:
                continue
            # Treat the model_dir itself as a single "no-quant" cell if
            # it directly contains `moe-*/expert_counts.json` (legacy
            # layout). Matches `aggregate_overview`'s convention.
            has_dataset = any(qdir.glob("moe-*/expert_counts.json"))
            if has_dataset:
                out.append((qdir, model_id, quant))
            elif mdir == qdir and any(mdir.glob("moe-*/expert_counts.json")):
                # Don't walk the model dir twice; skip if we already
                # covered it.
                out.append((mdir, model_id, None))
    return out


def _iter_models_multi(results_dir: Path, only_models: list[str] | None
                       ) -> list[tuple[Path, str]]:
    """Yield `(model_dir, model_id)` for every model dir with >= 2 quants."""
    out: list[tuple[Path, str]] = []
    if not results_dir.is_dir():
        return out
    for mdir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        model_id = _reverse_model_safe(mdir.name)
        if only_models and model_id not in only_models:
            continue
        # Count <quant>/overall/counts_total_overview.json children.
        n = sum(1 for q in mdir.iterdir()
                if q.is_dir() and (q / "overall" / "counts_total_overview.json").is_file())
        if n >= 2:
            out.append((mdir, model_id))
    return out


# --------------------------------------------------------------------- main

def _setup_mpl(dpi: int) -> None:
    """Configure matplotlib rcParams for paper figures.

    - Strip every default label color / padding.
    - Disable the default font-weight / family overrides that paper
      authors usually want at the LaTeX side.
    """
    mpl.rcParams.update({
        "savefig.dpi": dpi,
        "figure.dpi": dpi,
        "font.size": 8,
        "axes.titlesize": 0,       # never used; defensive.
        "axes.labelsize": 0,       # never used; defensive.
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,        # embed real fonts (paper convention).
        "ps.fonttype": 42,
    })


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Root directory containing per-model "
                             "subdirectories (recommended mode).")
    parser.add_argument("--cell-dir", type=Path, default=None,
                        help="Single (model, quant) cell directory "
                             "(debug mode).")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Single model directory for cross-quant view "
                             "(debug mode).")
    parser.add_argument("--models", action="append", default=None,
                        help="Restrict to a subset of HF model ids "
                             "(repeatable, used with --results-dir).")
    parser.add_argument("--quants", action="append", default=None,
                        help="Restrict to a subset of quant subdirectories "
                             "(repeatable).")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model id (single-cell modes only).")
    parser.add_argument("--quant", type=str, default=None,
                        help="Override quant tag (single-cell modes only).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: "
                             "<results-dir>/paper_heatmaps for multi-mode, "
                             "<cell-dir>/paper_heatmaps for single-cell).")
    parser.add_argument("--k-min", type=int, default=1,
                        help="Minimum K in the sweep (default: 1).")
    parser.add_argument("--k-max-frac", type=float, default=0.50,
                        help="K_max = ceil(n_expert * k_max_frac) "
                             "(default: 0.50).")
    parser.add_argument("--num-ks", type=int, default=21,
                        help="Number of K values in the linear sweep "
                             "(default: 21).")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PNG DPI for the heatmap (default: 300).")
    parser.add_argument("--cmap", type=str, default=DEFAULT_CMAP,
                        help=f"Colormap for the Jaccard / routing-strip "
                             f"heatmaps (default: {DEFAULT_CMAP}).")
    parser.add_argument("--jsd-cmap", type=str, default=DEFAULT_JSD_CMAP,
                        help=f"Colormap for the JS-divergence heatmaps "
                             f"(default: {DEFAULT_JSD_CMAP}). The default "
                             f"is `viridis_r` (perceptually uniform, "
                             f"multi-hue) — alternatives include "
                             f"`magma_r`, `inferno_r`, `cividis`. "
                             f"Try a different one if the small-value "
                             f"variations are hard to see in your "
                             f"specific dataset.")
    parser.add_argument("--width-in", type=float, default=DEFAULT_WIDTH_IN,
                        help=f"Figure width in inches "
                             f"(default: {DEFAULT_WIDTH_IN}).")
    parser.add_argument("--per-cell", action="store_true",
                        help="Also emit one routing-strip heatmap per "
                             "(model, quant) cell.")
    parser.add_argument("--skip-cross-quant", action="store_true",
                        help="Skip the cross-quantization view.")
    parser.add_argument("--skip-cross-dataset", action="store_true",
                        help="Skip the cross-dataset view.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing PNGs.")
    args = parser.parse_args()

    # Single-cell modes can't use --results-dir or other filters.
    if args.cell_dir and args.results_dir:
        raise SystemExit("[error] --cell-dir and --results-dir mutually "
                         "exclusive")
    if args.model_dir and args.results_dir:
        raise SystemExit("[error] --model-dir and --results-dir mutually "
                         "exclusive")
    if args.cell_dir and args.model_dir:
        raise SystemExit("[error] --cell-dir and --model-dir mutually "
                         "exclusive")

    _setup_mpl(args.dpi)

    # ---- single-cell mode (debug) ---------------------------------------
    if args.cell_dir:
        if args.output_dir is None:
            out_dir = args.cell_dir.resolve() / "paper_heatmaps"
        else:
            out_dir = args.output_dir
        res = _process_one_dataset_cell(
            args.cell_dir, output_dir=out_dir,
            width_in=args.width_in, dpi=args.dpi, cmap=args.cmap,
            k_min=args.k_min, k_max_frac=args.k_max_frac,
            num_ks=args.num_ks, model_id=args.model_id, quant=args.quant,
            force=args.force, per_cell=args.per_cell,
            jsd_cmap=args.jsd_cmap,
        )
        print(f"[done] {res}")
        return 0

    if args.model_dir:
        if args.output_dir is None:
            out_dir = args.model_dir.resolve() / "paper_heatmaps"
        else:
            out_dir = args.output_dir
        res = _process_one_cross_quant(
            args.model_dir, output_dir=out_dir,
            width_in=args.width_in, dpi=args.dpi, cmap=args.cmap,
            k_min=args.k_min, k_max_frac=args.k_max_frac,
            num_ks=args.num_ks, model_id=args.model_id, force=args.force,
            jsd_cmap=args.jsd_cmap,
        )
        print(f"[done] {res}")
        return 0

    # ---- multi-mode -----------------------------------------------------
    if args.results_dir is None:
        raise SystemExit("[error] either --cell-dir, --model-dir, or "
                         "--results-dir is required (use --help for full "
                         "options)")
    results_dir = args.results_dir
    if args.output_dir is None:
        out_dir = results_dir / "paper_heatmaps"
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cells = 0
    n_cell_failures = 0
    n_models = 0
    n_model_failures = 0

    if not args.skip_cross_dataset:
        cells = _iter_cells_multi(results_dir, args.models, args.quants)
        if cells:
            print(f"[plan] {len(cells)} cell(s) for cross-dataset view")
            for cell_dir, model_id, quant in cells:
                # Output one subdirectory per model (so models with
                # many quants don't drown out small ones).
                cell_out = out_dir / _safe_filename(
                    model_id or _reverse_model_safe(cell_dir.parent.name)
                )
                cell_out.mkdir(parents=True, exist_ok=True)
                print(f"\n[cell] {cell_dir}  (model={model_id}, quant={quant})")
                try:
                    r = _process_one_dataset_cell(
                        cell_dir, output_dir=cell_out,
                        width_in=args.width_in, dpi=args.dpi,
                        cmap=args.cmap,
                        k_min=args.k_min, k_max_frac=args.k_max_frac,
                        num_ks=args.num_ks,
                        model_id=model_id, quant=quant,
                        force=args.force, per_cell=args.per_cell,
                        jsd_cmap=args.jsd_cmap,
                    )
                    n_cells += 1
                    if r.get("skipped"):
                        print(f"[skip-cell] reason={r.get('reason')}")
                except SystemExit:
                    raise
                except Exception as e:  # noqa: BLE001
                    print(f"[error] {cell_dir}: {e}", file=sys.stderr)
                    n_cell_failures += 1
        else:
            print(f"[plan] no cells found under {results_dir}")

    if not args.skip_cross_quant:
        models = _iter_models_multi(results_dir, args.models)
        if models:
            print(f"\n[plan] {len(models)} model(s) for cross-quant view")
            for model_dir, model_id in models:
                model_out = out_dir / _safe_filename(model_id)
                model_out.mkdir(parents=True, exist_ok=True)
                print(f"\n[model] {model_dir}  (model_id={model_id})")
                try:
                    r = _process_one_cross_quant(
                        model_dir, output_dir=model_out,
                        width_in=args.width_in, dpi=args.dpi,
                        cmap=args.cmap,
                        k_min=args.k_min, k_max_frac=args.k_max_frac,
                        num_ks=args.num_ks,
                        model_id=model_id, force=args.force,
                        jsd_cmap=args.jsd_cmap,
                    )
                    n_models += 1
                    if r.get("skipped"):
                        print(f"[skip-model] reason={r.get('reason')}")
                except SystemExit:
                    raise
                except Exception as e:  # noqa: BLE001
                    print(f"[error] {model_dir}: {e}", file=sys.stderr)
                    n_model_failures += 1
        else:
            print(f"[plan] no models with >=2 quants under {results_dir}")

    failures = n_cell_failures + n_model_failures
    print(f"\n[done] cross-dataset: {n_cells} cell(s), "
          f"{n_cell_failures} failure(s); "
          f"cross-quant: {n_models} model(s), {n_model_failures} failure(s); "
          f"outputs under {out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
