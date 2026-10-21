#!/usr/bin/env python3
# type: ignore

"""Publication-quality cross-quantization K-sweep figure for the MoE-routing paper.

Cross-quant counterpart of `paper_jaccard_global_sweep.py` (which is
its cross-dataset twin). For each model, this script renders the
**mean pairwise Jaccard** as a function of K (top-K experts per
quantization, K = K_model marker):

    A_q_K[l]  = top-K experts in quant q of model m, layer l
                (descending count, ties -> ascending expert id)
    For each (K, layer l):
      J_K[l][i,j] = |A_i_K[l] ∩ A_j_K[l]| / |A_i_K[l] ∪ A_j_K[l]|
                    averaged across the C(N_q, 2) off-diagonal quant pairs
    Bold mean line:  mean over the L layers of J_K[l]
    Thin per-layer traces:  one trace per layer

Layout: 1 row x N_models columns (shared K axis). Each column shows
one model. K_model is drawn as a vertical dotted line on every column
AND labeled in-axes ("K = K_model = N") right above the curve at the
top, plus a shared figure-level legend at the bottom explaining
the per-layer trace, bold mean line, and K_model marker. The
generalized |∩|/|∪| (strict consensus) curve is intentionally NOT
included: it collapses to 0 the moment one dataset (here, one quant)
diverges, drowning the signal in noise — the mean pairwise Jaccard
already conveys the same information without that floor.

Output (under `<output-dir>/`, default
`<results-dir>/paper_heatmaps/cross_quant_sweep/`):

  - `cross_quant_jaccard_sweep.png`     the 1 x N paper figure (300 DPI)

No CSV outputs — single-responsibility script (figure only). The
underlying cross-quant Jaccard matrices at K = K_model are produced
by `cross_quantization_comparison.py` and tabulated by
`cross_quant_matrix_table.py`.

Mathematical contract (identical to `paper_heatmaps.py::` and the
existing eval-moe-overview Jaccard helpers):

  Off-diagonal mean (i != j), per layer:
    mean_offdiag_K[l] = mean over all C(N_q, 2) pairs
                        (i, j) with i != j of J_K[l][i, j]
  Per-layer traces:  L lines, one per layer l, of mean_offdiag_K[l].
  Mean line:        mean over l of mean_offdiag_K[l].
  Off-diagonal convention matches the cross-quantization heatmap
  (which also averages J across all C(N_q, 2) off-diagonal pairs at
  K = K_model): at K = K_model the bold line's value equals the
  off-diagonal mean of the heatmap's Jaccard matrix to floating-point
  precision.

Edge cases (mirrors `paper_heatmaps._jaccard`):
  both empty sets        -> 1.0
  only one non-empty set -> 0.0
  general case           -> |a ∩ b| / |a ∪ b|
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# Make the sibling `paper_heatmaps.py` and `cross_quantization_comparison.py`
# importable. All three files live in `examples/eval-heatmaps/` so
# Python's per-script sys.path entry resolves these imports without
# any setup.py / PYTHONPATH plumbing.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paper_heatmaps import (  # noqa: E402
    _jaccard,
    _make_k_values,
    _top_k_set_for_layer,
)

from cross_quantization_comparison import (  # noqa: E402
    DEFAULT_MODELS,
    _compute_model_cross_quant_matrices,
    _discover_quants,
    _load_one_quant_overview_safe,
)


# ----------------------------------------------------------------- constants

# Figure sizing. 1 row x N_cols means the figure is wider than tall.
# Subplots end up ~2in wide each which is roomy enough for K tick
# labels + the in-axes K_model annotation without crowding.
FIGURE_WIDTH_IN: float = 11.0
FIGURE_HEIGHT_IN: float = 3.5
DPI: int = 300

# Color scheme (same as paper_jaccard_global_sweep.py so the two
# paper K-sweep figures read as siblings to reviewers).
LAYER_COLOR: str = "#4477aa"
LAYER_ALPHA: float = 0.18
LAYER_LINEWIDTH: float = 0.7
MEAN_COLOR: str = "#1f3a5f"
MEAN_LINEWIDTH: float = 2.4
K_MODEL_COLOR: str = "#888888"
REFERENCE_COLOR: str = "#cccccc"

# y-axis range. Allows headroom for the K_model annotation sitting
# just under y = 1.0.
Y_MIN: float = -0.02
Y_MAX: float = 1.05

# Font sizes, matched to `cross_quantization_comparison.py` so this
# figure and the cross-quantization heatmap read as one system when
# they sit side by side in the paper:
#   - tick text            == heatmap's `tick_label_fontsize`
#   - column titles        == heatmap's `ax.set_title(...)`
#   - gutter + axis labels == heatmap's `row_label_fontsize`
#   - legend text         == heatmap's `ax.set_title(...)` (9), one
#     step up from the tick text: the legend is the figure's only
#     prose, so it carries more meaning per character than an axis
#     number and is legible from further away.
TICK_FONTSIZE: int = 8
TITLE_FONTSIZE: int = 9
LEGEND_FONTSIZE: int = 9
ROW_LABEL_FONTSIZE: int = 10

# Default CLI K-sweep grid (matches paper_jaccard_global_sweep.py).
DEFAULT_K_MIN: int = 1
DEFAULT_K_MAX_FRAC: float = 0.50
DEFAULT_NUM_KS: int = 21


# ---------------------------------------------------------- K-sweep math

def _pairwise_jaccard_per_layer(counts_list: list[np.ndarray], K: int, L: int
                                  ) -> np.ndarray:
    """Return (L, N, N) pairwise Jaccard tensor at fixed K.

    L[i, q, r] = Jaccard(top-K of counts_list[q][i], top-K of counts_list[r][i])
    Diagonal is 1.0 by construction. Identical layer-by-layer computation
    to `paper_heatmaps._layer_averaged_pairwise_jaccard`, kept as a
    separate tensor so the K-sweep can compute per-layer means without
    re-running the O(N_q^2 * L) top-K extraction per K.
    """
    N_q = len(counts_list)
    if N_q == 0:
        return np.zeros((L, 0, 0), dtype=np.float64)
    jk = np.zeros((L, N_q, N_q), dtype=np.float64)
    # Pre-compute per-(quant, layer) top-K sets.
    sets: list[list[set[int]]] = [
        [_top_k_set_for_layer(c, layer, K) for layer in range(L)]
        for c in counts_list
    ]
    for layer in range(L):
        # Diagonal = 1.0
        for q in range(N_q):
            jk[layer, q, q] = 1.0
        # Off-diagonal symmetric
        for q in range(N_q):
            for r in range(q + 1, N_q):
                v = _jaccard(sets[q][layer], sets[r][layer])
                jk[layer, q, r] = v
                jk[layer, r, q] = v
    return jk


def _offdiag_pairwise_mean_per_layer(jk: np.ndarray
                                       ) -> np.ndarray:
    """Reduce (L, N_q, N_q) -> (L,) by averaging over the C(N_q, 2)
    off-diagonal entries per layer. Diagonal (=1.0) is excluded.

    Concretely:
      mean_offdiag_K[l] = (1 / C(N_q, 2)) * sum over (q != r) of J[l, q, r]

    `jk` is symmetric (set at both [q,r] and [r,q]) with diagonal 1.0,
    so `jk.sum - N_q = 2 * sum over unique off-diag entries`. Dividing
    by `N_q * (N_q - 1) = 2 * C(N_q, 2)` yields the **mean over the
    C(N_q, 2) unique pairs**, not double-counted.

    Matches the convention used by the cross-quantization heatmap
    (which reports the same mean at K = K_model in its off-diagonal
    cells).
    """
    L, N_q, _ = jk.shape
    if N_q < 2:
        return np.full(L, 1.0 if N_q == 1 else np.nan, dtype=np.float64)
    s = jk.sum(axis=(1, 2))  # (L,)  -> includes diagonal N_q copies of 1.0
    n_off = N_q * (N_q - 1)  # = 2 * C(N_q, 2)
    return (s - N_q) / n_off


def _compute_k_sweep(counts: list[np.ndarray], K_values: list[int], L: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Run the full K-sweep.

    Returns:
      per_layer:  (n_K, L) array. `per_layer[k_idx, l]` is the
                  off-diagonal mean Jaccard at K_values[k_idx] for
                  layer l.
      mean_curve: (n_K,) array. `mean_curve[k_idx]` is the layer-
                  averaged value (mean over l of per_layer[k_idx, l]).

    The K-sweep is O(num_ks * L * N_q^2) on top-K extraction +
    set ops, dominated by the top-K itself. Same `np.mean`
    convention as `_layer_averaged_pairwise_jaccard`, so the
    bold line at K = K_model equals the heatmap's off-diagonal
    mean to floating-point precision.
    """
    n_K = len(K_values)
    per_layer = np.zeros((n_K, L), dtype=np.float64)
    for k_idx, K in enumerate(K_values):
        jk = _pairwise_jaccard_per_layer(counts, K, L)
        per_layer[k_idx] = _offdiag_pairwise_mean_per_layer(jk)
    mean_curve = per_layer.mean(axis=1)
    return per_layer, mean_curve


# --------------------------------------------------------------- render

def _cross_quant_jaccard_reference(*, model_dir_ref: Path, K_model: int,
                                     n_layers: int, n_quants: int,
                                     only_quants: list[str] | None = None,
                                     ) -> float:
    """Off-diagonal mean of the cross-quantization heatmap's Jaccard
    matrix, recomputed from scratch via the *heatmap's own* loader.

    This is the independent reference the `--self-check` compares the
    K-sweep's bold-line value against. It deliberately goes through
    `cross_quantization_comparison._compute_model_cross_quant_matrices`
    (the exact function the heatmap script calls) rather than reusing
    anything from this module, so agreement is real evidence that the
    two views describe the same quantity.

    Raises SystemExit when the heatmap's quant count differs from
    `n_quants` — that would mean this script is sweeping a different
    quant set than the heatmap plots, which invalidates the check.
    """
    data = _compute_model_cross_quant_matrices(model_dir_ref, only_quants)
    if data is None:
        raise SystemExit(
            f"[error] {model_dir_ref}: heatmap matrices unavailable; "
            f"cannot self-check"
        )
    mat = data["jaccard"]
    n = int(mat.shape[0])
    if n != n_quants:
        raise SystemExit(
            f"[error] {model_dir_ref}: self-check quant-set mismatch — "
            f"K-sweep swept {n_quants} quants but the heatmap has {n}. "
            f"Pass the same --quants filter to both scripts."
        )
    diag_sum = float(np.trace(mat))
    return (float(mat.sum()) - diag_sum) / (n * (n - 1))


def render_paper_cross_quant_jaccard_sweep(
        per_model: dict[str, dict[str, Any]],
        output_dir: Path,
        *,
        width_in: float, height_in: float, dpi: int,
        k_min: int, k_max_frac: float, num_ks: int,
        force: bool = False,
        self_check: bool = False,
        model_dir_ref_for: dict[str, Path] | None = None,
        only_quants: list[str] | None = None,
        ) -> dict[str, Any]:
    """Render the publication-quality cross-quant Jaccard K-sweep PNG.

    `per_model`: dict mapping short_name -> {
        "model_safe": str,
        "arch": {name, n_layer, n_expert, n_expert_used},
        "counts_list": list[np.ndarray],  # one [L, E] per quant
        "quant_labels": list[str],
    }
    Returns a summary dict for stdout reporting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "cross_quant_jaccard_sweep.png"
    # `--self-check` needs the sweep to actually execute, so it implies
    # `--force` — otherwise the skip-if-exists early-return below would
    # silently skip the very computation being verified.
    force = force or self_check
    if out_png.exists() and not force:
        print(f"[skip] {out_png} exists; pass --force to regenerate")
        return {"output_dir": str(output_dir), "skipped": True}

    n_models = len(per_model)
    if n_models < 1:
        raise SystemExit("[error] no models loaded successfully")

    # Each model's K grid is computed independently because `n_expert`
    # (and therefore the grid's upper bound, ceil(E * k_max_frac))
    # differs across models. We do NOT force a shared x range: each
    # panel is normalized to its own K sweep so the curve fills the
    # panel width regardless of expert count.
    k_grids: dict[str, list[int]] = {}
    archs: dict[str, dict[str, Any]] = {}
    for short_name, data in per_model.items():
        arch = data["arch"]
        archs[short_name] = arch
        K_model = arch["n_expert_used"]
        E = arch["n_expert"]
        K_values = _make_k_values(E, K_model, k_min, k_max_frac, num_ks)
        k_grids[short_name] = K_values

    fig, axes = plt.subplots(
        1, n_models, figsize=(width_in, height_in), dpi=dpi,
        sharex=False, sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    # Wrap a single subplot in a 1-tuple so the loop below is uniform.
    if n_models == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes)

    # K_grid is the same `n_points` per model but may have different
    # integer values per model (since E differs). Render per-model.
    for ax, (short_name, data) in zip(axes_list, per_model.items()):
        arch = archs[short_name]
        K_values = k_grids[short_name]
        K_model = arch["n_expert_used"]
        L = arch["n_layer"]
        E = arch["n_expert"]
        counts = data["counts_list"]
        quant_labels = data["quant_labels"]

        per_layer, mean_curve = _compute_k_sweep(counts, K_values, L)
        K_arr = np.asarray(K_values, dtype=np.float64)

        # ---- Optional self-check: the bold-line value at
        # K = K_model must equal the off-diagonal mean of the
        # cross-quantization heatmap's Jaccard matrix for this
        # model. Both derive from the same per-layer top-K sets via
        # the same `np.mean`, so they must agree to floating-point
        # precision. A mismatch means the K-sweep's quant set (or
        # its off-diagonal reduction) drifted away from the heatmap.
        if self_check:
            ref_dir = (model_dir_ref_for or {}).get(short_name)
            if ref_dir is None:
                raise SystemExit(
                    f"[error] --self-check needs the source model "
                    f"directory for {short_name} but none was recorded; "
                    f"call render_paper_cross_quant_jaccard_sweep() with "
                    f"model_dir_ref_for."
                )
            ref = _cross_quant_jaccard_reference(model_dir_ref=ref_dir,
                                                 K_model=K_model,
                                                 n_layers=L,
                                                 n_quants=len(counts),
                                                 only_quants=only_quants)
            diff = abs(mean_curve[K_values.index(K_model)] - ref)
            status = "OK" if diff <= 1e-12 else "MISMATCH"
            print(f"[self-check] {short_name}: "
                  f"curve@K_model={mean_curve[K_values.index(K_model)]:.12f} "
                  f"heatmap_offdiag_mean={ref:.12f} "
                  f"diff={diff:.2e} [{status}]")
            if diff > 1e-12:
                raise SystemExit(
                    f"[error] {short_name}: bold line at K_model disagrees "
                    f"with the cross-quantization heatmap by {diff:.3e} "
                    f"(> 1e-12); the figure would contradict the heatmap."
                )

        # Thin per-layer traces.
        for layer in range(L):
            ax.plot(K_arr, per_layer[:, layer],
                    color=LAYER_COLOR, alpha=LAYER_ALPHA,
                    linewidth=LAYER_LINEWIDTH)
        # Bold mean-across-layers line.
        ax.plot(K_arr, mean_curve,
                color=MEAN_COLOR, linewidth=MEAN_LINEWIDTH,
                marker="o", markersize=4)
        # K_model vertical marker.
        ax.axvline(K_model, color=K_MODEL_COLOR, linestyle=":",
                   linewidth=0.8)
        # Perfect-agreement horizontal reference.
        ax.axhline(1.0, color=REFERENCE_COLOR, linestyle="--",
                   linewidth=0.6)
        # In-axes K_model annotation. Places the text just under
        # y = 1 at the K_model x position so it sits ABOVE the curve
        # (curves are mostly < 1.0 at K_model). Use a small offset so
        # it doesn't kiss the reference line.
        ax.text(K_model, Y_MAX - 0.04, f"K={K_model}",
                ha="center", va="top", fontsize=TICK_FONTSIZE,
                color=K_MODEL_COLOR)
        # Clean axes.
        ax.set_ylim(Y_MIN, Y_MAX)
        # Keep each subplot's x extent local so the K_model line stays
        # roughly centered. Reset the default xlim after plotting so
        # internal calls (axvline + markers) don't fight it.
        ax.set_xlim(K_arr.min() - (K_arr.max() - K_arr.min()) * 0.04,
                    K_arr.max() + (K_arr.max() - K_arr.min()) * 0.04)
        ax.tick_params(axis="both", length=0, labelsize=TICK_FONTSIZE)
        # K grids differ by an order of magnitude across models
        # (Mixtral tops out at K=6, Qwen3 at K=64), so a fixed tick
        # count would crowd the wide panels and leave the narrow one
        # nearly blank. Cap the density instead.
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=6, integer=True, prune="both"))
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Column title = short model name + arch line (kept tight so
        # readers get the model + K_model in one glance).
        ax.set_title(f"{short_name}\nL={L} E={E} K={K_model}",
                     fontsize=TITLE_FONTSIZE, pad=4)
        # Per-column x label. Each panel sweeps its own K grid (E
        # differs by 16x across models), so the axis name belongs to
        # each panel rather than to the figure — this also keeps it
        # clear of the figure-level legend below.
        ax.set_xlabel("K", fontsize=ROW_LABEL_FONTSIZE, labelpad=6)
        # Gutter row label, matching the cross-quantization heatmap's
        # `_draw_one_heatmap(row_label=...)` styling (rotated, left of
        # the axes, same font size). Only the leftmost panel is
        # labeled; the rest share the y axis.
        if ax is axes_list[0]:
            ax.set_ylabel("Jaccard Similarity",
                          fontsize=ROW_LABEL_FONTSIZE, rotation=90,
                          labelpad=8, va="center")

    # One figure-level legend explaining the trace / mean line /
    # K_model marker. Bottom of the figure, centered. Use Line2D
    # handles so the marker styles match the actual drawing.
    legend_handles = [
        Line2D([0], [0], color=LAYER_COLOR, alpha=LAYER_ALPHA,
               linewidth=LAYER_LINEWIDTH, label="Per-Layer Mean"),
        Line2D([0], [0], color=MEAN_COLOR, linewidth=MEAN_LINEWIDTH,
               marker="o", markersize=4, label="Layer-Averaged Mean"),
        Line2D([0], [0], color=K_MODEL_COLOR, linestyle=":",
               linewidth=0.8, label="K = K_model"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, frameon=False, fontsize=LEGEND_FONTSIZE,
               bbox_to_anchor=(0.5, 0.0))

    # `bbox_inches="tight"` at save time overrides figure margins, so
    # reserve the gutters explicitly: the extra bottom band holds the
    # per-column "K" label and the legend beneath it, the extra left
    # band holds the rotated "Jaccard Similarity" gutter label.
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.24,
                        wspace=0.08)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_png}")

    return {
        "output_dir": str(output_dir),
        "n_models": n_models,
        "k_grids": k_grids,
        "skipped": False,
    }


# ------------------------------------------------------------------- CLI

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("/mnt/data/results-llama-cpp"),
        help="Root results directory containing per-model subdirectories "
             "(default: /mnt/data/results-llama-cpp).",
    )
    parser.add_argument(
        "--models", action="append", default=None,
        help="Restrict to a subset of filesystem-safe model ids "
             "(repeatable). Default: the 4 hardcoded models from "
             "cross_quantization_comparison.DEFAULT_MODELS so the "
             "paper figure matches the cross-quantization heatmap.",
    )
    parser.add_argument(
        "--quants", action="append", default=None,
        help="Restrict to a subset of quant subdirectories "
             "(repeatable). Default: all quants with valid "
             "counts_total_overview.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory "
             "(default: <results-dir>/paper_heatmaps/cross_quant_sweep).",
    )
    parser.add_argument(
        "--width-in", type=float, default=FIGURE_WIDTH_IN,
        help=f"Figure width in inches (default: {FIGURE_WIDTH_IN}).",
    )
    parser.add_argument(
        "--height-in", type=float, default=FIGURE_HEIGHT_IN,
        help=f"Figure height in inches (default: {FIGURE_HEIGHT_IN}).",
    )
    parser.add_argument(
        "--dpi", type=int, default=DPI,
        help=f"Output DPI (default: {DPI}).",
    )
    parser.add_argument(
        "--k-min", type=int, default=DEFAULT_K_MIN,
        help="Minimum K in the sweep (default: 1).",
    )
    parser.add_argument(
        "--k-max-frac", type=float, default=DEFAULT_K_MAX_FRAC,
        help="K_max = ceil(n_expert * k_max_frac) (default: 0.50). "
             "Same default as paper_jaccard_global_sweep.py so the "
             "two K-sweep figures share their K grid conventions.",
    )
    parser.add_argument(
        "--num-ks", type=int, default=DEFAULT_NUM_KS,
        help="Number of K values in the linear sweep (default: 21).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-render even if cross_quant_jaccard_sweep.png "
             "already exists.",
    )
    parser.add_argument(
        "--self-check", action="store_true",
        help="Verify that the bold line's value at K = K_model equals "
             "the off-diagonal mean of the cross-quantization "
             "heatmap's Jaccard matrix (agreement to 1e-12), then "
             "render normally. Exits non-zero on mismatch.",
    )
    args = parser.parse_args()

    # Resolve the model set. Same convention as
    # cross_quantization_comparison.py: a dict keyed by the
    # filesystem-safe id gives us both the existing order AND a
    # fallback for unrecognized ids.
    if args.models:
        canonical = dict(DEFAULT_MODELS)
        models = [(m, canonical.get(m, m)) for m in args.models]
    else:
        models = list(DEFAULT_MODELS)

    # Resolve the output directory.
    out_dir = (args.output_dir if args.output_dir is not None
               else args.results_dir / "paper_heatmaps"
                     / "cross_quant_sweep")

    # ---- Step 1: load every model's cross-quant matrices via the
    # same loader the heatmap uses. We only need `counts_total`
    # (per quant, [L, E]) to drive the K-sweep — the Jaccard matrix
    # is re-derived per K, not reused from the heatmap.
    per_model: dict[str, dict[str, Any]] = {}
    model_dir_ref_for: dict[str, Path] = {}
    for model_safe, short_name in models:
        model_dir = args.results_dir / model_safe
        if not model_dir.is_dir():
            print(f"[warn] {model_dir}: not found; skipping model "
                  f"{short_name}")
            continue
        data = _compute_model_cross_quant_matrices(model_dir, args.quants)
        if data is None:
            print(f"[warn] {model_dir}: <2 valid quantisations found; "
                  f"skipping model {short_name}")
            continue
        records = _iter_quant_records(model_dir, args.quants)
        model_dir_ref_for[short_name] = model_dir
        per_model[short_name] = {
            "model_safe": model_safe,
            "arch": data["arch"],
            # Keep ALL quants (including the `UnifRand` baseline) so the
            # bold-line value at K = K_model matches the heatmap's
            # off-diagonal mean exactly. The baseline exists in the
            # cross-quant result tree and is part of the paper's
            # cross-quantization view; excluding it here would diverge
            # from `cross_quantization_comparison.py` and confuse
            # readers comparing the two figures.
            "quant_labels": [r["publish_label"] for r in records],
            "counts_list": [
                _load_one_counts_total(model_dir, r["quant_dir_name"])
                for r in records
            ],
        }
        if len(per_model[short_name]["counts_list"]) < 2:
            print(f"[warn] {model_dir}: <2 valid quantisations "
                  f"found; skipping model {short_name}")
            per_model.pop(short_name, None)
            continue
        print(f"[load] {short_name}: {data['arch']['name']} "
              f"L={data['arch']['n_layer']} E={data['arch']['n_expert']} "
              f"K_model={data['arch']['n_expert_used']}; "
              f"quants={per_model[short_name]['quant_labels']}")

    if not per_model:
        raise SystemExit("[error] no models loaded successfully")

    render_paper_cross_quant_jaccard_sweep(
        per_model, out_dir,
        width_in=args.width_in, height_in=args.height_in, dpi=args.dpi,
        k_min=args.k_min, k_max_frac=args.k_max_frac, num_ks=args.num_ks,
        force=args.force, self_check=args.self_check,
        model_dir_ref_for=model_dir_ref_for, only_quants=args.quants,
    )
    return 0


# ------------------------------------------------------------- helpers

def _iter_quant_records(model_dir: Path,
                         only_quants: list[str] | None
                         ) -> list[dict[str, Any]]:
    """Discover per-quant records for one model.

    Pulls `counts_total` from each quant's
    `<model>/<quant>/overall/counts_total_overview.json`. Delegates to
    `cross_quantization_comparison._discover_quants` so this script
    walks quant directories in exactly the same order (QUANT_PRIORITY,
    then label) as the cross-quantization heatmap — the figure's
    "which quants are in the average" must match the heatmap's.

    Returns the same shape of record as
    `cross_quantization_comparison._discover_quants` (with keys
    `arch`, `counts_total`, `tokens`, `publish_label`,
    `quant_dir_name`).
    """
    return _discover_quants(model_dir, only_quants)


def _load_one_counts_total(model_dir: Path, quant_dir_name: str
                             ) -> np.ndarray:
    """Load the [L, E] `counts_total` matrix for one quant.

    Thin wrapper around the same JSON reader used by the heatmap
    loader. Returned matrix is `int64` to match `paper_heatmaps`'s
    `_top_k_set_for_layer` expectations.
    """
    rec = _load_one_quant_overview_safe(model_dir, quant_dir_name)
    if rec is None:
        raise SystemExit(f"[error] {model_dir}/{quant_dir_name}: "
                         f"unable to load counts_total_overview.json")
    return np.asarray(rec["counts_total"], dtype=np.int64)


if __name__ == "__main__":
    sys.exit(main())
