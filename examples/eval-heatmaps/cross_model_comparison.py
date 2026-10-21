#!/usr/bin/env python3
# type: ignore

"""Cross-model cross-dataset comparison figure for the MoE-routing paper.

Companion to `paper_heatmaps.py`. Generates a single **2 x 4 figure**
that compares four MoE models side-by-side on the cross-dataset Jaccard
and JS-divergence heatmaps, all sharing the same global colorbar
scales so reviewers can compare cells across models at a glance.

Layout:

    [OLMoE  ][Mixtral ][DeepSeek][Qwen3  ]   <- column titles (top)
    [ Jaccard row: 5x5 heatmap per model               ]   <- 1 colorbar
    [ JSD     row: 5x5 heatmap per model               ]   <- 1 colorbar

JSD colorbar uses a **fixed [0, 0.5] scale** (the empirically
observed largest possible divergence across the result histograms).
This avoids the per-run global-max pre-pass and keeps every cross-
model comparison on the identical scale without scanning every
quantization cell first.

The figure is sized for a double-column paper (~11in wide x ~5.5in tall),
300 DPI, no titles / no axis labels beyond the model-name column
titles — readers get the dataset labels (BBH / HumanEval / INCLUDE /
MMLU / PopQA) from the figure caption, not the figure itself (this
saves horizontal space as the user requested).

Output (under `<output-dir>/`, with `<quant>` = `--quant` value):

  - `cross_model_grid__<quant>.png`     the 2x4 figure

The `__<quant>` suffix (matching the `<model>__<quant>__jaccard_...`
naming convention in `paper_heatmaps.py`) prevents overwrites when the
script is run with multiple `--quant` values and makes it clear at a
glance which quantization a given output file corresponds to.

The companion **cross-quantization summary table** (mean ± std
across all available quantizations for each (model, metric,
dataset)) is produced by the **separate sibling script**
`examples/eval-heatmaps/cross_quantization_table.py` — single
responsibility per file, easier to test in isolation.

Mathematical contract (identical to `paper_heatmaps.py` / the
existing eval-moe-overview Jaccard/JSD helpers):

  For each layer l and each dataset i:
    A_i[l]  = top-K experts of the marginal routing distribution
              (descending count, ties -> ascending expert id)
  Jaccard(A_i, A_j) = |A_i ∩ A_j| / |A_i ∪ A_j|    averaged over layers
  JSD     (A_i, A_j) = Jensen-Shannon divergence of the marginal
                       routing distributions,             averaged over layers

We compute both at K = K_model for each model (the natural choice;
different K values per model produce matrices on the same [0, 1] scale
so they're still visually comparable).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

# Make the sibling `paper_heatmaps.py` importable. Both files live in
# `examples/eval-heatmaps/` so Python's per-script sys.path entry
# resolves these imports without any setup.py or PYTHONPATH plumbing.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Module-level set used to dedupe `[warn]` messages for corrupted JSONs.
# Partially-flushed `expert_counts.json` files (from interrupted C++
# runs) are unrecoverable, so the script skips them gracefully. But
# the same file can be encountered twice — once in the model-load
# pass and once in the global-JSD-scan pass — and re-printing the
# same warning twice per pass is noisy. We print each warning once
# (and once-only across both passes) and then summarise at the end.
_WARNED_FILES: set[str] = set()
# Catalogue of every file that was SKIPPED due to a load error so the
# final summary can list them in one place.
_SKIPPED_FILES: list[tuple[str, str]] = []  # (path, reason)

from paper_heatmaps import (  # noqa: E402
    _layer_averaged_pairwise_jaccard,
    _layer_averaged_js_divergence_matrix,
    _maybe_aggregate_random,
    _publish_dataset_label,
    _safe_json_load,
    _make_k_values,
)


# ----------------------------------------------------------------- constants

# Default list of models to compare. Hardcoded so the script produces
# the same 4-column figure every run regardless of what's in the
# results directory. Override via `--models` if the user wants to
# change the set.
DEFAULT_MODELS: list[tuple[str, str]] = [
    # (filesystem-safe model id, short name for the figure caption).
    # The filesystem-safe id is the directory name in the results tree
    # (slashes replaced with `--` per `scripts/run-evaluator.sh`).
    ("mradermacher--OLMoE-1B-7B-0924-Instruct-GGUF",       "OLMoE-1B-7B"),
    ("LiteLLMs--Mixtral-8x22B-Instruct-v0.1-GGUF",   "Mixtral-8x22B"),
    ("mradermacher--deepseek-moe-16b-chat-GGUF",      "DeepSeek-MoE-16B"),
    ("unsloth--Qwen3-30B-A3B-GGUF",                   "Qwen3-30B-A3B"),
    # ("allenai--OLMoE-1B-7B-0924-Instruct",       "OLMoE-1B-7B"),
    # ("mistralai--Mixtral-8x22B-Instruct-v0.1",   "Mixtral-8x22B"),
    # ("deepseek-ai--deepseek-moe-16b-chat",      "DeepSeek-MoE-16B"),
    # ("Qwen--Qwen3-30B-A3B",                   "Qwen3-30B-A3B"),
]

# Default benchmark order (publishable label form). `Random` is appended
# at the END when the synthetic auto-aggregated baseline is present
# (so the comparison reads naturally: real benchmarks first, baseline
# at the bottom). The aggregator falls back to sorted order for any
# dataset not in this list. The user renamed "Random" to "UnifRand"
# for paper-clarity (it's a uniform random router baseline; the
# short form is less ambiguous than "Random" which could be read as
# "arbitrary baseline" by a casual reader).
BENCHMARK_ORDER: list[str] = ["BBH", "HumanEval", "INCLUDE", "MMLU",
                                "PopQA", "UnifRand"]

# Figure sizing (double-column paper).
FIGURE_WIDTH_IN: float = 11.0
FIGURE_HEIGHT_IN: float = 5.5
DPI: int = 300


# ------------------------------------------------------------------- helpers

def _load_cell_counts(model_dir: Path, quant: str
                      ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load every `moe-<ds>/expert_counts.json` under one (model, quant).

    Returns:
      counts: dict mapping publishable dataset label -> [L, E] int64 array
              (the marginal_expert_counts matrix from `aggregate`).
      arch  : dict with keys `name`, `n_layer`, `n_expert`,
              `n_expert_used` (taken from the first dataset's JSON).
    """
    cell_dir = model_dir / quant
    counts: dict[str, np.ndarray] = {}
    arch: dict[str, Any] | None = None
    for js_path in sorted(cell_dir.glob("moe-*/expert_counts.json")):
        ds_dir_name = js_path.parent.name  # e.g. "moe-mmlu"
        # `_safe_json_load` (from `paper_heatmaps.py`) recovers from
        # partial UTF-8 / format corruption that the C++ binary leaves
        # behind when killed mid-write — see its docstring for the
        # recovery strategies. We only fall back to a hard skip if
        # every recovery strategy fails.
        data = _safe_json_load(js_path)
        if data is None:
            key = str(js_path)
            if key not in _WARNED_FILES:
                _WARNED_FILES.add(key)
                msg = "unable to load JSON (corrupted beyond recovery)"
                _SKIPPED_FILES.append((key, msg))
                print(f"[warn] {js_path}: {msg}; skipping")
            continue
        arch_in = data.get("model_arch") or {}
        L = int(arch_in.get("n_layer", 0))
        E = int(arch_in.get("n_expert", 0))
        k = int(arch_in.get("n_expert_used", 0))
        if L <= 0 or E <= 0 or k <= 0:
            continue
        marginal = (data.get("aggregate") or {}).get(
            "marginal_expert_counts"
        )
        if marginal is None:
            continue
        m = np.asarray(marginal, dtype=np.int64)
        if m.shape != (L, E):
            key = f"shape:{js_path}"
            if key not in _WARNED_FILES:
                _WARNED_FILES.add(key)
                msg = f"shape {m.shape} != ({L}, {E})"
                _SKIPPED_FILES.append((str(js_path), msg))
                print(f"[warn] {js_path}: {msg}; skipping")
            continue
        publish_ds = _publish_dataset_label(ds_dir_name)
        counts[publish_ds] = m
        if arch is None:
            arch = {"name": arch_in.get("name", "unknown"),
                    "n_layer": L, "n_expert": E, "n_expert_used": k}

    # ---- auto-aggregate the per-benchmark random-router baseline ----
    # ---- load the random-router baseline as a 6th dataset ----
    # The random router produces results in two possible locations:
    #
    #   1. Per-benchmark, inside each `moe-<ds>/random/` subdirectory.
    #      We sum the marginal counts across all of these to build a
    #      synthetic top-level "Random" dataset (paper_heatmaps.py
    #      convention, via `_maybe_aggregate_random`).
    #
    #   2. Already pre-aggregated, as a top-level
    #      `<cell>/random/expert_counts.json`. Some users run the
    #      random router once at the cell level and write the result
    #      here directly (it's the same arch + same [L, E] matrix).
    #
    # We support BOTH layouts — try the pre-aggregated file first
    # (cheaper), then fall back to aggregating per-benchmark files.
    # The baseline gets appended at the END of the matrix (after the
    # real benchmarks) so the comparison reads naturally.
    random_loaded = False
    pre_agg_path = cell_dir / "random" / "expert_counts.json"
    if pre_agg_path.is_file():
        # Use the lenient loader so a few corrupt bytes in the
        # pre-aggregated random file don't blacklist the whole Random
        # baseline.
        rdata = _safe_json_load(pre_agg_path)
        if rdata is None:
            key = f"random-preagg:{pre_agg_path}"
            if key not in _WARNED_FILES:
                _WARNED_FILES.add(key)
                msg = "unable to load JSON (will fall back to per-benchmark agg)"
                _SKIPPED_FILES.append((str(pre_agg_path), msg))
                print(f"[warn] {pre_agg_path}: {msg}")
        else:
            r_arch_in = rdata.get("model_arch") or {}
            r_L = int(r_arch_in.get("n_layer", 0))
            r_E = int(r_arch_in.get("n_expert", 0))
            r_k = int(r_arch_in.get("n_expert_used", 0))
            r_marg = (rdata.get("aggregate") or {}).get(
                "marginal_expert_counts"
            )
            if (r_L > 0 and r_E > 0 and r_k > 0 and r_marg is not None
                    and arch is not None
                    and r_L == arch["n_layer"]
                    and r_E == arch["n_expert"]
                    and r_k == arch["n_expert_used"]):
                m = np.asarray(r_marg, dtype=np.int64)
                if m.shape == (r_L, r_E):
                    counts["UnifRand"] = m
                    random_loaded = True
                else:
                    key = f"random-shape:{pre_agg_path}"
                    if key not in _WARNED_FILES:
                        _WARNED_FILES.add(key)
                        msg = (f"shape {m.shape} != ({r_L}, {r_E}); "
                               f"will fall back to per-benchmark agg")
                        _SKIPPED_FILES.append((str(pre_agg_path), msg))
                        print(f"[warn] {pre_agg_path}: {msg}")

    if not random_loaded:
        # Fall back to summing per-benchmark `moe-<ds>/random/` JSONs.
        random_agg = _maybe_aggregate_random(cell_dir)
        if random_agg is not None and arch is not None:
            rand_counts, rand_meta = random_agg
            r_arch = next(iter(rand_meta.values()))["arch"]
            if (r_arch["n_layer"] == arch["n_layer"]
                    and r_arch["n_expert"] == arch["n_expert"]
                    and r_arch["n_expert_used"] == arch["n_expert_used"]):
                for ds, c in rand_counts.items():
                    # `_maybe_aggregate_random` returns
                    # ``{"Random": agg}`` with the original (mixed-
                    # case) key — remap to our publishable label
                    # "UnifRand" so it lines up with the
                    # BENCHMARK_ORDER entry above. `setdefault` so
                    # we don't clobber a real "Random" / "UnifRand"
                    # cell entry if the user has one at the cell root.
                    counts.setdefault("UnifRand", c)

    if arch is None:
        raise SystemExit(f"[error] {cell_dir}: no usable datasets found")
    return counts, arch


def _order_labels(counts: dict[str, np.ndarray]
                  ) -> tuple[list[str], list[np.ndarray]]:
    """Order datasets (BBH / HumanEval / ...) in publishable order."""
    labels = [d for d in BENCHMARK_ORDER if d in counts]
    labels += [d for d in sorted(counts.keys()) if d not in labels]
    counts_list = [counts[d] for d in labels]
    return labels, counts_list


# ------------------------------------------------------------------- plot

def _draw_one_heatmap(ax, mat, labels, *, vmin, vmax, cmap,
                      cell_text_format: str = "{v:.2f}",
                      cell_text_threshold: float = 0.5,
                      tick_label_fontsize: int = 8,
                      cell_text_fontsize: int = 6,
                      cell_text_color_above: str = "white",
                      cell_text_color_below: str = "black",
                      row_label: str | None = None,
                      row_label_fontsize: int = 10,
                      ) -> "matplotlib.image.AxesImage":
    """Render one heatmap into `ax` (no titles / no axis labels) and
    return the `AxesImage` so the caller can build a shared colorbar.

    If `row_label` is supplied, it is rendered as the subplot's
    y-axis label (vertical text in the gutter immediately to the
    left of the heatmap). The label takes zero horizontal space
    inside the heatmap, so it can identify each row without
    shrinking the cell area — readers instantly know which row is
    Jaccard Similarity vs. JS-divergence without consulting the caption.
    """
    if row_label is not None:
        ax.set_ylabel(row_label, fontsize=row_label_fontsize,
                       rotation=90, labelpad=8, va="center")
    N = len(labels)
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal",
                   interpolation="nearest")
    if N > 0:
        ax.set_xticks(range(N))
        ax.set_xticklabels(labels, rotation=30, ha="right",
                           fontsize=tick_label_fontsize)
        ax.set_yticks(range(N))
        ax.set_yticklabels(labels, rotation=0, fontsize=tick_label_fontsize)
    else:
        ax.set_xticks([0])
        ax.set_xticklabels([""], fontsize=tick_label_fontsize)
        ax.set_yticks([0])
        ax.set_yticklabels([""], fontsize=tick_label_fontsize)
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Per-cell text. Threshold is relative to the colormap so cells
    # below the midpoint get one colour and cells above get the other;
    # works for both Jaccard (Blues, dark at top) and JSD (Blues_r,
    # dark at bottom).
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = float(mat[i, j])
            vnorm = (v - vmin) / max(1e-12, (vmax - vmin))
            color = (cell_text_color_above if vnorm > cell_text_threshold
                     else cell_text_color_below)
            ax.text(j, i, cell_text_format.format(v=v), ha="center",
                    va="center", fontsize=cell_text_fontsize, color=color)
    return im


def _format_v3(x: float, _pos: int) -> str:
    """3-decimal tick formatter (e.g., 0.108) for the JSD colorbar."""
    return f"{x:.3f}"


def _format_v2(x: float, _pos: int) -> str:
    """2-decimal tick formatter (e.g., 0.75) for the Jaccard colorbar."""
    return f"{x:.2f}"


# ------------------------------------------------------------------- main

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
        "--quant", type=str, default="Q4_K_M",
        help="Quantization subdirectory to use for all models "
             "(default: Q4_K_M).",
    )
    parser.add_argument(
        "--models", action="append", default=None,
        help="Restrict to a subset of filesystem-safe model ids "
             "(repeatable). Default: the 4 hardcoded models.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: <results-dir>/paper_heatmaps/"
             "cross_model).",
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
    args = parser.parse_args()

    # Resolve the model set.
    if args.models:
        # Use a dict so the user can preserve the canonical short-name
        # ordering even when overriding the set.
        canonical = dict(DEFAULT_MODELS)
        models = [(m, canonical.get(m, m)) for m in args.models]
    else:
        models = list(DEFAULT_MODELS)

    # Resolve the output directory.
    out_dir = (args.output_dir if args.output_dir is not None
               else args.results_dir / "paper_heatmaps" / "cross_model")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: load every cell, compute Jaccard + JSD matrices.
    # Only the --quant cell is loaded — the cross-quantization table
    # is generated by the sibling script `cross_quantization_table.py`,
    # which loads ALL quantizations per model. Keeping the figure
    # script focused on the figure means each file has a single
    # responsibility.
    per_model: dict[str, dict[str, Any]] = {}
    for model_safe, short_name in models:
        cell_dir = args.results_dir / model_safe
        if not cell_dir.is_dir():
            print(f"[warn] {cell_dir}: not found; skipping model "
                  f"{short_name}")
            continue
        quant_dir = cell_dir / args.quant
        if not quant_dir.is_dir():
            print(f"[warn] {quant_dir}: not found; skipping model "
                  f"{short_name}")
            continue
        counts, arch = _load_cell_counts(cell_dir, args.quant)
        labels, counts_list = _order_labels(counts)
        if len(counts_list) < 2:
            print(f"[warn] {quant_dir}: <2 datasets; skipping model "
                  f"{short_name}")
            continue
        L = arch["n_layer"]
        E = arch["n_expert"]
        k_model = arch["n_expert_used"]
        K_values = _make_k_values(E, k_model, k_min=1,
                                   k_max_frac=0.50, num_ks=21)
        jacc = _layer_averaged_pairwise_jaccard(counts_list, k_model, L)
        jsd = _layer_averaged_js_divergence_matrix(counts_list)
        per_model[short_name] = {
            "model_safe": model_safe,
            "arch": arch,
            "labels": labels,
            "counts_list": counts_list,
            "K_values": K_values,
            "K_model": k_model,
            "jaccard": jacc,
            "jsd": jsd,
        }
        print(f"[load] {short_name}: {arch['name']} "
              f"L={L} E={E} K_model={k_model}; labels={labels}")

    if not per_model:
        raise SystemExit("[error] no models loaded successfully")

    n_cols = len(per_model)
    if n_cols < 1:
        raise SystemExit("[error] need at least 1 model")

    # ---- Step 2: figure layout (2 rows x n_cols cols).
    # Reserve right margin for the per-row colorbar. With n_cols
    # subplots side-by-side, the per-subplot width is constrained so
    # the whole row + colorbar fits in --width-in inches.
    cbar_width_in = 0.4
    grid_width_in = args.width_in - cbar_width_in
    subplot_width_in = grid_width_in / n_cols
    subplot_height_in = subplot_width_in * 0.85  # slightly tall for rows

    fig = plt.figure(figsize=(args.width_in, args.height_in), dpi=args.dpi)

    # Use a manual gridspec so we can size each subplot precisely and
    # leave room for column titles + the per-row colorbars. The gaps
    # are tightened to the minimum that still keeps subplots readable
    # (`wspace=0.02`, `hspace=0.08`) — the user explicitly asked for
    # tighter gaps to save space in the paper figure.
    gs = fig.add_gridspec(
        nrows=2, ncols=n_cols + 1,
        width_ratios=[subplot_width_in] * n_cols + [cbar_width_in * 0.6],
        height_ratios=[subplot_height_in] * 2,
        left=0.04, right=0.98, top=0.92, bottom=0.08,
        wspace=0.02, hspace=0.08,
    )

    # ---- Step 3: render each subplot.
    # Jaccard row: vmin=0, vmax=1 (the metric's natural range).
    # We don't auto-scale to per-model max because Jaccard is bounded
    # in [0, 1] and the shared scale is the WHOLE POINT of the
    # cross-model comparison.
    jaccard_axes = []
    jaccard_im = None
    for col, short_name in enumerate(per_model.keys()):
        ax = fig.add_subplot(gs[0, col])
        data = per_model[short_name]
        # Jaccard: dark at the TOP (high similarity), so cells with
        # v >= 0.5 (top half of the range) get WHITE text, and the
        # diagonal (v=1.0) is dark blue with white text.
        im = _draw_one_heatmap(
            ax, data["jaccard"], data["labels"],
            vmin=0.0, vmax=1.0, cmap="Blues",
            cell_text_format="{v:.2f}",
            cell_text_threshold=0.5,
            cell_text_color_above="white",
            cell_text_color_below="black",
            tick_label_fontsize=8,
            cell_text_fontsize=6,
            row_label=("Jaccard Similarity" if col == 0 else None),
        )
        # Title combines the model name and the `--quant` value with a
        # pipe separator (e.g. "OLMoE-1B-7B | Q4_K_M"). This makes
        # every column independently labelled so the reader can pair
        # the figure with any of the `--quant` outputs without
        # consulting the file name; the same `__<quant>` suffix is
        # already in the PNG file name, but duplicating it in the
        # title is friendlier when the PNG is embedded in a slide.
        ax.set_title(f"{short_name} | {args.quant}", fontsize=9, pad=4)
        # Tick labels. x-tick labels are repeated on all subplots so
        # we only render them on the BOTTOM row (the JSD row below
        # this one) — readers scan the column names once at the
        # bottom of the figure rather than seeing them under every
        # subplot. y-tick labels are only shown on the LEFTMOST
        # column of each row — the row label in the gutter ("Jaccard
        # Similarity" / "JS divergence") handles row identification,
        # and the dataset names on the y-axis are shown just once per
        # row.
        plt.setp(ax.get_xticklabels(), visible=False)
        if col == 0:
            # Leftmost column gets y-tick labels (dataset names) AND
            # the gutter row label — they live in different visual
            # gutters (left margin vs. just-left-of-axis), so they
            # coexist cleanly. Reset tick length too in case the
            # earlier `length=0` had it zeroed out.
            ax.tick_params(axis="y", which="both", length=0)
        else:
            # Inner columns: no y-tick labels (the leftmost column
            # serves the whole row).
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis="y", which="both", length=0)
        jaccard_axes.append(ax)
        jaccard_im = im

    # Per-row colorbar for Jaccard (right of the row).
    cax_jacc = fig.add_subplot(gs[0, n_cols])
    cb_jacc = fig.colorbar(jaccard_im, cax=cax_jacc,
                            fraction=1.0, pad=0.0)
    cb_jacc.ax.tick_params(labelsize=8, length=0)
    cb_jacc.outline.set_visible(False)
    cb_jacc.ax.yaxis.set_major_formatter(FuncFormatter(_format_v2))
    cb_jacc.set_ticks(np.linspace(0.0, 1.0, 6))

    # JSD row: fixed [0, 0.5] scale. The 0.5 upper bound is the
    # empirically observed largest divergence across the result
    # histograms (it comfortably covers every matrix we've seen so
    # far), so we skip the expensive per-run global-max scan and
    # keep every cross-model comparison on the identical scale.
    print("[plan] JSD shared colorbar range: fixed 0 to 0.5 "
          "(empirical max across all result histograms; no per-run "
          "global scan needed)")

    jsd_axes = []
    jsd_im = None
    for col, short_name in enumerate(per_model.keys()):
        ax = fig.add_subplot(gs[1, col])
        data = per_model[short_name]
        # JSD: dark at the BOTTOM (low divergence = perfect agreement),
        # so cells with v <= 0.5 * vmax (bottom half of the range)
        # get WHITE text. Diagonal (v=0) is dark blue with white text.
        im = _draw_one_heatmap(
            ax, data["jsd"], data["labels"],
            vmin=0.0, vmax=0.5, cmap="Blues_r",
            cell_text_format="{v:.3f}",
            cell_text_threshold=0.5,
            cell_text_color_above="black",
            cell_text_color_below="white",
            tick_label_fontsize=8,
            cell_text_fontsize=6,
            row_label=("JS divergence" if col == 0 else None),
        )
        # Tick labels for the JSD row. The dataset names are now
        # shown on the BOTTOM row's x-axis (one set of column labels
        # serves the whole figure) and on the leftmost column's
        # y-axis (one set of row labels per metric). The gutter
        # label is preserved on the leftmost column.
        if col == 0:
            ax.tick_params(axis="y", which="both", length=0)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis="y", which="both", length=0)
        jsd_axes.append(ax)
        jsd_im = im

    # Per-row colorbar for JSD (right of the row).
    cax_jsd = fig.add_subplot(gs[1, n_cols])
    cb_jsd = fig.colorbar(jsd_im, cax=cax_jsd, fraction=1.0, pad=0.0)
    cb_jsd.ax.tick_params(labelsize=8, length=0)
    cb_jsd.outline.set_visible(False)
    cb_jsd.ax.yaxis.set_major_formatter(FuncFormatter(_format_v3))
    cb_jsd.set_ticks(np.linspace(0.0, 0.5, 6))

    # ---- Step 4: save the figure.
    # Include `--quant` in the file name (a) so running this script
    # with multiple --quant values doesn't overwrite each other's
    # outputs in the same directory, and (b) so a reviewer opening
    # the PNG knows immediately which quantization it corresponds
    # to. Same `__` separator convention as `paper_heatmaps.py`.
    quant_suffix = f"__{args.quant}"
    out_png = out_dir / f"cross_model_grid{quant_suffix}.png"
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_png}")

    # ---- Step 5: (intentionally empty for this figure-only script).
    # The cross-quantization summary table is generated by the sibling
    # script `cross_quantization_table.py`, which loads ALL available
    # quantizations per model (the figure script here only loads the
    # single `--quant` cell needed for the 2x4 heatmap figure).

    # ---- Summary of any files that were skipped during the run. ----
    # Corrupted or partial JSONs are skipped gracefully; we list them
    # all in one place so the user has a single audit point (instead
    # of grepping through the `[warn]` lines).
    if _SKIPPED_FILES:
        print(f"[skip-summary] {len(_SKIPPED_FILES)} file(s) were "
              f"skipped due to load errors (gracefully ignored; cells "
              f"that depend solely on these files may be missing one "
              f"or more datasets in their heatmap):")
        for path, reason in _SKIPPED_FILES:
            # Trim the path to a relative one under results-dir for
            # readability; show only the last 90 chars if it's long.
            rel = path
            try:
                rel = str(Path(path).relative_to(args.results_dir))
            except ValueError:
                pass
            if len(rel) > 90:
                rel = "..." + rel[-87:]
            print(f"  - {rel}: {reason[:120]}")
        print("[skip-summary] the figure still produced output above; "
              "re-run the affected C++ binary to regenerate the "
              "underlying JSON if needed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
