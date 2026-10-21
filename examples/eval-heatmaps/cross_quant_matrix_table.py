#!/usr/bin/env python3
# type: ignore

"""Cross-QUANTISATION numeric matrices for the cross-quantization figure.

Numeric backing for `cross_quantization_comparison.py`, which renders
the 2 x N heatmap `cross_quantization_grid.png` (top row:
layer-averaged pairwise top-K Jaccard across a model's
quantisations; bottom row: layer-averaged Jensen-Shannon divergence
across the same quantisations; one column per model). A heatmap
cannot be the only record of its numbers in a printed paper, so this
script tabulates exactly the same matrices.

Sibling of `cross_dataset_table.py`, which backs the cross-DATASET
figure.

Scope note: this script is NOT the cross-quantisation *stability
summary*. That quantity (mean ± std of a per-dataset scalar taken
ACROSS quantisations) belongs to `cross_quantization_table.py`, which
answers a different question and is left untouched. This script
answers the narrower question the figure asks: what are the pairwise
values, verbatim.

This script deliberately borrows the figure's own quant discovery and
ordering helpers from `cross_quantization_comparison` rather than
re-implementing them. A duplicated loader would silently desync from
the published figure the first time anyone adds a quant override or
changes the arch-mismatch filter.

Outputs (under `<output-dir>/`, default
`<results-dir>/paper_heatmaps/cross_quant_tables/`):

  - `cross_quant_pairwise.csv`    long form, full float64 precision,
                                  one row per (model, metric,
                                  quant_i, quant_j), diagonal and
                                  lower triangle included
  - `cross_quant_pairwise.md`     one upper-triangular table per
                                  (model, metric), each cell
                                  `value ± layer_std`, plus a
                                  condensed summary table

Mathematical contract (identical to the figure):

  A_q_K[l]  = top-K experts selected under quantisation q, layer l
               (descending count, ties -> ascending expert id)
  Jaccard(q_a, q_b) = |A_a ∩ A_b| / |A_a ∪ A_b|,   averaged over L
  JSD    (q_a, q_b) = JS divergence of the marginal activation
                       distributions,               averaged over L

All quantisations of a model share `K_model`, so the matrices are
directly comparable within a block. `layer_std` is the population
standard deviation (ddof=0) of the same per-layer quantity across
layers.

Usage:
    python examples/eval-heatmaps/cross_quant_matrix_table.py \
        --results-dir /mnt/data/results-llama-cpp

    # verify the tables transcribe the figure's matrices exactly
    python examples/eval-heatmaps/cross_quant_matrix_table.py --self-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Sibling modules live in the same directory, so Python's per-script
# sys.path entry resolves these without any PYTHONPATH plumbing.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Discovery and matrix computation are IMPORTED from the figure script
# so the tables can never drift from the published heatmap.
from cross_quantization_comparison import (  # noqa: E402
    DEFAULT_MODELS,
    _QUANT_PUBLISH_OVERRIDES,
    _compute_model_cross_quant_matrices,
    _discover_quants,
)
from paper_heatmaps import (  # noqa: E402
    _layer_averaged_js_divergence_matrix_with_std,
    _layer_averaged_pairwise_jaccard_with_std,
)
from pairwise_tables import (  # noqa: E402
    PairwiseBlock,
    summary_rows,
    verify_block,
    write_csv,
    write_markdown,
)


# ----------------------------------------------------------------- caption

# The random-router baseline's publishable label, sourced from the
# figure script's own override map so a rename there propagates here
# instead of desyncing the two.
BASELINE_LABEL: str = _QUANT_PUBLISH_OVERRIDES["RANDOM"]

CAPTION_TEMPLATE: str = (
    "Numeric backing for the cross-quantisation comparison figure "
    "`cross_quantization_grid.png`. Each block is one model; rows and "
    "columns are its quantisations in figure order (full precision "
    "first, then descending precision), with **{baseline}** the "
    "uniform-random router baseline. Every value is the mean over all "
    "`L` transformer layers of the pairwise top-`K` Jaccard similarity "
    "(`K = K_model`, ties broken by ascending expert id) for the "
    "Jaccard blocks, and of the Jensen-Shannon divergence between the "
    "two quantisations' marginal expert-activation distributions for "
    "the JSD blocks. The JSD is computed with a base-2 logarithm, so "
    "it lies in `[0, 1]`. `±` reports the population standard "
    "deviation of the same per-layer quantity across layers — that is, "
    "how stable the agreement is with depth. The diagonal is exactly "
    "1.000 (Jaccard) and 0.000 (JSD) by construction and the matrices "
    "are symmetric, so both are omitted here and reported in full in "
    "the CSV. The summary table's `quants` means exclude {baseline}; "
    "its `vs. baseline` means average only the {baseline} row. "
    "Cross-quantisation JS-divergence values are typically several "
    "orders of magnitude below the cross-dataset ones, so blocks whose "
    "largest off-diagonal magnitude is below 1e-2 are printed in "
    "scientific notation rather than collapsing to `0.000`."
)


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
        "--models", action="append", default=None,
        help="Restrict to a subset of filesystem-safe model ids "
             "(repeatable). Default: the same hardcoded model set as "
             "cross_quantization_comparison.py, so the tables back the "
             "published figure.",
    )
    parser.add_argument(
        "--quants", action="append", default=None,
        help="Restrict to a subset of quant subdirectories "
             "(repeatable). Default: all quants with a valid "
             "overall/counts_total_overview.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: <results-dir>/paper_heatmaps/"
             "cross_quant_tables).",
    )
    parser.add_argument(
        "--self-check", action="store_true",
        help="Recompute the std-free matrices the figure script draws "
             "and assert the tables agree to 1e-12. Exits non-zero on "
             "any mismatch.",
    )
    args = parser.parse_args()

    if args.models:
        canonical = dict(DEFAULT_MODELS)
        models = [(m, canonical.get(m, m)) for m in args.models]
    else:
        models = list(DEFAULT_MODELS)

    out_dir = (args.output_dir if args.output_dir is not None
               else args.results_dir / "paper_heatmaps"
                     / "cross_quant_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: load each model's cross-quant matrices and pair them
    # with the per-layer std.
    blocks: list[PairwiseBlock] = []
    failures: list[str] = []
    for model_safe, short_name in models:
        model_dir = args.results_dir / model_safe
        if not model_dir.is_dir():
            print(f"[warn] {model_dir}: not found; skipping model "
                  f"{short_name}")
            continue

        # Delegate discovery AND matrix computation to the figure
        # script so the label set, the arch-mismatch filter, and the
        # std-free reference matrices are exactly the figure's.
        data = _compute_model_cross_quant_matrices(model_dir, args.quants)
        if data is None:
            print(f"[warn] {model_dir}: <2 valid quantisations found; "
                  f"skipping model {short_name}")
            continue
        labels = data["quant_labels"]
        arch = data["arch"]

        # Re-derive the counts in the figure's own label order. Joining
        # by publish label (rather than re-running the arch filter)
        # guarantees these are the very arrays the figure averaged.
        records = _discover_quants(model_dir, args.quants)
        by_label = {r["publish_label"]: r for r in records}
        ordered = [by_label[l] for l in labels if l in by_label]
        if len(ordered) != len(labels):
            print(f"[warn] {short_name}: could not recover counts for "
                  f"all {len(labels)} quantisations; skipping model")
            continue
        counts_list = [r["counts_total"] for r in ordered]

        L = arch["n_layer"]
        K_model = arch["n_expert_used"]
        j_mean, j_std = _layer_averaged_pairwise_jaccard_with_std(
            counts_list, K_model, L)
        s_mean, s_std = _layer_averaged_js_divergence_matrix_with_std(
            counts_list)
        blocks.append(PairwiseBlock(
            model=short_name,
            model_safe=model_safe,
            quant_ref="",
            unit_kind="quant",
            labels=labels,
            arch=arch,
            jaccard_mean=j_mean,
            jaccard_std=j_std,
            jsd_mean=s_mean,
            jsd_std=s_std,
        ))
        print(f"[load] {short_name}: {arch['name']} "
              f"L={L} E={arch['n_expert']} K_model={K_model}; "
              f"quants={labels}")

        if args.self_check:
            failures += verify_block(
                blocks[-1],
                jaccard_ref=data["jaccard"],
                jsd_ref=data["jsd"],
            )

    if not blocks:
        raise SystemExit("[error] no models loaded successfully")

    if args.self_check:
        if failures:
            print("\n[self-check] FAILED:")
            for f in failures:
                print(f"  - {f}")
            raise SystemExit(
                f"[error] {len(failures)} self-check failure(s); the "
                f"tables do not transcribe the figure's matrices")
        print(f"[self-check] OK: {len(blocks)} block(s) agree with the "
              "figure's matrices to 1e-12")

    # ---- Step 2: write CSV (full precision, machine-readable).
    csv_path = write_csv(out_dir / "cross_quant_pairwise.csv", blocks)
    print(f"[save] {csv_path}")

    # ---- Step 3: write Markdown (reviewable, value ± layer_std).
    md_path = write_markdown(
        out_dir / "cross_quant_pairwise.md",
        blocks,
        title="Cross-quantisation routing stability",
        caption=CAPTION_TEMPLATE.format(baseline=BASELINE_LABEL),
        baseline_label=BASELINE_LABEL,
    )
    print(f"[save] {md_path}")

    # ---- Step 4: short stdout echo of the summary rows.
    rows = summary_rows(blocks, BASELINE_LABEL)
    print(f"\n{'Model':<18} {'Quants':>6} {'Jaccard':>9} {'JSD':>11} "
          f"{'Jacc/base':>10} {'JSD/base':>11}")
    for r in rows:
        def _f(v: float, width: int = 9) -> str:
            if not np.isfinite(v):
                return "n/a".rjust(width)
            # Small cross-quant JSD values read better in scientific
            # form here too.
            if 0.0 < abs(v) < 1e-2:
                return f"{v:.1e}".rjust(width)
            return f"{v:{width}.3f}"
        print(f"{r['model']:<18} {r['units']:>6} "
              f"{_f(r['jaccard_real'])} {_f(r['jsd_real'], 11)} "
              f"{_f(r['jaccard_baseline'])} {_f(r['jsd_baseline'], 11)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())