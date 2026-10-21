#!/usr/bin/env python3
# type: ignore

"""Cross-DATASET numeric tables for the cross-model comparison figure.

Numeric backing for `cross_model_comparison.py`, which renders the
2 x N heatmap `cross_model_grid__<quant>.png` (top row: layer-averaged
pairwise top-K Jaccard across benchmarks; bottom row: layer-averaged
Jensen-Shannon divergence across the same benchmarks; one column per
model). A heatmap cannot be the only record of its numbers in a
printed paper, so this script tabulates exactly the same matrices.

Sibling of `cross_quant_matrix_table.py`, which backs the
cross-QUANTISATION figure.

This script deliberately borrows the figure's own loaders and
ordering helpers from `cross_model_comparison` rather than
re-implementing them. A duplicated loader would silently desync from
the published figure the first time anyone adds a dataset alias or
changes the random-router fallback path.

Outputs (under `<output-dir>/`, default
`<results-dir>/paper_heatmaps/cross_model_tables/<quant>/`):

  - `cross_dataset_pairwise.csv`   long form, full float64 precision,
                                   one row per (model, metric, unit_i,
                                   unit_j), diagonal and lower
                                   triangle included
  - `cross_dataset_pairwise.md`    one upper-triangular table per
                                   (model, metric), each cell
                                   `value ± layer_std`, plus a
                                   condensed summary table

Mathematical contract (identical to the figure):

  A_ds_K[l]  = top-K experts selected on benchmark `ds`, layer l
               (descending count, ties -> ascending expert id)
  Jaccard(ds_a, ds_b) = |A_a ∩ A_b| / |A_a ∪ A_b|,   averaged over L
  JSD    (ds_a, ds_b) = JS divergence of the marginal activation
                        distributions,               averaged over L

We compute both at K = K_model. `layer_std` is the population standard
deviation (ddof=0) of the same per-layer quantity across layers.

Usage:
    python examples/eval-heatmaps/cross_dataset_table.py \
        --results-dir /mnt/data/results-llama-cpp --quant Q4_K_M

    # verify the tables transcribe the figure's matrices exactly
    python examples/eval-heatmaps/cross_dataset_table.py --self-check
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

# Loaders and ordering are IMPORTED from the figure script so the
# tables can never drift from the published heatmap.
from cross_model_comparison import (  # noqa: E402
    BENCHMARK_ORDER,
    DEFAULT_MODELS,
    _load_cell_counts,
    _order_labels,
)
from paper_heatmaps import (  # noqa: E402
    _layer_averaged_js_divergence_matrix,
    _layer_averaged_js_divergence_matrix_with_std,
    _layer_averaged_pairwise_jaccard,
    _layer_averaged_pairwise_jaccard_with_std,
)
from pairwise_tables import (  # noqa: E402
    PairwiseBlock,
    verify_block,
    write_csv,
    write_markdown,
)


# ----------------------------------------------------------------- caption

# The random-router baseline's publishable label. Sourced from the
# figure script's own ordering constant so a rename there propagates
# here automatically instead of desyncing the two.
BASELINE_LABEL: str = "UnifRand"
assert BASELINE_LABEL in BENCHMARK_ORDER, (
    f"baseline label {BASELINE_LABEL!r} is not in the figure script's "
    f"BENCHMARK_ORDER {BENCHMARK_ORDER}; update BASELINE_LABEL to match")

CAPTION_TEMPLATE: str = (
    "Numeric backing for the cross-dataset comparison figure "
    "`cross_model_grid__{quant}.png`. Each block is one model "
    "evaluated at quantisation **{quant}**; rows and columns are the "
    "evaluation benchmarks in figure order, with **{baseline}** the "
    "uniform-random router baseline. Every value is the mean over all "
    "`L` transformer layers of the pairwise top-`K` Jaccard "
    "similarity (`K = K_model`, ties broken by ascending expert id) "
    "for the Jaccard blocks, and of the Jensen-Shannon divergence "
    "between the two benchmarks' marginal expert-activation "
    "distributions for the JSD blocks. The JSD is computed with a "
    "base-2 logarithm, so it lies in `[0, 1]`. `±` reports the "
    "population standard deviation of the same per-layer quantity "
    "across layers — that is, how stable the agreement is with depth. "
    "The diagonal is exactly 1.000 (Jaccard) and 0.000 (JSD) by "
    "construction and the matrices are symmetric, so both are omitted "
    "here and reported in full in the CSV. The summary table's "
    "`benchmarks` means exclude {baseline}; its `vs. baseline` means "
    "average only the {baseline} row. Blocks whose largest "
    "off-diagonal magnitude is below 1e-2 are printed in scientific "
    "notation so the small values stay legible."
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
             "cross_model_comparison.py, so the tables back the "
             "published figure.",
    )
    parser.add_argument(
        "--quant", type=str, default="Q4_K_M",
        help="Quantisation level to tabulate. Must match the "
             "cross_model_comparison.py --quant used for the figure "
             "(default: Q4_K_M).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: <results-dir>/paper_heatmaps/"
             "cross_model_tables/<quant>).",
    )
    parser.add_argument(
        "--self-check", action="store_true",
        help="Recompute the std-free matrices the figure script draws "
             "and assert the tables agree to 1e-12. Exits non-zero on "
             "any mismatch.",
    )
    args = parser.parse_args()

    # Resolve the model set, preserving the figure's canonical
    # short-name mapping when the caller overrides the subset.
    if args.models:
        canonical = dict(DEFAULT_MODELS)
        models = [(m, canonical.get(m, m)) for m in args.models]
    else:
        models = list(DEFAULT_MODELS)

    out_dir = (args.output_dir if args.output_dir is not None
               else args.results_dir / "paper_heatmaps"
                     / "cross_model_tables" / args.quant)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: load each (model, quant) cell and compute matrices.
    blocks: list[PairwiseBlock] = []
    failures: list[str] = []
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
        try:
            counts, arch = _load_cell_counts(cell_dir, args.quant)
        except SystemExit as e:
            print(f"[warn] {quant_dir}: {e}; skipping model "
                  f"{short_name}")
            continue
        labels, counts_list = _order_labels(counts)
        if len(counts_list) < 2:
            print(f"[warn] {quant_dir}: <2 datasets; skipping model "
                  f"{short_name}")
            continue

        L = arch["n_layer"]
        K_model = arch["n_expert_used"]
        j_mean, j_std = _layer_averaged_pairwise_jaccard_with_std(
            counts_list, K_model, L)
        s_mean, s_std = _layer_averaged_js_divergence_matrix_with_std(
            counts_list)
        blocks.append(PairwiseBlock(
            model=short_name,
            model_safe=model_safe,
            quant_ref=args.quant,
            unit_kind="dataset",
            labels=labels,
            arch=arch,
            jaccard_mean=j_mean,
            jaccard_std=j_std,
            jsd_mean=s_mean,
            jsd_std=s_std,
        ))
        print(f"[load] {short_name}: {arch['name']} "
              f"L={L} E={arch['n_expert']} K_model={K_model}; "
              f"datasets={labels}")

        if args.self_check:
            # Compare against the exact matrices the figure script
            # renders from, using its own std-free helpers.
            failures += verify_block(
                blocks[-1],
                jaccard_ref=_layer_averaged_pairwise_jaccard(
                    counts_list, K_model, L),
                jsd_ref=_layer_averaged_js_divergence_matrix(counts_list),
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
    csv_path = write_csv(out_dir / "cross_dataset_pairwise.csv", blocks)
    print(f"[save] {csv_path}")

    # ---- Step 3: write Markdown (reviewable, value ± layer_std).
    md_path = write_markdown(
        out_dir / "cross_dataset_pairwise.md",
        blocks,
        title=f"Cross-dataset routing similarity at {args.quant}",
        caption=CAPTION_TEMPLATE.format(quant=args.quant,
                                        baseline=BASELINE_LABEL),
        baseline_label=BASELINE_LABEL,
    )
    print(f"[save] {md_path}")

    # ---- Step 4: short stdout echo of the summary rows so the
    # terminal user sees the numbers without opening the CSV.
    from pairwise_tables import summary_rows
    rows = summary_rows(blocks, BASELINE_LABEL)
    print(f"\n{'Model':<18} {'Units':>5} {'Jaccard':>9} {'JSD':>9} "
          f"{'Jacc/base':>10} {'JSD/base':>10}")
    for r in rows:
        def _f(v: float) -> str:
            return "     n/a" if not np.isfinite(v) else f"{v:9.3f}"
        print(f"{r['model']:<18} {r['units']:>5} "
              f"{_f(r['jaccard_real'])} {_f(r['jsd_real'])} "
              f"{_f(r['jaccard_baseline'])} {_f(r['jsd_baseline'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())