#!/usr/bin/env python3
# type: ignore

"""Build the cross-quantization summary table for the MoE-routing paper.

Sibling peer of `cross_model_comparison.py` (which produces the
2 x 4 cross-model heatmap figure). This script summarises **cross-
quantization** agreement: for each (model, metric, dataset) cell, it
computes `mean ± std` of the metric value across all available
quantizations for that model.

Why this matters: the 2 x 4 figure shows routing similarity across
datasets **at one quantization** (`--quant`); the table summarises
how stable that similarity is **across quantizations** for each
model. Together they answer:

- *figure*: at Q4_K_M, how similar is BBH routing to MMLU routing?
- *table*: when we compare Q2_K vs Q4_K_M routing distributions, does
  the routing similarity between BBH and MMLU change much?

For each `(model, metric, dataset)` cell:
    mean  = average of metric(dataset, dataset') over dataset' != dataset,
            averaged across all quants with that model+dataset
    std   = std of the same per-quant values
    n_quants = how many quants actually contributed (RANDOM is often
              skipped — it has a different label set because it's a
              pre-aggregated random-router baseline, not a real benchmark)

Output (under `<output-dir>/`, default
`<results-dir>/paper_heatmaps/cross_quantization/`):

  - `cross_quantization_table.csv`   long-form: rows = one per
                                      (model, metric, dataset)
  - `cross_quantization_table.txt`    pretty-printed wide table (rows =
                                      `(model, metric)`, columns =
                                      datasets, cells = "mean ± std")

The CSV is the machine-readable form; the TXT mirrors it for
eyeballing in the terminal without opening a file.

Usage:
    python examples/eval-heatmaps/cross_quantization_table.py \
        --results-dir /mnt/data/results-llama-cpp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make the sibling `paper_heatmaps.py` importable so we can reuse the
# Jaccard / JSD math helpers and the lenient JSON loader. Both files
# live in `examples/eval-heatmaps/` so Python's per-script sys.path
# entry resolves these imports without any setup.py / PYTHONPATH plumbing.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from paper_heatmaps import (  # noqa: E402
    _layer_averaged_pairwise_jaccard,
    _layer_averaged_js_divergence_matrix,
    _make_k_values,
    _maybe_aggregate_random,
    _publish_dataset_label,
    _safe_json_load,
)


# ----------------------------------------------------------------- constants

# Default list of models to include in the table. Hardcoded so the
# script produces the same 4-model table every run regardless of
# what's in the results directory. Override via `--models` if the
# user wants to change the set.
DEFAULT_MODELS: list[tuple[str, str]] = [
    # (filesystem-safe model id, short name for the table).
    ("mradermacher--OLMoE-1B-7B-0924-Instruct-GGUF",       "OLMoE-1B-7B"),
    ("LiteLLMs--Mixtral-8x22B-Instruct-v0.1-GGUF",   "Mixtral-8x22B"),
    ("mradermacher--deepseek-moe-16b-chat-GGUF",      "DeepSeek-MoE-16B"),
    ("unsloth--Qwen3-30B-A3B-GGUF",                   "Qwen3-30B-A3B"),
    # ("allenai--OLMoE-1B-7B-0924-Instruct",       "OLMoE-1B-7B"),
    # ("mistralai--Mixtral-8x22B-Instruct-v0.1",   "Mixtral-8x22B"),
    # ("deepseek-ai--deepseek-moe-16b-chat",      "DeepSeek-MoE-16B"),
    # ("Qwen--Qwen3-30B-A3B",                   "Qwen3-30B-A3B"),
]

# Dataset order — UnifRand last so the random-router baseline
# always appears at the right edge of the table (matching the figure's
# BENCHMARK_ORDER convention from cross_model_comparison.py, which
# uses the same short label so the figure and this table line up
# column-for-column).
BENCHMARK_ORDER: list[str] = ["BBH", "HumanEval", "INCLUDE", "MMLU",
                                "PopQA", "UnifRand"]


# ----------------------------------------------------------------- caption

# Paper caption for the cross-quantization summary table. Embedded as
# the header of `cross_quantization_table.txt` and at the top of the
# stdout print so reviewers don't have to open the README to make
# sense of the numbers. Edit `TABLE_N_PLACEHOLDER` to your paper's
# actual table number before submission.
TABLE_N_PLACEHOLDER = "Table N"
TABLE_CAPTION: str = (
    f"{TABLE_N_PLACEHOLDER}: Cross-quantisation stability of routing "
    "similarity. For each (model, metric, dataset) cell, the value "
    "reports the mean ± std of the metric across all available "
    "quantisations of that model. The per-quant \"metric value\" is "
    "the mean of the metric's row in that quant's cross-dataset "
    "pairwise matrix, excluding the diagonal — i.e. how similar "
    "this dataset's routing distribution is to the average of all "
    "OTHER datasets in this quant. Rows are organised as (model, "
    "metric) pairs; columns are dataset labels. `n_quants` reports "
    "the number of quantisations that contributed (the `RANDOM` "
    "quant is skipped because its label set lacks the `UnifRand` "
    "row). Low `std` indicates that the cross-dataset routing "
    "similarity for this dataset is stable across quantisations; "
    "high `std` indicates that quantisation meaningfully affects the "
    "routing distribution."
)


# ------------------------------------------------------------------- helpers

def _load_cell_counts(model_dir: Path, quant: str
                      ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load every `moe-<ds>/expert_counts.json` under one (model, quant).

    Returns:
      counts: dict mapping publishable dataset label -> [L, E] int64 array
              (the marginal_expert_counts matrix from `aggregate`).
      arch  : dict with keys `name`, `n_layer`, `n_expert`,
              `n_expert_used` (taken from the first dataset's JSON).

    Self-contained (does not import from cross_model_comparison.py)
    so this file can be used independently.

    Uses `_safe_json_load` from paper_heatmaps.py to recover from
    partial UTF-8 / format corruption that the C++ binary leaves behind
    when killed mid-write.
    """
    cell_dir = model_dir / quant
    counts: dict[str, np.ndarray] = {}
    arch: dict[str, Any] | None = None
    for js_path in sorted(cell_dir.glob("moe-*/expert_counts.json")):
        ds_dir_name = js_path.parent.name  # e.g. "moe-mmlu"
        data = _safe_json_load(js_path)
        if data is None:
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
            continue
        publish_ds = _publish_dataset_label(ds_dir_name)
        counts[publish_ds] = m
        if arch is None:
            arch = {"name": arch_in.get("name", "unknown"),
                    "n_layer": L, "n_expert": E, "n_expert_used": k}

    # ---- load the random-router baseline as a 6th dataset ----
    # The random router produces results in two possible locations:
    #
    #   1. Per-benchmark, inside each `moe-<ds>/random/` subdirectory.
    #      We sum the marginal counts across all of these to build a
    #      synthetic top-level "UnifRand" dataset.
    #
    #   2. Already pre-aggregated, as a top-level
    #      `<cell>/random/expert_counts.json`. Some users run the
    #      random router once at the cell level and write the result
    #      here directly (it's the same arch + same [L, E] matrix).
    random_loaded = False
    pre_agg_path = cell_dir / "random" / "expert_counts.json"
    if pre_agg_path.is_file():
        rdata = _safe_json_load(pre_agg_path)
        if rdata is not None:
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

    if not random_loaded:
        # Fall back to summing per-benchmark `moe-<ds>/random/` JSONs.
        random_agg = _maybe_aggregate_random(cell_dir)
        if random_agg is not None and arch is not None:
            rand_counts, rand_meta = random_agg
            r_arch = next(iter(rand_meta.values()))["arch"]
            if (r_arch["n_layer"] == arch["n_layer"]
                    and r_arch["n_expert"] == arch["n_expert"]
                    and r_arch["n_expert_used"] == arch["n_expert_used"]):
                # The aggregator returns {"Random": agg} with the
                # mixed-case key — remap to our publishable label
                # "UnifRand".
                counts.setdefault("UnifRand",
                                   rand_counts["Random"])

    if arch is None:
        raise SystemExit(f"[error] {cell_dir}: no usable datasets found")
    return counts, arch


def _load_all_quants(model_dir: Path, short_name: str
                      ) -> list[dict[str, Any]]:
    """Load every quantization for one model.

    Returns a list of records (one per quant dict), each with:
        - 'quant':        str  (e.g. "Q2_K")
        - 'arch':         dict (from _load_cell_counts)
        - 'labels':       list[str]
        - 'counts_list':  list[np.ndarray]
        - 'K_model':      int
        - 'jaccard':      np.ndarray  (per-dataset pairwise Jaccard)
        - 'jsd':          np.ndarray  (per-dataset pairwise JSD)
    """
    records: list[dict[str, Any]] = []
    for qdir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        quant_name = qdir.name
        try:
            counts, arch = _load_cell_counts(model_dir, quant_name)
        except SystemExit:
            continue
        labels = [d for d in BENCHMARK_ORDER if d in counts]
        labels += [d for d in sorted(counts.keys()) if d not in labels]
        counts_list = [counts[d] for d in labels]
        if len(counts_list) < 2:
            continue
        L = arch["n_layer"]
        E = arch["n_expert"]
        k_model = arch["n_expert_used"]
        jacc = _layer_averaged_pairwise_jaccard(counts_list, k_model, L)
        jsd = _layer_averaged_js_divergence_matrix(counts_list)
        records.append({
            "quant": quant_name,
            "arch": arch,
            "labels": labels,
            "counts_list": counts_list,
            "K_values": _make_k_values(
                E, k_model, k_min=1, k_max_frac=0.50, num_ks=21
            ),
            "K_model": k_model,
            "jaccard": jacc,
            "jsd": jsd,
        })
    return records


def _order_labels(counts: dict[str, np.ndarray]
                  ) -> tuple[list[str], list[np.ndarray]]:
    """Order datasets (BBH / HumanEval / ...) in publishable order."""
    labels = [d for d in BENCHMARK_ORDER if d in counts]
    labels += [d for d in sorted(counts.keys()) if d not in labels]
    counts_list = [counts[d] for d in labels]
    return labels, counts_list


def _compute_cross_quantization_table(
        per_model_records: dict[str, list[dict[str, Any]]]
        ) -> list[dict[str, Any]]:
    """Build the cross-quantization summary table.

    For each (model, metric, dataset), compute mean ± std of the
    metric value across all available quantizations for that model.

    For each (model, quant) we compute the per-dataset scalar metric
    value as the mean of the metric between this dataset and all OTHER
    datasets (excluding the diagonal). Then we take mean ± std across
    quants — this gives a single "central tendency" per cell with a
    stability indicator (the std) that tells the reviewer how much the
    routing similarity for this dataset varies across quantizations.

    Returns: list of dicts with keys
        `model, metric, dataset, mean, std, n_quants`.

    Quants whose dataset label list doesn't match the canonical (first
    quant's) list are silently skipped — they can't contribute to a
    cross-quantization mean that requires matching datasets.
    """
    rows: list[dict[str, Any]] = []
    for short_name, quant_records in per_model_records.items():
        if not quant_records:
            continue

        # Determine the canonical dataset label list from the first
        # quant (the "anchor" set of datasets). All later quants are
        # expected to match this; mismatches are silently dropped from
        # the cross-quantization mean.
        labels = quant_records[0]["labels"]
        n_datasets = len(labels)

        # For each quant, compute the per-dataset scalar metric value:
        # the mean of the metric row (excluding the diagonal).
        per_quant_metric: dict[str, dict[str, list[float]]] = {
            "jaccard": {}, "jsd": {},
        }
        for qrec in quant_records:
            if qrec["labels"] != labels:
                continue
            for metric_name in ("jaccard", "jsd"):
                mat = qrec[metric_name]
                for i, d in enumerate(labels):
                    # Mean of the row excluding the diagonal.
                    others = [mat[i, j] for j in range(n_datasets)
                              if j != i]
                    per_quant_metric[metric_name].setdefault(d, []).append(
                        float(np.mean(others))
                    )

        # Aggregate mean ± std across quants for each (metric, dataset).
        for metric_name in ("jaccard", "jsd"):
            for d in labels:
                values = per_quant_metric[metric_name].get(d, [])
                if not values:
                    continue
                arr = np.asarray(values, dtype=np.float64)
                rows.append({
                    "model": short_name,
                    "metric": metric_name,
                    "dataset": d,
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "n_quants": int(arr.size),
                })
    return rows


def _print_wide_table(rows: list[dict[str, Any]]) -> None:
    """Pretty-print the cross-quantization table in wide format.

    Layout (rows = (model, metric), columns = datasets, cells =
    "mean ± std"):
                                BBH        HumanEval   INCLUDE     MMLU       PopQA     UnifRand
        OLMoE-1B-7B   Jaccard    0.30±0.05  0.20±0.03   ...
                      JSD        ...
        Mixtral-8x22B Jaccard    ...
        ...

    Datasets are sorted with `UnifRand` last (matching the
    BENCHMARK_ORDER ordering) so the baseline always appears at the
    right edge of the table.
    """
    if not rows:
        print("[table] no cross-quantization rows to display")
        return

    models = sorted({r["model"] for r in rows})
    metrics = ("jaccard", "jsd")

    canonical_order = BENCHMARK_ORDER
    datasets = sorted(
        {r["dataset"] for r in rows},
        key=lambda d: (canonical_order.index(d)
                       if d in canonical_order
                       else len(canonical_order) + 1),
    )

    def _fmt(r: dict[str, Any]) -> str:
        return f"{r['mean']:.3f}±{r['std']:.3f}"

    sample_value = _fmt(rows[0])
    value_width = max(14, len(sample_value))
    metric_width = max(len(m) for m in metrics) + 2
    model_width = max(len(m) for m in models) + 2

    # Header
    header = " " * (model_width + metric_width)
    for d in datasets:
        header += f" {d:>{value_width}}"
    print(header)

    # Rows
    metric_label = {"jaccard": "Jaccard", "jsd": "JSD"}
    for model in models:
        for j, metric in enumerate(metrics):
            label = metric_label[metric]
            if j == 0:
                row = f"{model:<{model_width}} {label:<{metric_width}}"
            else:
                # Continuation row — leave the model column blank so
                # the metric-label aligns vertically under its sibling.
                row = f"{'':<{model_width}} {label:<{metric_width}}"
            for d in datasets:
                match = [r for r in rows
                         if r["model"] == model
                         and r["metric"] == metric
                         and r["dataset"] == d]
                cell = _fmt(match[0]) if match else "n/a"
                row += f" {cell:>{value_width}}"
            print(row)
        # Blank line between models for readability.
        print()


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
             "(repeatable). Default: the 4 hardcoded models.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: <results-dir>/paper_heatmaps/"
             "cross_quantization).",
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
               else args.results_dir / "paper_heatmaps"
               / "cross_quantization")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: load every model × quant cell.
    per_model_records: dict[str, list[dict[str, Any]]] = {}
    for model_safe, short_name in models:
        model_dir = args.results_dir / model_safe
        if not model_dir.is_dir():
            print(f"[warn] {model_dir}: not found; skipping model "
                  f"{short_name}")
            continue
        records = _load_all_quants(model_dir, short_name)
        if not records:
            print(f"[warn] {model_safe}: no usable quantizations found; "
                  f"skipping model {short_name}")
            continue
        per_model_records[short_name] = records
        print(f"[load] {short_name}: {len(records)} quant(s) loaded: "
              f"{[r['quant'] for r in records]}")

    if not per_model_records:
        raise SystemExit("[error] no models loaded successfully")

    # ---- Step 2: build the table.
    rows = _compute_cross_quantization_table(per_model_records)

    # ---- Step 3: write CSV (long-form).
    csv_path = out_dir / "cross_quantization_table.csv"
    with open(csv_path, "w") as f:
        f.write("model,metric,dataset,mean,std,n_quants\n")
        for row in rows:
            f.write(f"{row['model']},{row['metric']},{row['dataset']},"
                    f"{row['mean']:.6f},{row['std']:.6f},{row['n_quants']}\n")
    print(f"[save] {csv_path}")

    # ---- Step 4: write TXT (pretty-printed wide table for eyeballing).
    txt_path = out_dir / "cross_quantization_table.txt"
    lines: list[str] = []
    # Caption first so reviewers see it before the numbers.
    lines.append(TABLE_CAPTION)
    lines.append("")
    lines.append("=" * 80)
    lines.append(" CROSS-QUANTIZATION TABLE "
                 "(mean ± std across all available quantizations)")
    lines.append("=" * 80)
    # Capture stdout to a list by redirecting print.
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_wide_table(rows)
    lines.append(buf.getvalue().rstrip("\n"))
    lines.append("")
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"[save] {txt_path}")

    # ---- Step 5: pretty-print to stdout too.
    print()
    print("\n".join(lines))

    return 0


if __name__ == "__main__":
    sys.exit(main())
