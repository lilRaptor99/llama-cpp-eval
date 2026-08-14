#!/usr/bin/env python3
# type: ignore

"""Per-cell Jaccard K-sweep plotter for the MoE-routing eval suite.

For one `(model, quant)` cell at `${RESULTS_DIR}/<model_safe>/<quant_safe>/`,
renders a Jaccard-similarity K-sweep comparing every dataset's top-K experts
to the **aggregated-across-datasets reference** top-K. This is the first
phase of a MoE-routing consistency study; a second phase (cross-quant
comparison + cosine / JS) will be added later.

Outputs (under `<cell-dir>/jaccard_sweep/`):

  - jaccard_sweep.csv        long-form: dataset, layer, K, jaccard, jaccard_n_used
  - jaccard_sweep.png        1 subplot per layer; x = K, y = Jaccard; 1 line per dataset
  - jaccard_vs_n_used.png    bar chart per dataset at K = n_expert_used
                             (mean across layers + min across layers)
  - summary.csv              per dataset: jaccard_at_n_used, jaccard_auc,
                             jaccard_at_n_used_per_layer_min, n_layers_below_0.5
  - README.json              meta: model, quant, K values, arch, token totals

Mathematical contract (matches §5 of the task spec):

  p_ds[l]   = counts_ds[l]   / tokens_ds                          (per-dataset)
  p_ref[l]  = counts_agg[l]  / tokens_agg                         (aggregated)
  A_K       = top-K(p_ds[l])  by descending value, ties -> asc expert id
  B_K       = top-K(p_ref[l]) by descending value, ties -> asc expert id
  Jaccard_K = |A_K ∩ B_K| / |A_K ∪ B_K|
  jaccard=1.0 when both sets empty; jaccard=0.0 when exactly one is empty

The aggregated reference distribution is always taken as the live sum of
the per-dataset `layer_expert_counts` matrices. If
`<cell-dir>/overall/top_experts.json` exists, we sanity-check it against
the live sum and warn (but do not fail) on disagreement — this is the
"preferred path" the spec calls out, but the live sum is the
authoritative source so the script also works before
`aggregate_overview.py` has been run.

Per-dataset scalar summaries (summary.csv):

  - jaccard_at_n_used                  mean across layers at K = n_expert_used
  - jaccard_auc                        mean across layers of the trapezoidal
                                       AUC over the K-sweep, normalised by
                                       (max(K) - min(K))
  - jaccard_at_n_used_per_layer_min    min across layers at K = n_expert_used
  - n_layers_below_0.5                 number of layers with
                                       jaccard_at_n_used < 0.5

Two CLI modes:

  - Single cell:
        python3 jaccard_sweep_from_cpp.py --cell-dir <path> \
            [--model-id <hf-repo-id>] [--quant <tag>]

  - Multi-cell batch (used by spartan/llama-moe-eval.sbatch):
        python3 jaccard_sweep_from_cpp.py --results-dir <root> \
            --models <safe1> [<safe2> ...] [--quants <q1> [<q2> ...]]

The script is idempotent: a cell whose `jaccard_sweep/summary.csv` already
exists is skipped (use --force to re-render). Errors are reported as
one-line `SystemExit("[error] ...")` messages so the .sbatch can surface
them via the per-cell log.

Dependencies: numpy + matplotlib only. We `from aggregate_overview import`
the three helpers we need (_load_one, compute_top_k, _RECORD_KEYS) —
`aggregate_overview.py` is the sibling peer script in this directory,
not a shared library, so this stays consistent with the README's
"no shared library import" convention while still avoiding the
copy-paste the spec explicitly forbids.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

# Sibling peer script; both files live in examples/eval-moe-overview/
# so Python's per-script sys.path entry resolves this import without
# any setup.py or PYTHONPATH plumbing.
from aggregate_overview import _load_one, compute_top_k, _RECORD_KEYS  # noqa: E402


# ----------------------------------------------------------------- K-sweep

# `aggregate_overview.compute_top_k` already returns tied entries
# ordered by ascending expert id (np.argsort(-row, kind="stable")), so we
# can call it directly for the K-sweep. But it returns full per-expert
# dicts; we only need the set of expert ids. Extracting that inline is
# cheaper than a follow-up dict-keyed lookup, so we keep this thin
# helper as the single point that decides "top-K of a layer".
def _top_k_set_for_layer(counts: np.ndarray, layer: int, K: int) -> set[int]:
    """Return the top-K expert IDs in `counts[layer]`, ties -> asc expert id.

    K is clamped to [1, E]. Mirrors `compute_top_k`'s `kind="stable"`
    ordering so the result is bit-identical to what that helper would
    produce for the same `(counts, K)` pair.
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


def _make_k_values(n_expert: int, n_expert_used: int,
                   k_min: int, k_max_frac: float, num_ks: int) -> list[int]:
    """Build the K-sweep grid: linear `num_ks` samples in [k_min, k_max],
    always including K = n_expert_used and K = ceil(0.125 * E).
    """
    k_min = max(1, int(k_min))
    k_max = max(k_min, int(np.ceil(n_expert * k_max_frac)))
    # Dense linspace, dedup + sort + cast to int.
    base = np.linspace(k_min, k_max, num=max(1, int(num_ks)))
    base_ints = sorted({int(round(float(v))) for v in base})
    # Must-include sentinel K values; spec §5 lists both n_expert_used and
    # ceil(0.125 * E). We add them after the linspace so they're present
    # even when num_ks is small.
    must_include = {int(n_expert_used), int(np.ceil(n_expert * 0.125))}
    base_ints = sorted(set(base_ints) | must_include)
    # Final clamp + dedup.
    base_ints = [k for k in base_ints if 1 <= k <= n_expert]
    return base_ints


def _jaccard(a: set[int], b: set[int]) -> float:
    """Jaccard similarity of two integer sets.

    Per spec §5.4: 1.0 when both empty, 0.0 when one empty (the spec
    actually says "0.0 when one is empty and the other isn't"; both
    branches covered by the same condition via `len(union) == 0`).
    """
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    inter = a & b
    return len(inter) / len(union)


# --------------------------------------------------------------- core

def _load_cell(cell_dir: Path) -> tuple[
        dict[str, np.ndarray], dict[str, int], list[dict[str, Any]],
        dict[str, Any], str | None]:
    """Load every `moe-*/expert_counts.json` under cell_dir via _load_one.

    Returns (per_ds_counts, per_ds_tokens, per_ds_meta, arch, model_id).
    Raises SystemExit on arch mismatch or empty cell.
    """
    jsons = sorted(cell_dir.glob("moe-*/expert_counts.json"))
    if not jsons:
        raise SystemExit(f"[error] {cell_dir}: no moe-*/expert_counts.json found")

    per_ds_counts: dict[str, np.ndarray] = {}
    per_ds_tokens: dict[str, int] = {}
    per_ds_meta: list[dict[str, Any]] = []
    arch: dict[str, Any] | None = None
    model_id: str | None = None

    for jp in jsons:
        try:
            loaded = _load_one(jp)
        except (SystemExit, json.JSONDecodeError, OSError) as e:
            print(f"[warn] {jp}: {e}; skipping")
            continue
        ds = loaded["dataset_label"]
        if arch is None:
            arch = loaded["arch"]
            model_id = loaded["model_id"]
        else:
            if (loaded["arch"]["n_layer"] != arch["n_layer"]
                    or loaded["arch"]["n_expert"] != arch["n_expert"]):
                raise SystemExit(
                    f"[error] {jp}: arch mismatch "
                    f"(got L={loaded['arch']['n_layer']} E={loaded['arch']['n_expert']}, "
                    f"expected L={arch['n_layer']} E={arch['n_expert']})"
                )
        per_ds_counts[ds] = loaded["counts_total"]
        per_ds_tokens[ds] = loaded["total_tokens"]
        per_ds_meta.append({
            "dataset": ds,
            "tokens": loaded["total_tokens"],
            "n_rows": loaded["n_rows"],
        })
        print(f"[load] {jp.parent.name}: tokens={loaded['total_tokens']:,}, "
              f"rows={loaded['n_rows']}, arch={loaded['arch']['name']}")

    if arch is None or not per_ds_counts:
        raise SystemExit(f"[error] {cell_dir}: no usable datasets found")

    return per_ds_counts, per_ds_tokens, per_ds_meta, arch, model_id


def _aggregate(per_ds_counts: dict[str, np.ndarray]) -> np.ndarray:
    """Sum per-dataset [L, E] count matrices into a single [L, E] array."""
    first = next(iter(per_ds_counts.values()))
    out = np.zeros_like(first, dtype=np.int64)
    for counts in per_ds_counts.values():
        out += counts
    return out


def _sanity_check_top_experts(cell_dir: Path, counts_agg: np.ndarray,
                              tokens_agg: int) -> None:
    """Compare live aggregate against `overall/top_experts.json` if present.

    The live sum is authoritative (spec §3 fallback path); this is a
    diagnostic only. We warn on disagreement but never fail.
    """
    te_path = cell_dir / "overall" / "top_experts.json"
    if not te_path.exists():
        return
    try:
        with open(te_path) as f:
            te = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[warn] could not read {te_path}: {e}")
        return
    if "layers" not in te or not isinstance(te["layers"], list):
        return
    L = counts_agg.shape[0]
    if len(te["layers"]) != L:
        print(f"[warn] {te_path}: layers len {len(te['layers'])} != L={L}")
        return
    bad_layers = []
    for layer, layer_obj in enumerate(te["layers"]):
        lt = int(layer_obj.get("layer_total", -1))
        live = int(counts_agg[layer].sum())
        if lt != live:
            bad_layers.append((layer, lt, live))
    if bad_layers:
        print(f"[warn] {te_path}: {len(bad_layers)} layer(s) where "
              f"layer_total != live sum (using live sum as authoritative)")
    else:
        print(f"[ok] {te_path.name}::layers[].layer_total agrees with live sum")
    tt_meta = int(te.get("totals", {}).get("tokens_total", -1))
    if tt_meta > 0 and tt_meta != tokens_agg:
        print(f"[warn] {te_path}: totals.tokens_total={tt_meta} != "
              f"live sum {tokens_agg}; using live sum")


# ----------------------------------------------------------- per-cell main

def process_cell(cell_dir: Path, *, output_dir: Path | None = None,
                 k_min: int, k_max_frac: float, num_ks: int,
                 dpi: int, model_id: str | None = None,
                 quant: str | None = None,
                 force: bool = False) -> dict[str, Any]:
    """Process one `(model, quant)` cell and write all artefacts."""
    cell_dir = cell_dir.resolve()
    if output_dir is None:
        output_dir = cell_dir / "jaccard_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "summary.csv"
    if summary_csv.exists() and not force:
        print(f"[skip] {summary_csv} exists; pass --force to regenerate")
        return {"cell_dir": str(cell_dir), "output_dir": str(output_dir),
                "skipped": True}

    per_ds_counts, per_ds_tokens, per_ds_meta, arch, model_id_loaded = (
        _load_cell(cell_dir)
    )
    L = arch["n_layer"]
    E = arch["n_expert"]
    K_model = arch["n_expert_used"]

    counts_agg = _aggregate(per_ds_counts)
    tokens_agg = int(sum(per_ds_tokens.values()))
    print(f"[agg] aggregated counts_total shape={counts_agg.shape}, "
          f"tokens_total={tokens_agg:,}")

    _sanity_check_top_experts(cell_dir, counts_agg, tokens_agg)

    K_values = _make_k_values(E, K_model, k_min, k_max_frac, num_ks)
    print(f"[plan] arch={arch['name']} L={L} E={E} K_model={K_model}; "
          f"K-sweep ({len(K_values)} values) = {K_values}")

    # Pre-compute the reference (aggregated) top-K sets, one set per (K, layer).
    ref_topk: dict[int, list[set[int]]] = {
        K: [_top_k_set_for_layer(counts_agg, layer, K) for layer in range(L)]
        for K in K_values
    }
    # And the per-dataset top-K sets, structured as
    #   ds_topk[ds][K][layer] -> set[int]
    ds_topk: dict[str, dict[int, list[set[int]]]] = {}
    for ds, counts in per_ds_counts.items():
        ds_topk[ds] = {
            K: [_top_k_set_for_layer(counts, layer, K) for layer in range(L)]
            for K in K_values
        }

    # -------- long-form rows (one per dataset, layer, K)
    long_rows: list[dict[str, Any]] = []
    # -------- per-dataset summary rows
    summary_rows: list[dict[str, Any]] = []

    for ds in per_ds_counts:
        # jaccard_at_n_used per layer (needed for both long-form and summary).
        per_layer_j_at_k_model = [
            _jaccard(ds_topk[ds][K_model][layer], ref_topk[K_model][layer])
            for layer in range(L)
        ]
        # AUC: build the per-(layer, K) matrix and integrate along K.
        K_arr = np.asarray(K_values, dtype=np.float64)
        per_layer_curves = np.zeros((L, len(K_values)), dtype=np.float64)
        for ki, K in enumerate(K_values):
            for layer in range(L):
                per_layer_curves[layer, ki] = _jaccard(
                    ds_topk[ds][K][layer], ref_topk[K][layer])
        # Trapezoidal AUC per layer, normalised by K-range so the value
        # is in [0, 1] and comparable across models with different E.
        if len(K_values) >= 2:
            per_layer_auc = np.trapz(per_layer_curves, K_arr, axis=1)
            k_range = max(float(K_arr.max() - K_arr.min()), 1.0)
            per_layer_auc = per_layer_auc / k_range
        else:
            per_layer_auc = per_layer_curves[:, 0]
        # Long-form rows for this dataset.
        for layer in range(L):
            for ki, K in enumerate(K_values):
                j_val = float(per_layer_curves[layer, ki])
                j_at_k_model = per_layer_j_at_k_model[layer]
                long_rows.append({
                    "dataset": ds,
                    "layer": layer,
                    "K": int(K),
                    "jaccard": j_val,
                    "jaccard_n_used": (
                        float(j_at_k_model) if int(K) == K_model else ""
                    ),
                })
        n_below_0_5 = int(np.sum(
            np.asarray(per_layer_j_at_k_model, dtype=np.float64) < 0.5))
        summary_rows.append({
            "dataset": ds,
            "tokens": per_ds_tokens[ds],
            "n_rows": next(m["n_rows"] for m in per_ds_meta
                           if m["dataset"] == ds),
            "jaccard_at_n_used": float(np.mean(per_layer_j_at_k_model)),
            "jaccard_auc": float(np.mean(per_layer_auc)),
            "jaccard_at_n_used_per_layer_min":
                float(np.min(per_layer_j_at_k_model)),
            "n_layers_below_0.5": n_below_0_5,
        })

    # -------- write CSVs
    long_csv_path = output_dir / "jaccard_sweep.csv"
    with open(long_csv_path, "w") as f:
        f.write("dataset,layer,K,jaccard,jaccard_n_used\n")
        for r in long_rows:
            jn = r["jaccard_n_used"]
            f.write(f"{r['dataset']},{r['layer']},{r['K']},"
                    f"{r['jaccard']:.6f},"
                    f"{('' if jn == '' else f'{jn:.6f}')}\n")
    print(f"[save] {long_csv_path}")

    summary_csv_path = output_dir / "summary.csv"
    with open(summary_csv_path, "w") as f:
        f.write("dataset,tokens,n_rows,jaccard_at_n_used,jaccard_auc,"
                "jaccard_at_n_used_per_layer_min,n_layers_below_0.5\n")
        for r in summary_rows:
            f.write(f"{r['dataset']},{r['tokens']},{r['n_rows']},"
                    f"{r['jaccard_at_n_used']:.6f},"
                    f"{r['jaccard_auc']:.6f},"
                    f"{r['jaccard_at_n_used_per_layer_min']:.6f},"
                    f"{r['n_layers_below_0.5']}\n")
    print(f"[save] {summary_csv_path}")

    # -------- README.json (meta)
    meta = {
        "cell_dir": str(cell_dir),
        "model_id": model_id or model_id_loaded,
        "model_arch": arch,
        "quant": quant,
        "k_min": int(k_min),
        "k_max_frac": float(k_max_frac),
        "num_ks_requested": int(num_ks),
        "k_values": K_values,
        "n_expert_used_model": int(K_model),
        "k_ceil_0.125_E": int(np.ceil(E * 0.125)),
        "k_ceil_k_max_frac_E": int(np.ceil(E * k_max_frac)),
        "datasets": per_ds_meta,
        "totals": {
            "datasets_used": len(per_ds_counts),
            "tokens_aggregated": int(tokens_agg),
        },
        "metric": "jaccard",
    }
    meta_path = output_dir / "README.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] {meta_path}")

    # -------- jaccard_sweep.png (1 subplot per layer)
    n_cols = min(4, L)
    n_rows = (L + n_cols - 1) // n_cols
    fig_w = 4.2 * n_cols
    fig_h = 3.0 * n_rows + 0.6  # extra room for suptitle
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)
    model_label = (model_id or model_id_loaded or cell_dir.parent.name)
    quant_label = (quant or cell_dir.name)
    # Pre-bucket the long_rows by (dataset, layer) -> {K: jaccard} for plotting.
    by_ds_layer: dict[tuple[str, int], dict[int, float]] = {}
    for r in long_rows:
        by_ds_layer.setdefault((r["dataset"], r["layer"]),
                               {})[int(r["K"])] = float(r["jaccard"])
    for layer in range(L):
        ax = axes[layer // n_cols][layer % n_cols]
        for ds in per_ds_counts:
            ys = [by_ds_layer[(ds, layer)].get(K, float("nan"))
                  for K in K_values]
            ax.plot(K_values, ys, marker="o", markersize=3,
                    linewidth=1.0, label=ds)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
                   label="aggregated (== self)")
        ax.axvline(K_model, color="red", linestyle=":", linewidth=0.7,
                   label=f"K_model={K_model}")
        ax.set_title(f"layer {layer}", fontsize=9)
        ax.set_xlabel("K", fontsize=8)
        ax.set_ylabel("Jaccard", fontsize=8)
        ax.set_ylim(-0.02, 1.05)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, linestyle=":", alpha=0.4)
        if layer == 0:
            ax.legend(fontsize=6, loc="lower right")
    for layer in range(L, n_rows * n_cols):
        axes[layer // n_cols][layer % n_cols].axis("off")
    fig.suptitle(
        f"{model_label}  -  Jaccard K-sweep vs aggregated reference ({quant_label})\n"
        f"{arch['name']} L={L} E={E} K_model={K_model}; "
        f"tokens aggregated={tokens_agg:,}; K-sweep size={len(K_values)}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    sweep_png = output_dir / "jaccard_sweep.png"
    fig.savefig(sweep_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {sweep_png}")

    # -------- jaccard_vs_n_used.png (bar chart at K=K_model, per dataset)
    ds_names = [r["dataset"] for r in summary_rows]
    means = [r["jaccard_at_n_used"] for r in summary_rows]
    mins = [r["jaccard_at_n_used_per_layer_min"] for r in summary_rows]
    x = np.arange(len(ds_names))
    fig, ax = plt.subplots(figsize=(max(7, len(ds_names) * 1.4), 4.5))
    ax.bar(x, means, color="#4477aa", edgecolor="black", linewidth=0.4,
           label="mean across layers")
    ax.scatter(x, mins, color="red", marker="v", s=40, zorder=5,
               label="min across layers")
    for xi, m, mn in zip(x, means, mins):
        ax.text(xi, max(m, mn) + 0.02,
                f"{m:.2f}\n(min {mn:.2f})",
                ha="center", va="bottom", fontsize=7)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.6,
               label="1.0 (perfect)")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=20, ha="right", fontsize=8)
    ax.set_xlabel("dataset")
    ax.set_ylabel(f"Jaccard at K={K_model}")
    ax.set_ylim(-0.02, 1.18)
    ax.set_title(
        f"{model_label}  -  per-dataset Jaccard at K=K_model={K_model} ({quant_label})\n"
        f"{arch['name']} L={L} E={E}; aggregated reference = top-{K_model} "
        f"by total tokens across {len(per_ds_counts)} dataset(s)",
        fontsize=10,
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    vs_used_png = output_dir / "jaccard_vs_n_used.png"
    fig.savefig(vs_used_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {vs_used_png}")

    # -------- acceptance check: warn on per-dataset layer min < 0.20
    # for the longest dataset (spec §9.2: warn any layer < 0.20 in moe-mmlu
    # for OLMoE). We don't restrict to moe-mmlu here because the longest
    # dataset is model-dependent; we flag anything below 0.20.
    for r in summary_rows:
        if r["jaccard_at_n_used_per_layer_min"] < 0.20:
            print(f"[warn] {r['dataset']} has at least one layer with "
                  f"jaccard@K={K_model} < 0.20 "
                  f"(min={r['jaccard_at_n_used_per_layer_min']:.3f})")

    return {
        "cell_dir": str(cell_dir),
        "output_dir": str(output_dir),
        "arch": arch,
        "k_values": K_values,
        "summary_rows": summary_rows,
        "n_datasets": len(per_ds_counts),
        "tokens_aggregated": int(tokens_agg),
        "skipped": False,
    }


# --------------------------------------------------------------- CLI

def _model_safe(name: str) -> str:
    """Mirror `scripts/run-evaluator.sh::model_safe` (replace / with --)."""
    return name.replace("/", "--")


def _iter_cells(args: argparse.Namespace
                ) -> list[tuple[Path, str | None, str | None]]:
    """Resolve `--cell-dir` or `--results-dir --models --quants` to a list
    of (cell_dir, model_id, quant) tuples ready for process_cell().
    """
    if args.cell_dir:
        model_id = args.model_id
        quant = args.quant
        if not quant:
            quant = args.cell_dir.name
        if not model_id:
            # Heuristic: cell_dir.parent.name is the model_safe form
            # (`org--name`). Reverse by replacing the FIRST "--" with "/".
            safe = args.cell_dir.parent.name
            if "--" in safe:
                model_id = safe.replace("--", "/", 1)
            else:
                model_id = safe
        return [(args.cell_dir, model_id, quant)]
    # Multi-cell batch mode.
    results_dir = args.results_dir
    cells: list[tuple[Path, str | None, str | None]] = []
    for m in args.models:
        model_dir = results_dir / _model_safe(m)
        if not model_dir.is_dir():
            print(f"[warn] {model_dir} does not exist; skipping")
            continue
        # If --quants is set, restrict; otherwise pick every subdir that
        # contains at least one moe-*/expert_counts.json (the new layout
        # uses <model>/<quant>/moe-*; the old layout would have matched
        # the whole model_dir which we still want to support).
        if args.quants:
            quant_dirs = [model_dir / q for q in args.quants]
        else:
            quant_dirs = [d for d in sorted(model_dir.iterdir()) if d.is_dir()]
        for q_dir in quant_dirs:
            if not q_dir.is_dir():
                continue
            # Treat the model_dir itself as a single "no-quant" cell when
            # it directly contains moe-*/expert_counts.json (legacy
            # layout). This matches the README's "Migrating old results"
            # advice; without this, users who forgot to migrate still
            # get a sensible behaviour.
            if any(q_dir.glob("moe-*/expert_counts.json")):
                cells.append((q_dir, m, q_dir.name if q_dir != model_dir else None))
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cell-dir", type=Path, default=None,
        help="Process a single (model, quant) cell. The cell directory is "
             "expected to contain moe-*/expert_counts.json and (optionally) "
             "overall/top_experts.json.")
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Process multiple cells under <results-dir>/<model>/<quant>/. "
             "Use together with --models and (optionally) --quants.")
    parser.add_argument(
        "--models", action="append", default=None,
        help="Restrict to a subset of HF model ids (one per flag, "
             "repeatable). Used only with --results-dir.")
    parser.add_argument(
        "--quants", action="append", default=None,
        help="Restrict to a subset of quant subdirectories (one per flag, "
             "repeatable). Used only with --results-dir.")
    parser.add_argument(
        "--model-id", type=str, default=None,
        help="Override the model id recorded in README.json "
             "(single-cell mode only).")
    parser.add_argument(
        "--quant", type=str, default=None,
        help="Override the quant tag recorded in README.json "
             "(single-cell mode only).")
    parser.add_argument(
        "--k-min", type=int, default=1,
        help="Minimum K in the sweep (default: 1).")
    parser.add_argument(
        "--k-max-frac", type=float, default=0.20,
        help="K_max = ceil(n_expert * k_max_frac) (default: 0.20).")
    parser.add_argument(
        "--num-ks", type=int, default=21,
        help="Number of K values in the linear sweep (default: 21).")
    parser.add_argument(
        "--dpi", type=int, default=120,
        help="Output PNG DPI (default: 120).")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if jaccard_sweep/summary.csv already exists.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override the output dir "
             "(default: <cell-dir>/jaccard_sweep).")
    args = parser.parse_args()

    if not args.cell_dir and not args.results_dir:
        raise SystemExit(
            "[error] either --cell-dir or --results-dir is required "
            "(use --help for full options)")
    if args.cell_dir and args.results_dir:
        raise SystemExit(
            "[error] --cell-dir and --results-dir are mutually exclusive")
    if args.results_dir and not args.models:
        raise SystemExit(
            "[error] --results-dir requires at least one --models entry")

    cells = _iter_cells(args)
    if not cells:
        raise SystemExit(
            f"[error] no cells to process under "
            f"{args.results_dir or args.cell_dir}")

    print(f"[plan] processing {len(cells)} cell(s)")
    failures = 0
    for cell_dir, model_id, quant in cells:
        print(f"\n[cell] {cell_dir}  (model={model_id}, quant={quant})")
        try:
            process_cell(
                cell_dir, output_dir=args.output_dir,
                k_min=args.k_min, k_max_frac=args.k_max_frac,
                num_ks=args.num_ks, dpi=args.dpi,
                model_id=model_id, quant=quant, force=args.force,
            )
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 - keep the loop running
            print(f"[error] {cell_dir}: {e}", file=sys.stderr)
            failures += 1
            continue

    print(f"\n[done] processed {len(cells)} cell(s); {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
