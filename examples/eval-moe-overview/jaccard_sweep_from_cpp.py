#!/usr/bin/env python3
# type: ignore

"""Per-cell cross-dataset top-K alignment plotter for the MoE-routing eval suite.

For one `(model, quant)` cell at `${RESULTS_DIR}/<model_safe>/<quant_safe>/`,
renders two views of cross-dataset alignment on the **top-K expert sets**:

  1. A pairwise heatmap (dataset × dataset) at K = n_expert_used,
     averaged across layers — the visual.
  2. A single-number-per-K view: x = K, y = cross-dataset Jaccard.
     Two complementary metrics overlaid (generalized |∩|/|∪| + mean pairwise).

Outputs (under `<cell-dir>/jaccard_sweep/`):

  - jaccard_pairwise.csv     long-form: K, layer, dataset_a, dataset_b, jaccard
                             (every pair of datasets; one row per (K, layer))
  - jaccard_pairwise.png         heatmap of pairwise Jaccard at K = K_model,
                                 averaged across layers; primary heatmap
  - jaccard_pairwise_K{K}.png    additional heatmap at K = 2 * K_model
                                 and K = 3 * K_model (one file per K, with
                                 K embedded in the filename), same
                                 averaging/colormap as the primary
  - jaccard_pairwise_K2x.png     convenience alias for the 2*K_model heatmap
                                 (always present when E >= 2*K_model)
  - jaccard_pairwise_K3x.png     convenience alias for the 3*K_model heatmap
                                 (always present when E >= 3*K_model)
  - jaccard_global.csv       long-form: K, layer, jaccard_union, jaccard_pairwise_mean
                             (single-number-per-(K, layer) view of cross-dataset alignment)
  - jaccard_global.png       x = K, y = cross-dataset Jaccard; thin per-layer lines
                             + bold mean across layers; both union and pairwise-mean
                             metrics overlaid for comparison
  - summary.csv              per dataset: tokens, n_rows,
                             mean_jaccard_to_others_at_n_used,
                             min_jaccard_to_others_at_n_used
  - README.json              meta: model, quant, K values, arch, token totals,
                             cross-dataset alignment summary at K=K_model

Mathematical contract:

  For each layer l and K value:
    A_i_K[l]  = top-K experts in dataset i, layer l
                (descending count, ties -> ascending expert id)
    |∩|_K[l]  = |∩_i A_i_K[l]|              (intersection across datasets)
    |∪|_K[l]  = |∪_i A_i_K[l]|              (union across datasets)
    generalized Jaccard (strict consensus): |∩|_K[l] / |∪|_K[l]
    mean pairwise Jaccard:                  mean over all C(N, 2) pairs of
                                            |A_i ∩ A_j| / |A_i ∪ A_j|

  Edge cases follow the same convention as `_jaccard`:
    1.0 when all sets empty, 0.0 when only one is non-empty.

The aggregated cross-dataset reference is *not* used here — these views
compare datasets directly to one another. For "dataset vs aggregated
reference" views, see the older `jaccard_sweep` output of
`aggregate_overview.py`.

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
    always including K = n_expert_used, ceil(0.125 * E), 2*n_expert_used
    and 3*n_expert_used (clamped to E).
    """
    k_min = max(1, int(k_min))
    k_max = max(k_min, int(np.ceil(n_expert * k_max_frac)))
    # Dense linspace, dedup + sort + cast to int.
    base = np.linspace(k_min, k_max, num=max(1, int(num_ks)))
    base_ints = sorted({int(round(float(v))) for v in base})
    # Must-include sentinel K values: n_expert_used, ceil(0.125 * E), and
    # the 2x / 3x K_model values that drive the extra heatmaps. We add
    # them after the linspace so they're present even when num_ks is
    # small. The 2x/3x Ks are clamped to E so we don't ask for more top-K
    # experts than the model has.
    must_include = {
        int(n_expert_used),
        int(np.ceil(n_expert * 0.125)),
        int(min(2 * n_expert_used, n_expert)),
        int(min(3 * n_expert_used, n_expert)),
    }
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


def _generalized_jaccard(sets_list: list[set[int]]) -> float:
    """Generalized Jaccard for N>=1 sets:  |∩ A_i| / |∪ A_i|.

    "Strict consensus" measure — 1.0 iff every dataset has the *exact
    same* top-K set, drops fast if any dataset diverges. Returns:

      - 1.0 when all sets are empty (no data -> trivially agree)
      - 0.0 when only one set is non-empty (no comparison possible)
      - `len(intersection) / len(union)` in the general case

    Mirrors the empty-set conventions of `_jaccard` so the two metrics
    stay numerically comparable.
    """
    non_empty = [s for s in sets_list if s]
    if not non_empty:
        return 1.0
    if len(non_empty) == 1:
        return 0.0
    inter = non_empty[0].copy()
    union = non_empty[0].copy()
    for s in non_empty[1:]:
        inter &= s
        union |= s
    if not union:
        return 1.0
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
    """Sum per-dataset [L, E] count matrices into a single [L, E] array.

    Retained for potential future use but not referenced by the
    cross-dataset outputs anymore (the two remaining plots compare
    datasets directly, not against the aggregated reference).
    """
    first = next(iter(per_ds_counts.values()))
    out = np.zeros_like(first, dtype=np.int64)
    for counts in per_ds_counts.values():
        out += counts
    return out


def _sanity_check_top_experts(cell_dir: Path, counts_agg: np.ndarray,
                              tokens_agg: int) -> None:
    """Compare live aggregate against `overall/top_experts.json` if present.

    Diagnostic only — the cross-dataset outputs do not use the aggregated
    reference, so we no longer call this from `process_cell`. Kept for
    future extensions and as a regression test for `aggregate_overview.py`.
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

    tokens_agg = int(sum(per_ds_tokens.values()))

    K_values = _make_k_values(E, K_model, k_min, k_max_frac, num_ks)
    print(f"[plan] arch={arch['name']} L={L} E={E} K_model={K_model}; "
          f"K-sweep ({len(K_values)} values) = {K_values}; "
          f"datasets={len(per_ds_counts)}; tokens_aggregated={tokens_agg:,}")

    # Per-dataset top-K sets, structured as
    #   ds_topk[ds][K][layer] -> set[int]
    ds_topk: dict[str, dict[int, list[set[int]]]] = {}
    for ds, counts in per_ds_counts.items():
        ds_topk[ds] = {
            K: [_top_k_set_for_layer(counts, layer, K) for layer in range(L)]
            for K in K_values
        }

    # -------- cross-dataset alignment: do all datasets agree on top-K?
    # Two complementary metrics at each (K, layer):
    #   jaccard_union[K][layer]         = |∩ A_i| / |∪ A_i|  (strict consensus)
    #   jaccard_pairwise_mean[K][layer] = mean of all C(N,2) pairwise Jaccards
    # plus the full N×N pairwise matrix per (K, layer) for the heatmap output.
    # Single-number-per-K view is the mean-across-layers of each metric.
    ds_names_sorted = sorted(per_ds_counts.keys())
    N_ds = len(ds_names_sorted)
    jaccard_union: dict[int, list[float]] = {K: [0.0] * L for K in K_values}
    jaccard_pairwise_mean: dict[int, list[float]] = {
        K: [0.0] * L for K in K_values
    }
    pair_matrices: dict[int, np.ndarray] = {}  # K -> [L, N_ds, N_ds]
    for K in K_values:
        per_layer_mats = np.zeros((L, N_ds, N_ds), dtype=np.float64)
        for layer in range(L):
            sets_at_kl = [ds_topk[ds][K][layer] for ds in ds_names_sorted]
            jaccard_union[K][layer] = _generalized_jaccard(sets_at_kl)
            for i in range(N_ds):
                per_layer_mats[layer, i, i] = 1.0
                for j in range(i + 1, N_ds):
                    jv = _jaccard(sets_at_kl[i], sets_at_kl[j])
                    per_layer_mats[layer, i, j] = jv
                    per_layer_mats[layer, j, i] = jv
            if N_ds >= 2:
                triu = per_layer_mats[layer][np.triu_indices(N_ds, k=1)]
                jaccard_pairwise_mean[K][layer] = float(np.mean(triu))
            else:
                jaccard_pairwise_mean[K][layer] = 1.0
        pair_matrices[K] = per_layer_mats

    # Per-dataset "agreement with the rest of the pack" at K=K_model,
    # averaged across layers. For each dataset, mean Jaccard with every
    # OTHER dataset at K=K_model (across all layers) plus the worst layer.
    mean_to_others: dict[str, float] = {}
    min_to_others: dict[str, float] = {}
    if N_ds <= 1:
        for ds in ds_names_sorted:
            mean_to_others[ds] = 1.0
            min_to_others[ds] = 1.0
    else:
        pair_at_kmodel = pair_matrices[K_model]  # [L, N_ds, N_ds]
        for i, ds in enumerate(ds_names_sorted):
            per_layer_means = np.array(
                [np.mean([pair_at_kmodel[layer, i, j]
                          for j in range(N_ds) if j != i])
                 for layer in range(L)],
                dtype=np.float64,
            )
            mean_to_others[ds] = float(np.mean(per_layer_means))
            min_to_others[ds] = float(np.min(per_layer_means))

    # -------- per-dataset summary rows (cross-dataset alignment only)
    summary_rows: list[dict[str, Any]] = []
    for ds in ds_names_sorted:
        summary_rows.append({
            "dataset": ds,
            "tokens": per_ds_tokens[ds],
            "n_rows": next(m["n_rows"] for m in per_ds_meta
                           if m["dataset"] == ds),
            "mean_jaccard_to_others_at_n_used": mean_to_others[ds],
            "min_jaccard_to_others_at_n_used": min_to_others[ds],
        })

    # -------- write summary.csv (cross-dataset alignment only)
    summary_csv_path = output_dir / "summary.csv"
    with open(summary_csv_path, "w") as f:
        f.write("dataset,tokens,n_rows,"
                "mean_jaccard_to_others_at_n_used,"
                "min_jaccard_to_others_at_n_used\n")
        for r in summary_rows:
            f.write(f"{r['dataset']},{r['tokens']},{r['n_rows']},"
                    f"{r['mean_jaccard_to_others_at_n_used']:.6f},"
                    f"{r['min_jaccard_to_others_at_n_used']:.6f}\n")
    print(f"[save] {summary_csv_path}")

    # -------- write cross-dataset CSVs
    pair_csv_path = output_dir / "jaccard_pairwise.csv"
    with open(pair_csv_path, "w") as f:
        f.write("K,layer,dataset_a,dataset_b,jaccard\n")
        for K in K_values:
            for layer in range(L):
                mat = pair_matrices[K][layer]
                for i in range(N_ds):
                    for j in range(i + 1, N_ds):
                        f.write(f"{K},{layer},"
                                f"{ds_names_sorted[i]},"
                                f"{ds_names_sorted[j]},"
                                f"{mat[i, j]:.6f}\n")
    print(f"[save] {pair_csv_path}")

    global_csv_path = output_dir / "jaccard_global.csv"
    with open(global_csv_path, "w") as f:
        f.write("K,layer,jaccard_union,jaccard_pairwise_mean\n")
        for K in K_values:
            for layer in range(L):
                f.write(f"{K},{layer},"
                        f"{jaccard_union[K][layer]:.6f},"
                        f"{jaccard_pairwise_mean[K][layer]:.6f}\n")
    print(f"[save] {global_csv_path}")

    # -------- README.json (meta)
    # Cross-dataset alignment summary at K=K_model (averaged across layers).
    gj_at_kmodel = float(np.mean(jaccard_union[K_model]))
    mp_at_kmodel = float(np.mean(jaccard_pairwise_mean[K_model]))
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
        "cross_dataset": {
            "metric_definition": {
                "jaccard_union":
                    "|intersection of top-K sets| / |union of top-K sets| "
                    "across all datasets (strict consensus; drops fast on outliers)",
                "jaccard_pairwise_mean":
                    "mean of all C(N,2) pairwise top-K Jaccards across datasets",
            },
            "at_k_n_used": {
                "jaccard_union_mean_across_layers": gj_at_kmodel,
                "jaccard_pairwise_mean_mean_across_layers": mp_at_kmodel,
            },
            "datasets_sorted": ds_names_sorted,
        },
    }
    meta_path = output_dir / "README.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] {meta_path}")

    # -------- jaccard_pairwise.png (heatmap at K=K_model, avg across layers)
    # Three heatmaps in total, all sharing the same Blues colormap and
    # threshold rule for legible text:
    #   - K = K_model       (primary;  named jaccard_pairwise.png)
    #   - K = 2 * K_model   (alias    jaccard_pairwise_K2x.png)
    #   - K = 3 * K_model   (alias    jaccard_pairwise_K3x.png)
    # Each heatmap is also written with K embedded in the filename
    # (jaccard_pairwise_K{K}.png). The 2x/3x Ks may exceed E for very
    # small models (e.g. Mixtral has E=8, K_model=2 -> 3*K_model=6 ok,
    # but 2*K_model=4 ok too; for gpt-oss with K_model=4, E=128 -> 12
    # and 24 always fine). We clamp the 2x/3x Ks to E inside
    # `_make_k_values` already; here we only render when the K lands
    # within K_values (otherwise we emit a [warn] and skip).
    model_label = (model_id or model_id_loaded or cell_dir.parent.name)
    quant_label = (quant or cell_dir.name)
    # Candidate Ks for the additional heatmaps, in priority order. We
    # dedupe against K_model so we don't write the same heatmap twice
    # on models that overlap (e.g. K_model=8 with 2*8=16 != 3*8=24 but
    # if K_model=12 and 2*K_model=24 != 3*K_model=36 on E=64, no
    # collision; we still keep the explicit dedupe for safety).
    heatmap_ks: list[tuple[int, str]] = [(K_model, "")]
    heatmap_ks.append((int(min(2 * K_model, E)), "K2x"))
    heatmap_ks.append((int(min(3 * K_model, E)), "K3x"))
    rendered_ks: list[int] = []
    for k_render, alias in heatmap_ks:
        if k_render not in pair_matrices:
            print(f"[warn] K={k_render} not in sweep "
                  f"(K_values={K_values}); skipping extra heatmap")
            continue
        if k_render in rendered_ks:
            # Already rendered under an earlier name; skip the alias.
            print(f"[skip] K={k_render} already rendered; "
                  f"suppressing duplicate alias '{alias}'")
            continue
        rendered_ks.append(k_render)
        pair_avg = pair_matrices[k_render].mean(axis=0)  # [N_ds, N_ds]
        fig_w = max(5.5, N_ds * 1.2 + 1.0)
        fig_h = max(4.5, N_ds * 1.0 + 0.8)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(pair_avg, vmin=0.0, vmax=1.0, cmap="Blues",
                       aspect="auto")
        ax.set_xticks(range(N_ds))
        ax.set_yticks(range(N_ds))
        ax.set_xticklabels(ds_names_sorted, rotation=30, ha="right",
                           fontsize=8)
        ax.set_yticklabels(ds_names_sorted, fontsize=8)
        ax.set_xlabel("dataset")
        ax.set_ylabel("dataset")
        for i in range(N_ds):
            for j in range(N_ds):
                v = pair_avg[i, j]
                # Blues cmap: low v -> near-white, high v -> deep blue.
                # Diagonal cells (i == j) are always 1.0 -> deep blue,
                # so they MUST be white; off-diagonal low-overlap cells
                # are light blue, so they read best in black.
                color = "black" if v < 0.55 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color=color)
        ax.set_title(
            f"{model_label}  -  pairwise top-K expert agreement ({quant_label})\n"
            f"{arch['name']} L={L} E={E} K={k_render} "
            f"(={k_render // K_model}x K_model); "
            f"averaged across {L} layer(s); {N_ds} datasets",
            fontsize=10,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"pairwise Jaccard at K={k_render}", fontsize=8)
        fig.tight_layout()
        # Filenames:
        #   - K = K_model (alias=="")  -> jaccard_pairwise.png
        #     (kept for backward compat with existing references)
        #   - K = 2 * K_model / 3 * K_model
        #     -> jaccard_pairwise_K{K}.png AND jaccard_pairwise_K2x.png /
        #        K3x.png (one of the two, the {K}-embedded form)
        pair_png = (output_dir / "jaccard_pairwise.png"
                    if alias == ""
                    else output_dir / f"jaccard_pairwise_K{k_render}.png")
        fig.savefig(pair_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {pair_png}")
        if alias in ("K2x", "K3x"):
            alias_png = output_dir / f"jaccard_pairwise_{alias}.png"
            if alias_png != pair_png:
                pair_png.rename(alias_png)
                print(f"[save] {alias_png}  (renamed)")

    # -------- jaccard_global.png (x=K, y=cross-dataset Jaccard)
    # Two complementary metrics overlaid; thin per-layer traces underneath
    # + bold mean-across-layers lines. This is the "single number per K"
    # view of cross-dataset alignment.
    K_arr = np.asarray(K_values, dtype=np.float64)
    union_per_layer = np.array([jaccard_union[K] for K in K_values],
                               dtype=np.float64).T  # [L, n_K]
    pairwise_per_layer = np.array(
        [jaccard_pairwise_mean[K] for K in K_values], dtype=np.float64).T
    union_mean_curve = union_per_layer.mean(axis=0)
    pairwise_mean_curve = pairwise_per_layer.mean(axis=0)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    # Faint per-layer lines for both metrics.
    for layer in range(L):
        ax.plot(K_arr, union_per_layer[layer], color="#4477aa",
                alpha=0.15, linewidth=0.7)
        ax.plot(K_arr, pairwise_per_layer[layer], color="#cc6677",
                alpha=0.15, linewidth=0.7)
    # Bold mean-across-layers lines.
    ax.plot(K_arr, union_mean_curve, color="#4477aa", linewidth=2.4,
            marker="o", markersize=4,
            label="generalized Jaccard  (|∩|/|∪|)")
    ax.plot(K_arr, pairwise_mean_curve, color="#cc6677", linewidth=2.4,
            marker="s", markersize=4,
            label="mean pairwise Jaccard")
    ax.axvline(K_model, color="black", linestyle=":", linewidth=0.8,
               label=f"K_model={K_model}")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.6,
               label="perfect agreement")
    ax.set_xlabel("K")
    ax.set_ylabel("cross-dataset Jaccard")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        f"{model_label}  -  cross-dataset top-K alignment ({quant_label})\n"
        f"{arch['name']} L={L} E={E} K_model={K_model}; "
        f"{N_ds} datasets; {tokens_agg:,} aggregated tokens\n"
        f"thin lines = per-layer; bold lines = mean across layers",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    global_png = output_dir / "jaccard_global.png"
    fig.savefig(global_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {global_png}")

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
        "--k-max-frac", type=float, default=0.50,
        help="K_max = ceil(n_expert * k_max_frac) (default: 0.50). "
             "Default chosen so the 2x K_model and 3x K_model must-include "
             "points (clamped to E) land inside the sweep on common models.")
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
