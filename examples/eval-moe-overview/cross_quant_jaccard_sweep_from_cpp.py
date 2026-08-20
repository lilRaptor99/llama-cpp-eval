#!/usr/bin/env python3
# type: ignore

"""Cross-quantization top-K alignment plotter for the MoE-routing eval suite.

Companion to `jaccard_sweep_from_cpp.py` (which is cross-dataset within a
single `(model, quant)` cell). This tool is cross-quantization for a
single model: it reads every `<model>/<quant>/overall/counts_total_overview.json`
under one model directory (the per‑quant aggregated‑across‑datasets `[L, E]`
matrix already produced by `aggregate_overview.py`), then renders two
views of cross-quantization alignment on the **top-K expert sets**:

  1. A pairwise heatmap (quant × quant) at K = n_expert_used, averaged
     across layers — the visual.
  2. A single-number-per-K view: x = K, y = cross-quantization Jaccard.
     Two complementary metrics overlaid (generalized |∩|/|∪| + mean pairwise).

Outputs (under `<model-dir>/cross_quant_jaccard_sweep/`):

  - summary.csv              per quant: tokens_total, n_datasets,
                             mean_jaccard_to_others_at_n_used,
                             min_jaccard_to_others_at_n_used
  - jaccard_pairwise.csv     long-form: K, layer, quant_a, quant_b, jaccard
                             (every pair of quants; one row per (K, layer))
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
                             (single-number-per-(K, layer) view of cross-quant alignment)
  - jaccard_global.png       x = K, y = cross-quant Jaccard; thin per-layer lines
                             + bold mean across layers; both union and pairwise-mean
                             metrics overlaid for comparison
  - README.json              meta: model_id, arch, quants, K values, totals,
                             cross-quant alignment summary at K = K_model

Mathematical contract:

  For each layer l and K value, using the per-quant aggregated-across-datasets
  counts matrix from `counts_total_overview.json::counts_total` (shape [L, E]):
    A_q_K[l]  = top-K experts in quant q, layer l
                (descending count, ties -> ascending expert id)
    |∩|_K[l]  = |∩_q A_q_K[l]|              (intersection across quants)
    |∪|_K[l]  = |∪_q A_q_K[l]|              (union across quants)
    generalized Jaccard (strict consensus): |∩|_K[l] / |∪|_K[l]
    mean pairwise Jaccard:                  mean over all C(N, 2) pairs of
                                            |A_i ∩ A_j| / |A_i ∪ A_j|

  Edge cases follow the same convention as `_jaccard`:
    1.0 when all sets empty, 0.0 when only one is non-empty.

The cross-quant comparison is computed **per quant on the aggregated-across-
datasets reference** (the same per-quant counts the overview heatmap uses).
This is the same convention `aggregate_overview.py` uses for its top-K bar
chart and `top_experts.json` — we don't re-aggregate from per-dataset JSONs.

Two CLI modes:

  - Single model:
        python3 cross_quant_jaccard_sweep_from_cpp.py --model-dir <path> \
            [--model-id <hf-repo-id>] [--output-dir <override>]

  - Multi-model batch:
        python3 cross_quant_jaccard_sweep_from_cpp.py --results-dir <root> \
            --models <safe1> [<safe2> ...] [--quants <q1> [<q2> ...]]

The script is idempotent: a model whose
`cross_quant_jaccard_sweep/summary.csv` already exists is skipped
(use --force to re-render). Errors are reported as one-line
`SystemExit("[error] ...")` messages so a future sbatch wrapper can
surface them via the per-model log.

Dependencies: numpy + matplotlib only. We `from aggregate_overview import`
the four pure helpers we need (`_top_k_set_for_layer`, `_make_k_values`,
`_jaccard`, `_generalized_jaccard`) — `aggregate_overview.py` is the sibling
peer script in this directory, not a shared library, so this stays
consistent with the README's "no shared library import" convention while
still avoiding the copy-paste the spec explicitly forbids.

The filename contract (`summary.csv`, `jaccard_pairwise.csv`,
`jaccard_global.csv`, `jaccard_pairwise.png`, `jaccard_pairwise_K{K}.png`,
`jaccard_pairwise_K2x.png`, `jaccard_pairwise_K3x.png`, `jaccard_global.png`,
`README.json`) is bit-identical to the cross-dataset version's. Only the
CSV header column swaps `dataset` -> `quant` (and `tokens` -> `tokens_total`,
`n_rows` -> `n_datasets` in `summary.csv`). Downstream consumers can treat
the two formats identically — directory name is the only discriminator.
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
# any setup.py or PYTHONPATH plumbing. The four pure helpers below
# (`_top_k_set_for_layer`, `_make_k_values`, `_jaccard`,
# `_generalized_jaccard`) live in `jaccard_sweep_from_cpp.py` — they
# were originally written for the cross-dataset comparison and reused
# verbatim here. We import from the sibling peer script rather than
# copy-paste, matching the convention the cross-dataset version set
# for `aggregate_overview`'s helpers.
from jaccard_sweep_from_cpp import (  # noqa: E402
    _top_k_set_for_layer,
    _make_k_values,
    _jaccard,
    _generalized_jaccard,
)


# --------------------------------------------------------------- I/O

def _load_quant_overview(model_dir: Path, quant: str
                         ) -> dict[str, Any] | None:
    """Load one `<model>/<quant>/overall/counts_total_overview.json` and
    the sibling `metadata_overview.json`.

    Returns a dict with keys:
      quant             : str (the quant tag, e.g. "Q4_K_M")
      arch              : dict (name, n_layer, n_expert, n_expert_used)
      counts_total      : np.ndarray [L, E] int64
      totals_tokens     : int (totals.tokens_total from metadata)
      n_datasets        : int (totals.datasets_used from metadata)
      per_quant_meta    : dict (tokens_total, n_datasets, etc.)

    Returns None on missing file, JSON parse error, or shape mismatch
    (the caller logs a `[skip]` line and moves on). SystemExit on
    arch shape mismatches that suggest a malformed overview file.
    """
    counts_path = model_dir / quant / "overall" / "counts_total_overview.json"
    meta_path = model_dir / quant / "overall" / "metadata_overview.json"
    if not counts_path.exists():
        print(f"[skip] {counts_path}: not found")
        return None
    try:
        with open(counts_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[skip] {counts_path}: {e}")
        return None

    arch = data.get("model_arch") or {}
    L = int(arch.get("n_layer", 0))
    E = int(arch.get("n_expert", 0))
    k = int(arch.get("n_expert_used", 0))
    if L <= 0 or E <= 0:
        print(f"[skip] {counts_path}: malformed model_arch "
              f"(L={L} E={E} k={k})")
        return None

    raw_counts = data.get("counts_total")
    if raw_counts is None:
        print(f"[skip] {counts_path}: missing counts_total key")
        return None
    counts_total = np.asarray(raw_counts, dtype=np.int64)
    if counts_total.shape != (L, E):
        print(f"[skip] {counts_path}: shape {counts_total.shape} "
              f"!= expected ({L}, {E})")
        return None

    # Metadata sidecar is best-effort: when present, surface its
    # `totals.tokens_total` and `totals.datasets_used` in the per-quant
    # summary rows. When missing, fall back to values derived from the
    # counts matrix itself (sum-of-row-totals for tokens; no fallback
    # for n_datasets -> 0).
    totals_tokens = int(data.get("totals_tokens", 0))
    n_datasets = 0
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            totals = (meta.get("totals") or {})
            if totals_tokens <= 0:
                totals_tokens = int(totals.get("tokens_total", 0))
            n_datasets = int(totals.get("datasets_used", 0))
        except (json.JSONDecodeError, OSError):
            pass
    # If metadata sidecar didn't yield tokens_total, derive it from the
    # counts matrix (it's the sum of all layer totals in the raw
    # aggregated matrix, identical to the C++ per-dataset sum).
    if totals_tokens <= 0:
        totals_tokens = int(counts_total.sum())

    return {
        "quant": quant,
        "arch": {
            "name": arch.get("name", "unknown"),
            "n_layer": L,
            "n_expert": E,
            "n_expert_used": k,
        },
        "counts_total": counts_total,
        "totals_tokens": totals_tokens,
        "n_datasets": n_datasets,
        "per_quant_meta": {
            "quant": quant,
            "tokens_total": int(totals_tokens),
            "n_datasets": int(n_datasets),
        },
    }


def _iter_quants(model_dir: Path, only_quants: list[str] | None
                 ) -> list[tuple[str, Path]]:
    """Discover `<model>/<quant>/overall/counts_total_overview.json` files.

    Returns a sorted list of `(quant_name, overall_dir)` pairs. The quant
    tag is the directory name immediately under `model_dir` (i.e. the
    layer in the `<model>/<quant>/overall/` layout). Apply `--quants`
    filter before returning.
    """
    candidates: list[tuple[str, Path]] = []
    if not model_dir.is_dir():
        return candidates
    for sub in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        overall_dir = sub / "overall"
        if not (overall_dir / "counts_total_overview.json").exists():
            continue
        if only_quants and sub.name not in only_quants:
            continue
        candidates.append((sub.name, overall_dir))
    return candidates


def _model_safe(name: str) -> str:
    """Mirror `scripts/run-evaluator.sh::model_safe` (replace / with --)."""
    return name.replace("/", "--")


def _reverse_model_safe(safe: str) -> str:
    """Reverse `scripts/run-evaluator.sh::model_safe` (first "--" -> "/")."""
    if "--" in safe:
        return safe.replace("--", "/", 1)
    return safe


# ----------------------------------------------------------- per-model main

def process_model(model_dir: Path, *, output_dir: Path | None = None,
                  k_min: int, k_max_frac: float, num_ks: int,
                  dpi: int, model_id: str | None = None,
                  force: bool = False,
                  only_quants: list[str] | None = None,
                  ) -> dict[str, Any]:
    """Process one model directory and write all cross-quant artifacts.

    Iterates over every `<model>/<quant>/overall/counts_total_overview.json`
    under `model_dir`, computes the per-quant top-K expert sets at every
    layer, and renders the cross-quant Jaccard sweep under
    `<output_dir>/` (default: `<model_dir>/cross_quant_jaccard_sweep/`).

    Returns a summary dict for stdout reporting. Skips when the sentinel
    `summary.csv` already exists (unless `force=True`).
    """
    model_dir = model_dir.resolve()
    if output_dir is None:
        output_dir = model_dir / "cross_quant_jaccard_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "summary.csv"
    if summary_csv.exists() and not force:
        print(f"[skip] {summary_csv} exists; pass --force to regenerate")
        return {"model_dir": str(model_dir), "output_dir": str(output_dir),
                "skipped": True}

    # -------- discover quants and load their overview JSONs
    quant_candidates = _iter_quants(model_dir, only_quants)
    if not quant_candidates:
        raise SystemExit(
            f"[error] {model_dir}: no <quant>/overall/counts_total_overview.json "
            f"found (filter={only_quants or 'all'})"
        )

    per_quant_counts: dict[str, np.ndarray] = {}
    per_quant_tokens: dict[str, int] = {}
    per_quant_n_datasets: dict[str, int] = {}
    per_quant_meta: list[dict[str, Any]] = []
    arch: dict[str, Any] | None = None
    n_quants_used = 0
    n_quants_skipped = 0

    for quant, _overall_dir in quant_candidates:
        loaded = _load_quant_overview(model_dir, quant)
        if loaded is None:
            n_quants_skipped += 1
            continue
        if arch is None:
            arch = loaded["arch"]
        else:
            # All quants of one model must share arch shape.
            if (loaded["arch"]["n_layer"] != arch["n_layer"]
                    or loaded["arch"]["n_expert"] != arch["n_expert"]
                    or loaded["arch"]["n_expert_used"] != arch["n_expert_used"]):
                raise SystemExit(
                    f"[error] {model_dir}/{quant}: arch mismatch "
                    f"(got L={loaded['arch']['n_layer']} "
                    f"E={loaded['arch']['n_expert']} "
                    f"k={loaded['arch']['n_expert_used']}, "
                    f"expected L={arch['n_layer']} "
                    f"E={arch['n_expert']} "
                    f"k={arch['n_expert_used']})"
                )
        per_quant_counts[quant] = loaded["counts_total"]
        per_quant_tokens[quant] = loaded["totals_tokens"]
        per_quant_n_datasets[quant] = loaded["n_datasets"]
        per_quant_meta.append(loaded["per_quant_meta"])
        n_quants_used += 1
        print(f"[load] quant={quant}: "
              f"tokens={loaded['totals_tokens']:,}, "
              f"n_datasets={loaded['n_datasets']}, "
              f"arch={loaded['arch']['name']}")

    if arch is None or not per_quant_counts:
        raise SystemExit(
            f"[error] {model_dir}: no usable quants found "
            f"(candidates={len(quant_candidates)}, skipped={n_quants_skipped})"
        )

    L = arch["n_layer"]
    E = arch["n_expert"]
    K_model = arch["n_expert_used"]

    tokens_agg = int(sum(per_quant_tokens.values()))

    K_values = _make_k_values(E, K_model, k_min, k_max_frac, num_ks)
    print(f"[plan] arch={arch['name']} L={L} E={E} K_model={K_model}; "
          f"K-sweep ({len(K_values)} values) = {K_values}; "
          f"quants={len(per_quant_counts)}; "
          f"tokens_aggregated={tokens_agg:,}")

    # Per-quant top-K sets, structured as
    #   ds_topk[quant][K][layer] -> set[int]
    # Variable named `ds_topk` to mirror the cross-dataset version even
    # though here it's per-quant, not per-dataset.
    ds_topk: dict[str, dict[int, list[set[int]]]] = {}
    for quant, counts in per_quant_counts.items():
        ds_topk[quant] = {
            K: [_top_k_set_for_layer(counts, layer, K) for layer in range(L)]
            for K in K_values
        }

    # -------- cross-quant alignment: do all quants agree on top-K?
    # Two complementary metrics at each (K, layer):
    #   jaccard_union[K][layer]         = |∩ A_q| / |∪ A_q|  (strict consensus)
    #   jaccard_pairwise_mean[K][layer] = mean of all C(N,2) pairwise Jaccards
    # plus the full N×N pairwise matrix per (K, layer) for the heatmap output.
    # Single-number-per-K view is the mean-across-layers of each metric.
    quant_names_sorted = sorted(per_quant_counts.keys())
    N_q = len(quant_names_sorted)
    jaccard_union: dict[int, list[float]] = {K: [0.0] * L for K in K_values}
    jaccard_pairwise_mean: dict[int, list[float]] = {
        K: [0.0] * L for K in K_values
    }
    pair_matrices: dict[int, np.ndarray] = {}  # K -> [L, N_q, N_q]
    for K in K_values:
        per_layer_mats = np.zeros((L, N_q, N_q), dtype=np.float64)
        for layer in range(L):
            sets_at_kl = [ds_topk[quant][K][layer] for quant in quant_names_sorted]
            jaccard_union[K][layer] = _generalized_jaccard(sets_at_kl)
            for i in range(N_q):
                per_layer_mats[layer, i, i] = 1.0
                for j in range(i + 1, N_q):
                    jv = _jaccard(sets_at_kl[i], sets_at_kl[j])
                    per_layer_mats[layer, i, j] = jv
                    per_layer_mats[layer, j, i] = jv
            if N_q >= 2:
                triu = per_layer_mats[layer][np.triu_indices(N_q, k=1)]
                jaccard_pairwise_mean[K][layer] = float(np.mean(triu))
            else:
                jaccard_pairwise_mean[K][layer] = 1.0
        pair_matrices[K] = per_layer_mats

    # Per-quant "agreement with the rest of the pack" at K=K_model,
    # averaged across layers. For each quant, mean Jaccard with every
    # OTHER quant at K=K_model (across all layers) plus the worst layer.
    mean_to_others: dict[str, float] = {}
    min_to_others: dict[str, float] = {}
    if N_q <= 1:
        for q in quant_names_sorted:
            mean_to_others[q] = 1.0
            min_to_others[q] = 1.0
    else:
        pair_at_kmodel = pair_matrices[K_model]  # [L, N_q, N_q]
        for i, q in enumerate(quant_names_sorted):
            per_layer_means = np.array(
                [np.mean([pair_at_kmodel[layer, i, j]
                          for j in range(N_q) if j != i])
                 for layer in range(L)],
                dtype=np.float64,
            )
            mean_to_others[q] = float(np.mean(per_layer_means))
            min_to_others[q] = float(np.min(per_layer_means))

    # -------- per-quant summary rows (cross-quant alignment only)
    summary_rows: list[dict[str, Any]] = []
    for q in quant_names_sorted:
        summary_rows.append({
            "quant": q,
            "tokens_total": per_quant_tokens[q],
            "n_datasets": per_quant_n_datasets[q],
            "mean_jaccard_to_others_at_n_used": mean_to_others[q],
            "min_jaccard_to_others_at_n_used": min_to_others[q],
        })

    # -------- write summary.csv (cross-quant alignment only)
    summary_csv_path = output_dir / "summary.csv"
    with open(summary_csv_path, "w") as f:
        f.write("quant,tokens_total,n_datasets,"
                "mean_jaccard_to_others_at_n_used,"
                "min_jaccard_to_others_at_n_used\n")
        for r in summary_rows:
            f.write(f"{r['quant']},{r['tokens_total']},{r['n_datasets']},"
                    f"{r['mean_jaccard_to_others_at_n_used']:.6f},"
                    f"{r['min_jaccard_to_others_at_n_used']:.6f}\n")
    print(f"[save] {summary_csv_path}")

    # -------- write cross-quant CSVs
    pair_csv_path = output_dir / "jaccard_pairwise.csv"
    with open(pair_csv_path, "w") as f:
        f.write("K,layer,quant_a,quant_b,jaccard\n")
        for K in K_values:
            for layer in range(L):
                mat = pair_matrices[K][layer]
                for i in range(N_q):
                    for j in range(i + 1, N_q):
                        f.write(f"{K},{layer},"
                                f"{quant_names_sorted[i]},"
                                f"{quant_names_sorted[j]},"
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
    gj_at_kmodel = float(np.mean(jaccard_union[K_model]))
    mp_at_kmodel = float(np.mean(jaccard_pairwise_mean[K_model]))
    model_label = model_id or _reverse_model_safe(model_dir.name)
    meta = {
        "model_dir": str(model_dir),
        "model_id": model_label,
        "model_arch": arch,
        "quants": quant_names_sorted,
        "k_min": int(k_min),
        "k_max_frac": float(k_max_frac),
        "num_ks_requested": int(num_ks),
        "k_values": K_values,
        "n_expert_used_model": int(K_model),
        "k_ceil_0.125_E": int(np.ceil(E * 0.125)),
        "k_ceil_k_max_frac_E": int(np.ceil(E * k_max_frac)),
        "totals": {
                "quants_used": n_quants_used,
                "quants_skipped": n_quants_skipped,
                "tokens_aggregated": int(tokens_agg),
        },
        "metric": "jaccard",
        "cross_quant": {
            "metric_definition": {
                "jaccard_union":
                    "|intersection of top-K sets| / |union of top-K sets| "
                    "across all quants (strict consensus; drops fast on outliers)",
                "jaccard_pairwise_mean":
                    "mean of all C(N,2) pairwise top-K Jaccards across quants",
            },
            "input_source":
                "per-quant aggregated-across-datasets counts matrix "
                "(<model>/<quant>/overall/counts_total_overview.json::counts_total)",
            "at_k_n_used": {
                "jaccard_union_mean_across_layers": gj_at_kmodel,
                "jaccard_pairwise_mean_mean_across_layers": mp_at_kmodel,
            },
            "quants_sorted": quant_names_sorted,
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
    # small models; we clamp them to E inside `_make_k_values` already,
    # so we only need to dedupe against K_model when rendering.
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
        pair_avg = pair_matrices[k_render].mean(axis=0)  # [N_q, N_q]
        fig_w = max(5.5, N_q * 1.2 + 1.0)
        fig_h = max(4.5, N_q * 1.0 + 0.8)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(pair_avg, vmin=0.0, vmax=1.0, cmap="Blues",
                       aspect="auto")
        ax.set_xticks(range(N_q))
        ax.set_yticks(range(N_q))
        ax.set_xticklabels(quant_names_sorted, rotation=30, ha="right",
                           fontsize=8)
        ax.set_yticklabels(quant_names_sorted, fontsize=8)
        ax.set_xlabel("quant")
        ax.set_ylabel("quant")
        for i in range(N_q):
            for j in range(N_q):
                v = pair_avg[i, j]
                # Blues cmap: low v -> near-white, high v -> deep blue.
                # Diagonal cells (i == j) are always 1.0 -> deep blue,
                # so they MUST be white; off-diagonal low-overlap cells
                # are light blue, so they read best in black.
                color = "black" if v < 0.55 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color=color)
        ax.set_title(
            f"{model_label}  -  pairwise top-K expert agreement across quantizations\n"
            f"{arch['name']} L={L} E={E} K={k_render} "
            f"(={k_render // K_model}x K_model); "
            f"averaged across {L} layer(s); {N_q} quant(s)",
            fontsize=10,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"pairwise Jaccard at K={k_render}", fontsize=8)
        fig.tight_layout()
        # Filenames:
        #   - K = K_model (alias=="")  -> jaccard_pairwise.png
        #     (kept for backward compat with the cross-dataset layout)
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

    # -------- jaccard_global.png (x=K, y=cross-quant Jaccard)
    # Two complementary metrics overlaid; thin per-layer traces underneath
    # + bold mean-across-layers lines. This is the "single number per K"
    # view of cross-quant alignment.
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
    ax.set_ylabel("cross-quant Jaccard")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        f"{model_label}  -  cross-quantization top-K alignment\n"
        f"{arch['name']} L={L} E={E} K_model={K_model}; "
        f"{N_q} quant(s); {tokens_agg:,} aggregated tokens\n"
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
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "model_id": model_label,
        "arch": arch,
        "k_values": K_values,
        "summary_rows": summary_rows,
        "n_quants": len(per_quant_counts),
        "tokens_aggregated": int(tokens_agg),
        "skipped": False,
    }


# --------------------------------------------------------------- CLI

def _iter_models(args: argparse.Namespace
                 ) -> list[tuple[Path, str | None]]:
    """Resolve `--model-dir` or `--results-dir --models --quants` to a list
    of `(model_dir, model_id)` tuples ready for `process_model()`.
    """
    if args.model_dir:
        model_id = args.model_id
        if not model_id:
            model_id = _reverse_model_safe(args.model_dir.name)
        return [(args.model_dir, model_id)]
    # Multi-model batch mode.
    results_dir = args.results_dir
    if results_dir is None or not args.models:
        # Defensive: main() already guards this, but keep the
        # post-condition explicit for type checkers.
        return []
    models: list[tuple[Path, str | None]] = []
    for m in args.models:
        model_dir = results_dir / _model_safe(m)
        if not model_dir.is_dir():
            print(f"[warn] {model_dir} does not exist; skipping")
            continue
        models.append((model_dir, m))
    return models


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="Process a single model directory. Expected layout: "
             "<model-dir>/<quant>/overall/counts_total_overview.json.")
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Process multiple model directories under "
             "<results-dir>/<model_safe>/. Use with --models and "
             "(optionally) --quants.")
    parser.add_argument(
        "--models", action="append", default=None,
        help="Restrict to a subset of HF model ids (one per flag, "
             "repeatable). Used only with --results-dir.")
    parser.add_argument(
        "--quants", action="append", default=None,
        help="Restrict to a subset of quantization subdirectories "
             "(one per flag, repeatable). Used with both --model-dir "
             "and --results-dir.")
    parser.add_argument(
        "--model-id", type=str, default=None,
        help="Override the model id recorded in README.json "
             "(single-model mode only).")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override the output dir "
             "(default: <model-dir>/cross_quant_jaccard_sweep).")
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
        help="Re-run even if cross_quant_jaccard_sweep/summary.csv "
             "already exists.")
    args = parser.parse_args()

    if not args.model_dir and not args.results_dir:
        raise SystemExit(
            "[error] either --model-dir or --results-dir is required "
            "(use --help for full options)")
    if args.model_dir and args.results_dir:
        raise SystemExit(
            "[error] --model-dir and --results-dir are mutually exclusive")
    if args.results_dir and not args.models:
        raise SystemExit(
            "[error] --results-dir requires at least one --models entry")

    models = _iter_models(args)
    if not models:
        raise SystemExit(
            f"[error] no models to process under "
            f"{args.results_dir or args.model_dir}")

    print(f"[plan] processing {len(models)} model(s)")
    failures = 0
    for model_dir, model_id in models:
        print(f"\n[model] {model_dir}  (model_id={model_id})")
        try:
            process_model(
                model_dir, output_dir=args.output_dir,
                k_min=args.k_min, k_max_frac=args.k_max_frac,
                num_ks=args.num_ks, dpi=args.dpi,
                model_id=model_id,
                force=args.force,
                only_quants=args.quants,
            )
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 - keep the loop running
            print(f"[error] {model_dir}: {e}", file=sys.stderr)
            failures += 1
            continue

    print(f"\n[done] processed {len(models)} model(s); {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())