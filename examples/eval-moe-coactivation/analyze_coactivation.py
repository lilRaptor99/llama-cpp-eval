#!/usr/bin/env python3
# type: ignore
"""Compute derived co-activation metrics from a `llama-eval-moe-coactivation` JSON.

Reads the raw counts produced by the C++ tool and computes:
  * intra_conditional_p[L, E, E] = P(e_j | e_i) per layer
  * marginal_p[L, E]            = marginal firing probability per expert
  * intra_pmi[L, E, E]          = log(intra_conditional_p / marginal_p[j])
  * intra_lift[L, E, E]         = P(e_i, e_j) / (P(e_i) * P(e_j))
  * inter_conditional_p[L, L, E, E]  (analogous across layer pairs)
  * inter_pmi[L, L, E, E]
  * inter_lift[L, L, E, E]

Outputs:
  * enriched.json         (raw counts + derived metrics, both per-subject and aggregate)
  * top_pairs.txt         (top-K PMI / lift pairs across all layers for quick inspection)

This is intentionally minimal — it demonstrates the math, prints the top-K pairs, and
writes an enriched JSON. A more featureful plotter can build on this foundation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np


def compute_metrics(intra: np.ndarray, marginal: np.ndarray, tokens_total: int) -> Dict[str, np.ndarray]:
    """Given a [L, E, E] co-occurrence count matrix and a [L, E] marginal vector,
    return a dict of derived metrics. `tokens_total` is the number of distinct tokens
    that contributed (prefill + generated)."""
    if tokens_total <= 0:
        L, E = intra.shape[0], intra.shape[1]
        nan = np.full((L, E, E), np.nan)
        return {
            "intra_conditional_p": nan,
            "intra_pmi":           nan,
            "intra_lift":          nan,
            "marginal_p":          np.full((L, E), np.nan),
        }

    # P(e_i) per layer: marginal count (already k-multiplied) / (k * tokens_total).
    # The k-multiplication is consistent across both numerator and denominator, so
    # the probability is correct without needing to know k explicitly.
    L, E = intra.shape[0], intra.shape[1]
    marginal_p = marginal.astype(np.float64) / float(tokens_total)  # already /k, so this is P(e)
    # P(e_i, e_j) per layer
    joint_p = intra.astype(np.float64) / float(tokens_total)
    # P(e_j | e_i) = joint_p[i,j] / marginal_p[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_p = joint_p / marginal_p[:, :, None]
        cond_p = np.where(np.isfinite(cond_p), cond_p, np.nan)
    # PMI = log(P(e_j | e_i) / P(e_j)) = log(joint_p / (marginal_p[i] * marginal_p[j]))
    denom = marginal_p[:, :, None] * marginal_p[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(joint_p / denom)
        pmi = np.where(np.isfinite(pmi) & (joint_p > 0), pmi, np.nan)
    # Lift = P(e_i, e_j) / (P(e_i) * P(e_j)) = joint_p / (marginal_p[i] * marginal_p[j])
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = joint_p / denom
        lift = np.where(np.isfinite(lift) & (joint_p > 0), lift, np.nan)
    return {
        "intra_conditional_p": cond_p,
        "intra_pmi":           pmi,
        "intra_lift":          lift,
        "marginal_p":          marginal_p,
    }


def compute_inter_metrics(
    inter: np.ndarray, marg_l1: np.ndarray, marg_l2: np.ndarray, tokens_total: int
) -> Dict[str, np.ndarray]:
    """Given a [L, L, E, E] cross-layer co-occurrence matrix and the marginals at
    L1 and L2 (both [L, E]), return P / PMI / lift for cross-layer pairs.
    Only the upper triangle (L1 <= L2) has data; the lower triangle is NaN."""
    L = inter.shape[0]
    E = inter.shape[2]
    if tokens_total <= 0:
        nan = np.full((L, L, E, E), np.nan)
        return {"inter_conditional_p": nan, "inter_pmi": nan, "inter_lift": nan}

    joint_p = inter.astype(np.float64) / float(tokens_total)
    # marginal_p at L1 is marg_l1[L1]; at L2 is marg_l2[L2]
    marg_l1_f = marg_l1.astype(np.float64) / float(tokens_total)
    marg_l2_f = marg_l2.astype(np.float64) / float(tokens_total)
    # denom[L1, L2, e_i, e_j] = marg_l1[L1, e_i] * marg_l2[L2, e_j]
    denom = marg_l1_f[:, None, :, None] * marg_l2_f[None, :, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_p = np.where(
            marg_l1_f[:, None, :, None] > 0,
            joint_p / marg_l1_f[:, None, :, None],
            np.nan,
        )
        pmi = np.log(np.where(denom > 0, joint_p / denom, np.nan))
        lift = np.where(denom > 0, joint_p / denom, np.nan)
    # Mask out the lower triangle (L1 > L2) explicitly with NaN
    for L1 in range(L):
        for L2 in range(L1):
            cond_p[L1, L2] = np.nan
            pmi[L1, L2]     = np.nan
            lift[L1, L2]    = np.nan
    return {"inter_conditional_p": cond_p, "inter_pmi": pmi, "inter_lift": lift}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Path to coactivation.json")
    ap.add_argument("-o", "--output-dir", required=True, help="Output directory for enriched.json and top_pairs.txt")
    ap.add_argument("--top-k", type=int, default=20, help="Number of top pairs to print per metric")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)

    L = data["model_arch"]["n_layer"]
    E = data["model_arch"]["n_expert"]

    # --- aggregate metrics
    agg = data["aggregate"]
    agg_intra = np.array(agg["intra_pair_counts"])
    agg_marg = np.array(agg["marginal_expert_counts"])

    # Detect inter output format (handles all 4 Phase 3 formats).
    config = data.get("config", {})
    inter_k_lag = int(config.get("inter_k_lag", 0))
    sparse_min_count = int(config.get("sparse_min_count", 0))

    def load_agg_inter(agg):
        """Return a [L, L, E, E] array from the aggregate inter field, regardless of format."""
        if inter_k_lag > 0:
            key = "inter_klag_counts_sparse" if sparse_min_count > 0 else "inter_klag_counts"
            if sparse_min_count > 0:
                coo = agg[key]
                arr = np.zeros(tuple(coo["shape"]), dtype=np.int64)
                for idx, v in zip(coo["indices"], coo["values"]):
                    arr[tuple(idx)] = v
            else:
                arr = np.array(agg[key])  # [L, K, E, E]
            # Scatter into [L, L, E, E] upper triangular
            full = np.zeros((L, L, E, E), dtype=np.int64)
            K = arr.shape[1]
            for L1 in range(L):
                for k in range(K):
                    L2 = L1 + k + 1
                    if L2 < L:
                        full[L1, L2] = arr[L1, k]
            return full
        else:
            key = "inter_pair_counts_sparse" if sparse_min_count > 0 else "inter_pair_counts"
            if sparse_min_count > 0:
                coo = agg[key]
                arr = np.zeros(tuple(coo["shape"]), dtype=np.int64)
                for idx, v in zip(coo["indices"], coo["values"]):
                    arr[tuple(idx)] = v
                return arr
            else:
                return np.array(agg[key])  # [L, L, E, E]

    agg_inter = load_agg_inter(agg)
    agg_tokens = agg["tokens_total"]

    agg_intra_metrics = compute_metrics(agg_intra, agg_marg, agg_tokens)
    agg_inter_metrics = compute_inter_metrics(agg_inter, agg_marg, agg_marg, agg_tokens)

    # --- print top-K pairs
    lines: list[str] = []
    lines.append(f"model={data['model']} arch={data['model_arch']['name']} L={L} E={E} k={data['model_arch']['n_expert_used']}")
    lines.append(f"tokens_total={agg_tokens} (prefill={agg['tokens_prefill']}, gen={agg['tokens_generated']})")
    lines.append("")

    cond = agg_intra_metrics["intra_conditional_p"]
    pmi = agg_intra_metrics["intra_pmi"]
    lift = agg_intra_metrics["intra_lift"]
    # Mask diagonal in PMI (always 0) and P(e|e) = 1 in cond_p
    diag_mask = np.eye(E, dtype=bool)
    pmi_off = pmi.copy()
    pmi_off[:, diag_mask] = np.nan

    lines.append(f"=== Top-{args.top_k} intra-layer PMI pairs (e_i, e_j, pmi, cond_p, count) ===")
    flat = pmi_off.flatten()
    top_idx = np.argsort(np.where(np.isnan(flat), -np.inf, flat))[-args.top_k:][::-1]
    for idx in top_idx:
        L1 = idx // (E * E)
        rest = idx % (E * E)
        ei = rest // E
        ej = rest % E
        lines.append(
            f"  L={L1:2d} e_i={ei:2d} e_j={ej:2d} pmi={pmi[L1,ei,ej]:+.3f} "
            f"cond_p={cond[L1,ei,ej]:.4f} count={int(agg_intra[L1,ei,ej])}"
        )
    lines.append("")
    lines.append(f"=== Top-{args.top_k} intra-layer LIFT pairs ===")
    flat = lift.flatten()
    top_idx = np.argsort(np.where(np.isnan(flat), -np.inf, flat))[-args.top_k:][::-1]
    for idx in top_idx:
        L1 = idx // (E * E)
        rest = idx % (E * E)
        ei = rest // E
        ej = rest % E
        lines.append(
            f"  L={L1:2d} e_i={ei:2d} e_j={ej:2d} lift={lift[L1,ei,ej]:.3f} "
            f"cond_p={cond[L1,ei,ej]:.4f} count={int(agg_intra[L1,ei,ej])}"
        )
    lines.append("")

    # --- top inter-layer pairs by PMI
    inter_pmi = agg_inter_metrics["inter_pmi"]
    inter_cond = agg_inter_metrics["inter_conditional_p"]
    inter_lift = agg_inter_metrics["inter_lift"]
    inter_count = agg_inter
    lines.append(f"=== Top-{args.top_k} inter-layer PMI pairs (L1, L2, e_i, e_j, pmi, cond_p, count) ===")
    flat = inter_pmi.flatten()
    valid = ~np.isnan(flat)
    top_idx = np.argsort(np.where(valid, flat, -np.inf))[-args.top_k:][::-1]
    for idx in top_idx:
        L1 = idx // (L * E * E)
        rest = idx % (L * E * E)
        L2 = rest // (E * E)
        rest2 = rest % (E * E)
        ei = rest2 // E
        ej = rest2 % E
        lines.append(
            f"  L1={L1:2d} L2={L2:2d} e_i={ei:2d} e_j={ej:2d} pmi={inter_pmi[L1,L2,ei,ej]:+.3f} "
            f"cond_p={inter_cond[L1,L2,ei,ej]:.4f} count={int(inter_count[L1,L2,ei,ej])}"
        )
    lines.append("")

    out_txt = out_dir / "top_pairs.txt"
    out_txt.write_text("\n".join(lines))
    print(f"[analyze] wrote {out_txt}")
    print()
    print("\n".join(lines))

    # --- enriched JSON (compact: only aggregate metrics to keep size reasonable)
    enriched: Dict[str, Any] = {
        "model":                 data["model"],
        "model_arch":            data["model_arch"],
        "config":                data["config"],
        "totals":                data["totals"],
        "aggregate": {
            "tokens_total":      agg_tokens,
            "marginal_p":        agg_intra_metrics["marginal_p"].tolist(),
            "intra_conditional_p": agg_intra_metrics["intra_conditional_p"].tolist(),
            "intra_pmi":         agg_intra_metrics["intra_pmi"].tolist(),
            "intra_lift":        agg_intra_metrics["intra_lift"].tolist(),
            # inter metrics are too large to emit cleanly; print top-K only.
            "inter_pmi_top_k":   [
                {
                    "L1": int(L1), "L2": int(L2), "e_i": int(ei), "e_j": int(ej),
                    "pmi": float(inter_pmi[L1, L2, ei, ej]),
                    "cond_p": float(inter_cond[L1, L2, ei, ej]),
                    "lift":  float(inter_lift[L1, L2, ei, ej]),
                    "count": int(inter_count[L1, L2, ei, ej]),
                }
                for idx in (np.argsort(np.where(~np.isnan(inter_pmi.flatten()),
                                              inter_pmi.flatten(), -np.inf))[-args.top_k:][::-1])
                for L1 in [int(idx // (L * E * E))]
                for L2 in [int((idx % (L * E * E)) // (E * E))]
                for ei in [int(((idx % (L * E * E)) % (E * E)) // E)]
                for ej in [int(((idx % (L * E * E)) % (E * E)) % E)]
            ],
        },
    }
    out_json = out_dir / "enriched.json"
    with open(out_json, "w") as f:
        json.dump(enriched, f, allow_nan=True)
    print(f"[analyze] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
