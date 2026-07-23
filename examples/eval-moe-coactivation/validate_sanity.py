#!/usr/bin/env python3
"""Sanity-check the eval-moe-coactivation JSON output.

Validates:
  1. tokens_total == tokens_prefill + tokens_generated (per subject and aggregate)
  2. sum_j intra_pair_counts[L][e1][j] == marginal_expert_counts[L][e1] (k-multiplied marginals)
  3. intra_pair_counts[L][e1][e2] == intra_pair_counts[L][e2][e1] (symmetry)
  4. inter_pair_counts[L][L][e1][e2] == intra_pair_counts[L][e1][e2] (inter on the diagonal equals intra)
  5. inter_pair_counts[L1][L2][e1][e2] is non-zero for L1 <= L2 and zero for L1 > L2
"""
import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "build/moe-coactivation/smoke.json"
with open(path) as f:
    data = json.load(f)

n_layer = data["model_arch"]["n_layer"]
n_expert = data["model_arch"]["n_expert"]
n_expert_used = data["model_arch"]["n_expert_used"]
print(f"Model arch: {data['model_arch']['name']}, L={n_layer}, E={n_expert}, k={n_expert_used}")

# Determine the inter output format from the config block
config = data.get("config", {})
inter_k_lag = int(config.get("inter_k_lag", 0))
per_subject_inter = bool(config.get("per_subject_inter", False))
sparse_min_count = int(config.get("sparse_min_count", 0))
print(f"Config: inter_k_lag={inter_k_lag}, per_subject_inter={per_subject_inter}, sparse_min_count={sparse_min_count}")

def load_inter(agg):
    """Return a dense numpy array of shape [L, L, E, E] from the aggregate inter
    field, regardless of which format was emitted (default / k-lag / sparse)."""
    if inter_k_lag > 0:
        # k-lag format: [L, K, E, E] (k = 1..K, off-diagonal only)
        key = "inter_klag_counts_sparse" if sparse_min_count > 0 else "inter_klag_counts"
        if sparse_min_count > 0:
            coo = agg[key]
            arr = np.zeros(tuple(coo["shape"]), dtype=np.int64)
            for idx, v in zip(coo["indices"], coo["values"]):
                arr[tuple(idx)] = v
            # The shape is [L, K, E, E] (k-lag). We need to scatter into a [L, L, E, E]
            # upper-triangular array (k = L2 - L1, 1-based).
            full = np.zeros((n_layer, n_layer, n_expert, n_expert), dtype=np.int64)
            L, K, E, _ = arr.shape
            for L1 in range(L):
                for k in range(K):
                    L2 = L1 + k + 1  # 0-based lag, so k=0 means L1 -> L1+1
                    if L2 < L:
                        full[L1, L2] = arr[L1, k]
            return full
        else:
            arr = np.array(agg[key])  # [L, K, E, E]
            # Scatter into [L, L, E, E]
            full = np.zeros((n_layer, n_layer, n_expert, n_expert), dtype=np.int64)
            L, K, E, _ = arr.shape
            for L1 in range(L):
                for k in range(K):
                    L2 = L1 + k + 1
                    if L2 < L:
                        full[L1, L2] = arr[L1, k]
            return full
    else:
        # full upper triangular: [L, L, E, E]
        key = "inter_pair_counts_sparse" if sparse_min_count > 0 else "inter_pair_counts"
        if sparse_min_count > 0:
            coo = agg[key]
            arr = np.zeros(tuple(coo["shape"]), dtype=np.int64)
            for idx, v in zip(coo["indices"], coo["values"]):
                arr[tuple(idx)] = v
            return arr
        else:
            return np.array(agg[key])  # [L, L, E, E]

# Check 1: token accounting (aggregate)
agg = data["aggregate"]
total_check = agg["tokens_prefill"] + agg["tokens_generated"]
print(f"\n[1] Aggregate tokens: prefill={agg['tokens_prefill']}, gen={agg['tokens_generated']}, "
      f"total={agg['tokens_total']} (expected={total_check}): {'OK' if total_check == agg['tokens_total'] else 'FAIL'}")

# Per-subject token accounting
token_failures = 0
for subj, s in data["subjects"].items():
    expected = s["tokens_prefill"] + s["tokens_generated"]
    if expected != s["tokens_total"]:
        print(f"  [1] {subj}: FAIL (expected={expected}, got={s['tokens_total']})")
        token_failures += 1
print(f"[1] Per-subject token accounting: {len(data['subjects'])-token_failures}/{len(data['subjects'])} OK")

# Aggregate pair counts
agg_intra = np.array(agg["intra_pair_counts"])  # [L, E, E]
agg_inter = load_inter(agg)  # [L, L, E, E] (see load_inter for format details)
agg_marg = np.array(agg["marginal_expert_counts"])  # [L, E]

# Check 2: marginals = sum over j of intra pair counts
marginal_check = agg_intra.sum(axis=2)  # [L, E]
diff = np.abs(marginal_check - agg_marg)
print(f"\n[2] Marginal = sum_j(intra[L,:,j]): max diff = {diff.max()}, total mismatches = {(diff>0).sum()}")
if diff.max() == 0:
    print("[2] OK: marginals exactly match sum of intra pair counts")

# Check 3: intra symmetry
sym_diff = np.abs(agg_intra - agg_intra.transpose(0, 2, 1))
print(f"\n[3] Intra symmetry: max |a - a.T| = {sym_diff.max()}, total mismatches = {(sym_diff>0).sum()}")
if sym_diff.max() == 0:
    print("[3] OK: intra_pair_counts is exactly symmetric")

# Check 4: inter on diagonal == intra. This check only applies to the default
# (full upper-triangular) format and to the k-lag format at k = 0 (which we
# don't emit; the diagonal in k-lag format is the intra layer). So we skip
# this check for k-lag mode.
if inter_k_lag == 0 and sparse_min_count == 0:
    diag_diff = np.abs(agg_inter[np.arange(n_layer), np.arange(n_layer)] - agg_intra)
    print(f"\n[4] inter[L,L] == intra[L]: max diff = {diag_diff.max()}, total mismatches = {(diag_diff>0).sum()}")
    if diag_diff.max() == 0:
        print("[4] OK: inter on the diagonal matches intra")
elif inter_k_lag > 0:
    print(f"\n[4] skipped: k-lag format (inter_k_lag={inter_k_lag}) does not include the diagonal (k=0); intra layer is the canonical source")
else:
    print(f"\n[4] skipped: sparse format (sparse_min_count={sparse_min_count}) filters out cells with value <= N; the diagonal is lossy by design")

# Check 5: inter lower-triangular (L1 > L2 should be zero; we only stored L1 <= L2)
lower_zero = agg_inter[np.tril_indices(n_layer, k=-1)]
print(f"\n[5] inter[L1>L2] (should be zero): max={lower_zero.max()}, nonzero count={np.sum(lower_zero>0)}")
if lower_zero.max() == 0:
    print("[5] OK: lower triangle of inter is zero (L1 > L2 entries untouched)")

# Diagnostic: sparsity of intra
intra_nonzero_frac = (agg_intra > 0).mean()
print(f"\n[diag] Intra non-zero fraction: {intra_nonzero_frac:.4f} "
      f"({(agg_intra>0).sum()}/{agg_intra.size} cells)")

# Diagnostic: top intra pair per layer
print("\n[diag] Top intra pair (e_i, e_j) by count for each layer (top-3):")
for L in range(n_layer):
    flat = agg_intra[L].flatten()
    top3 = np.argsort(flat)[-3:][::-1]
    pairs = [(int(i // n_expert), int(i % n_expert), int(flat[i])) for i in top3]
    print(f"  L={L:2d}: " + ", ".join(f"({p[0]},{p[1]})={p[2]}" for p in pairs))

# Diagnostic: top inter pair per layer-pair lag
print("\n[diag] Top inter pair (e_i in L, e_j in L+k) by count for k=1,2,3 (top-3):")
for k in (1, 2, 3):
    print(f"  k={k}:")
    for L in range(n_layer - k):
        block = agg_inter[L, L + k]  # [E, E]
        flat = block.flatten()
        top3 = np.argsort(flat)[-3:][::-1]
        pairs = [(int(i // n_expert), int(i % n_expert), int(flat[i])) for i in top3]
        print(f"    L={L}->L+{k}: " + ", ".join(f"({p[0]},{p[1]})={p[2]}" for p in pairs))

print("\nALL DONE.")
