# Random MoE Router Baseline - Implementation Progress

## Status: All 6 binaries built; smoke test passed; reproducibility verification pending

## What is done

### Shared header
- `examples/_shared/moe_random_router.h` — defines `moe_random_router_state` with:
  - `enabled`, `base_seed`, `n_expert`, `expert_ids`, `std::mt19937_64 rng`
  - `configure(n_expert, base_seed)` and `seed_subject(model, arch, subject)`
  - `fill_topk_buffer(n_tokens, n_expert_k, out)` — uniform sample-without-replacement
    per token, stride1 = n_expert layout matching the existing ggml_view.
- Deterministic seed derivation via splitmix64 chain over (base_seed, model_name,
  arch, subject). This keeps the random stream stable across reruns for the
  same (model, arch, subject) and varies across subject boundaries inside one run.

### Per-binary plumbing (all 6 binaries)
- `examples/eval-moe-{mmlu,popqa,bigbench,humaneval,include,coactivation}/*.cpp`
- Each binary now:
  1. Includes `../_shared/moe_random_router.h`.
  2. Declares a static `g_random_router`/`g_random_seed` (or main-local) and
     a `moe_random_router_state random_router` member on the global accumulator.
  3. Adds `--random-router` (no-arg) and `--random-seed <N>` (uint64_t) to the
     custom usage string.
  4. Parses them in the first-pass custom-arg extractor.
  5. Strips them in the second-pass argv filter so llama.cpp's parameter parser
     doesn't see them.
  6. Calls `g_acc.random_router.configure(g_acc.n_expert, seed)` after metadata
     is read, and `g_acc.random_router.seed_subject(...)` each time
     `current_subject` (or equivalent) is assigned.
  7. In the eval callback, replaces the bare `ggml_backend_tensor_get(...)` call
     with a branch:
        if (g_acc.random_router.enabled) {
            g_acc.random_router.fill_topk_buffer(ntok, g_acc.n_expert_k, buf.data());
        } else {
            ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        }
     The downstream tally code (marginal counts, per-subject pair counts,
     aggregate sums) is unchanged, so the JSON output keeps the standard schema.

### Build verification
- All six targets compile cleanly with the existing build cache:
    [100%] Built target llama-eval-moe-mmlu
    [100%] Built target llama-eval-moe-popqa
    [100%] Built target llama-eval-moe-bigbench
    [100%] Built target llama-eval-moe-humaneval
    [100%] Built target llama-eval-moe-include
    [100%] Built target llama-eval-moe-coactivation

## Standalone smoke test
- `/tmp/random_baseline_test/random_router_smoke.cpp` + `make + run`
  exercises the random_router helper directly without the model.
- All 5 checks pass:
    [1] same-seed reproducibility:           OK
    [2] different-seed sensitivity:          OK
    [3] ids in [0,E) and no duplicates:       OK  (oob=0 dup=0)
    [4] marginals ~ uniform (expected=512):   OK  (min=466 max=551 max_dev=8.98%)
    [5] coverage >= 95% of E:                 OK  (distinct=64)

## End-to-end smoke test
- OLMoE-1B-7B Q4_K_M, --n-samples 1, --n-shots 0, --random-router --random-seed 0
- Completed in 114 s on CPU. Output: /tmp/random_baseline_test/seed0.json (15.6 MB).
- Schema preserved exactly:
    model_arch: olmoe L=16 E=64 K=8
    totals.tokens_total = 7454
    aggregate.marginal_expert_counts: (16, 64), sum = 59,632 per layer × 15 layers
      + 456 (collapsed last layer, expected from --no-embeddings default).
    aggregate.intra_pair_counts:        (16, 64, 64)
    aggregate.adjacent_pair_counts:     (15, 64, 64)
    aggregate.layer_min_topk_n_tokens:  populated
    per-subject.layer_expert_counts:    (16, 64)

## Still to do
1. Bit-identical reproducibility check across two same-seed runs (in flight).
2. Different-seed check (seed 0 vs seed 1) on the same model.
3. Wire `--random-router` and `--random-seed` into `scripts/run-evaluator.sh`
   with an opt-in `--random-router` flag and an optional seed override.
4. Update the README/CLAUDE.md notes for the eval-moe-* suite to mention the
   random baseline mode and the new CLI flags.
5. Document the shared header in `examples/_shared/` (currently has only the
   routing graph library — adding the random router header is a new file).
