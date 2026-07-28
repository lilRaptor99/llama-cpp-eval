# Handoff: `llama-eval-moe-coactivation`

## What this tool does

A C++ binary in the `llama.cpp` examples tree that runs an MoE LLM over a
slice of the MMLU dataset and records **pairwise expert co-activation
statistics** for downstream analysis in Python:

- **Intra-layer** `[L, E, E]`: which experts tend to fire together within
  the same MoE layer.
- **Inter-layer** `[L, L, E, E]` (default) or `[L, K, E, E]` (k-lag mode):
  position-aligned co-occurrence across layers (same token position in both).

Built on the same `ggml_backend_sched_eval_callback` hook as the existing
`eval-moe-mmlu` tool, but extends it to capture the **full per-token topk
trace** (not just marginal counts) so cross-layer pair statistics are
recoverable.

Also supports autoregressive generation (`--gen-tokens N`, default 4) so
routing during the answer-writing phase is captured too.

---

## Current state

**Phase 1 (MVP) and Phase 3 (multi-arch memory control) are complete and
validated.** Phase 2 (Python analysis with derived metrics) is partially done
as `analyze_coactivation.py`.

| Status | Phase | Description |
|--------|-------|-------------|
| ✅ Done | Phase 1 | C++ binary MVP for OLMoE (prefill + 4-token gen, intra + inter pair counts) |
| ✅ Done | Phase 2 (partial) | Python analysis script: derives conditional P, PMI, lift from raw counts; handles all 4 output formats |
| ✅ Done | Phase 3 | Memory controls: `--inter-k-lag K`, `--per-subject-inter`, `--sparse-min-count N` |
| ⬜ Open | Phase 4 (partial) | README polish, full-scale validation run, multi-arch testing, plotting |

**Smoke-tested on OLMoE-1B-7B-0125-Instruct-GGUF (16L, 64E, k=8)** with all
4 output formats. All 5 structural sanity checks pass (with format-aware
skips for k-lag and sparse modes).

---

## Architecture

```
ggml_backend_sched_eval_callback  (per-decode, per-layer)
        |
        v
moe_eval_callback (eval-moe-coactivation.cpp)
        |
        v   copies tensor data via ggml_backend_tensor_get
decode_slice buffers  (one per llama_decode call: prefill slice + 1 gen slice per gen step)
        |
        v   after all decodes for a question complete
tally_pair_counts_for_question()
        |
        v   walks slices once, fills per-subject counts
subject_pair_counts (one per MMLU subject)
        |     - marginal_expert_counts   [L, E]
        |     - intra_pair_counts        [L, E, E]
        |     - inter_pair_counts        [L, L, E, E]  (default)
        |     - inter_klag_counts        [L, K, E, E]  (when --inter-k-lag K>0)
        |
        v   end-of-main aggregation
aggregate counts
        |
        v   JSON output (one of 4 formats)
analyze_coactivation.py  (Python: conditional P, PMI, lift, top-K)
validate_sanity.py       (Python: 5 structural invariants)
```

**Key design choices** (do not change without care):
1. The per-question slice buffer is **flushed after every question** —
   `current_question_slices.clear()` in `tally_pair_counts_for_question`.
   Without this, we'd hold ~750 MB in flight for a 2850-question run.
2. `params.embedding = true` is required to force every token through every
   layer (not just the last). This is the same trick `eval-moe-mmlu` uses.
3. The `ggml_backend_tensor_get` non-contiguous-view copy logic (stride1 =
   `nb[1] / sizeof(int32)`) is critical — `ffn_moe_topk-<il>` is a view with
   `nb[1] = n_expert * sizeof(int32)`, not `k * sizeof(int32)`.
4. The greedy sampler mirrors `eval-moe-popqa`; EOS detection via
   `llama_vocab_is_eog(vocab, id)` breaks the gen loop early.

---

## File map

| Path | Purpose |
|---|---|
| `examples/eval-moe-coactivation/eval-moe-coactivation.cpp` | C++ binary, 1429 lines |
| `examples/eval-moe-coactivation/analyze_coactivation.py` | Python: derives metrics, prints top-K pairs |
| `examples/eval-moe-coactivation/validate_sanity.py` | Python: 5 structural invariants, handles all 4 formats |
| `examples/eval-moe-coactivation/README.md` | Build, run, schema, analysis docs |
| `examples/eval-moe-coactivation/CMakeLists.txt` | Build target (links `llama-common llama ${CMAKE_THREAD_LIBS_INIT}`) |
| `examples/CMakeLists.txt:23` | Registration: `add_subdirectory(eval-moe-coactivation)` |

**Reference files (read but don't modify):**
- `examples/eval-moe-mmlu/eval-moe-mmlu.cpp` — original marginal-count harness this tool was modeled on
- `examples/eval-moe-popqa/eval-moe-popqa.cpp` — generation pattern (sampler + EOS detection) at lines 733–833

**Plan and notes:**
- `/memories/session/plan.md` — the plan (Phases 1–4 with decisions and trade-offs)
- `/memories/repo/llama-cpp-mixtral-arch.md` — note that Mixtral uses `general.architecture = "llama"` (relevant for multi-arch testing)
- `/memories/repo/llama-cpp-popqa-research.md` — broader context on the existing eval-moe-* family

**Source files (read-only references):**
- `src/llama-graph.cpp:1913-1918` — `ffn_moe_topk` creation site
- `src/llama-graph.cpp:1799-2110` — `build_moe_ffn` body (tensor inventory)
- `src/llama-context.cpp:1307-1321` — where `cb_eval` is wired into the scheduler
- `src/llama-context.cpp:1684-1723` — where `output_all = cparams.embeddings` is set
- `ggml/src/ggml.c:5338-5356` — `ggml_argsort_top_k` (non-contiguous view semantics)
- `ggml/src/ggml-backend.cpp:1675-1715` — the scheduler callback loop

---

## Build & smoke test

```sh
# Configure (one-time)
cd /home/ubuntu/llama-cpp-eval
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build the new target
cmake --build build --target llama-eval-moe-coactivation -j$(nproc)

# Smoke test (prefill only, 1 question/subject)
MODEL=/home/ubuntu/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0125-Instruct-GGUF/snapshots/2ac1d27317927518c9ef7bd99f91f3f1e4ee288d/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf
./build/bin/llama-eval-moe-coactivation \
    -m "$MODEL" --questions-per-subject 1 --n-shots 0 --gen-tokens 0 \
    -o build/moe-coactivation/smoke.json

# Validate
source .venv/bin/activate
python examples/eval-moe-coactivation/validate_sanity.py \
    build/moe-coactivation/smoke.json

# Analyze
python examples/eval-moe-coactivation/analyze_coactivation.py \
    -i build/moe-coactivation/smoke.json \
    -o build/moe-coactivation/analyze
```

All 5 sanity checks should print `[N] OK`. The analyzer writes
`top_pairs.txt` (top-K PMI / lift pairs) and `enriched.json`.

**Multi-format smoke test** (each must complete without errors):

```sh
for flags in "" "--inter-k-lag 4" "--per-subject-inter" \
             "--sparse-min-count 10" \
             "--inter-k-lag 4 --per-subject-inter --sparse-min-count 10"; do
    f="build/moe-coactivation/phase3_$(echo $flags | tr ' ' '_' | tr -d '-').json"
    [ -z "$flags" ] && f="build/moe-coactivation/phase3_default.json"
    ./build/bin/llama-eval-moe-coactivation -m "$MODEL" \
        --questions-per-subject 1 --n-shots 0 --gen-tokens 0 $flags -o "$f"
    python3 examples/eval-moe-coactivation/validate_sanity.py "$f"
done
```

---

## JSON output schema (4 formats)

The schema depends on the `--inter-k-lag` and `--sparse-min-count` flags.
The `config` block always records the format choices so downstream tools can
dispatch on them.

```jsonc
{
  "model": "...",
  "model_arch": {"name": "olmoe", "n_layer": 16, "n_expert": 64, "n_expert_used": 8},
  "config": {
    "questions_per_subject": 50, "n_shot": 5, "gen_tokens": 4,
    "inter_k_lag": 0,            // 0 = full [L, L, E, E]; K>0 = [L, K, E, E]
    "per_subject_inter": false,  // also emit inter per subject
    "sparse_min_count": 0        // 0 = dense; >0 = COO sparse
  },
  "totals":   { "subjects_run", "questions_total", "tokens_prefill_total",
                "tokens_generated_total", "tokens_total" },
  "aggregate": {
    "tokens_prefill", "tokens_generated", "tokens_total",
    "marginal_expert_counts":  [[L, E] int64],         // k-multiplied
    "intra_pair_counts":       [[L, E, E] int64],
    // One of:
    "inter_pair_counts":         [[L, L, E, E] int64],   // default
    "inter_klag_counts":         [[L, K, E, E] int64],   // k-lag mode
    "inter_pair_counts_sparse":  {"shape", "indices", "values", "min_count", "nnz"},
    "inter_klag_counts_sparse":  {"shape", "indices", "values", "min_count", "nnz"}
  },
  "subjects": {
    "<subj>": {
      "questions", "tokens_prefill", "tokens_generated", "tokens_total",
      "marginal_expert_counts": [[L, E] int64],
      "intra_pair_counts":      [[L, E, E] int64]
      // inter_*_counts: only when --per-subject-inter is set
    }
  }
}
```

---

## Sanity invariants

The validator (`validate_sanity.py`) checks:

1. `tokens_total == tokens_prefill + tokens_generated` (per subject and aggregate)
2. `sum_j intra[L][e_i][j] == marginal[L][e_i]` (k-multiplied marginals)
3. `intra[L][e_i][e_j] == intra[L][e_j][e_i]` (symmetry)
4. `inter[L][L][e_i][e_j] == intra[L][e_i][e_j]` (cross-layer diagonal = intra)
   — **skipped when `--inter-k-lag K>0` or `--sparse-min-count N>0`**
5. `inter[L1 > L2] == 0` (lower triangle untouched)
6. `per_subject_inter: true` ⇒ all subjects have an inter field
7. `sparse_min_count: N` ⇒ the COO `min_count` matches the config

For the k-lag format, the validator **scatters** the `[L, K, E, E]` array
back into a dense `[L, L, E, E]` upper triangular array (with L1+k+1 → L2)
before running checks 4 and 5.

---

## Known limitations / gotchas

1. **No multi-arch testing yet.** The dynamic arch-key read (`<arch>.expert_count`,
   `<arch>.expert_used_count`) should work for Mixtral (uses `general.architecture = "llama"`),
   Qwen3-MoE (`qwen3moe`), DeepSeek-V3 (`deepseek2`), etc., but only OLMoE
   has actually been exercised. The smoke test for each arch requires the
   corresponding GGUF model to be cached at `~/.cache/huggingface/`.

2. **JSON size.** Aggregate inter for DeepSeek-V3 (60L × 256E) is 1.9 GB
   as JSON. Use `--inter-k-lag 4 --sparse-min-count 10` to bring this down to
   ~30 MB.

3. **Sparse format is lossy.** `--sparse-min-count N` filters cells with value ≤ N.
   The diagonal (intra cells) is included only if it exceeds N, so check 4
   (intra = diagonal-of-inter) is correctly skipped for sparse outputs.

4. **Per-question memory.** Each question's slice buffer is flushed at the
   end of `tally_pair_counts_for_question`. If you ever change this to
   accumulate across questions, you'll need to recompute the memory budget
   (≈260 KB per question for OLMoE prefill+4gen, ~750 MB for 2850 questions).

5. **No generation at k-lag K>0 with per-subject.** The combined flag smoke
   test passed all checks, but this combination is the largest memory footprint
   for OLMoE (~19 MB JSON). It has not been exercised on larger archs.

6. **Validator's load_inter helper assumes `n_layer` from `model_arch`.**
   If you add a model whose `n_layer` differs from the JSON's `model_arch.n_layer`,
   the scatter may misbehave.

7. **Some `insert_edit_into_file` edits in the early development truncated
   the file unexpectedly.** Always re-read the file after an `insert_edit_into_file`
   patch and verify the function structure is intact. (`apply_patch` with V4A
   diff syntax is more reliable for large multi-function rewrites.)

---

## Suggested next steps

### Short-term (a few hours)

1. **Full-scale validation run on OLMoE** (50 q × 5-shot × 4-gen):
   ```sh
   ./build/bin/llama-eval-moe-coactivation -m "$MODEL" \
       --questions-per-subject 50 --n-shots 5 --gen-tokens 4 \
       -o build/moe-coactivation/full_run.json
   ```
   Expected: ~3000+ seconds, ~3 GB JSON (default). Compare intra pair counts
   against `build/moe-mmlu-olmoe-1/expert_counts.json` to verify the
   marginals match.

2. **Cross-check marginals against `eval-moe-mmlu`.** With both tools run on
   the same data, `sum_e intra_pair_counts[L][e][j]` should match
   `MMLU_<subj>.layer_expert_counts[L][e]` per subject.

3. **Plotting.** Add a small `plot_coactivation.py` that renders
   intra-conditional-P heatmaps per layer and inter-conditional-P heatmaps
   per k-lag. Use `analyze_coactivation.py`'s top-K output as a starter.

### Medium-term (a day)

4. **Multi-arch smoke tests.** For each of Mixtral-8x7B, Qwen3-MoE,
   DeepSeek-V3, run a 1-question/subject smoke test and verify
   `validate_sanity.py` passes. The arch-specific GGUF key reads
   (`<arch>.expert_count`) should "just work" but verify.

5. **Streaming JSON output.** For DeepSeek-V3 at full scale, even `--inter-k-lag 4`
   aggregate output is ~60 MB. Consider streaming writes (one subject at a time)
   if memory or write-buffer issues emerge.

6. **Cross-token inter-layer analysis.** Add a `--inter-cross-token` flag
   that emits a separate `[L, L, E, E]` matrix for "any expert in L1 fires
   on the same question as any expert in L2" (coarser, cheaper). Phase 1
   design supports this — just a different accumulator.

### Long-term

7. **Routing-weight capture.** The C++ tool currently only counts which
   experts fired. The `ffn_moe_weights_norm` tensor (final normalized router
   weights) is also exposed through `cb(...)` at `src/llama-graph.cpp:1942`.
   Adding a flag to capture these (per-token expert weights) would enable
   weighted co-activation analysis (weighted PMI, etc.).

8. **Comparison with expert-group annotations.** For DeepSeek-V3-style
   `n_expert_groups`, the `ffn_moe_group_topk` tensor at
   `src/llama-graph.cpp:1903` exposes which expert groups were selected.
   A natural extension is to test whether the cross-layer co-activation
   patterns respect expert groups (i.e., "if group A fires in L1, does
   group B fire in L2 more than chance?").

9. **Integration with `llama-graph.cpp` for native reporting.** A cleaner
   long-term design would expose the pair-count accumulator through a
   public `llama.cb_pair_eval` callback in `include/llama.h`, letting
   downstream tools (Python bindings, etc.) consume pair counts in real time.

---

## Testing checklist before declaring done

- [ ] All 4 output formats produce JSON with the documented schema
- [ ] `validate_sanity.py` returns 5/0 OK on the default format
- [ ] `validate_sanity.py` returns 4/0 OK (with check 4 skipped) on k-lag
- [ ] `validate_sanity.py` returns 4/0 OK (with check 4 skipped) on sparse
- [ ] `analyze_coactivation.py` runs on all 4 formats without error
- [ ] Full-scale OLMoE run completes in a reasonable time
- [ ] Marginal counts from this tool match `eval-moe-mmlu` per subject
      (within tolerance for differing default `--gen-tokens 4` vs `--gen-tokens 0`)
- [ ] At least one non-OLMoE arch (Mixtral or Qwen3-MoE) smoke-tested
- [ ] README reflects any schema changes
- [ ] No compiler warnings on the latest build
