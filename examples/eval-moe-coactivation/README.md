# llama-eval-moe-coactivation

Evaluate an MoE LLM (e.g. `allenai/OLMoE-1B-7B-0125-Instruct-GGUF`) on a slice
of the MMLU dataset and record, per subject, **pairwise expert co-activation
statistics** both within a single MoE layer (intra-layer) and across pairs of
MoE layers (inter-layer, position-aligned on the same token). Output is a
single JSON file suitable for post-hoc analysis in Python (e.g. conditional
probability, PMI, lift heatmaps, expert-group detection).

The tool is built around the same `ggml_backend_sched_eval_callback` hook used
by `eval-moe-mmlu`: every MoE forward pass materialises an `int32` tensor
named `ffn_moe_topk-<il>` of shape `[n_expert_used, n_tokens]`. The callback
filters on that tensor name, copies the data via `ggml_backend_tensor_get`,
and stores it into a per-decode slice buffer. After **all** decodes for a
question complete (prefill + up to `--gen-tokens` generation steps), the tool
walks the slices once and tallies:

- **intra-layer pair counts** `[L, E, E]` — `intra[L][e_i][e_j]` = number of
  tokens where both `e_i` and `e_j` fired in the topk of layer `L` (with
  multiplicity across the `k` topk slots).
- **inter-layer pair counts** `[L, L, E, E]` — `inter[L1][L2][e_i][e_j]` =
  number of tokens `t` such that `e_i` fired in the topk of layer `L1` at
  position `t` **and** `e_j` fired in the topk of layer `L2` at position `t`
  (position-aligned). Stored for `L1 <= L2`; the lower triangle is zero.

Plus **marginal expert counts** `[L, E]` (k-multiplied, matching the convention
of `eval-moe-mmlu`).

It currently targets `olmoe`. Other archs (`mixtral`, `qwen2moe`, `qwen3moe`,
`deepseek2`, `gpt-oss`, etc.) are discovered through their respective
metadata keys; the top-k tensor name is the same, so pair counts will work
for them too.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON            # or -DGGML_METAL=ON, etc.
cmake --build build --target llama-eval-moe-coactivation -j
```

The binary is emitted to `build/bin/llama-eval-moe-coactivation`.

## Prepare MMLU

```sh
python examples/eval-moe-mmlu/download_mmlu.py
```

This writes (under `build/moe-mmlu/`):

- `mmlu.jsonl` one record per question
- `subjects.txt` one subject per line (all 57 by default)

`eval-moe-coactivation` reads these same files; no separate download is needed.

## Run

```sh
./build/bin/llama-eval-moe-coactivation \
    -hf allenai/OLMoE-1B-7B-0125-Instruct-GGUF \
    --questions-per-subject 50 \
    --n-shots 5 \
    --gen-tokens 4
```

The tool-specific options are:

| Flag                        | Meaning                                                                            | Default                                    |
| --------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------ |
| `--mmlu <path>`             | `mmlu.jsonl`                                                                       | `build/moe-mmlu/mmlu.jsonl`                |
| `--subjects <path>`         | subjects list                                                                      | `build/moe-mmlu/subjects.txt`              |
| `--questions-per-subject N` | questions per MMLU subject                                                         | 50                                         |
| `--n-shots N`               | few-shot exemplars from each dev split                                             | 5                                          |
| `--gen-tokens N`            | autoregressive decode cap (0 = prompt-only)                                        | 4                                          |
| `--inter-k-lag K`           | restrict inter-layer to k=1..K only (memory control for large archs); 0 = full     | 0                                          |
| `--per-subject-inter`       | also emit inter-layer pair counts per subject (default: aggregate only)            | off                                        |
| `--sparse-min-count N`      | emit inter-layer as COO {shape, indices, values} keeping only cells with value > N | 0 (dense)                                  |
| `-o, --output <path>`       | output JSON                                                                        | `build/moe-coactivation/coactivation.json` |

Standard llama.cpp flags (`-ngl`, `-c`, `--seed`, `-t`, ...) are also accepted.

Quick smoke test (1 question/subject, 0-shot, no generation):

```sh
./build/bin/llama-eval-moe-coactivation \
    -m models/olmoe-1b-7b-0125-instruct-q4_k_m.gguf \
    -ngl 999 \
    --questions-per-subject 1 --n-shots 0 --gen-tokens 0
```

## Output JSON schema

The schema depends on the `--inter-k-lag` and `--sparse-min-count` flags. The
`config` block always records the format choices so downstream tools can
dispatch on them.

```jsonc
{
  "model": "...",
  "model_arch": {"name": "olmoe", "n_layer": 16, "n_expert": 64, "n_expert_used": 8},
  "config": {
    "questions_per_subject": 50,
    "n_shot": 5,
    "gen_tokens": 4,
    "inter_k_lag": 0,            // 0 = full [L, L, E, E]; K>0 = [L, K, E, E]
    "per_subject_inter": false,  // also emit inter per subject
    "sparse_min_count": 0,       // 0 = dense; >0 = COO sparse (cells > N)
    "few_shot_pool": "cais/mmlu dev split",
    "prompt_format": "few_shot_chat"
  },
  "totals": {
    "subjects_run": 57,
    "questions_total": 2850,
    "tokens_prefill_total": 1862535,
    "tokens_generated_total": 11400,
    "tokens_total": 1873935
  },
  "aggregate": {
    "tokens_prefill": 1862535,
    "tokens_generated": 11400,
    "tokens_total": 1873935,
    "marginal_expert_counts": [[L, E] int64],          // k-multiplied
    "intra_pair_counts":      [[L, E, E] int64],

    // One of the following four, depending on the flags:
    "inter_pair_counts":         [[L, L, E, E] int64],  // default (inter_k_lag=0, sparse=0)
    "inter_klag_counts":         [[L, K, E, E] int64],  // --inter-k-lag K>0, sparse=0
    "inter_pair_counts_sparse":  {"shape": [...], "indices": [...], "values": [...], "min_count": N, "nnz": N},
    "inter_klag_counts_sparse":  {"shape": [...], "indices": [...], "values": [...], "min_count": N, "nnz": N}
  },
  "subjects": {
    "<subj>": {
      "questions": 50,
      "tokens_prefill": ...,
      "tokens_generated": ...,
      "tokens_total": ...,
      "marginal_expert_counts": [[L, E] int64],
      "intra_pair_counts":      [[L, E, E] int64],
      // inter_*_counts is only present when --per-subject-inter is set
      "inter_*_counts":          (one of the four formats above)
    },
    ...
  }
}
```

## Analysis (Python)

```sh
source .venv/bin/activate
python examples/eval-moe-coactivation/analyze_coactivation.py \
    -i build/moe-coactivation/coactivation.json \
    -o build/moe-coactivation/analyze
```

Writes:

- `analyze/top_pairs.txt` — top-K intra-layer and inter-layer pairs by PMI and
  lift (with conditional P and raw counts)
- `analyze/enriched.json` — derived metrics for the aggregate:
  `marginal_p`, `intra_conditional_p`, `intra_pmi`, `intra_lift`, and the
  top-K inter-layer PMI / lift pairs

## Expert graph visualization (Python)

Render expert nodes in a fixed grid (x = expert index, y = layer index) and
draw coactivation edges where line thickness is proportional to pair count.
Inter-layer edges are only drawn between adjacent layers (L -> L+1).

```sh
source .venv/bin/activate
python examples/eval-moe-coactivation/plot_expert_graph.py \
    -i build/moe-coactivation/coactivation.json \
    -o build/moe-coactivation/expert_graph.png
```

Default behavior:

- includes both intra-layer and inter-layer edges
- uses raw pair count for edge thickness
- keeps top-20 edges per layer-pair block (`--top-k-per-layer-pair 20`)
- renders all layers by default (override with `--layer-min` / `--layer-max`)

Useful options:

```sh
# Layer window and stronger pruning
python examples/eval-moe-coactivation/plot_expert_graph.py \
    -i build/moe-coactivation/coactivation.json \
    --layer-min 8 --layer-max 15 \
    --top-k-per-layer-pair 12 \
    -o build/moe-coactivation/expert_graph_L8_L15.png

# Plot a subject (requires --per-subject-inter at data collection time for inter edges)
python examples/eval-moe-coactivation/plot_expert_graph.py \
    -i build/moe-coactivation/coactivation.json \
    --subject abstract_algebra \
    -o build/moe-coactivation/expert_graph_abstract_algebra.png
```

The plotter supports all four inter output formats automatically:

- `inter_pair_counts` (dense full)
- `inter_klag_counts` (dense k-lag)
- `inter_pair_counts_sparse` (sparse COO full)
- `inter_klag_counts_sparse` (sparse COO k-lag)

## Sanity validation

```sh
source .venv/bin/activate
python examples/eval-moe-coactivation/validate_sanity.py \
    build/moe-coactivation/coactivation.json
```

Verifies structural invariants:

1. `tokens_total == tokens_prefill + tokens_generated` (per subject and aggregate)
2. `sum_j intra_pair_counts[L][e][j] == marginal_expert_counts[L][e]` (k-multiplied)
3. `intra[L][e_i][e_j] == intra[L][e_j][e_i]` (symmetric)
4. `inter[L][L][e_i][e_j] == intra[L][e_i][e_j]` (cross-layer on the diagonal matches intra) — _skipped when `--inter-k-lag K>0` (k=0 not emitted) or `--sparse-min-count N>0` (sparse is lossy by design)_
5. `inter[L1][L2] == 0` for `L1 > L2` (lower triangle untouched)
6. `per_subject_inter: true` ⇒ all subjects have an inter field (only checked when the flag is set)
7. `sparse_min_count: N` ⇒ the `min_count` field in the COO output matches the config (only checked when the flag is set)

The validator handles all four output formats (default / k-lag / per-subject
/ sparse) automatically.

## Differences from `eval-moe-mmlu`

- Captures the full per-token topk trace, not just marginal counts, so
  position-aligned cross-layer correlation is recoverable.
- Emits a 3D `[L, E, E]` intra pair-count matrix and a 4D `[L, L, E, E]`
  inter pair-count matrix (raw counts only; derived metrics are computed in
  Python).
- Supports autoregressive generation (`--gen-tokens N`); routing during the
  model's answer-writing phase is also captured.
- Tracks `tokens_prefill` and `tokens_generated` separately.

## Phase 3: large-arch memory control

For very large MoE models the dense `[L, L, E, E]` inter output quickly becomes
unwieldy. Three flags control the trade-off:

### `--inter-k-lag K` (default 0)

Restrict the inter-layer output to lag `k=1..K` only. The output shape
becomes `[L, K, E, E]` instead of `[L, L, E, E]`. For DeepSeek-V3 (60L × 256E)
this brings the aggregate inter from 235M int64 entries (~1.9 GB) down to
60 × 4 × 256 × 256 × 8 = 60 MB for `K=4`. The diagonal `k=0` (= the intra
layer) is **not** duplicated; consumers should use `intra_pair_counts` for that.

```sh
# DeepSeek-V3 with k-lag K=4
./build/bin/llama-eval-moe-coactivation -m <deepseek.gguf> \
    --inter-k-lag 4 --questions-per-subject 1
```

### `--per-subject-inter`

By default only the aggregate inter is emitted (subjects only get marginals
and intra-layer pairs). Setting this flag also emits inter-layer pair counts
per subject. This is useful for subject-conditional analysis (e.g. "does
mathematics activate different cross-layer expert groups than philosophy?")
but can be large — for OLMoE it adds ~8 MB per subject × 57 subjects = ~450 MB
to the JSON.

### `--sparse-min-count N`

Emit the inter-layer output in COO (coordinate, value) format keeping only
cells with value > N. Useful when most (e_i, e_j) pairs have small counts and
you want to focus on the strong co-occurrences. For dense matrices (e.g. OLMoE
where ~96% of cells are non-zero) sparse encoding is **larger** than dense; use
it only for very sparse or large-E models.

```jsonc
"inter_pair_counts_sparse": {
  "shape": [16, 16, 64, 64],
  "min_count": 10,
  "nnz": 469618,
  "indices": [[0, 0, 6, 6], [0, 0, 41, 41], ...],
  "values":  [3092, 2212, ...]
}
```

### Memory and runtime

The dense `[L, L, E, E]` aggregate inter output is the dominant memory
consumer. Phase 3 controls the trade-off:

| Arch         | n_layer | n_expert | Default full `[L,L,E,E]` | k-lag K=4 `[L,K,E,E]` |
| ------------ | ------- | -------- | ------------------------ | --------------------- |
| OLMoE        | 16      | 64       | 8 MB                     | 2 MB                  |
| Mixtral-8x7B | 32      | 8        | 0.5 MB                   | 0.06 MB               |
| Qwen3-MoE    | 48      | 128      | 300 MB                   | 25 MB                 |
| DeepSeek-V3  | 60      | 256      | 1.9 GB                   | 60 MB                 |

Per-subject inter multiplies the aggregate size by the number of subjects.
For large archs use the default (aggregate-only) plus the Python analysis
script for subject-conditional views, or enable `--per-subject-inter`
together with `--inter-k-lag K` to cap the per-subject cost.
