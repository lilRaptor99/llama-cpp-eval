# llama-eval-moe-bigbench

Evaluate an MoE LLM on [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard)
(Suzgun et al., 2022) and record, per BBH task config, how often each expert
cell was activated at each MoE layer. Output is a single JSON file suitable
for post-hoc analysis (e.g. expert-usage heatmaps, accuracy by task).

The tool uses the `ggml_backend_sched_eval_callback` hook exposed by
llama.cpp: every MoE forward pass materialises an `int32` tensor named
`ffn_moe_topk-<il>` of shape `[n_expert_used, n_tokens]`. The callback
filters on that name, copies the data via `ggml_backend_tensor_get`, and
tallies counts into a per-task `[n_layer][n_expert]` matrix.

Like [llama-eval-moe-popqa](../eval-moe-popqa/README.md), this tool also
**autoregressively decodes** up to `--gen-tokens` after prefill and
whitespace-normalises the first non-empty line of the completion, then
exact-matches against the BBH target. Both prefill and generation tokens
feed the routing counters in a **single combined matrix** so the result is
directly comparable to the PopQA output.

It is arch-agnostic: any MoE model whose GGUF exposes `<arch>.expert_count`
and `<arch>.expert_used_count` works (e.g. `allenai/OLMoE-1B-7B-0125-Instruct-GGUF`,
Mixtral, DeepSeek-MoE, Qwen2/3 MoE).

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_ZENDNN=ON            # or -DGGML_CUDA=ON -DGGML_METAL=ON, etc.
cmake --build build --target llama-eval-moe-bigbench -j
```

The binary is emitted to `build/bin/llama-eval-moe-bigbench`.

## Prepare BIG-Bench Hard

```sh
pip install -r requirements/requirements-server-bench.txt
python examples/eval-moe-bigbench/download_bigbench.py
```

This writes:

- `build/moe-bigbench/bigbench.jsonl` one record per question
- `build/moe-bigbench/tasks.txt` one BBH config per line, **27 configs**
- `build/moe-bigbench/sources.json` per-task provenance (which loader ran)
- `build/moe-bigbench/_cache/*.json` cached upstream JSON when the GitHub
  fallback is used

Each `bigbench.jsonl` record contains the four fields the C++ evaluator
needs:

```text
{"task": "<config>", "index": <int>, "input": "<question>", "target": "<gold>"}
```

Useful flags:

| Flag             | Meaning                                          |
| ---------------- | ------------------------------------------------ |
| `--outdir <dir>` | output directory (default: `build/moe-bigbench`) |
| `--tasks <list>` | restrict to these BBH configs (default: all 27)  |
| `--limit N`      | keep at most N rows per task (default: no limit) |

> **Note.** `maveriq/bigbenchhard` ships only an executable dataset
> builder script, which `datasets>=4` no longer supports. The downloader
> therefore tries `load_dataset(...)` first and falls back to a direct
> HTTPS download of the canonical per-task JSON from
> `suzgunmirac/BIG-Bench-Hard/bbh/<task>.json` when that fails. The
> fallback uses Python's stdlib `urllib.request`, so no extra dependency
> is added. Which loader served each task is recorded in `sources.json`.

## Dataset shape

The Hugging Face repository `maveriq/bigbenchhard` exposes **27
configurations** of Suzgun et al.'s 23 BIG-Bench Hard tasks. Two task
families are split by object count and therefore appear as three configs
each:

- `logical_deduction_three_objects`, `_five_objects`, `_seven_objects`
- `tracking_shuffled_objects_three_objects`, `_five_objects`, `_seven_objects`

This tool evaluates and reports **all 27 configs independently** and does
not aggregate them into 23 paper-level families; if you want the
paper-level numbers, sum the three configs of each family downstream.
The dataset exposes only a single `train` split, so all evaluations are
zero-shot (no dev split, no held-out test split, no example leakage into
prompts by design).

## Run

```sh
./build/bin/llama-eval-moe-bigbench \
    -hf allenai/OLMoE-1B-7B-0125-Instruct-GGUF \
    --questions-per-task 50 \
    --gen-tokens 128 \
    --numa distribute
```

All standard llama.cpp flags are accepted (`-ngl`, `-c`, `--seed`, `-t`,
`-b`, ...). The tool-specific options are:

| Flag                     | Meaning                                     | Default                                 |
| ------------------------ | ------------------------------------------- | --------------------------------------- |
| `--bigbench <path>`      | `bigbench.jsonl`                            | `build/moe-bigbench/bigbench.jsonl`     |
| `--tasks <path>`         | tasks list                                  | `build/moe-bigbench/tasks.txt`          |
| `--questions-per-task N` | questions per task config                   | 50                                      |
| `--gen-tokens N`         | autoregressive decode cap (0 = prompt-only) | 128                                     |
| `-o, --output <path>`    | output JSON                                 | `build/moe-bigbench/expert_counts.json` |

### Prompt and scoring

We use a zero-shot **direct** `Q: <input>\nA:` prompt. No chat template,
no examples, no chain-of-thought preamble. The model is expected to emit
the answer on the first generated line; we stop on newline to avoid
inflating the generated-token counter with trailing prose.

BBH targets are heterogeneous (choice letters, booleans, integers, Dyck
sequences, space-separated word lists), so scoring is **whitespace-
normalised exact match**: trim, collapse internal whitespace, take the
first non-empty line of the completion, and compare. Case and significant
punctuation (parens, brackets, signs, apostrophes) are preserved.

### Routing semantics

For each BBH task we record one combined `[n_layer][n_expert]` matrix
that sums both the prefill routing decisions and the generated-token
routing decisions. Token totals are kept separate as `n_tokens_prefill`
and `n_tokens_generated` so downstream tools can reason about phase-
specific traffic.

### Quick smoke test

```sh
./build/bin/llama-eval-moe-bigbench \
    -m models/olmoe-1b-7b-0125-instruct-q4_k_m.gguf \
    -ngl 999 \
    --tasks build/moe-bigbench/tasks.txt \
    --questions-per-task 1 --gen-tokens 0
```

(With `--gen-tokens 0` the tool runs in prompt-only mode; useful for
quick routing sanity checks without paying the decode cost.)

## Visualize

```sh
python examples/eval-moe-bigbench/heatmap_from_cpp.py -i build/moe-bigbench/expert_counts.json
```

This writes `routing_heatmap.png` (per-token activation rate overview),
`routing_heatmap_by_task.png` (27 tasks x L·E cells, log1p), and
`accuracy_by_task.png` (per-task exact-match accuracy bars), plus
`metadata.json` and `counts_total.json` for downstream tooling.

## Output schema

```text
{
  "model": "<hf model id or path>",
  "model_arch": {"name": "...", "n_layer": L, "n_expert": E, "n_expert_used": k},
  "config": {
    "questions_per_task": N,
    "gen_tokens": G,
    "prompt_format": "zero_shot_direct_qa",
    "match_metric": "normalized_exact_match"
  },
  "dataset": "maveriq/bigbenchhard",
  "split":   "train",
  "totals": {
    "tasks_run": <int>, "questions_total": <int>,
    "tokens_total_prefill": <int>, "tokens_total_generated": <int>,
    "correct": <int>, "accuracy": <float>
  },
  "tasks": {
    "<config>": {
      "questions": <int>,
      "n_tokens_prefill": <int>,
      "n_tokens_generated": <int>,
      "n_correct": <int>,
      "match_rate": <float>,
      "layer_expert_counts": [[int x E] x L]
    },
    ...
  }
}
```
