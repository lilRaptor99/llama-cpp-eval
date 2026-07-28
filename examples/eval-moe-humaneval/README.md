# llama-eval-moe-humaneval

Evaluate an MoE LLM on [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval)
(Chen et al., 2021, "Evaluating Large Language Models Trained on Code",
arXiv:2107.03374) and record, per problem (identified by upstream
`task_id`), how often each expert cell was activated at each MoE layer.

The tool uses the `ggml_backend_sched_eval_callback` hook exposed by
llama.cpp: every MoE forward pass materialises an `int32` tensor named
`ffn_moe_topk-<il>` of shape `[n_expert_used, n_tokens]`. The callback
filters on that name, copies the data via `ggml_backend_tensor_get`, and
tallies counts into a per-`task_id` `[n_layer][n_expert]` matrix.

Like [llama-eval-moe-bigbench](../eval-moe-bigbench/README.md) and
[llama-eval-moe-popqa](../eval-moe-popqa/README.md), this tool
**autoregressively decodes** up to `--gen-tokens` after prefill. The
generated text is persisted verbatim per problem in the output JSON so
downstream tooling can decide what to do with it (run pass@1 scoring,
compare to the canonical solution, etc.).

**Crucially, this tool never executes the generated Python and never
scores it for correctness.** The `match_metric` field of the output JSON
is the literal string `"none"`. The canonical HumanEval pass@1 evaluation
must be performed by the user against the persisted `completion` field
(e.g. by writing the completion to a temporary file and running the
upstream `human_eval` `check(candidate)` harness in a sandboxed
interpreter).

Both prefill and generation tokens feed the routing counters in a single
combined `[n_layer][n_expert]` matrix per problem so the result is
directly comparable to the PopQA and BIG-Bench-Hard outputs.

It is arch-agnostic: any MoE model whose GGUF exposes `<arch>.expert_count`
and `<arch>.expert_used_count` works (e.g. `allenai/OLMoE-1B-7B-0125-Instruct-GGUF`,
Mixtral, DeepSeek-MoE, Qwen2/3 MoE).

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_ZENDNN=ON            # or -DGGML_CUDA=ON -DGGML_METAL=ON, etc.
cmake --build build --target llama-eval-moe-humaneval -j
```

The binary is emitted to `build/bin/llama-eval-moe-humaneval`.

## Prepare HumanEval

```sh
pip install -r requirements/requirements-server-bench.txt
python examples/eval-moe-humaneval/download_humaneval.py
```

This writes:

- `build/moe-humaneval/humaneval.jsonl` one record per problem (164 rows)
- `build/moe-humaneval/tasks.txt` one `task_id` per line, in upstream order

Each `humaneval.jsonl` record contains the five fields the upstream
dataset exposes:

```text
{"task_id": "<upstream id, e.g. HumanEval/0>",
 "prompt": "<function header + docstring>",
 "canonical_solution": "<gold solution body>",
 "test": "<check(candidate) driver>",
 "entry_point": "<function name>"}
```

The C++ binary reads only `task_id`, `prompt`, and `entry_point`; the
other two fields are kept in the JSONL for downstream pass@1 tooling.

Useful flags:

| Flag             | Meaning                                           |
| ---------------- | ------------------------------------------------- |
| `--outdir <dir>` | output directory (default: `build/moe-humaneval`) |
| `--tasks <list>` | restrict to these `task_id` values (default: all) |
| `--limit N`      | keep at most N rows total (default: no limit)     |

> **Note.** HumanEval's HF dataset ships a single `test` split of 164
> rows and exposes the upstream `task_id` strings directly. There is no
> train / validation / dev split, so this tool runs **zero-shot only**;
> there is no `--n-shots` flag by design (matches the published Chen
> et al. protocol). There is also no GitHub-fallback path: HumanEval is
> not a script-based dataset, so `load_dataset(..., split="test")` works
> directly with modern `datasets>=4`.

## Dataset shape

HumanEval has 164 Python programming problems. Each problem provides a
function signature + docstring (`prompt`), a gold canonical solution
(`canonical_solution`), a `check(candidate)` driver (`test`), and the
function's entry point name (`entry_point`). The dataset is intentionally
hand-crafted to avoid leakage from GitHub; the original paper uses
zero-shot pass@1 with greedy decoding and a 200-token generation cap.

## Run

```sh
./build/bin/llama-eval-moe-humaneval \
    -hf allenai/OLMoE-1B-7B-0125-Instruct-GGUF \
    --questions-per-task 164 \
    --gen-tokens 256 \
    --numa distribute
```

All standard llama.cpp flags are accepted (`-ngl`, `-c`, `--seed`, `-t`,
`-b`, ...). The tool-specific options are:

| Flag                     | Meaning                                     | Default                                  |
| ------------------------ | ------------------------------------------- | ---------------------------------------- |
| `--humaneval <path>`     | `humaneval.jsonl`                           | `build/moe-humaneval/humaneval.jsonl`    |
| `--tasks <path>`         | tasks list                                  | `build/moe-humaneval/tasks.txt`          |
| `--questions-per-task N` | problems per `task_id`                      | 164                                      |
| `--gen-tokens N`         | autoregressive decode cap (0 = prompt-only) | 256                                      |
| `-o, --output <path>`    | output JSON                                 | `build/moe-humaneval/expert_counts.json` |

### Prompt and scoring

We feed each problem's upstream `prompt` directly as a Python
continuation prompt (the function header + docstring). No chat
template, no `Q: ... A:` wrapper, no manual indentation -- any wrapper
would dominate the prompt budget on small context windows and would
contaminate the routing signal.

The default sampler is **greedy**, matching the canonical HumanEval
pass@1 protocol. Generation stops only on EOS/EOG **or** after
`--gen-tokens` non-EOG tokens have been decoded and counted. There is
**no newline / dedent / blank-line early-stop heuristic**: the user
asked for a fixed token budget, so we always decode exactly
`--gen-tokens` non-EOG tokens (or stop early on EOG). The
`completion_truncated` flag in the output JSON tells you which problems
hit the cap without emitting EOG.

### Routing semantics

For each `task_id` we record one combined `[n_layer][n_expert]` matrix
that sums both the prefill routing decisions and the generated-token
routing decisions. Token totals are kept separate as `n_tokens_prefill`
and `n_tokens_generated` so downstream tools can reason about phase-
specific traffic.

### Quick smoke test

```sh
./build/bin/llama-eval-moe-humaneval \
    -m models/olmoe-1b-7b-0125-instruct-q4_k_m.gguf \
    -ngl 999 \
    --questions-per-task 1 --gen-tokens 0
```

(With `--gen-tokens 0` the tool runs in prompt-only mode; useful for
quick routing sanity checks without paying the decode cost.)

## Visualize

```sh
python examples/eval-moe-humaneval/heatmap_from_cpp.py -i build/moe-humaneval/expert_counts.json
```

This writes `routing_heatmap.png` (per-token activation rate overview,
L×E), `routing_heatmap_by_task.png` (164 problems in a `sqrt(164)` ×
`sqrt(164)` grid of small L×E heatmaps, log1p color), and
`routing_heatmap_by_task_normalized.png` (same layout, row-normalized),
plus `metadata.json` and `counts_total.json` for downstream tooling.

There is **no accuracy / match-rate plot** by design -- the tool never
scores generated code for correctness.

## Output schema

```text
{
  "model": "<hf model id or path>",
  "model_arch": {"name": "...", "n_layer": L, "n_expert": E, "n_expert_used": k},
  "config": {
    "questions_per_task": <int>,
    "gen_tokens": <int>,
    "prompt_format": "humaneval_zero_shot_continuation",
    "match_metric": "none"
  },
  "dataset": "openai/openai_humaneval",
  "split":   "test",
  "totals": {
    "tasks_run": <int>, "questions_total": <int>,
    "tokens_total_prefill": <int>, "tokens_total_generated": <int>,
    "completions_truncated": <int>
  },
  "tasks": {
    "<task_id>": {
      "entry_point": "<function name>",
      "questions": <int>,
      "n_tokens_prefill": <int>,
      "n_tokens_generated": <int>,
      "completion_truncated": <bool>,
      "layer_expert_counts": [[int x E] x L],
      "completion": "<full generated text, json_escaped>"
    },
    ...
  }
}
```

## Re-scoring the generated code (optional)

The output JSON persists the raw `completion` for each `task_id` so you
can re-run pass@1 yourself without paying the model's decode cost again:

```python
import json

with open("build/moe-humaneval/expert_counts.json") as f:
    d = json.load(f)

for task_id, body in d["tasks"].items():
    completion = body["completion"]
    entry_point = body["entry_point"]
    if not completion.strip():
        continue  # nothing to score
    # Reconstruct a candidate function and run `check(candidate)` in a
    # sandboxed interpreter; see https://github.com/openai/human-eval
    # for the canonical scoring harness.
```

The `canonical_solution` and `test` fields are deliberately persisted
in the `humaneval.jsonl` (downloaded by `download_humaneval.py`) for
exactly this purpose -- they are not read by the C++ binary.

> **Safety.** Generated Python must never be executed on a trusted
> machine. Use a sandbox (Docker, gVisor, firejail, or the upstream
> `human_eval` `execution_check` flag) and never pipe completion text
> into a shell.
