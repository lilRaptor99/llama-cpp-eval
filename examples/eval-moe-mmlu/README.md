# llama-eval-moe-mmlu

Evaluate an MoE LLM on a slice of MMLU and record, per subject, how often each
expert cell was activated at each MoE layer. Output is a single JSON file
suitable for post-hoc analysis (e.g. expert-usage heatmaps).

The tool is designed around the `ggml_backend_sched_eval_callback` hook exposed
by llama.cpp: every MoE forward pass materialises an `int32` tensor named
`ffn_moe_topk-<il>` of shape `[n_expert_used, n_tokens]`. The callback filters
on that name, copies the data via `ggml_backend_tensor_get`, and tallies counts
into a per-subject `[n_layer][n_expert]` matrix.

It currently targets `olmoe` (e.g. `allenai/OLMoE-1B-7B-0125-Instruct-GGUF`).
Other archs (`mixtral`, `qwen2moe`, `qwen3moe`, `deepseek2`, `gpt-oss`) are
discovered through their respective metadata keys; the top-k tensor name is the
same, so routing counts will work for them too.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON            # or -DGGML_METAL=ON, etc.
cmake --build build --target llama-eval-moe-mmlu -j
```

The binary is emitted to `build/bin/llama-eval-moe-mmlu`.

## Prepare MMLU

```sh
pip install -r requirements/requirements-server-bench.txt
python examples/eval-moe-mmlu/download_mmlu.py
```

This writes:

- `build/moe-mmlu/mmlu.jsonl`    one record per question
- `build/moe-mmlu/subjects.txt`  one subject per line (all 57 by default)

Pass `--subjects <list>` to restrict to a subset.

## Run

```sh
./build/bin/llama-eval-moe-mmlu \
    -hf allenai/OLMoE-1B-7B-0125-Instruct-GGUF \
    --questions-per-subject 100
     --numa distribute
```

All standard llama.cpp flags are accepted (`-ngl`, `-c`, `--seed`, `-t`, ...).
The tool-specific options are:

| Flag | Meaning | Default |
|---|---|---|
| `--mmlu <path>` | `mmlu.jsonl` | `build/moe-mmlu/mmlu.jsonl` |
| `--subjects <path>` | subjects list | `build/moe-mmlu/subjects.txt` |
| `--questions-per-subject N` | questions per MMLU subject | 50 |
| `--n-shots N` | few-shot exemplars from each dev split | 5 |
| `-o, --output <path>` | output JSON | `build/moe-mmlu/expert_counts.json` |

Quick smoke test (1 question/subject, 0-shot):

```sh
./build/bin/llama-eval-moe-mmlu \
    -m models/olmoe-1b-7b-0125-instruct-q4_k_m.gguf \
    -ngl 999 \
    --questions-per-subject 1 --n-shots 0
```

## Visualize

```sh
python examples/eval-moe-mmlu/heatmap_from_cpp.py -i build/moe-mmlu/expert_counts.json
```
