# llama-eval-moe-popqa

Evaluate an MoE LLM on a slice of [PopQA](https://huggingface.co/datasets/akariasai/PopQA)
(Mallen et al., 2022) and record, per relation type (`prop`), how often each
expert cell was activated at each MoE layer. Output is a single JSON file
suitable for post-hoc analysis (e.g. expert-usage heatmaps, match-rate by
relation type).

The tool is designed around the `ggml_backend_sched_eval_callback` hook exposed
by llama.cpp: every MoE forward pass materialises an `int32` tensor named
`ffn_moe_topk-<il>` of shape `[n_expert_used, n_tokens]`. The callback filters
on that name, copies the data via `ggml_backend_tensor_get`, and tallies counts
into a per-`prop` `[n_layer][n_expert]` matrix.

Unlike `llama-eval-moe-mmlu` (which is decode-only), this tool also
**autoregressively decodes** up to `--gen-tokens` after prefill and substring-
matches the completion against `possible_answers` (the canonical self-rag
metric). Both prefill and generation tokens feed the routing counters.

It currently targets `olmoe` (e.g. `allenai/OLMoE-1B-7B-0125-Instruct-GGUF`).
Other archs (`mixtral`, `qwen2moe`, `qwen3moe`, `deepseek2`, `gpt-oss`) are
discovered through their respective metadata keys; the top-k tensor name is the
same, so routing counts will work for them too.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON            # or -DGGML_METAL=ON, etc.
cmake --build build --target llama-eval-moe-popqa -j
```

The binary is emitted to `build/bin/llama-eval-moe-popqa`.

## Prepare PopQA

```sh
pip install -r requirements/requirements-server-bench.txt
python examples/eval-moe-popqa/download_popqa.py
```

This writes:

- `build/moe-popqa/popqa.jsonl`    one record per question
- `build/moe-popqa/props.txt`      one `prop` (relation type) per line, ~16

Pass `--props <list>` to restrict to a subset, `--limit N` to subsample evenly
across props.

## Run

```sh
./build/bin/llama-eval-moe-popqa \
    -hf allenai/OLMoE-1B-7B-0125-Instruct-GGUF \
    --questions-per-prop 50 \
    --gen-tokens 16 \
     --numa distribute
```

All standard llama.cpp flags are accepted (`-ngl`, `-c`, `--seed`, `-t`, ...).
The tool-specific options are:

| Flag | Meaning | Default |
|---|---|---|
| `--popqa <path>` | `popqa.jsonl` | `build/moe-popqa/popqa.jsonl` |
| `--props <path>` | props list | `build/moe-popqa/props.txt` |
| `--questions-per-prop N` | questions per relation type | 50 |
| `--gen-tokens N` | autoregressive decode cap (0 = prompt-only) | 16 |
| `-o, --output <path>` | output JSON | `build/moe-popqa/expert_counts.json` |

> **Note.** PopQA's HF dataset ships with a single `test` split (14,267 rows).
> There is no dev split, so this tool runs **zero-shot only**; there is no
> `--n-shots` flag by design (matches the published Mallen et al. protocol).

Quick smoke test (1 question/prop, prompt-only):

```sh
./build/bin/llama-eval-moe-popqa \
    -m models/olmoe-1b-7b-0125-instruct-q4_k_m.gguf \
    -ngl 999 \
    --questions-per-prop 1 --gen-tokens 0
```

## Visualize

```sh
python examples/eval-moe-popqa/heatmap_from_cpp.py -i build/moe-popqa/expert_counts.json
```

This writes `routing_heatmap.png` (per-token activation rate overview),
`routing_heatmap_by_prop.png` (16 rows x L·E cells, log1p), and
`match_rate_by_prop.png` (per-prop substring-match accuracy bars).
