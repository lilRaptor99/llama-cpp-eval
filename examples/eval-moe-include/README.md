# llama-eval-moe-include

Evaluate an MoE LLM on a slice of [INCLUDE](https://huggingface.co/datasets/CohereLabs/include-base-44)
(Romanou et al., 2024, [arXiv:2411.19799](http://arxiv.org/abs/2411.19799)) and
record, per `(language, domain)` bucket, how often each expert cell was
activated at each MoE layer. Output is a single JSON file suitable for
post-hoc analysis in Python (expert-usage heatmaps, per-language /
per-domain / per-(lang, dom) aggregations, accuracy bars).

INCLUDE is a 44-language multilingual knowledge / reasoning MCQ benchmark
spanning academic + professional-licensing exams. The HuggingFace dataset
exposes 45 real language configs (the 45th is a duplicate `Dutch-Flemish`
variant whose `main` parquet has no live rows). This tool relies on the
`Dutch` (Schema A, 551 test rows) config — see
`examples/eval-moe-include/../../memories/repo/llama-cpp-include-research.md`
for the full data-quality investigation.

The tool uses the `ggml_backend_sched_eval_callback` hook exposed by
llama.cpp: every MoE forward pass materialises an `int32` tensor named
`ffn_moe_topk-<il>` of shape `[n_expert_used, n_tokens]`. The callback
filters on that name, copies the data via `ggml_backend_tensor_get`, and
tallies counts into a per-(lang, dom) `[n_layer][n_expert]` matrix.

It currently targets `olmoe` (e.g. `allenai/OLMoE-1B-7B-0125-Instruct-GGUF`).
Other archs (`mixtral`, `qwen2moe`, `qwen3moe`, `deepseek2`, `gpt-oss`) are
discovered through their respective metadata keys; the top-k tensor name is
the same, so routing counts will work for them too.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON            # or -DGGML_METAL=ON, etc.
cmake --build build --target llama-eval-moe-include -j
```

The binary is emitted to `build/bin/llama-eval-moe-include`.

## Prepare INCLUDE

```sh
pip install -r requirements/requirements-server-bench.txt
python examples/eval-moe-include/download_include.py
```

This writes (under `build/moe-include/`):

- `include.jsonl` one record per question (~23k rows for the 44-language set)
- `languages.txt` one language per line (44 by default)
- `domains.txt` one domain per line
- `languages_domains.txt` one `"<language>::<domain>"` per line, sorted
- `metadata.json` per-language + per-(lang, dom) row counts

Useful flags:

| Flag                    | Meaning                             | Default              |
| ----------------------- | ----------------------------------- | -------------------- |
| `--outdir <dir>`        | output directory                    | `build/moe-include/` |
| `--languages <list>`    | restrict to a subset of languages   | all 44               |
| `--domains <list>`      | restrict to a subset of domains     | all                  |
| `--limit-per-langdom N` | keep at most N rows per (lang, dom) | no limit             |
| `--include-validation`  | also download the validation split  | off                  |

## Run

```sh
./build/bin/llama-eval-moe-include \
    -hf allenai/OLMoE-1B-7B-0125-Instruct-GGUF \
    --questions-per-langdom 5 \
    --n-shots 5 \
    --gen-tokens 16 \
     --numa distribute
```

All standard llama.cpp flags are accepted (`-ngl`, `-c`, `--seed`, `-t`, ...).
The tool-specific options are:

| Flag                        | Meaning                                        | Default                                   |
| --------------------------- | ---------------------------------------------- | ----------------------------------------- |
| `--include <path>`          | `include.jsonl`                                | `build/moe-include/include.jsonl`         |
| `--langdom-list <path>`     | `languages_domains.txt`                        | `build/moe-include/languages_domains.txt` |
| `--questions-per-langdom N` | test questions per `(language, domain)` bucket | 5                                         |
| `--n-shots N`               | few-shot exemplars drawn from each bucket      | 5                                         |
| `--gen-tokens N`            | autoregressive decode cap (0 = prompt-only)    | 16                                        |
| `-o, --output <path>`       | output JSON                                    | `build/moe-include/expert_counts.json`    |

> **Note.** The first 5 rows of each `(language, domain)` bucket (sorted by
> question text) are reserved as **few-shot exemplars**; the remaining rows
> are tested up to `--questions-per-langdom`. This avoids test-row leakage
> into exemplars, matches what the canonical Harness does, and gives
> reproducible numbers across runs. The paper's protocol is **5-shot
> in-language** (Romanou et al. 2024 §4.2). To replicate the paper, leave
> `--n-shots 5`. Set `--n-shots 0` for 0-shot (faster, less stable).

Quick smoke test (1 test question per langdom, 0-shot, prompt-only):

```sh
./build/bin/llama-eval-moe-include \
    -m models/olmoe-1b-7b-0125-instruct-q4_k_m.gguf \
    -ngl 999 \
    --questions-per-langdom 1 --n-shots 0 --gen-tokens 0
```

## Visualize

```sh
python examples/eval-moe-include/heatmap_from_cpp.py -i build/moe-include/expert_counts.json
```

This writes:

- `routing_heatmap.png` layer x expert overview
- `routing_heatmap_by_language.png` ~44 languages x (L·E) cells (log1p)
- `routing_heatmap_by_language_normalized.png` ~44 languages x (L·E) cells (row-normalized)
- `routing_heatmap_by_domain.png` ~11 domains x (L·E) cells + per-layer-pair
- `routing_heatmap_by_langdom.png` ~484 buckets x (L·E) cells (log1p)
- `routing_heatmap_by_langdom_normalized.png` row-normalized (~484 buckets)
- `accuracy_by_langdom.png` per-(lang, dom) substring-match accuracy
- `counts_total.json` aggregated [L, E] matrix
- `metadata.json` pass-through + computed totals

Use `--heatmap languages-norm` to render only the new normalized-by-language
plot; `--heatmap all` (default) renders every plot in the list above.
