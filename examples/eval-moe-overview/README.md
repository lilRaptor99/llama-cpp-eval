# llama-eval-moe-overview (cross-dataset aggregator)

Cross-dataset aggregator for the MoE expert-routing statistics produced by
the per-dataset `llama-eval-moe-*` C++ binaries.

Unlike [`eval-moe-humaneval/heatmap_from_cpp.py`](../eval-moe-humaneval/heatmap_from_cpp.py)
(and its bigbench / mmlu / popqa / include siblings), which operate on a
single `expert_counts.json`, this tool walks every
`build/results/<model>/moe-*/expert_counts.json` under the results tree,
sums the [n_layer, n_expert] count matrices across all datasets for each
model, and writes one overview set of artifacts per model under
`build/results/<model>/overall/`.

## What it produces (per model)

For each `<model>` subdirectory found under `--results-dir`, the script
writes `<results-dir>/<model>/overall/`:

| File | Description |
|------|-------------|
| `routing_heatmap_overview.png` | Layer × expert selections/token rate (aggregated across all datasets). |
| `routing_heatmap_overview_highlighted.png` | Same heatmap with the top ⌈n_expert × `--top-k-fraction`⌉ experts per layer outlined in red. |
| `top_experts_bars.png` | One panel per layer: top-K activation counts (descending). |
| `top_experts.json` | Per-layer top-K list with `expert_id`, `rank`, `count`, `fraction_of_layer_total`, `cumulative_share`. Optionally includes `per_dataset_layer_topk` when `--include-per-dataset` is passed. |
| `counts_total_overview.json` | Raw aggregated L×E matrix (sum of raw counts, not normalized). |
| `metadata_overview.json` | Model id, arch, per-dataset token split, total tokens, generation timestamp. |

## Aggregation

For each model we compute:

```
counts_total_overview[L, E] = sum over all datasets of (sum over tasks/subjects/.../langdoms of layer_expert_counts[L, E])
total_tokens_overview      = sum over all datasets of (prefill + gen tokens; for mmlu: subject.n_tokens)
selections_per_token[L, E] = counts_total_overview[L, E] / total_tokens_overview
```

This is the same convention used by the per-dataset `save_overview_heatmap`
functions: selections / token, where `top_k / n_expert` is the uniform
expectation.

## Top-K selection

Per model, `top_k_count = ceil(n_expert × --top-k-fraction)`. Defaults:

| Model | n_expert | ceil(n_expert × 0.125) |
|-------|----------|------------------------|
| unsloth/gpt-oss-120b | 128 | 16 |
| LiteLLMs/Mixtral-8x22B-Instruct-v0.1 | 8 | 1 |
| allenai/OLMoE-1B-7B-0125-Instruct | 64 | 8 |
| mradermacher/deepseek-moe-16b-chat | 64 | 8 |

Per layer, experts are sorted by `counts_total_overview[layer]` descending
and the top-K indices + statistics are persisted.

## Usage

```sh
python examples/eval-moe-overview/aggregate_overview.py --results-dir build/results
```

Flags:

| Flag | Meaning | Default |
|------|---------|---------|
| `--results-dir` | Root directory containing per-model subdirectories | `build/results` |
| `--top-k-fraction` | Fraction of experts per layer to highlight / report as top-K | `0.125` |
| `--include-per-dataset` | Add `per_dataset_layer_topk` to `top_experts.json` | off |
| `--models` | Restrict to a subset of model directory names (repeatable) | (all) |
| `--colormap` | Matplotlib colormap | `viridis` |
| `--dpi` | Output PNG DPI | `120` |

## Known data quirks

- `unsloth--gpt-oss-120b-GGUF/moe-mmlu/expert_counts.json` reports
  `model_arch.name = "olmoe"` even though the model is gpt-oss (L/E/k
  values are correct: 36/128/4). This script reads L/E/k directly from
  the JSON, so the misnamed field does not affect the output.
- `LiteLLMs--Mixtral-8x22B-Instruct-v0.1-GGUF/moe-mmlu` is incomplete
  (only `run.log` present). The script logs a skipped warning and
  aggregates the remaining datasets (humaneval, include).
- The script tolerates `expert_counts.json` files with zero tasks (e.g.
  failed runs) by logging and continuing.
