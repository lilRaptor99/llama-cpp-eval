# llama-eval-moe-overview (cross-dataset aggregator)

Cross-dataset aggregator for the MoE expert-routing statistics produced by
the per-dataset `llama-eval-moe-*` C++ binaries.

Unlike [`eval-moe-humaneval/heatmap_from_cpp.py`](../eval-moe-humaneval/heatmap_from_cpp.py)
(and its bigbench / mmlu / popqa / include siblings), which operate on a
single `expert_counts.json`, this tool walks every
`build/results/<model>/<quant>/moe-*/expert_counts.json` under the
results tree, sums the [n_layer, n_expert] count matrices across all
datasets for each `(model, quant)` cell, and writes one overview set of
artifacts per cell under `build/results/<model>/<quant>/overall/`.

A legacy (no-quant) layout is also tolerated when the results tree has
not been migrated yet:

```text
<results-dir>/<model>/moe-*/expert_counts.json        # legacy
    -> <results-dir>/<model>/overall/                  # (quant_name = "")

<results-dir>/<model>/<quant>/moe-*/expert_counts.json  # new (preferred)
    -> <results-dir>/<model>/<quant>/overall/
```

When both structures are present under the same model directory, the
new layout takes precedence.

## What it produces (per (model, quant) cell)

For each `(model, quant)` cell the script writes
`<results-dir>/<model>/<quant>/overall/` (or
`<results-dir>/<model>/overall/` for legacy cells):

| File                                       | Description                                                                                                                                                                                                                                                                                                                                               |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `routing_heatmap_overview.png`             | L × E circle grid, one circle per expert. Circle fill is the **raw** activation count (linear scale, Blues colormap). Layer 0 is at the top; circle fill darker = more activity. **Dark-blue lines** connect the top `--top-k-pairs` adjacent-layer expert co-activations (line thickness proportional to raw count, same contract as the routing graph). |
| `routing_heatmap_overview_highlighted.png` | Same circle grid as the overview, with a thin **red ring** (matching the routing-graph ring colour in `eval-moe-mmlu/routing_graph_from_cpp.py`) around the top `⌈n_expert × --top-k-fraction` experts per layer. Includes the same co-activation lines.                                                                                                  |
| `top_experts_bars.png`                     | One panel per layer: top-K activation counts (descending, fractions of layer total).                                                                                                                                                                                                                                                                      |
| `top_experts.json`                         | Per-layer top-K list with `expert_id`, `rank`, `count`, `fraction_of_layer_total`, `cumulative_share`. Optionally includes `per_dataset_layer_topk` when `--include-per-dataset` is passed.                                                                                                                                                               |
| `counts_total_overview.json`               | Raw aggregated L×E matrix (sum of raw counts, not normalized). When the per-dataset JSONs include the `aggregate` block, the aggregated `[L-1, E, E] int64` `adjacent_pair_counts` array is also persisted here under the same key, with a `adjacent_pair_counts_shape` header.                                                                           |
| `metadata_overview.json`                   | Model id, arch, quant, per-dataset token split, total tokens, generation timestamp, and a `coactivation_lines` block documenting the `--top-k-pairs` / `--line-scale` values that produced the PNG.                                                                                                                                                       |

## Aggregation

For each `(model, quant)` cell we compute:

```
counts_total_overview[L, E] = sum over all datasets of (sum over tasks/subjects/.../langdoms of layer_expert_counts[L, E])
total_tokens_overview      = sum over all datasets of (prefill + gen tokens; for mmlu: subject.n_tokens)
selections_per_token[L, E] = counts_total_overview[L, E] / total_tokens_overview
```

The per-token rate is what `top_k / n_expert` is compared against in
the metadata sanity check. The PNG heatmaps themselves use **raw
counts** (no per-token normalisation) to match the visual style of the
routing-graph heatmaps in `eval-moe-mmlu/routing_graph_from_cpp.py`.

## Heatmap visual style

Both `routing_heatmap_overview.png` and `*_highlighted.png` use a
**circle grid** layout: one row per layer, one column per expert, with
each cell drawn as a filled circle. The fill colour is linearly scaled
to the raw activation count using the Blues colormap (no log scale),
so darker blue = more activity. The light-grey ring around each
circle is just a border so zero-count cells are still visible.

The highlighted variant adds a thin red ring (`#ff1744`, width 1.5)
around each top-K cell per layer. The colour matches the routing
graph's highlight ring in `eval-moe-mmlu/routing_graph_from_cpp.py`;
the width is intentionally thinner (the routing graph uses 4.0) so
the ring does not visually dominate the smaller overview circles.

Both variants also draw **dark-blue lines** (`#1f3a93`, same colour
as the routing graph) connecting the top `--top-k-pairs`
adjacent-layer `(L, e_i) -> (L+1, e_j)` co-activations, sorted by
raw count desc per layer pair. Line thickness is `0.1 + line_scale *
raw_count` (default `line_scale = 5e-7`, same default as the routing
graph). The aggregated pair counts are summed across datasets from
the per-dataset `aggregate.adjacent_pair_counts` block; if any
dataset lacks that block its contribution is skipped with a `[note]`
on stdout and the title reports how many datasets contributed. Set
`--top-k-pairs 0` to disable the lines.

A vertical colorbar on the right encodes the raw count scale, with
tick labels formatted in K/M (`37.2M`, `450.0K`, etc.) by the same
millions-formatter helper used by the routing graph.

## Top-K selection

Per cell, `top_k_count = ceil(n_expert × --top-k-fraction)`. Defaults:

| Model                                | n_expert | ceil(n_expert × 0.125) |
| ------------------------------------ | -------- | ---------------------- |
| unsloth/gpt-oss-120b                 | 128      | 16                     |
| LiteLLMs/Mixtral-8x22B-Instruct-v0.1 | 8        | 1                      |
| allenai/OLMoE-1B-7B-0125-Instruct    | 64       | 8                      |
| mradermacher/deepseek-moe-16b-chat   | 64       | 8                      |

Per layer, experts are sorted by `counts_total_overview[layer]` descending
and the top-K indices + statistics are persisted.

## Usage

```sh
python examples/eval-moe-overview/aggregate_overview.py --results-dir build/results
```

Flags:

| Flag                    | Meaning                                                             | Default         |
| ----------------------- | ------------------------------------------------------------------- | --------------- |
| `--results-dir`         | Root directory containing per-model subdirectories                  | `build/results` |
| `--top-k-fraction`      | Fraction of experts per layer to highlight / report as top-K        | `0.125`         |
| `--top-k-pairs`         | Top-K adjacent-layer co-activation pairs to draw on both heatmaps   | `64`            |
| `--line-scale`          | `line_width = 0.1 + line_scale * raw_count` for co-activation lines | `5e-7`          |
| `--include-per-dataset` | Add `per_dataset_layer_topk` to `top_experts.json`                  | off             |
| `--models`              | Restrict to a subset of model directory names (repeatable)          | (all)           |
| `--quants`              | Restrict to a subset of quantization directory names (repeatable)   | (all)           |
| `--dpi`                 | Output PNG DPI                                                      | `120`           |

The `--colormap` flag was removed when the overview heatmap was switched
to the Blues circle-grid style (matching the routing graph). The new
style is hardcoded so the visual contract is identical across models.

## Known data quirks

- `unsloth--gpt-oss-120b-GGUF/Q4_K_M/moe-mmlu/expert_counts.json` reports
  `model_arch.name = "olmoe"` even though the model is gpt-oss (L/E/k
  values are correct: 36/128/4). This script reads L/E/k directly from
  the JSON, so the misnamed field does not affect the output.
- `LiteLLMs--Mixtral-8x22B-Instruct-v0.1-GGUF/Q4_K_M/moe-mmlu` is
  incomplete (only `run.log` present). The script logs a skipped warning
  and aggregates the remaining datasets (humaneval, include, bigbench,
  popqa).
- `unsloth--gpt-oss-120b-GGUF/Q4_K_M` has no `expert_counts.json` files
  at all (only `run.log` in `moe-mmlu/`). The model is silently skipped
  during discovery.
- The script tolerates `expert_counts.json` files with zero tasks (e.g.
  failed runs) by logging and continuing.

---

# llama-eval-moe-overview (cross-quantization alignment)

Companion to `aggregate_overview.py` (which is cross-dataset within a
single `(model, quant)` cell). This tool is **cross-quantization** for a
single model: it reads every
`build/results/<model_safe>/<quant>/overall/counts_total_overview.json`
under one model directory — the per‑quant aggregated‑across‑datasets
`[L, E]` count matrices already produced by `aggregate_overview.py` —
and renders two views of cross-quantization alignment on the **top-K
expert sets**:

1. A pairwise heatmap (quant × quant) at K = n_expert_used, averaged
   across layers — the visual.
2. A single-number-per-K view: x = K, y = cross-quantization Jaccard.
   Two complementary metrics overlaid (generalized |∩|/|∪| + mean
   pairwise).

The filename contract (`summary.csv`, `jaccard_pairwise.csv`,
`jaccard_global.csv`, `jaccard_pairwise.png`,
`jaccard_pairwise_K{K}.png`, `jaccard_pairwise_K2x.png`,
`jaccard_pairwise_K3x.png`, `jaccard_global.png`, `README.json`) is
**bit-identical** to the cross-dataset version's. The only differences
are the CSV header column names (`quant` instead of `dataset`,
`tokens_total` instead of `tokens`, `n_datasets` instead of `n_rows` in
`summary.csv`) and the output directory (one per model, instead of one
per `(model, quant)` cell). Downstream consumers can treat the two
formats identically — directory name is the only discriminator.

## Run order

`aggregate_overview.py` first (produces `<quant>/overall/`); then this
script. The cross-quant tool reads `counts_total_overview.json`
directly, so no re-aggregation from per-dataset JSONs is needed.

## What it produces (per model)

For each `<model_safe>/` the script writes
`<model_safe>/cross_quant_jaccard_sweep/`:

| File                        | Description                                                                                                                                                                |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `summary.csv`               | Per quant: `quant,tokens_total,n_datasets,mean_jaccard_to_others_at_n_used,min_jaccard_to_others_at_n_used` (quant alignment averaged across layers at K = n_expert_used). |
| `jaccard_pairwise.csv`      | Long-form: `K,layer,quant_a,quant_b,jaccard` (every pair of quants; one row per (K, layer)).                                                                               |
| `jaccard_pairwise.png`      | Heatmap of pairwise Jaccard at K = n_expert_used, averaged across layers; primary visual.                                                                                  |
| `jaccard_pairwise_K{K}.png` | Additional heatmap at K = 2 _ n_expert_used and K = 3 _ n_expert_used (one file per K, with K embedded in the filename).                                                   |
| `jaccard_pairwise_K2x.png`  | Convenience alias for the 2× K_model heatmap (always present when E ≥ 2 \* K_model).                                                                                       |
| `jaccard_pairwise_K3x.png`  | Convenience alias for the 3× K_model heatmap (always present when E ≥ 3 \* K_model).                                                                                       |
| `jaccard_global.csv`        | Long-form: `K,layer,jaccard_union,jaccard_pairwise_mean` (single-number-per-(K, layer) view of cross-quant alignment).                                                     |
| `jaccard_global.png`        | x = K, y = cross-quant Jaccard; thin per-layer lines + bold mean across layers; both union and pairwise-mean metrics overlaid for comparison.                              |
| `README.json`               | Model id, arch, list of quants used, K values, totals, cross-quant alignment summary at K = n_expert_used.                                                                 |

## Mathematical contract

For each layer `l` and K value, using the per-quant aggregated-across-
datasets counts matrix from `counts_total_overview.json::counts_total`
(shape `[L, E]`):

```
A_q_K[l]  = top-K experts in quant q, layer l
            (descending count, ties -> ascending expert id)
|∩|_K[l]  = |∩_q A_q_K[l]|              (intersection across quants)
|∪|_K[l]  = |∪_q A_q_K[l]|              (union across quants)
generalized Jaccard (strict consensus): |∩|_K[l] / |∪|_K[l]
mean pairwise Jaccard:                  mean over all C(N, 2) pairs of
                                        |A_i ∩ A_j| / |A_i ∪ A_j|
```

Edge cases (from `_jaccard` and `_generalized_jaccard`, shared with the
cross-dataset version):

- All sets empty → 1.0 (trivially agree)
- Only one set non-empty → 0.0 (no comparison possible)

## Usage

```sh
# Single model:
python examples/eval-moe-overview/cross_quant_jaccard_sweep_from_cpp.py \
    --model-dir build/results/unsloth--gpt-oss-120b-GGUF

# Multi-model batch:
python examples/eval-moe-overview/cross_quant_jaccard_sweep_from_cpp.py \
    --results-dir build/results \
    --models unsloth/gpt-oss-120b-GGUF LiteLLMs/Mixtral-8x22B-Instruct-v0.1-GGUF
```

Flags:

| Flag            | Meaning                                                                                                 | Default                   |
| --------------- | ------------------------------------------------------------------------------------------------------- | ------------------------- |
| `--model-dir`   | Process a single model directory (mutually exclusive with `--results-dir`).                             | (none)                    |
| `--results-dir` | Process multiple model directories under the given root.                                                | (none)                    |
| `--models`      | Restrict to a subset of HF model ids (repeatable; required with `--results-dir`).                       | (none)                    |
| `--quants`      | Restrict to a subset of quantization subdirectories (repeatable).                                       | (all)                     |
| `--model-id`    | Override the model id recorded in `README.json` (single-mode only).                                     | inferred from `model_dir` |
| `--output-dir`  | Override the output directory (default: `<model-dir>/cross_quant_jaccard_sweep/`).                      | (default)                 |
| `--k-min`       | Minimum K in the sweep                                                                                  | `1`                       |
| `--k-max-frac`  | `K_max = ceil(n_expert * k_max_frac)` (chosen so 2x/3x K_model land inside the sweep on common models). | `0.50`                    |
| `--num-ks`      | Number of K values in the linear sweep                                                                  | `21`                      |
| `--dpi`         | Output PNG DPI                                                                                          | `120`                     |
| `--force`       | Re-run even if `cross_quant_jaccard_sweep/summary.csv` already exists.                                  | off                       |

## Heatmap visual style

Identical to the cross-dataset version: Blues colormap, `[black, white]`
text-threshold rule at v = 0.55, aspect = auto, x/y tick labels rotated
30° for legibility. For the cross-quant case the matrix is usually
very small (2×2 to 5×5 for typical model + quant runs), so the
"averaged across L layers" title and the colorbar are the most useful
visual cues.

## Known data quirks

- **Run order matters.** This script reads `counts_total_overview.json`,
  which `aggregate_overview.py` writes. Running cross-quant before
  aggregate will yield a one-line `[error] no <quant>/overall/...` per
  model.
- **Single-quant degeneracy.** A model with only one available quant
  still writes the artifacts (a 1×1 heatmap at 1.0 and trivially-1.0
  curves); the comparison is degenerate by construction.
- **Arch mismatch across quants.** If two quants of the same model
  disagree on `n_layer`, `n_expert`, or `n_expert_used` (e.g. one
  aggregate run was performed on an old C++ binary), the script
  SystemExits with one line `[error] ... arch mismatch`. Re-run
  `aggregate_overview.py --force` on the offending quant to recover.
- **Idempotency.** A model whose `cross_quant_jaccard_sweep/summary.csv`
  already exists is skipped (one-line `[skip] ... summary.csv exists;
pass --force to regenerate`). Pass `--force` to re-render.
