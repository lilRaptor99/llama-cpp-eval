# llama-eval-moe-mmlu

Evaluate an MoE LLM on a slice of MMLU and record, per subject, how often each
expert cell was activated at each MoE layer, plus how often each pair of
experts fires (i) together within the same layer and (ii) across adjacent
layers. Output is a single JSON file suitable for post-hoc analysis (e.g.
expert-usage heatmaps, integrated routing graphs).

The tool is designed around the `ggml_backend_sched_eval_callback` hook exposed
by llama.cpp: every MoE forward pass materialises an `int32` tensor named
`ffn_moe_topk-<il>` of shape `[n_expert_used, n_tokens]`. The callback filters
on that name, copies the data via `ggml_backend_tensor_get`, and tallies counts
into a per-subject `[n_layer][n_expert]` matrix. After every `llama_decode`,
the same raw top-k slice is walked once on the main thread to tally
intra-layer `[L, E, E]` and adjacent-layer `[L-1, E, E]` pair counts.

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

- `build/moe-mmlu/mmlu.jsonl` one record per question
- `build/moe-mmlu/subjects.txt` one subject per line (all 57 by default)

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

| Flag                        | Meaning                                | Default                             |
| --------------------------- | -------------------------------------- | ----------------------------------- |
| `--mmlu <path>`             | `mmlu.jsonl`                           | `build/moe-mmlu/mmlu.jsonl`         |
| `--subjects <path>`         | subjects list                          | `build/moe-mmlu/subjects.txt`       |
| `--questions-per-subject N` | questions per MMLU subject             | 50                                  |
| `--n-shots N`               | few-shot exemplars from each dev split | 5                                   |
| `-o, --output <path>`       | output JSON                            | `build/moe-mmlu/expert_counts.json` |

Quick smoke test (1 question/subject, 0-shot):

```sh
./build/bin/llama-eval-moe-mmlu \
    -m models/olmoe-1b-7b-0125-instruct-q4_k_m.gguf \
    -ngl 999 \
    --questions-per-subject 1 --n-shots 0
```

## Output schema (`expert_counts.json`)

The JSON has three top-level blocks: a per-subject detail block, an aggregate
block (sum across subjects), and the run metadata.

```jsonc
{
  "model":      "<hf model id or local GGUF name>",
  "model_arch": { "name": "olmoe", "n_layer": 16, "n_expert": 64, "n_expert_used": 8 },
  "config":     { "questions_per_subject": 50, "n_shot": 5,
                  "few_shot_pool": "cais/mmlu dev split", "prompt_format": "few_shot_chat" },
  "totals":     { "subjects_run": 57, "questions_total": 2850, "tokens_total": 1862535 },

  // Aggregate = sum of per-subject counts (see aggregation section below).
  "aggregate": {
    "marginal_expert_counts":  [[L, E] int64],   // sum of layer_expert_counts across subjects
    "intra_pair_counts":       [[L, E, E] int64],
    "adjacent_pair_counts":    [[L-1, E, E] int64]
  },

  // Per-subject detail (57 entries by default).
  "subjects": {
    "<subject>": {
      "questions":      50,
      "n_tokens":       19460,
      "layer_expert_counts":   [[L, E] int64],         // marginal firing count
      "intra_pair_counts":     [[L, E, E] int64],      // same-layer pair count
      "adjacent_pair_counts":  [[L-1, E, E] int64]     // (L, L+1) pair count
    }
  }
}
```

### Field semantics

| Field                               | Shape         | Counts one increment per                                                |
| ----------------------------------- | ------------- | ----------------------------------------------------------------------- |
| `layer_expert_counts[L][e]`         | `[L, E]`      | top-k slot in layer L where expert e fired                              |
| `intra_pair_counts[L][e_i][e_j]`    | `[L, E, E]`   | token where e_i and e_j both fired in layer L (any of k × k slot pairs) |
| `adjacent_pair_counts[L][e_i][e_j]` | `[L-1, E, E]` | token where e_i fired in L and e_j fired in L+1 at the same position    |

Identity invariants (verified by the visualisation):

- `layer_expert_counts[L][e] * k == sum_{e'} intra_pair_counts[L][e][e']` (each token's k top-k slots in L pair with k in L).
- `sum_{e_i, e_j} adjacent_pair_counts[L][e_i][e_j] == k * k * tokens_total` per layer pair.
- `aggregate.X == sum_{subjects} subjects[*].X` for every X.

## Visualize

Two Python scripts consume the JSON and emit PNGs (both consume the
`aggregate` block produced by the updated C++ binary):

```sh
# Four 2D heatmaps: layer×expert overview + per-subject + per-category,
# in both raw and row-normalised forms.
python examples/eval-moe-mmlu/heatmap_from_cpp.py -i build/moe-mmlu/expert_counts.json

# One integrated routing graph: experts as circles on a decoupled grid (rows
# spaced wider than columns), linear Blues fill for marginal activation,
# lines (thickness ∝ raw count) for the top-K adjacent-layer pairs per layer
# pair, and a thick red ring on the top 12.5% of experts per layer (selection
# rule controlled by --highlight-mode; default: marginal ∩ outgoing-pair-sum).
python examples/eval-moe-mmlu/routing_graph_from_cpp.py -i build/moe-mmlu/expert_counts.json
```

The first script produces:

- `routing_heatmap.png` layer × expert overview
- `routing_heatmap_by_subject.png` 57 subjects × 1024 cells (log1p)
- `routing_heatmap_by_category.png` 6 categories × 1024 cells (log1p)
- `routing_heatmap_by_category_normalized.png` same with rows normalised to 1
- `counts_total.json` aggregated `[L, E]` matrix
- `metadata.json` pass-through + computed totals

The second script produces:

- `routing_graph.png` integrated graph (overwrite with `-o`)

`routing_graph_from_cpp.py` accepts optional flags for the visual encoding:

| Flag                      | Default                         | Meaning                                                                                                                                                                                   |
| ------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-o, --output <path>`     | `<input_dir>/routing_graph.png` | output PNG path                                                                                                                                                                           |
| `--col-spacing <float>`   | `1.0`                           | horizontal distance between expert columns in data units (also controls circle radius)                                                                                                    |
| `--row-spacing <float>`   | `2.5`                           | vertical distance between layer rows in data units. Increase to make the cross-layer connection lines more visible.                                                                       |
| `--top-k <int>`           | `64`                            | top-K adjacent-layer pairs to draw per layer pair (capped at E²)                                                                                                                          |
| `--top-frac <float>`      | `0.125`                         | fraction of experts per layer to highlight with a red ring. Set to `0` to disable.                                                                                                        |
| `--highlight-mode <mode>` | `pair`                          | highlight selection rule: `pair` (default — intersection of top by marginal AND top by outgoing pair-sum), `marginal` (top by marginal only), `pair-sum` (top by outgoing pair-sum only). |
| `--line-scale <float>`    | `5e-7`                          | line width = `0.1 + line_scale * raw_count` (tune to your count magnitude; OLMoE-scale ≈ `1e-4`, Mixtral-scale ≈ `1e-5`)                                                                  |

The visual encoding is:

- **Circles**: one per `(layer, expert)` on a simple grid; fill colour encodes marginal firing count linearly (Blues: white = 0, dark blue = max). The colorbar is labelled "Token Processing Load" and shows values in millions (e.g. `40.0M`).
- **Highlight rings**: a thick red ring (`#ff1744`, linewidth 4.0) is drawn on experts selected by `--highlight-mode`:
    - `pair` (default): experts in BOTH the top-12.5%-by-marginal AND top-12.5%-by-outgoing-pair-sum for their layer — the experts that are both heavily fired AND heavily involved in cross-layer routing.
    - `marginal`: top-12.5%-by-marginal firing count only — the most-used experts per layer.
    - `pair-sum`: top-12.5%-by-outgoing-pair-sum only — the experts with the highest cross-layer routing footprint.
- **Lines**: connect adjacent-layer `(L, L+1)` expert pairs; only the top-K pairs by raw count per layer pair are drawn. Line thickness is proportional to the raw count (no per-layer normalisation), so absolute routing volume is comparable across the figure.
- **Row spacing** is deliberately larger than column spacing (default 2.5 vs 1.0) so the cross-layer connection lines have a clearly visible vertical distance to traverse.
- **Axis labels**: "Layer Index" (vertical) and "Expert Index (within same layer)" (horizontal).

It requires the `aggregate` block in the input JSON. Running it on a
pre-coactivation JSON exits with a clear error pointing to the C++ rebuild.
