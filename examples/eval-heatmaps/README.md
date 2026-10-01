# eval-heatmaps (publication-quality heatmaps)

Companion to [`eval-moe-overview/`](../eval-moe-overview/) that produces
**paper-ready** heatmaps for the MoE-routing evaluation suite. The
heatmaps in the overview module use 120 DPI, multi-line titles, and
30° rotated labels — those are fine for screen inspection but they
look cramped in a printed paper. This module renders at **300 DPI** with
**no titles, no axis headings, and only the benchmark / quantization
labels** that the paper actually needs.

The script is `paper_heatmaps.py`. It generates three views at three
K-snapshots each for the Jaccard view, plus one JS-divergence heatmap
per (cell, view) and one optional routing strip per cell:

| View                       | Description                                                                                          | Output layout                                                              |
| -------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Cross-dataset Jaccard**  | Pairwise top-K Jaccard across the five benchmarks (BBH / HumanEval / INCLUDE / MMLU / PopQA), plus a sixth "Random" baseline when per-benchmark `random/` JSONs are present. | One subdirectory per model, three PNGs per `(model, quant)` at K_model / 2x / 3x. |
| **Cross-dataset JS**       | Layer-averaged pairwise Jensen-Shannon divergence across the same datasets (incl. Random if available). | One PNG per `(model, quant)` cell.                                          |
| **Cross-quant Jaccard**    | Pairwise top-K Jaccard across quantizations for a single model.                                      | Three PNGs per model at K_model / 2x / 3x.                                 |
| **Cross-quant JS**         | Layer-averaged pairwise Jensen-Shannon divergence across quantizations for a single model.          | One PNG per model.                                                          |
| **Routing strip** (optional) | Compact `[L, E]` heatmap of the aggregated expert counts (raw, no log) per cell, with layer & expert indices on the axes. | One PNG per `(model, quant)` cell. |

Every PNG is named so the file itself carries the model and quant:

```text
<model_safe>__<quant_safe>__jaccard_dataset.png          # cross-dataset Jaccard at K = K_model
<model_safe>__<quant_safe>__jaccard_dataset_K2x.png      # cross-dataset Jaccard at K = 2 * K_model
<model_safe>__<quant_safe>__jaccard_dataset_K3x.png      # cross-dataset Jaccard at K = 3 * K_model
<model_safe>__<quant_safe>__js_divergence_dataset.png    # cross-dataset JS divergence (layer-averaged)
<model_safe>__<quant_safe>__routing_strip.png            # routing strip (--per-cell)

<model_safe>__jaccard_quant.png                          # cross-quant Jaccard at K = K_model
<model_safe>__jaccard_quant_K2x.png                      # cross-quant Jaccard at K = 2 * K_model
<model_safe>__jaccard_quant_K3x.png                      # cross-quant Jaccard at K = 3 * K_model
<model_safe>__js_divergence_quant.png                    # cross-quant JS divergence (layer-averaged)
```

`<model_safe>` is the HF model id with `/` replaced by `--`, the same
mapping `scripts/run-evaluator.sh::model_safe` uses. `<quant_safe>` is
the quantization tag (e.g. `Q4_K_M`, `BF16`, `FP8`) verbatim — already
in publishable form.

## What it produces

### Cross-dataset heatmap

For each `(model, quant)` cell under `<results-dir>/<model>/<quant>/`:

- Reads each `moe-<dataset>/expert_counts.json` (`aggregate.marginal_expert_counts`).
- Computes the top-K expert set at every layer for every dataset.
- Builds a `[N, N]` pairwise Jaccard matrix averaged across all `L` layers.
- Saves one PNG per K snapshot (`K_model`, `2 * K_model`, `3 * K_model`,
  clamped to `[1, E]`).
- All five benchmarks are labeled in their publishable form: **BBH,
  HumanEval, INCLUDE, MMLU, PopQA** (the on-disk directory name is
  `moe-<dataset>`; the alias lives in `_DATASET_PUBLISH_LABELS` at
  the top of `paper_heatmaps.py`).
- **Optional 6th dataset: `Random`.** When one or more
  `<moe-<ds>/random/expert_counts.json` files exist in the cell, the
  script auto-aggregates them into a synthetic top-level "Random"
  dataset (sum of `aggregate.marginal_expert_counts` across the
  per-benchmark random JSONs, identical to the convention
  `aggregate_overview.py` uses for the per-quant aggregated reference)
  and adds it as the 6th row/column of every cross-dataset heatmap.
  This lets the paper compare real-routing vs random-routing JSD /
  Jaccard directly, alongside the real benchmarks.

### Cross-quantization heatmap

For each model directory with **two or more** quants (only those that
have already been processed by
`aggregate_overview.py`, which writes the
`<quant>/overall/counts_total_overview.json` files this script reads):

- Computes the pairwise top-K Jaccard matrix averaged across layers.
- Saves one PNG per K snapshot.
- Quant labels are sorted in the conventional paper order:
  full-precision first (`F32`, `F16`, `BF16`, `FP8`), then descending
  by quantization aggressiveness (`Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`,
  `Q3_K_M`, `Q2_K`). Add an entry to `_QUANT_PUBLISH_LABELS` to force
  a label override.

### Routing strip (optional, `--per-cell`)

A compact `[L, E]` heatmap of the aggregated expert counts **on a
linear raw-count scale** (no log1p), suitable for showing **where**
experts are hot in the paper text. The strip carries explicit axis
labels (`Layer index` on the y-axis, `Expert index` on the x-axis,
`Activation count` on the colorbar) so the bare-integer ticks
(`0, 8, 16, ...`) are unambiguous — this is the only view in
`paper_heatmaps.py` that retains axis labels because the integers
otherwise have no obvious referent. Tick labels are 7pt for
readability; the colorbar uses a millions-formatter (`12.3M` /
`456.7K` / `42`) so the scale matches the routing-graph heatmaps
in `eval-moe-overview/`.

Layer height is adaptive: capped at `2.4 × width_in` inches so even
the 48-layer Qwen3 stays single-column (the slight extra over `2.0`
accommodates the new axis labels without crowding the data).

Tick density is auto-selected via `_integer_axis_ticks(N,
max_ticks=...)`: a 16-layer model shows every 2nd layer
(`0,2,...,16`), a 28-layer model every 2nd, a 48-layer model every
4th, a 64-expert model every 4th expert, a 128-expert model every
16th. The 0 and the last index are always included so the axis range
stays clear. The `max_ticks` budget keeps 3-digit labels like
`120` legible at single-column width.

### Cross-dataset / cross-quant JS divergence

Layer-averaged pairwise Jensen-Shannon divergence on the same units as
the Jaccard views (cross-dataset: one per `(model, quant)` cell;
cross-quant: one per model).

**Fixed colorbar at `vmin=0, vmax=1`** so the reader can compare
JS-divergence values across figures against the canonical metric
range, not against whatever happens to sit in this particular cell.

**Colormap: `viridis`** (default, see `--jsd-cmap` below), oriented
with dark purple = low divergence (most similar), teal = mid
divergence, bright yellow = high divergence (most different). The
user explicitly requested this orientation so that **purple-teal is
the dominant palette** in the paper (matches their preferred "quiet"
look next to surrounding text), with bright yellow reserved for the
rare off-diagonal cells that hit the upper end of the [0, 1] range.
A side-benefit: the diagonal cells (`v = 0`, perfect agreement)
now render dark purple with WHITE text — high contrast and easy to
scan, unlike the previous `viridis_r` orientation where the yellow
diagonal had white-on-yellow legibility issues.

The earlier `Blues_r` default collapsed the small divergence values
(most cross-quant matrices live in `0..0.05`) onto a near-uniform
dark-blue gradient; `viridis` is perceptually uniform AND multi-hue
(dark purple → teal → bright yellow), so even at a fixed 0..1 range
the hue drifts visibly as values climb from 0.0 toward the mid-range.
Alternatives include `viridis_r` (reversed direction),
`magma_r`, `inferno_r`, `cividis` — all perceptually-uniform
scientific colormaps.

> **Note on cross-quant values.** With `vmax=1` fixed, a cross-quant
> matrix whose largest off-diagonal is `2.4e-4` (typical for
> bit-identical quants) only occupies 0.024 % of the colour range.
> Even `viridis` cannot resolve sub-percent differences visually;
> the colour cue is essentially "all dark purple" for the whole
> matrix in that regime. The cell text (`0.0002`) is the only way to
> read the exact value. If this is a recurring need, override the
> colour gradient per-cell (e.g., per-figure `--jsd-cmap`) or open
> an issue for an "auto-scale" mode (the earlier prototype used it).

## How it differs from the existing overview heatmaps

| Concern                       | `eval-moe-overview/aggregate_overview.py` / `jaccard_sweep_from_cpp.py` / `cross_quant_jaccard_sweep_from_cpp.py` | `eval-heatmaps/paper_heatmaps.py`                          |
| ----------------------------- | ---------------------------------------------------------- | --------------------------------------------------------- |
| Default DPI                   | 120                                                        | 300                                                       |
| Title                         | Multi-line ("X - pairwise top-K agreement...")         | None                                                      |
| Axis labels (`xlabel`/`ylabel`)| `"dataset"` / `"quant"`                                | None (no axis heading), **except for the routing strip** (`Layer index` / `Expert index`) — integers need a referent to be unambiguous |
| Heatmap dimensions            | Often `aspect="auto"` with colorbar stealing width     | Auto-sized to fit paper column (~3.3 in)                  |
| Filenames                     | `jaccard_pairwise.png`, `jaccard_pairwise_K2x.png`, …  | `<model>__<quant>__jaccard_dataset.png`, `…K2x.png`, …   |
| Tick rotation                 | 30° always                                             | 0° for 2-3 quants, 30° for ≥4 benchmarks                  |
| Goal                          | Fast screen inspection                                 | Drop-in `\includegraphics` for a double-column paper       |

The math is identical to `jaccard_sweep_from_cpp.py` /
`cross_quant_jaccard_sweep_from_cpp.py`: same `_top_k_set_for_layer`,
same `_jaccard`, same K-sweep grid (K_model, 2× K_model, 3× K_model
sentinels must be in the sweep). The output is just a stripped-down
PNG that prints well.

## Usage

```sh
# Full multi-mode (recommended):
python examples/eval-heatmaps/paper_heatmaps.py \
    --results-dir build/results \
    --per-cell

# Restrict to one model:
python examples/eval-heatmaps/paper_heatmaps.py \
    --results-dir build/results \
    --models allenai/OLMoE-1B-7B-0125-Instruct-GGUF \
    --per-cell

# Single (model, quant) cell (debug mode):
python examples/eval-heatmaps/paper_heatmaps.py \
    --cell-dir build/results/allenai--OLMoE-1B-7B-0125-Instruct-GGUF/Q4_K_M \
    --per-cell

# Single model for cross-quant only (debug):
python examples/eval-heatmaps/paper_heatmaps.py \
    --model-dir build/results/LiteLLMs--Mixtral-8x22B-Instruct-v0.1-GGUF
```

### Flags

| Flag                   | Meaning                                                                                                       | Default                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `--results-dir`        | Root directory containing per-model subdirectories (multi-mode).                                             | (none)                           |
| `--cell-dir`           | Single `(model, quant)` cell directory (debug). Mutually exclusive with `--results-dir` / `--model-dir`.      | (none)                           |
| `--model-dir`          | Single model directory for the cross-quant view (debug).                                                      | (none)                           |
| `--models`             | Restrict to a subset of HF model ids (repeatable; used with `--results-dir`).                                | (all)                            |
| `--quants`             | Restrict to a subset of quant subdirectories (repeatable).                                                   | (all)                            |
| `--model-id`           | Override the model id recorded in titles / filenames (single-cell modes).                                     | inferred from the cell path      |
| `--quant`              | Override the quant tag (single-cell modes).                                                                  | inferred from the cell path      |
| `--output-dir`         | Output directory.                                                                                            | `<results-dir>/paper_heatmaps`   |
| `--k-min`              | Minimum K in the sweep (debug / single-cell).                                                                | `1`                              |
| `--k-max-frac`         | `K_max = ceil(n_expert * k_max_frac)` (debug / single-cell).                                                  | `0.50`                           |
| `--num-ks`             | Number of K values in the linear sweep (debug / single-cell).                                                | `21`                             |
| `--dpi`                | PNG DPI for the heatmap.                                                                                     | `300`                            |
| `--cmap`               | Colormap for the Jaccard / routing-strip heatmaps (publication convention: `Blues`).                         | `Blues`                          |
| `--jsd-cmap`           | Colormap for the JS-divergence heatmaps (perceptually uniform, multi-hue).                                   | `viridis`                        |
| `--width-in`           | Figure width in inches (paper-column budget).                                                                | `3.3`                            |
| `--per-cell`           | Also emit one routing-strip heatmap per `(model, quant)` cell.                                               | off                              |
| `--skip-cross-quant`   | Skip the cross-quantization view.                                                                            | off                              |
| `--skip-cross-dataset` | Skip the cross-dataset view.                                                                                 | off                              |
| `--force`              | Overwrite existing PNGs.                                                                                     | off                              |

### Idempotency

A destination PNG that already exists is **skipped** (with a
`[skip] ...; pass --force` on stdout), so rerunning on a populated
results tree is cheap. Pass `--force` to regenerate.

## Run order

1. Run the per-dataset `llama-eval-moe-*` C++ binaries (this produces
   `moe-<dataset>/expert_counts.json` per cell).
2. Run `examples/eval-moe-overview/aggregate_overview.py` (this
   produces `<quant>/overall/counts_total_overview.json` per cell,
   which the cross-quant view needs).
3. Run `examples/eval-heatmaps/paper_heatmaps.py`.

If the cross-quant step fails with `[skip] <quant>/overall/counts_total_overview.json:
not found`, you forgot step 2.

## Cross-model comparison figure (for the paper)

The script `cross_model_comparison.py` is a sibling peer of
`paper_heatmaps.py` that generates **one 2×4 figure comparing four
models side-by-side** on the cross-dataset Jaccard and JS-divergence
heatmaps, all sharing the same global colorbar scales. This is the
single figure reviewers need to compare cross-dataset routing
similarity across models at a glance.

```sh
python examples/eval-heatmaps/cross_model_comparison.py \
    --results-dir /mnt/data/results-llama-cpp \
    --quant Q4_K_M
```

Layout:

```
    [OLMoE-1B ][Mixtral-8x22B ][DeepSeek-MoE-16B][Qwen3-30B-A3B ]   <- column titles
    [  Jaccard row: 5x5 (or 6x6 with Random) heatmap per model        ]   <- 1 shared colorbar
    [  JSD     row: 5x5 (or 6x6 with Random) heatmap per model        ]   <- 1 shared colorbar
```

Output (under `<output-dir>/`, default
`<results-dir>/paper_heatmaps/cross_model/`; the `__<quant>` suffix
in each filename embeds the `--quant` value so running the script
with multiple quantizations doesn't overwrite each other and a
reviewer opening a file knows immediately which quantization it is):

- `cross_model_grid__<quant>.png`    the 2×4 figure (1 row per metric,
                                       1 column per model)

The cross-quantization **summary table** is produced by the sibling
script `cross_quantization_table.py` (run separately) — see the
"Cross-quantization table" section below.

The figure uses:
- **Jaccard colorbar: 0–1 (fixed)** — both metrics are bounded in [0,1]
  so the same scale applies to all 4 models.
- **JSD colorbar: 0 to global max across ALL cells in the results
  tree** — not just the 4 models in this figure. This matches the
  scale `paper_heatmaps.py::main()` uses for its individual JSD
  heatmaps, so the per-cell PNGs and the cross-model grid are
  directly comparable on the same colorbar scale (the user explicitly
  asked for this — "so the JSD max is similar to the other individual
  JSD heatmaps").
- **`UniformRand` (uniform random router) baseline is included** as a
  6th dataset row/column when per-benchmark `random/` JSONs (or a
  pre-aggregated `<cell>/random/expert_counts.json`) are present —
  same convention as `paper_heatmaps.py`. The label is "UniformRand"
  (not just "Random") because the latter could be read by a casual
  reviewer as "arbitrary baseline"; "UniformRand" makes the routing
  scheme explicit. The UniformRand row appears at the END of each
  matrix so the comparison reads naturally: real benchmarks first,
  baseline at the bottom.
- **Short model names** (`OLMoE-1B`, `Mixtral-8x22B`, etc.) as column
  titles, suitable for a paper caption.
- **Dataset labels (BBH / HumanEval / INCLUDE / MMLU / PopQA /
  Random)** appear on the leftmost column and the bottom row only —
  the user requested this to save horizontal space; a full set of
  row labels goes in the caption, not the figure.
- **Row titles ("Jaccard" / "JSD") are omitted** — also from the
  caption to save space.
- **Tight gaps** (`wspace=0.08`, `hspace=0.30`) — the user flagged
  the original `wspace=0.25, hspace=0.55` as too generous and asked
  to tighten.

### Cross-quantization table (companion to the figure)

The figure shows routing similarity across datasets, **at one
quantization** (`--quant`). The companion table summarises routing
similarity across **all available quantizations** for each model.
Together they answer:

- *figure*: at Q4_K_M, how similar is BBH routing to MMLU routing?
- *table*: when we compare Q2_K vs Q4_K_M routing distributions,
  does the routing similarity between BBH and MMLU change much?

The table is produced by a **separate script**,
`examples/eval-heatmaps/cross_quantization_table.py`, so the figure
script stays focused on the heatmap (single responsibility, easier
to test in isolation). Run it separately:

```sh
python examples/eval-heatmaps/cross_quantization_table.py \
    --results-dir /mnt/data/results-llama-cpp
```

**Table caption** (also embedded as the header of the saved `.txt`
file so reviewers don't have to open this README to make sense of
the numbers):

> **Table N: Cross-quantisation stability of routing similarity.**
> For each (model, metric, dataset) cell, the value reports the
> mean ± std of the metric across all available quantisations of
> that model. The per-quant "metric value" is the mean of the
> metric's row in that quant's cross-dataset pairwise matrix,
> excluding the diagonal — i.e. how similar this dataset's routing
> distribution is to the average of all OTHER datasets in this
> quant. Rows are organised as (model, metric) pairs; columns are
> dataset labels. `n_quants` reports the number of quantisations that
> contributed (the `RANDOM` quant is skipped because its label set
> lacks the `UniformRand` row). Low `std` indicates that the
> cross-dataset routing similarity for this dataset is stable across
> quantisations; high `std` indicates that quantisation meaningfully
> affects the routing distribution.

**Reading the table**: a low `std` means routing for this dataset is
stable across quantizations (the same experts fire). A high `std`
means quantization meaningfully affects the routing distribution.
The script does NOT depend on `--quant` because it summarises across
**all** quants for each model — running with `--quant Q2_K` vs
`--quant Q6_K` produces the same table. The CSV/TXT filenames
therefore have no quant suffix.

The CLI for `cross_quantization_table.py`:

| Flag            | Meaning                                                        | Default                                                  |
| --------------- | -------------------------------------------------------------- | -------------------------------------------------------- |
| `--results-dir` | Root results directory (multi-cell results tree).              | `/mnt/data/results-llama-cpp`                            |
| `--models`      | Restrict to a subset of filesystem-safe model ids.             | (all 4 hardcoded defaults)                                |
| `--output-dir`  | Output directory.                                              | `<results-dir>/paper_heatmaps/cross_quantization`        |

To customize the model set, edit the `DEFAULT_MODELS` list at the top
of `cross_quantization_table.py` (filesystem-safe model id → short name
mapping).

The figure script (`cross_model_comparison.py`) and the table script
(`cross_quantization_table.py`) share the same `DEFAULT_MODELS` list
and the same `BENCHMARK_ORDER` — both are kept in sync by convention
(make the same edit in both files if you want to change the model set).

### Cross-quantization comparison figure (sibling peer)

The companion figure to `cross_model_comparison.py` answers the
**same question but along the quantization axis**: for each model,
how stable is the routing distribution across quantizations?
The script generates a single **2×N figure** where columns are
models and each subplot is that model's N_q×N_q cross-quantization
Jaccard (top row) or JSD (bottom row) matrix. Together with the
cross-model figure, reviewers can see at a glance whether
quantisation affects routing uniformly across models or only
some. The script is `examples/eval-heatmaps/cross_quantization_comparison.py`:

```sh
# Default run (4 hardcoded models, all available quants):
python examples/eval-heatmaps/cross_quantization_comparison.py

# Restrict to a subset of models:
python examples/eval-heatmaps/cross_quantization_comparison.py \
    --models allenai--OLMoE-1B-7B-0125-Instruct-GGUF \
    --models unsloth--Qwen3-30B-A3B-GGUF

# Restrict to a subset of quants (e.g. just Q2_K vs Q4_K_M):
python examples/eval-heatmaps/cross_quantization_comparison.py \
    --quants Q2_K --quants Q4_K_M
```

Layout:

```
    [OLMoE-1B ][Mixtral ][DeepSeek][Qwen3  ]   <- column titles (top)
    [ N_q x N_q Jaccard heatmap per model       ]   <- 1 shared colorbar (0..1)
    [ N_q x N_q JSD     heatmap per model       ]   <- 1 shared colorbar (0..auto vmax)
```

`N_q` is the number of valid quantisations discovered per model
(typically 2–4: `Q2_K`, `Q4_K_M`, optionally `Q8_0` / `F16` / `F32`,
plus the `UniformRand` baseline when `<model>/RANDOM/overall/...`
exists). Models with different `N_q` still appear in the same figure
— cell sizes vary modestly across models (small `N_q` → bigger
cells, large `N_q` → smaller cells), which is fine for a
cross-model comparison.

- **Square cells** (`aspect="equal"`), tight gaps (`wspace=0.02`,
  `hspace=0.08`) — same convention as `cross_model_comparison.py`.
- **Short model names** as column titles (`OLMoE-1B-7B`,
  `Mixtral-8x22B`, `DeepSeek-MoE-16B`, `Qwen3-30B-A3B`) — same
  short names as `cross_model_comparison.py`.
- **`UniformRand` (uniform random router) baseline** is included as
  one of the `N_q` quants when its `RANDOM/` directory is present.
- **Quant labels sorted by `QUANT_PRIORITY`** (most-precise first:
  `F32`, `F16`, `BF16`, `FP8`, then descending by aggressiveness
  `Q8_0 → Q2_K`).
- **Tick labels only on leftmost column and bottom row** to save
  horizontal space; full labels go in the caption.
- **Jaccard colorbar: 0–1 (fixed)** — bounded metric, so the same
  scale applies across all models.
- **JSD colorbar: 0 to global max across ALL models' cross-quant
  matrices** (matches the global pre-pass in `paper_heatmaps.py`),
  so reviewers can compare cells across the paper without
  re-scaling. Pass `--jsd-vmax 1.0` to fall back to the canonical
  fixed `[0, 1]` range.
- **No CSV outputs** — single-responsibility script: figure only.
  The companion table is `cross_quantization_table.py` (see
  "Cross-quantization table" section above); the per-model
  single-figure views are produced by `paper_heatmaps.py`
  (`<model>__<quant>__jaccard_quant.png` /
  `<model>__js_divergence_quant.png`).

Output (under `<output-dir>/`, default
`<results-dir>/paper_heatmaps/cross_quantization/`):

- `cross_quantization_grid.png`     — the 2×N figure

The CLI:

| Flag            | Meaning                                                       | Default                                                |
| --------------- | ------------------------------------------------------------- | ------------------------------------------------------ |
| `--results-dir` | Root results directory (multi-model batch mode).              | `/mnt/data/results-llama-cpp`                          |
| `--models`      | Restrict to a subset of filesystem-safe model ids (repeatable). | (the 4 hardcoded defaults in `DEFAULT_MODELS`)        |
| `--quants`      | Restrict to a subset of quant subdirectories (repeatable).    | (all quants with valid `counts_total_overview.json`)   |
| `--output-dir`  | Output directory.                                             | `<results-dir>/paper_heatmaps/cross_quantization`      |
| `--width-in`    | Figure width in inches.                                       | `11.0` (double-column paper)                           |
| `--height-in`   | Figure height in inches.                                      | `5.5` (2 rows × N model columns + colorbars)           |
| `--dpi`         | Output DPI.                                                   | `300`                                                  |
| `--jsd-vmax`    | JSD colorbar vmax (default: auto from all models' matrices).  | auto                                                   |

The figure script (`cross_model_comparison.py`) and this one share
the same `DEFAULT_MODELS` list — both are kept in sync by convention
(make the same edit in both files if you want to change the model
set).

## Customizing labels

`_DATASET_PUBLISH_LABELS` (top of `paper_heatmaps.py`) maps each
`moe-<dataset>` directory name to its publishable label. The default
mapping is:

| On-disk directory | Published label |
| ----------------- | --------------- |
| `moe-bigbench`    | BBH             |
| `moe-humaneval`   | HumanEval       |
| `moe-include`     | INCLUDE         |
| `moe-mmlu`        | MMLU            |
| `moe-popqa`       | PopQA           |

Anything not in the map passes through with the `moe-` prefix removed
and uppercase (`moe-arc` → `ARC`), so adding a new benchmark doesn't
break this script — it just doesn't get a custom alias until you add
one.

`_QUANT_PUBLISH_LABELS` works the same way for quant tags. Add an
entry to force a label override (e.g. some repos ship `Q4_K_M-f16`
on disk and you want `Q4_K_M` on the figure); leave it absent to
pass the directory name through verbatim.

## Why a separate script?

The existing overview scripts (`aggregate_overview.py`,
`jaccard_sweep_from_cpp.py`,
`cross_quant_jaccard_sweep_from_cpp.py`) are tuned for screen
inspection at 120 DPI with full titles and a generous colorbar —
they're great for quickly reading off values during development but
they don't drop cleanly into a `\includegraphics` slot. This script
is a sibling peer: same math, same K-sweep contract, same file-naming
contract for cross-quant/cross-dataset directories (one cell dir per
view, bit-identical `summary.csv` / `jaccard_pairwise.csv` /
`README.json` if you ever want them) but with a stripped-down PNG
that prints well.
