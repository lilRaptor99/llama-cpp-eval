# Spartan HPC wrapper for `scripts/run-evaluator.sh`

Three artefacts that wrap the existing 5-MoE-routing eval pipeline so it
runs end-to-end on the Unimelb Spartan HPC under SLURM:

| File                                             | Where it runs         | Purpose                                                             |
| ------------------------------------------------ | --------------------- | ------------------------------------------------------------------- |
| [`download-models.sh`](download-models.sh)       | Login node            | Pre-populate the HF cache with the GGUF files the eval will consume |
| [`llama-moe-eval.sbatch`](llama-moe-eval.sbatch) | GPU node (`gpu-a100`) | Build (CUDA) + run the 4 × 5 × N eval sweep                         |
| [`README.md`](README.md)                         | —                     | This file                                                           |

The `scripts/run-evaluator.sh` Bash script is unchanged at its core; the
canvas extends to `--cuda` and `--quant` flags plus a new results tree
(`results/<model>/<quant>/<dataset>/`), but the per-cell logic is the
same.

---

## Prerequisites

| What                                                                                                   | Why                                                 | Where to set it up                    |
| ------------------------------------------------------------------------------------------------------ | --------------------------------------------------- | ------------------------------------- |
| Spartan account + `gpu-a100` group membership                                                          | Run on A100 nodes                                   | Spartan onboarding                    |
| `eval-moe` branch of [`lilRaptor99/llama-cpp-eval`](https://github.com/lilRaptor99/llama-cpp-eval.git) | The code under test                                 | `git clone` to `$HOME/llama-cpp-eval` |
| `$SCRATCH` set                                                                                         | Per-project Lustre scratch volume for build + cache | Default on Spartan                    |
| `huggingface_hub` + `numpy` + `matplotlib` + `datasets`                                                | Heatmap step + login-node downloads                 | `pip install --user …` on login node  |
| CUDA module on the GPU node                                                                            | Compiles the C++ binaries                           | `module avail cuda` on `gpu-a100`     |

The default `DEFAULT_MODELS` list in `download-models.sh` only contains
public HF repos, so `HF_TOKEN` is not required. If you add a gated repo
later, export `HF_TOKEN` before invoking either script.

---

## Workflow

### 1. Login node — pre-download the GGUF files

```bash
# Default: Q4_K_M for all 4 hardcoded models.
bash spartan/download-models.sh

# Multiple quants per model (will multiply disk usage by ~quants count):
bash spartan/download-models.sh --quant Q4_K_M Q8_0

# Enumerate quants for one repo without downloading:
bash spartan/download-models.sh --list-quants unsloth/gpt-oss-120b-GGUF

# Quiet mode (skip the disk-usage confirmation):
bash spartan/download-models.sh --quant Q4_K_M --yes
```

The script prints an estimate and asks for confirmation before pulling
anything heavier than a few GB. Worst-case (all 4 repos, all quant
variants) the cache can exceed 500 GB.

The cache layout used by `download-models.sh` and the C++ binary must
match — both write to
`${HF_CACHE}/hub/models--<org>--<name>/snapshots/<rev>/...`
(the standard HF hub layout; see `common/hf-cache.cpp` in the repo).

### 2. Submit the GPU job

```bash
# Default: 4 A100 GPUs, 24 CPUs, 128 GB RAM, 4-day wall clock.
sbatch spartan/llama-moe-eval.sbatch
```

The `.sbatch` builds the 5 `llama-eval-moe-*` binaries with
`DGGML_CUDA=ON` (or reuses them if already present) and then runs the
`model × quant × dataset` sweep. The summary table ends up printed
twice — once to the per-cell `run.log` (in
`${RESULTS_DIR}/<model>/<quant>/moe-<ds>/run.log`) and once to the
overall job log.

### 3. Monitor

```bash
# Job status
squeue -j $JOB_ID

# Top-level job log
tail -f spartan/logs/llama-moe-eval-$JOB_ID.out

# Per-cell run log
tail -f $SCRATCH/llama-cpp-eval/results/<model_safe>/<quant_safe>/moe-<ds>/run.log
```

---

## Smoke testing

The full sweep (4 models × 5 datasets × 1 default quant × ~100 questions
per dataset) takes hours to days. To verify the pipeline on a small
subset, temporarily edit the `#SBATCH` directives in
`llama-moe-eval.sbatch` to use the `gpu-a100-short` partition:

```bash
# In slurm/llama-moe-eval.sbatch (or a copy):
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --mem=32G
```

Then submit a single-cell smoke test:

```bash
sbatch \
  --export=ALL,MODELS_OVERRIDE=allenai/OLMoE-1B-7B-0125-Instruct-GGUF,QUANTS_OVERRIDE=Q4_K_M,DATASETS_OVERRIDE=mmlu \
  spartan/llama-moe-eval.sbatch
```

You should see:

- `ggml_cuda_init: found 1 CUDA device` in the job log (CUDA wired up).
- `models--allenai--OLMoE-1B-7B-0125-Instruct-GGUF/snapshots/<rev>/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf` reported
  by the C++ binary (pre-download was used).
- `expert_counts.json` at
  `${SCRATCH_BASE}/results/allenai--OLMoE-1B-7B-0125-Instruct-GGUF/Q4_K_M/moe-mmlu/expert_counts.json`.
- `routing_heatmap.png` and `accuracy_by_langdom.png` next to the JSON.

---

## Env-var overrides

All variables are optional. Set them before `sbatch` (or pass via
`--export=...`).

| Variable             | Default                                     | Notes                                                                   |
| -------------------- | ------------------------------------------- | ----------------------------------------------------------------------- |
| `REPO_ROOT`          | `$HOME/llama-cpp-eval`                      | Where the `eval-moe` branch is cloned                                   |
| `SCRATCH_BASE`       | `${SCRATCH:-/scratch/$USER}/llama-cpp-eval` | Root for build / datasets / results / hf_cache                          |
| `CUDA_MODULE`        | `cuda/12.6.0`                               | Lmod module name                                                        |
| `PYTHON_MODULE`      | `python/3.11`                               | Lmod module name                                                        |
| `GCC_MODULE`         | `gcc/12.2.0`                                | Lmod module name                                                        |
| `CUDA_ARCHITECTURES` | `80`                                        | A100 = SM_80 (single arch keeps compile time sane)                      |
| `EXTRA_CMAKE_FLAGS`  | `<unset>`                                   | Spaces-separated extras forwarded to cmake configure                    |
| `MODELS_OVERRIDE`    | `<all 4 defaults>`                          | Space-separated list of HF repo ids                                     |
| `DATASETS_OVERRIDE`  | `<all 5 defaults>`                          | Space-separated subset of `{mmlu, popqa, bigbench, humaneval, include}` |
| `QUANTS_OVERRIDE`    | `Q4_K_M`                                    | Space-separated list of quant tags                                      |
| `HF_TOKEN`           | `<unset>`                                   | Optional; for gated repos                                               |

---

## Results tree

Every cell under the sweep writes to:

```
${RESULTS_DIR}/<model_safe>/<quant_safe>/moe-<dataset>/
├── expert_counts.json   # the JSON the C++ binary writes
├── run.log              # stdout+stderr of the C++ binary
├── plot.log             # stdout+stderr of heatmap_from_cpp.py
├── routing_heatmap.png  # generated by heatmap_from_cpp.py
├── routing_heatmap_by_language.png
├── routing_heatmap_by_domain.png
├── routing_heatmap_by_langdom.png
├── routing_heatmap_by_langdom_normalized.png
├── accuracy_by_langdom.png
├── metadata.json
└── counts_total.json
```

`<model_safe>` is the HF repo id with `/` replaced by `--`
(e.g. `unsloth/gpt-oss-120b-GGUF` → `unsloth--gpt-oss-120b-GGUF`).
`<quant_safe>` is the quant tag (e.g. `Q4_K_M`); see `quant_safe()`
in `scripts/run-evaluator.sh` for the sanitisation rule.

### Migrating old results

Pre-`--quant` runs sit at `${RESULTS_DIR}/<model>/moe-<ds>/`. To bring
them under the new tree:

```bash
cd "${SCRATCH_BASE}/results/allenai--OLMoE-1B-7B-0125-Instruct-GGUF"
mkdir -p Q4_K_M
mv moe-* Q4_K_M/
```

---

## Resuming / wiping

The eval phase is naturally idempotent: `(model, quant, dataset)`
cells whose `expert_counts.json` already exists are skipped with
status `SKIPPED`. To resume an interrupted job, just re-submit:

```bash
sbatch spartan/llama-moe-eval.sbatch
```

To start fresh for a specific cell, delete its JSON:

```bash
rm -rf "${RESULTS_DIR}/<model>/<quant>/moe-<ds>/expert_counts.json"
```

To wipe everything:

```bash
rm -rf "${SCRATCH_BASE}/"{build,datasets,results,hf_cache}
```

To wipe just the HF cache (forces re-download but keeps results):

```bash
rm -rf "${SCRATCH_BASE}/hf_cache"
```

---

## Troubleshooting

| Symptom                                 | Likely cause                                    | Fix                                                                                                     |
| --------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `module: command not found`             | Lmod not in `.bashrc` on the GPU node           | `source /usr/local/lmod/lmod/init/bash` before `module` calls, or trust the .sbatch's `module purge`    |
| `nvcc: command not found`               | Wrong CUDA module                               | `module avail cuda` on `gpu-a100` and set `CUDA_MODULE`                                                 |
| `ggml_cuda_init: no CUDA devices found` | `CUDA_VISIBLE_DEVICES` empty or wrong GPU count | Check `squeue -j $JOBID -o "% Gres"`; request `--gres=gpu:N` to match                                   |
| C++ binary picks the wrong GGUF         | Quant tag didn't match anything in the repo     | Run `bash spartan/download-models.sh --list-quants <repo>` to see the available tags                    |
| `dataset/moe-*.jsonl` not found         | Datasets weren't downloaded                     | The script auto-downloads on first run; if it fails, set `--datasets-dir` to a writable path and re-run |
| OOM / `Killed` in job log               | 120B model + activations exceed `--mem`         | Raise `--mem` (A100 node has 495 GB total) or use a smaller quant                                       |

---

## File map

```
spartan/
├── README.md                 # this file
├── download-models.sh        # login-node pre-download
└── llama-moe-eval.sbatch     # SLURM job (gpu-a100, 4 GPU, 4 days)
```

Related (not in this directory):

- [`scripts/run-evaluator.sh`](../scripts/run-evaluator.sh) — the
  orchestration script that the .sbatch invokes. Holds `DEFAULT_MODELS`
    - the default `QUANTS=("Q4_K_M")`; `download-models.sh` mirrors these
      intentionally so the two stay in sync.
- [`examples/eval-moe-*/`](../examples/eval-moe-include/) — the 5 eval
  C++ binaries and their `download_*.py` + `heatmap_from_cpp.py`
  siblings.
- [`common/hf-cache.cpp`](../common/hf-cache.cpp) — defines the cache
  resolution order (`HF_HOME` → `HUGGINGFACE_HUB_CACHE` → …) that both
  scripts depend on.
