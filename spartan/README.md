# Spartan HPC wrapper for `scripts/run-evaluator.sh`

Three artefacts that wrap the existing 5-MoE-routing eval pipeline so it
runs end-to-end on the Unimelb Spartan HPC under SLURM:

| File                                             | Where it runs         | Purpose                                                                                      |
| ------------------------------------------------ | --------------------- | -------------------------------------------------------------------------------------------- |
| [`download-models.sh`](download-models.sh)       | Login node            | Pre-populate the HF cache with the GGUF files the eval will consume                          |
| [`download-datasets.sh`](download-datasets.sh)   | Login node            | Pre-populate the dataset cache (mmlu/popqa/bigbench/humaneval/include) the eval will consume |
| [`llama-moe-eval.sbatch`](llama-moe-eval.sbatch) | GPU node (`gpu-a100`) | Build (CUDA) + run the 4 × 5 × N eval sweep                                                  |
| [`README.md`](README.md)                         | —                     | This file                                                                                    |

The `scripts/run-evaluator.sh` Bash script is unchanged at its core; the
canvas extends to `--cuda` and `--quant` flags plus a new results tree
(`results/<model>/<quant>/<dataset>/`), but the per-cell logic is the
same.

---

## Prerequisites

| What                                                                                                                                       | Why                              | Where to set it up                                                                                                                                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Spartan account + `gpu-a100` group membership                                                                                              | Run on A100 nodes                | Spartan onboarding                                                                                                                                                                                                                                                                                                               |
| `eval-moe` branch of [`lilRaptor99/llama-cpp-eval`](https://github.com/lilRaptor99/llama-cpp-eval.git)                                     | The code under test              | `git clone` to `/data/gpfs/projects/uom00014/llama-cpp-eval`                                                                                                                                                                                                                                                                     |
| `/data/gpfs/projects/uom00014/llama-cpp-eval` (REPO_ROOT) and `/data/scratch/projects/uom00014/llama-cpp-eval` (SCRATCH_BASE) writable     | Build + cache + results storage  | Project quota path on Spartan                                                                                                                                                                                                                                                                                                    |
| `CUDA/12.4.1` + `NCCL/2.22.3-CUDA-12.4.1` + `Python/3.11.3` + `GCCcore/11.3.0` + `CMake/3.31.3` Lmod modules                               | Compiles + runs the C++ binaries | Already pre-selected as defaults; override via `*_MODULE` env vars if your partition shows different versions. Note that `GCCcore/11.3.0` (the older compiler family) is required, not the newer `GCC/13.3.0`. NCCL must be built against the same CUDA major.minor (12.4.x) since that's the only NCCL available on `gpu-a100`. |
| `huggingface_hub` + `datasets` (the heatmap + routing-graph steps' `numpy` + `matplotlib` come from `SciPy-bundle` + `matplotlib` modules) | Login-node downloads + plotting  | `pip install --user huggingface_hub datasets` on the login node. The `datasets` package is only needed on the login node (by `download-datasets.sh`); do NOT install it on the GPU node — GPU nodes are firewalled off from pypi anyway. The heatmap + routing-graph steps only need `huggingface_hub` + `numpy` + `matplotlib`. |
| `huggingface_hub` + `datasets` on the login node; `numpy` + `matplotlib` on the GPU node (auto-installed by the .sbatch)                   | Pre-downloads + plotting         | `pip install --user huggingface_hub datasets` on the login node. The `datasets` package is only needed on the login node (by `download-datasets.sh`); the .sbatch installs `numpy` + `matplotlib` to ~/.local automatically before invoking the heatmap + routing-graph steps.                                                   |

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

### 1b. Login node — pre-download the eval datasets

> **This step is required.** The Spartan GPU compute nodes are firewalled
> off from the public internet AND don't ship the `datasets` Python
> package. Skipping this step will cause the `prepare_dataset()` phase
> of the GPU job to fail with `error: the 'datasets' package is required`.

```bash
# Default: all 5 datasets (mmlu, popqa, bigbench, humaneval, include).
# Total ~100 MB.
bash spartan/download-datasets.sh

# Just one (e.g. for a smoke test):
bash spartan/download-datasets.sh --datasets mmlu

# Show what each dataset is without downloading:
bash spartan/download-datasets.sh --list
```

The script will `pip install --user datasets` on the login node if it's
missing, then write each dataset to
`${SCRATCH_BASE}/datasets/moe-<ds>/<ds>.jsonl` (plus a sidecar like
`subjects.txt` / `tasks.txt` / `languages_domains.txt`). It's
idempotent — re-runs skip anything that's already cached.

### 2. Submit the GPU job

```bash
# Default: 4 A100 GPUs, 24 CPUs, 128 GB RAM, 4-day wall clock.
# Default modules: CUDA/12.4.1, NCCL/2.22.3-CUDA-12.4.1, Python/3.11.3,
#                  GCCcore/11.3.0, CMake/3.31.3
sbatch spartan/llama-moe-eval.sbatch

# Override modules if `module avail` shows different versions on your
# partition:
CUDA_MODULE=CUDA/12.4.1 NCCL_MODULE=NCCL/2.22.3-CUDA-12.4.1 \
GCC_MODULE=GCCcore/11.3.0 \
    sbatch spartan/llama-moe-eval.sbatch
```

If the heatmap or routing-graph step fails on the GPU node with
`ModuleNotFoundError: No
The .sbatch automatically runs `pip install --user numpy matplotlib`at
job start if either is missing, so neither step should ever fail
with a `ModuleNotFoundError`. Neither step needs
`datasets`; that's only consumed by the login-node `download-datasets.sh`script. Do NOT install`datasets` on the GPU node — it would fail
because GPU nodes can't reach pypi.

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
- `routing_heatmap.png`, `routing_graph.png`, and `accuracy_by_langdom.png` next to the JSON.

---

## Env-var overrides

All variables are optional. Set them before `sbatch` (or pass via
`--export=...`).

| Variable                   | Default                                                                                                | Notes                                                                                                                                                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `REPO_ROOT`                | `/data/gpfs/projects/uom00014/llama-cpp-eval`                                                          | Where the `eval-moe` branch is cloned                                                                                                                                                                              |
| `SCRATCH_BASE`             | `/data/scratch/projects/uom00014/llama-cpp-eval` (or `${SCRATCH}/llama-cpp-eval` if `$SCRATCH` is set) | Root for build / datasets / results / hf_cache                                                                                                                                                                     |
| `CUDA_MODULE`              | `CUDA/12.4.1`                                                                                          | Lmod module name; canonical gpu-a100 default (from Core). Must match the CUDA version `NCCL_MODULE` was built against.                                                                                             |
| `NCCL_MODULE`              | `NCCL/2.22.3-CUDA-12.4.1`                                                                              | Lmod module name; only NCCL available on gpu-a100. Without it, cmake prints `Could NOT find NCCL` and multi-GPU tensor splitting falls back to peer-copy.                                                          |
| `PYTHON_MODULE`            | `Python/3.11.3`                                                                                        | Lmod module name; canonical gpu-a100 default (from Compiler/GCCcore/11.3.0)                                                                                                                                        |
| `GCC_MODULE`               | `GCCcore/11.3.0`                                                                                       | Lmod module name; canonical gpu-a100 default; **must be the older compiler family** that Python/3.11.3 is built against, not the newer GCC/13.3.0                                                                  |
| `CMAKE_MODULE`             | `CMake/3.31.3`                                                                                         | Lmod module name; required by the build step — `module purge` strips the base cmake that ships with `spartan/rhel9`, so we explicitly load this one                                                                |
| `CUDA_ARCHITECTURES`       | `80`                                                                                                   | A100 = SM_80 (single arch keeps compile time sane)                                                                                                                                                                 |
| `EXTRA_CMAKE_FLAGS`        | `<unset>`                                                                                              | Spaces-separated extras forwarded to cmake configure                                                                                                                                                               |
| `MODELS_OVERRIDE`          | `<all 4 defaults>`                                                                                     | Space-separated list of HF repo ids                                                                                                                                                                                |
| `DATASETS_OVERRIDE`        | `<all 5 defaults>`                                                                                     | Space-separated subset of `{mmlu, popqa, bigbench, humaneval, include}`                                                                                                                                            |
| `QUANTS_OVERRIDE`          | `Q4_K_M`                                                                                               | Space-separated list of quant tags                                                                                                                                                                                 |
| `ROUTING_GRAPH_LINE_SCALE` | `<wrapper default = 5e-7>`                                                                             | Override `--line-scale` for `routing_graph_from_cpp.py`. Tune this when pair counts are much smaller or larger than your usual scale — the wrapper's own default has been calibrated for OLMoE-style pair volumes. |
| `HF_TOKEN`                 | `<unset>`                                                                                              | Optional; for gated repos                                                                                                                                                                                          |

---

## Results tree

Every cell under the sweep writes to:

```
${RESULTS_DIR}/<model_safe>/<quant_safe>/moe-<dataset>/
├── expert_counts.json   # the JSON the C++ binary writes
├── run.log              # stdout+stderr of the C++ binary
├── plot.log             # stdout+stderr of heatmap_from_cpp.py
├── routing.log          # stdout+stderr of routing_graph_from_cpp.py
├── routing_heatmap.png  # generated by heatmap_from_cpp.py
├── routing_heatmap_by_language.png
├── routing_heatmap_by_domain.png
├── routing_heatmap_by_langdom.png
├── routing_heatmap_by_langdom_normalized.png
├── routing_graph.png    # generated by routing_graph_from_cpp.py
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

| Symptom                                                                                                                           | Likely cause                                                                                                                                                                                                                                                                                                                      | Fix                                                                                                                                                                                                                                                                    |
| --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `module: command not found`                                                                                                       | Lmod not in `.bashrc` on the GPU node                                                                                                                                                                                                                                                                                             | `source /usr/local/lmod/lmod/init/bash` before `module` calls, or trust the .sbatch's `module purge`                                                                                                                                                                   |
| `Lmod has detected the following error: The following module(s) are unknown: "CUDA/12.X"` (or similar)                            | The hard-coded default module name doesn't exist on `gpu-a100` (versions change over time)                                                                                                                                                                                                                                        | `module avail cuda/python/gcc` on a `gpu-a100` node, then re-submit with the exact name (e.g. `CUDA_MODULE=CUDA/12.4.1 sbatch …`)                                                                                                                                      |
| `Lmod has detected the following error: These module(s) or extension(s) exist but cannot be loaded as requested: "Python/3.11.3"` | The Python module isn't built against the GCC compiler you have loaded (e.g. you used `GCC/13.3.0` but need `GCCcore/11.3.0`)                                                                                                                                                                                                     | `module spider Python/3.11.3` to see the required parent compiler, then re-submit with `GCC_MODULE=<spider-suggested-gcc>` (default is `GCCcore/11.3.0`)                                                                                                               |
| `[fatal] CUDA_MODULE is empty`                                                                                                    | Forgot to export the module env vars                                                                                                                                                                                                                                                                                              | `module avail cuda/python/gcc` on `gpu-a100`, then re-submit with all three exported                                                                                                                                                                                   |
| `nvcc: command not found`                                                                                                         | Wrong CUDA module                                                                                                                                                                                                                                                                                                                 | `module avail cuda` on `gpu-a100` and set `CUDA_MODULE`                                                                                                                                                                                                                |
| `cmake: command not found` (during the build step)                                                                                | `module purge` stripped the base cmake; the .sbatch needs to load a `CMake/...` module explicitly                                                                                                                                                                                                                                 | `module avail cmake` on `gpu-a100`; default is `CMake/3.31.3`                                                                                                                                                                                                          |
| `Could NOT find NCCL` (during the build step)                                                                                     | NCCL module not loaded (or CUDA version mismatch)                                                                                                                                                                                                                                                                                 | `module avail nccl` (after loading CUDA) on `gpu-a100`; the only available one is `NCCL/2.22.3-CUDA-12.4.1`, which is why `CUDA_MODULE` defaults to `CUDA/12.4.1`                                                                                                      |
| `ggml_cuda_init: no CUDA devices found`                                                                                           | `CUDA_VISIBLE_DEVICES` empty or wrong GPU count                                                                                                                                                                                                                                                                                   | Check `squeue -j $JOBID -o "% Gres"`; request `--gres=gpu:N` to match                                                                                                                                                                                                  |
| C++ binary picks the wrong GGUF                                                                                                   | Quant tag didn't match anything in the repo                                                                                                                                                                                                                                                                                       | Run `bash spartan/download-models.sh --list-quants <repo>` to see the available tags                                                                                                                                                                                   |
| `mkdir: cannot create directory '...'`                                                                                            | The script is trying to write to a path you don't own                                                                                                                                                                                                                                                                             | Pick a writable `SCRATCH_BASE` (e.g. `/data/gpfs/projects/<your-project>/llama-cpp-eval`) and re-submit                                                                                                                                                                |
| `dataset/moe-*.jsonl` not found                                                                                                   | Datasets weren't pre-downloaded; the GPU node is firewalled off from the public internet AND doesn't ship the `datasets` Python package                                                                                                                                                                                           | Run on the login node first: `bash spartan/download-datasets.sh --datasets <names>`. The .sbatch will also print a `[warn] ... not staged on scratch` line at job start listing the missing ones.                                                                      |
| `error: the 'datasets' package is required` (in `${DATASETS_DIR}/moe-*.download.log`)                                             | Same as above — the C++ binary subprocess tried to download datasets on the GPU node                                                                                                                                                                                                                                              | Same fix: pre-stage on the login node                                                                                                                                                                                                                                  |
| `ModuleNotFoundError: No module named 'matplotlib'` (in `${RESULTS_DIR}/<model>/<quant>/moe-<ds>/plot.log`)                       | Heatmap step couldn't import matplotlib. The .sbatch's preflight should install it automatically; if this still appears the auto-install failed (e.g. pypi unreachable)                                                                                                                                                           | Re-submit the job — the install is idempotent and will retry. If it keeps failing, install manually after the job has allocated: `pip install --user numpy matplotlib`                                                                                                 |
| OOM / `Killed` in job log                                                                                                         | 120B model + activations exceed `--mem`                                                                                                                                                                                                                                                                                           | Raise `--mem` (A100 node has 495 GB total) or use a smaller quant                                                                                                                                                                                                      |
| `error while loading shared libraries: libllama-common.so.0` (in any per-cell `run.log`)                                          | The eval binary was built with `BUILD_SHARED_LIBS=ON` (the llama.cpp default on Linux) and its baked-in RPATH no longer points to a directory that contains the .so files (e.g. because the build dir moved from `gpfs` to `scratch`). New builds use `-DBUILD_SHARED_LIBS=OFF` so they're statically linked and don't need this. | If you're seeing this on a brand-new job, the auto-install of `LD_LIBRARY_PATH=${BUILD_DIR}/bin` should already have rescued you — check the banner for it. If it didn't, `rm -rf $BUILD_DIR && sbatch spartan/llama-moe-eval.sbatch` to force a clean static rebuild. |

---

## File map

```
spartan/
├── README.md                 # this file
├── download-models.sh        # login-node pre-download (HF GGUFs)
├── download-datasets.sh      # login-node pre-download (eval datasets)
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
