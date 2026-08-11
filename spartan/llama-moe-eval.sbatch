#!/usr/bin/env bash
#
# spartan/llama-moe-eval.sbatch - SLURM job that builds (CUDA) + runs the
# 5 MoE-routing evals from scripts/run-evaluator.sh on the Spartan HPC
# (Unimelb) gpu-a100 partition.
#
# Designed for a 4-day wall clock budget. The job is naturally idempotent:
# the eval phase skips (model, quant, dataset) cells whose
# `<quant_safe>/moe-<ds>/expert_counts.json` already exists, so a
# pre-empted run can be resumed by re-submitting without redoing work.
#
# Prerequisites
# -------------
# 1. The `eval-moe` branch of https://github.com/lilRaptor99/llama-cpp-eval
#    is checked out at $REPO_ROOT (default: $HOME/llama-cpp-eval).
# 2. The HF cache has been pre-populated with the GGUF files the eval will
#    consume. Run on the login node first:
#        bash spartan/download-models.sh --quant $QUANTS_OVERRIDE
#    (or just leave default quant=Q4_K_M).
# 3. Python deps for the heatmap step are available on the GPU node:
#        pip install --user huggingface_hub numpy matplotlib datasets
#    (huggingface_hub is also needed by download-models.sh; the eval
#    binaries themselves do not need it.)
#
# Quick start
# -----------
#   # 1. login node: pre-download
#   ssh spartan
#   cd ~/llama-cpp-eval && git pull origin eval-moe
#   bash spartan/download-models.sh --quant Q4_K_M Q8_0
#
#   # 2. submit the GPU job. The defaults below assume the canonical
#   #    gpu-a100 modules (CUDA/12.8.0, GCC/13.3.0, Python/3.11.3). If
#   #    module avail reports different versions on your partition, just
#   #    override the three *MODULE vars.
#   sbatch spartan/llama-moe-eval.sbatch
#   # or, to override:
#   CUDA_MODULE=CUDA/12.4.1 GCC_MODULE=GCC/12.3.0 \
#       sbatch spartan/llama-moe-eval.sbatch
#
#   # 3. tail logs
#   tail -f spartan/logs/llama-moe-eval-<jobid>.out
#
# Env-var overrides (export before sbatch, or pass via --export=...)
# ------------------------------------------------------------------
#   REPO_ROOT         path to the cloned repo        (default: $HOME/llama-cpp-eval)
#   SCRATCH_BASE      root for build/datasets/results/hf_cache
#                                                  (default: /data/gpfs/projects/uom00014/llama-cpp-eval)
#   CUDA_MODULE       Lmod module name for CUDA     (default: CUDA/12.8.0)
#   PYTHON_MODULE     Lmod module name for Python   (default: Python/3.11.3)
#   GCC_MODULE        Lmod module name for GCC      (default: GCC/13.3.0)
#   CUDA_ARCHITECTURES  passed to cmake              (default: 80  - A100 = SM_80)
#   EXTRA_CMAKE_FLAGS   extra args for cmake configure (default: unset)
#   MODELS_OVERRIDE     subset of hardcoded models   (default: all 4)
#   DATASETS_OVERRIDE   subset of {mmlu,popqa,bigbench,humaneval,include}
#                                                   (default: all 5)
#   QUANTS_OVERRIDE     quants to run                (default: Q4_K_M)
#   HF_TOKEN            optional, for gated repos    (default: unset)

# ---------------------------------------------------------------- SLURM
#SBATCH --job-name=llama-moe-eval
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=spartan/logs/llama-moe-eval-%j.out
#SBATCH --error=spartan/logs/llama-moe-eval-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@student.unimelb.edu.au

set -euo pipefail

# ---------------------------------------------------------------- paths
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "${SLURM_SUBMIT_DIR}/spartan/logs"

# Default values (overridable via env before sbatch).
# Default module names are taken from the `module avail` output on the
# gpu-a100 partition (EasyBuild /apps/easybuild-2022 tree). Override
# these if your partition shows different versions.
#   CUDA_MODULE     CUDA/12.8.0          (from Core)
#   GCC_MODULE      GCC/13.3.0           (from Core)
#   PYTHON_MODULE   Python/3.11.3        (from Compiler/GCCcore/11.3.0)
# Load order matters: GCC must come before Python (which depends on it),
# and CUDA should come last (it pulls in the matching cuDNN/NCCL).
CUDA_MODULE="${CUDA_MODULE:-CUDA/12.8.0}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3}"
GCC_MODULE="${GCC_MODULE:-GCC/13.3.0}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-80}"
REPO_ROOT="${REPO_ROOT:-$HOME/llama-cpp-eval}"
SCRATCH_BASE="${SCRATCH_BASE:-${SCRATCH:-/data/gpfs/projects/uom00014}/llama-cpp-eval}"

BUILD_DIR="${SCRATCH_BASE}/build"
DATASETS_DIR="${SCRATCH_BASE}/datasets"
RESULTS_DIR="${SCRATCH_BASE}/results"
HF_CACHE="${SCRATCH_BASE}/hf_cache"

mkdir -p "${BUILD_DIR}" "${DATASETS_DIR}" "${RESULTS_DIR}" "${HF_CACHE}"

# Defaults from scripts/run-evaluator.sh (kept in sync; see DEFAULTS below)
: "${MODELS_OVERRIDE:=}"      # empty = run all 4 hardcoded models
: "${DATASETS_OVERRIDE:=}"    # empty = run all 5 datasets
: "${QUANTS_OVERRIDE:=Q4_K_M}" # default quant
: "${EXTRA_CMAKE_FLAGS:=}"

# ---------------------------------------------------------------- banner
echo "============================================================"
echo "[$(date -Iseconds)] llama-moe-eval starting"
echo "[$(date -Iseconds)] job_id        = ${SLURM_JOB_ID:-}"
echo "[$(date -Iseconds)] node          = ${SLURM_JOB_NODELIST:-}"
echo "[$(date -Iseconds)] cpus          = ${SLURM_CPUS_PER_TASK:-}"
echo "[$(date -Iseconds)] gpus          = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[$(date -Iseconds)] REPO_ROOT     = ${REPO_ROOT}"
echo "[$(date -Iseconds)] SCRATCH_BASE  = ${SCRATCH_BASE}"
echo "[$(date -Iseconds)] HF_HOME       = ${HF_CACHE}"
echo "[$(date -Iseconds)] MODELS        = ${MODELS_OVERRIDE:-<all defaults>}"
echo "[$(date -Iseconds)] DATASETS      = ${DATASETS_OVERRIDE:-<all defaults>}"
echo "[$(date -Iseconds)] QUANTS        = ${QUANTS_OVERRIDE}"
echo "[$(date -Iseconds)] CUDA_MODULE   = ${CUDA_MODULE}"
echo "[$(date -Iseconds)] PYTHON_MODULE = ${PYTHON_MODULE}"
echo "[$(date -Iseconds)] GCC_MODULE    = ${GCC_MODULE}"
echo "[$(date -Iseconds)] CUDA_ARCHS    = ${CUDA_ARCHITECTURES}"
echo "[$(date -Iseconds)] EXTRA_CMAKE_FLAGS = ${EXTRA_CMAKE_FLAGS:-<unset>}"
echo "============================================================"

# ---------------------------------------------------------------- modules
# Preflight: error out with a clear message if any *_MODULE variable is
# empty. Without this, an empty CUDA_MODULE would silently become
# `module load ""` which Lmod might accept as a no-op - giving the user
# a confusing later error (no nvcc on PATH) instead of the right one.
if [[ -z "${GCC_MODULE}" ]]; then
    echo "[fatal] GCC_MODULE is empty. Run 'module avail gcc' on a gpu-a100 node and export GCC_MODULE=<name> before sbatch." >&2
    exit 3
fi
if [[ -z "${PYTHON_MODULE}" ]]; then
    echo "[fatal] PYTHON_MODULE is empty. Run 'module avail python' on a gpu-a100 node and export PYTHON_MODULE=<name> before sbatch." >&2
    exit 3
fi
if [[ -z "${CUDA_MODULE}" ]]; then
    echo "[fatal] CUDA_MODULE is empty. Run 'module avail cuda' on a gpu-a100 node and export CUDA_MODULE=<name> before sbatch." >&2
    exit 3
fi

# `set -u` would crash on the module command if it isn't installed; we
# unconditionally `module purge` first so we know exactly what we're getting.
set +u
module purge
module load "${GCC_MODULE}"
module load "${PYTHON_MODULE}"
module load "${CUDA_MODULE}"
module list
set -u

# Post-load sanity check: make sure the binaries we expect are actually on
# PATH. If they aren't, Lmod succeeded but our defaults were wrong.
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[fatal] nvcc not on PATH after loading ${CUDA_MODULE}. The module exists but doesn't expose nvcc - pick a different CUDA_MODULE." >&2
    exit 3
fi
if ! command -v gcc >/dev/null 2>&1; then
    echo "[fatal] gcc not on PATH after loading ${GCC_MODULE}. Pick a different GCC_MODULE." >&2
    exit 3
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "[fatal] python3 not on PATH after loading ${PYTHON_MODULE}. Pick a different PYTHON_MODULE." >&2
    exit 3
fi
echo "[info] nvcc: $(nvcc --version | tail -1)"
echo "[info] gcc : $(gcc --version | head -1)"
echo "[info] python: $(python3 --version)"

# Make sure C++ sees the loaded GCC and that nvcc is on PATH.
export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-$(command -v g++)}"
export PATH="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")"):/usr/local/cuda/bin}/bin:${PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${CUDA_HOME:-/usr/local/cuda}/lib64"

# OpenMP: per the Spartan docs, set OMP_NUM_THREADS to match the SLURM
# requested CPUs so multi-threaded math libraries don't oversubscribe.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"

# ----------------------------------------------- HF cache for the eval
# The C++ binary reads $HF_HOME (and $HUGGINGFACE_HUB_CACHE) before falling
# back to $HOME/.cache/huggingface. See common/hf-cache.cpp in this repo.
# By exporting both here, we guarantee the pre-downloaded cache lands at
# the same path the eval reads from.
export HF_HOME="${HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"

# CUDA runtime knobs from docs/build.md that improve behaviour on tight-VRAM
# scenarios (the 120B model + 4-bit quant will push close to the 4x80GB
# limit on the A100 node).
export GGML_CUDA_ENABLE_UNIFIED_MEMORY="${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1}"
# NB: -ngl 999 is set by run-evaluator.sh's run_eval() and offloads every
#     layer to GPU. We do not override that here.

# Compiler-visible CMAKE_ARCHS for the build phase.
export CUDA_ARCHITECTURES
# EXTRA_CMAKE_FLAGS is also passed through to run-evaluator.sh -> cmake.
export EXTRA_CMAKE_FLAGS

# ---------------------------------------------------------------- preflight
# 1. Make sure the repo is in place.
if [[ ! -d "${REPO_ROOT}/scripts" ]]; then
    echo "[fatal] REPO_ROOT=${REPO_ROOT} does not look like a llama-cpp-eval checkout (no scripts/ dir)" >&2
    exit 2
fi
if [[ ! -x "${REPO_ROOT}/scripts/run-evaluator.sh" ]]; then
    echo "[fatal] ${REPO_ROOT}/scripts/run-evaluator.sh is not executable" >&2
    exit 2
fi

# 2. Make sure the HF cache has been pre-populated. If it's empty, warn but
#    continue (the C++ binary's `-hf` flag will pull on demand).
if [[ -d "${HF_CACHE}/hub" ]]; then
    n_repos=$(find "${HF_CACHE}/hub" -maxdepth 1 -mindepth 1 -type d -name 'models--*' | wc -l)
    echo "[info] HF cache contains ${n_repos} repo(s)"
    if [[ "${n_repos}" -eq 0 ]]; then
        echo "[warn] ${HF_CACHE}/hub is empty."
        echo "[warn] Run on the login node first: bash spartan/download-models.sh --quant ${QUANTS_OVERRIDE}"
    fi
else
    echo "[warn] ${HF_CACHE}/hub does not exist yet."
    echo "[warn] Run on the login node first: bash spartan/download-models.sh --quant ${QUANTS_OVERRIDE}"
fi

# 3. Quick CUDA sanity check (nvidia-smi prints to stdout).
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
else
    echo "[warn] nvidia-smi not found on PATH; CUDA may not be active"
fi

# ---------------------------------------------------------------- run
# Compose the run-evaluator.sh argv. The ${VAR:+...} guards mean each flag
# only appears when its env var is set, so unset vars fall back to the
# script's own defaults.
RUN_ARGS=(
    --build-dir "${BUILD_DIR}"
    --datasets-dir "${DATASETS_DIR}"
    --results-dir "${RESULTS_DIR}"
    --cuda
)
if [[ -n "${MODELS_OVERRIDE}" ]]; then
    # Bash word-split on purpose: the env var is a space-separated list.
    # shellcheck disable=SC2206
    RUN_ARGS+=( --models ${MODELS_OVERRIDE} )
fi
if [[ -n "${DATASETS_OVERRIDE}" ]]; then
    # shellcheck disable=SC2206
    RUN_ARGS+=( --datasets ${DATASETS_OVERRIDE} )
fi
if [[ -n "${QUANTS_OVERRIDE}" ]]; then
    # shellcheck disable=SC2206
    RUN_ARGS+=( --quant ${QUANTS_OVERRIDE} )
fi

echo "============================================================"
echo "[$(date -Iseconds)] running: bash scripts/run-evaluator.sh ${RUN_ARGS[*]}"
echo "============================================================"

cd "${REPO_ROOT}"
bash scripts/run-evaluator.sh "${RUN_ARGS[@]}"

echo "============================================================"
echo "[$(date -Iseconds)] llama-moe-eval finished (rc=$?)"
echo "results root: ${RESULTS_DIR}"
echo "hf cache    : ${HF_CACHE}"
echo "logs        : ${SLURM_SUBMIT_DIR}/spartan/logs/llama-moe-eval-${SLURM_JOB_ID}.{out,err}"
echo "============================================================"
