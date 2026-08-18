    #!/usr/bin/env bash
#
# scripts/run-evaluator.sh - orchestrate the 5 MoE-routing evals across a
# hardcoded list of HF MoE GGUF repos.
#
# Behaviour:
#  - Downloads each of {MMLU, PopQA, BBH, HumanEval, INCLUDE} ONCE into a shared
#    build/datasets/<name>/ directory (idempotent - skipped if the .jsonl
#    already exists, or pass --redownload to force a refresh).
#  - Builds the five llama-eval-moe-* binaries if any are missing
#    (or pass --rebuild to force a clean rebuild).
#  - Loops every (model, dataset) cell, skipping cells whose
#    expert_counts.json already exists in the results tree.
#  - Captures each run's stdout+stderr to
#    build/results/<model_safe>/<dataset>/run.log.
#  - Runs the matching heatmap_from_cpp.py unless --skip-plots is passed.
#  - Prints a final OK / SKIPPED / FAILED summary table that survives
#    partial failures.

set -uo pipefail

# ---------------------------------------------------------------- paths

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ------------------------------------------------------------- model list

# Hardcoded HF MoE GGUF repos. Edit freely; each -hf <id> invocation
# will download + cache the GGUF on first use.
readonly DEFAULT_MODELS=(
    "allenai/OLMoE-1B-7B-0125-Instruct-GGUF"
    "LiteLLMs/Mixtral-8x22B-Instruct-v0.1-GGUF"
    "mradermacher/deepseek-moe-16b-chat-i1-GGUF"
    "unsloth/gpt-oss-120b-GGUF"
)

# --------------------------------------------------------- defaults / state

BUILD_DIR="${REPO_ROOT}/build"
DATASETS_DIR=""
RESULTS_DIR=""
MODELS=("${DEFAULT_MODELS[@]}")
DATASETS=(mmlu popqa bigbench humaneval include)
QUANTS=("Q4_K_M")
REBUILD=0
REDOWNLOAD=0
SKIP_PLOTS=0
SKIP_ROUTING_GRAPHS=0
USE_CUDA=0
PRINT_USAGE=0
# Per-routing-graph --line-scale. Tune to match the magnitude of your
# pair counts: OLMoE (~1e7 pair counts) wants ~1e-4, Mixtral (~1e6) wants
# ~1e-5. Empty means use the wrapper default (5e-7).
LINE_SCALE="${ROUTING_GRAPH_LINE_SCALE:-}"
# Forwarded to the C++ binary as --numa / -ngl / --split-mode /
# --tensor-split. Defaults below were tuned for single-A100 + 4xA100
# PCIe partitions (gpu-a100 on Spartan):
#   - NUMA_MODE=isolate avoids the "illegal memory access" race in
#     ggml_cuda's first launch (observed on OLMoE/include) and stops
#     pinning per-expert allocations to the wrong NUMA node on
#     multi-GPU PCIe boxes (observed slowing Mixtral by ~30%).
#   - NGL=99 fully offloads every model we run (all fit in 4x80GB) but
#     uses the safer integer the way llama.cpp's scheduler expects;
#     -ngl 999 occasionally trips a first-launch scratch reallocation.
#   - SPLIT_MODE controls how the model is laid out across GPUs visible
#     to the binary. llama.cpp's own default is `layer` (each layer on
#     one GPU, KV buffer sharded). Our default is also `layer` because
#     SLURM gives us >=1 GPU and we want to actually use them all. If
#     you really want single-GPU mode (e.g. to compare 1-GPU vs 4-GPU
#     throughput), set SPLIT_MODE=none. With n_gpu == 1 (CPU-only build
#     or --gres=gpu:1) the split-mode flag is a no-op, so it's always
#     safe to leave the default.
#   - EXTRA_TENSOR_SPLIT is comma-separated fractions summing to ~1.0
#     (e.g. "50,50" for 2 GPUs or "25,25,25,25" for 4). Empty means
#     auto-balance (llama.cpp's behaviour). We auto-derive a uniform
#     split when this is empty so the multi-GPU case actually spreads
#     layers instead of falling back to all-on-GPU-0.
NUMA_MODE="${NUMA_MODE:-isolate}"
NGL="${NGL:-99}"
SPLIT_MODE="${SPLIT_MODE:-layer}"
EXTRA_TENSOR_SPLIT="${EXTRA_TENSOR_SPLIT:-}"
# EMBD: forwarded as --embeddings / --no-embeddings to the eval binaries.
# Default 0 (no embeddings) to avoid the CUDA illegal-memory-access bug
# we hit with embedding=true on long multilingual prefills; the cb_eval
# hook still captures routing data either way. Set EMBD=1 to opt back
# into the old behaviour for A/B testing.
EMBD="${EMBD:-0}"

declare -A STATUS=()

# ----------------------------------------------------------- logging funcs

_log()      { printf '[%s] %s\n' "${1}" "${*:2}"; }
_log_info() { _log "info"  "$@"; }
_log_warn() { _log "warn"  "$@"; }
_log_err()  { _log "error" "$@" >&2; }
_die()      { _log_err "$@"; exit 1; }

# --------------------------------------------------- model / dataset maps

# Short name -> on-disk binary path under $BUILD_DIR
binary_for() {
    case "$1" in
        mmlu)     echo "${BUILD_DIR}/bin/llama-eval-moe-mmlu" ;;
        popqa)    echo "${BUILD_DIR}/bin/llama-eval-moe-popqa" ;;
        bigbench) echo "${BUILD_DIR}/bin/llama-eval-moe-bigbench" ;;
        humaneval) echo "${BUILD_DIR}/bin/llama-eval-moe-humaneval" ;;
        include)  echo "${BUILD_DIR}/bin/llama-eval-moe-include" ;;
        *) _die "binary_for: unknown dataset '$1'" ;;
    esac
}

# Short name -> python downloader script path (relative to REPO_ROOT)
downloader_for() {
    case "$1" in
        mmlu)     echo "${REPO_ROOT}/examples/eval-moe-mmlu/download_mmlu.py" ;;
        popqa)    echo "${REPO_ROOT}/examples/eval-moe-popqa/download_popqa.py" ;;
        bigbench) echo "${REPO_ROOT}/examples/eval-moe-bigbench/download_bigbench.py" ;;
        humaneval) echo "${REPO_ROOT}/examples/eval-moe-humaneval/download_humaneval.py" ;;
        include)  echo "${REPO_ROOT}/examples/eval-moe-include/download_include.py" ;;
        *) _die "downloader_for: unknown dataset '$1'" ;;
    esac
}

# Short name -> matching heatmap renderer
plotter_for() {
    case "$1" in
        mmlu)     echo "${REPO_ROOT}/examples/eval-moe-mmlu/heatmap_from_cpp.py" ;;
        popqa)    echo "${REPO_ROOT}/examples/eval-moe-popqa/heatmap_from_cpp.py" ;;
        bigbench) echo "${REPO_ROOT}/examples/eval-moe-bigbench/heatmap_from_cpp.py" ;;
        humaneval) echo "${REPO_ROOT}/examples/eval-moe-humaneval/heatmap_from_cpp.py" ;;
        include)  echo "${REPO_ROOT}/examples/eval-moe-include/heatmap_from_cpp.py" ;;
        *) _die "plotter_for: unknown dataset '$1'" ;;
    esac
}

# Short name -> matching integrated routing-graph renderer (added in the
# coactivation patch; consumes the `aggregate` block emitted by the
# updated C++ binary).
routing_graph_for() {
    case "$1" in
        mmlu)     echo "${REPO_ROOT}/examples/eval-moe-mmlu/routing_graph_from_cpp.py" ;;
        popqa)    echo "${REPO_ROOT}/examples/eval-moe-popqa/routing_graph_from_cpp.py" ;;
        bigbench) echo "${REPO_ROOT}/examples/eval-moe-bigbench/routing_graph_from_cpp.py" ;;
        humaneval) echo "${REPO_ROOT}/examples/eval-moe-humaneval/routing_graph_from_cpp.py" ;;
        include)  echo "${REPO_ROOT}/examples/eval-moe-include/routing_graph_from_cpp.py" ;;
        *) _die "routing_graph_for: unknown dataset '$1'" ;;
    esac
}

# Dataset-specific C++ flags to control eval size + few-shot + decoding.
# Tweak per hardware budget - the values match the per-dataset README
# "Quick smoke test" recommendations.
config_flags_for() {
    case "$1" in
        mmlu)     printf -- '--questions-per-subject 100 --n-shots 5' ;;
        popqa)    printf -- '--questions-per-prop 100 --gen-tokens 16' ;;
        bigbench) printf -- '--questions-per-task 100 --gen-tokens 128' ;;
        humaneval) printf -- '--questions-per-task 164 --gen-tokens 256' ;;
        include)  printf -- '--questions-per-langdom 50 --n-shots 5 --gen-tokens 16' ;;
        *) _die "config_flags_for: unknown dataset '$1'" ;;
    esac
}

# Dataset-specific input-file flags (--mmlu/--popqa/--bigbench/--include
# + the matching sidecar like --subjects/--props/--tasks/--langdom-list).
# Reads from $DATASETS_DIR/moe-<ds>/.
dataset_input_flags_for() {
    local ds="$1"
    local dir="${DATASETS_DIR}/moe-${ds}"
    case "$ds" in
        mmlu)
            printf -- '--mmlu %s --subjects %s' \
                "${dir}/mmlu.jsonl" "${dir}/subjects.txt" ;;
        popqa)
            printf -- '--popqa %s --props %s' \
                "${dir}/popqa.jsonl" "${dir}/props.txt" ;;
        bigbench)
            printf -- '--bigbench %s --tasks %s' \
                "${dir}/bigbench.jsonl" "${dir}/tasks.txt" ;;
        humaneval)
            printf -- '--humaneval %s --tasks %s' \
                "${dir}/humaneval.jsonl" "${dir}/tasks.txt" ;;
        include)
            printf -- '--include %s --langdom-list %s' \
                "${dir}/include.jsonl" "${dir}/languages_domains.txt" ;;
        *) _die "dataset_input_flags_for: unknown dataset '$ds'" ;;
    esac
}

# JSONL filename under the per-dataset dir - used by prepare_dataset to
# decide whether the dataset is already cached.
dataset_jsonl_for() {
    case "$1" in
        mmlu)     echo "mmlu.jsonl" ;;
        popqa)    echo "popqa.jsonl" ;;
        bigbench) echo "bigbench.jsonl" ;;
        humaneval) echo "humaneval.jsonl" ;;
        include)  echo "include.jsonl" ;;
        *) _die "dataset_jsonl_for: unknown dataset '$1'" ;;
    esac
}

# ------------------------------------------------------------- CLI parsing

print_usage() {
    cat <<'EOF'
Usage: bash scripts/run-evaluator.sh [options]

Options:
  --models <id> [<id> ...]      restrict to subset of hardcoded MODELS
    --datasets <ds> [<ds> ...]    restrict to subset of {mmlu,popqa,bigbench,humaneval,include}
    --quant <tag> [<tag> ...]     quantizations to run per model (default: Q4_K_M;
                                  forwarded as -hf <repo>:<tag>; case-insensitive)
  --build-dir <path>            llama.cpp build dir   (default: <repo>/build)
  --datasets-dir <path>         shared dataset cache  (default: <build-dir>/datasets)
  --results-dir <path>          per-model results tree (default: <build-dir>/results)
  --rebuild                     force cmake --build even if binaries exist
  --redownload                  force re-download of every dataset
  --skip-plots                  skip heatmap_from_cpp.py invocation
  --skip-routing-graphs         skip routing_graph_from_cpp.py invocation
  --line-scale <float>          routing graph --line-scale override
                                (default: 5e-7; tune to your pair count magnitude)
  --numa <mode>                  forwarded as --numa <mode> to the C++ binary
                                  (default: isolate; try "distribute" or empty
                                  if you have NUMA-aware workload issues)
  --ngl <int>                    forwarded as -ngl <int> (default: 99)
                                  99 fully offloads every model that fits in
                                  VRAM; safer than 999 against first-launch
                                  scratch races in ggml_cuda
  --split-mode <mode>            forwarded as --split-mode <mode>
                                  (default: layer; "none" forces single-GPU,
                                  even when SLURM allocated more)
  --tensor-split <a,b,...>       forwarded as --tensor-split <a,b,...>
                                  (default: empty; e.g. "50,50" for 2 GPUs)
  --cuda                        enable CUDA build (passes -DGGML_CUDA=ON to cmake)
  --no-cuda                     disable CUDA build (default)
  --embeddings                  pass --embeddings to each eval binary
                                (forces cparams.embeddings=true; default off
                                because embedding=true triggers a CUDA
                                illegal-memory-access bug on long multilingual
                                prefills. The cb_eval hook still captures
                                routing counts either way)
  --no-embeddings               opposite of --embeddings (default)
  -h, --help                    show this message and exit

Environment:
  EXTRA_CMAKE_FLAGS="..."        extra arguments forwarded to cmake configure
  CUDA_ARCHITECTURES="80;89"     forwarded as -DCMAKE_CUDA_ARCHITECTURES=<...>
                                 (only used when --cuda is set)
  ROUTING_GRAPH_LINE_SCALE=<f>   equivalent to --line-scale <f>
  NUMA_MODE=<mode>               equivalent to --numa <mode>
  NGL=<int>                      equivalent to --ngl <int>
  SPLIT_MODE=<mode>              equivalent to --split-mode <mode>
  EXTRA_TENSOR_SPLIT=<a,b,...>   equivalent to --tensor-split <a,b,...>
  EMBD=<0|1>                    equivalent to --embeddings/--no-embeddings
                                 (default 0 = no embeddings; the safer mode
                                 that avoids the CUDA OOB bug on long
                                 multilingual prefills)

The hardcoded MODELS list is at the top of the script (DEFAULT_MODELS).
The default quantization is Q4_K_M (see QUANTS= at the top of the script).

Results tree: results/<model_safe>/<quant_safe>/moe-<dataset>/
  ├── expert_counts.json    (C++ output: per-bucket + aggregate coactivation stats)
  ├── run.log               (C++ stdout+stderr)
  ├── plot.log              (heatmap_from_cpp.py stdout+stderr)
  ├── routing.log           (routing_graph_from_cpp.py stdout+stderr)
  ├── routing_heatmap*.png  (from heatmap_from_cpp.py)
  ├── routing_graph.png     (from routing_graph_from_cpp.py)
  └── metadata.json / counts_total.json (from heatmap_from_cpp.py)
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --models)
                shift; MODELS=()
                while [[ $# -gt 0 && "$1" != --* && "$1" != -* ]]; do
                    MODELS+=("$1"); shift
                done
                [[ ${#MODELS[@]} -gt 0 ]] || _die "--models requires at least one id"
                ;;
            --datasets)
                shift; DATASETS=()
                while [[ $# -gt 0 && "$1" != --* && "$1" != -* ]]; do
                    DATASETS+=("$1"); shift
                done
                [[ ${#DATASETS[@]} -gt 0 ]] || _die "--datasets requires at least one name"
                ;;
            --quant)
                shift; QUANTS=()
                while [[ $# -gt 0 && "$1" != --* && "$1" != -* ]]; do
                    QUANTS+=("$1"); shift
                done
                [[ ${#QUANTS[@]} -gt 0 ]] || _die "--quant requires at least one tag"
                ;;
            --build-dir)        BUILD_DIR="$2"; shift 2 ;;
            --datasets-dir)     DATASETS_DIR="$2"; shift 2 ;;
            --results-dir)      RESULTS_DIR="$2"; shift 2 ;;
            --rebuild)          REBUILD=1; shift ;;
            --redownload)       REDOWNLOAD=1; shift ;;
            --skip-plots)       SKIP_PLOTS=1; shift ;;
            --skip-routing-graphs)  SKIP_ROUTING_GRAPHS=1; shift ;;
            --line-scale)       LINE_SCALE="$2"; shift 2 ;;
            --numa)             NUMA_MODE="$2"; shift 2 ;;
            --ngl)              NGL="$2"; shift 2 ;;
            --split-mode)       SPLIT_MODE="$2"; shift 2 ;;
            --tensor-split)     EXTRA_TENSOR_SPLIT="$2"; shift 2 ;;
            --embeddings)       EMBD=1; shift ;;
            --no-embeddings)    EMBD=0; shift ;;
            --cuda)             USE_CUDA=1; shift ;;
            --no-cuda)          USE_CUDA=0; shift ;;
            -h|--help)          PRINT_USAGE=1; shift ;;
            *)
                _die "unknown argument: $1 (use --help)"
                ;;
        esac
    done
}

validate_choices() {
    local -a ALL_DS=(mmlu popqa bigbench humaneval include)
    local ds
    for ds in "${DATASETS[@]}"; do
        local known=0
        local k
        for k in "${ALL_DS[@]}"; do
            if [[ "$k" == "$ds" ]]; then known=1; break; fi
        done
        [[ $known -eq 1 ]] || _die "unknown dataset '$ds' (allowed: ${ALL_DS[*]})"
    done
}

# -------------------------------------------------------- ensure_build

ensure_build() {
    local ds
    local need_rebuild=$REBUILD
    if [[ $need_rebuild -eq 0 ]]; then
        for ds in "${DATASETS[@]}"; do
            if [[ ! -x "$(binary_for "$ds")" ]]; then
                need_rebuild=1; break
            fi
        done
    fi

    # Source-vs-binary staleness check. cmake's incremental build uses
    # mtime comparisons, but those only work within a single build tree.
    # If BUILD_DIR was previously populated from a different checkout
    # (e.g. a different $REPO_ROOT, or sources were edited and pushed
    # without a corresponding rebuild), the existing binary will be
    # newer than the source but still be the old code. Detect this by
    # comparing the source mtime against the binary mtime and force a
    # rebuild if any binary is older than its source.
    if [[ $need_rebuild -eq 0 ]]; then
        for ds in "${DATASETS[@]}"; do
            local bin bin_src
            bin="$(binary_for "$ds")"
            bin_src="${REPO_ROOT}/examples/eval-moe-${ds}/eval-moe-${ds}.cpp"
            if [[ -x "${bin}" && -f "${bin_src}" ]]; then
                if [[ "${bin_src}" -nt "${bin}" ]]; then
                    _log_info "source ${bin_src} is newer than ${bin##*/} - forcing rebuild"
                    need_rebuild=1; break
                fi
            fi
        done
    fi
    if [[ $need_rebuild -eq 1 ]]; then
        # Compose cmake configure flags. We always set CMAKE_BUILD_TYPE;
        # when --cuda is on we add -DGGML_CUDA=ON (and an optional
        # CMAKE_CUDA_ARCHITECTURES from the env). $EXTRA_CMAKE_FLAGS is
        # an escape hatch for users who want to add more options without
        # touching this script.
        # -DGGML_CCACHE=OFF silences the upstream "ccache not found"
        # warning. ccache is only a build-speed accelerator; the binaries
        # produced are identical. Disable unconditionally so the warning
        # never appears regardless of whether ccache happens to be on PATH.
        # -DBUILD_SHARED_LIBS=OFF statically links the eval binaries so they
        # don't need libllama-common.so / libggml.so / etc. on PATH at
        # runtime. Without this, llama.cpp's cmake defaults to
        # BUILD_SHARED_LIBS=ON on Linux (see top-level CMakeLists.txt:62),
        # which means the binaries have an RPATH baked in at link time.
        # If BUILD_DIR is moved/renamed/recreated between the link step and
        # the run step, the eval fails with:
        #     error while loading shared libraries: libllama-common.so.0
        # This bit us when SCRATCH_BASE changed from gpfs to scratch on
        # Spartan - the build dir moved and the existing dynamic binaries
        # could no longer find their libs. Static linking adds ~100 MB per
        # binary (negligible vs the 60-240 GB GGUF) and removes the
        # dependency on $BUILD_DIR/bin being on LD_LIBRARY_PATH.
        local -a CMAKE_FLAGS=( -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=OFF )
            CMAKE_FLAGS+=( -DBUILD_SHARED_LIBS=OFF )
        if [[ $USE_CUDA -eq 1 ]]; then
            CMAKE_FLAGS+=( -DGGML_CUDA=ON )
            if [[ -n "${CUDA_ARCHITECTURES:-}" ]]; then
                CMAKE_FLAGS+=( "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" )
            fi
        fi
        if [[ -n "${EXTRA_CMAKE_FLAGS:-}" ]]; then
            # shellcheck disable=SC2206  # intentional word-split on $EXTRA_CMAKE_FLAGS
            local extra
            extra=( ${EXTRA_CMAKE_FLAGS} )
            CMAKE_FLAGS+=( "${extra[@]}" )
        fi
        _log_info "running cmake configure + build (USE_CUDA=${USE_CUDA}, flags=${CMAKE_FLAGS[*]}) ..."
        ( cd "${REPO_ROOT}" && \
          cmake -B "${BUILD_DIR}" "${CMAKE_FLAGS[@]}" && \
          cmake --build "${BUILD_DIR}" \
              --target llama-eval-moe-mmlu llama-eval-moe-popqa \
                       llama-eval-moe-bigbench llama-eval-moe-humaneval \
                       llama-eval-moe-include -j
        ) || _die "cmake build failed"
    else
            _log_info "all 5 binaries present in ${BUILD_DIR}/bin, skipping rebuild"
    fi
}

# ---------------------------------------------------- prepare_dataset

prepare_dataset() {
    local ds="$1"
    local dir="${DATASETS_DIR}/moe-${ds}"
    local jsonl="${dir}/$(dataset_jsonl_for "$ds")"
    if [[ $REDOWNLOAD -eq 1 && -d "$dir" ]]; then
        _log_warn "--redownload: rm -rf ${dir}"
        rm -rf "$dir"
    fi
    mkdir -p "$dir"
    if [[ -f "$jsonl" ]]; then
        _log_info "already prepared (${jsonl} exists), skip"
        return 0
    fi
    _log_info "downloading ${ds} -> ${dir} ..."
    if ! python3 "$(downloader_for "$ds")" --outdir "$dir" \
         >"${dir}.download.log" 2>&1; then
        _log_err "downloader for ${ds} failed; see ${dir}.download.log"
        return 1
    fi
    if [[ ! -f "$jsonl" ]]; then
        _log_err "downloader for ${ds} finished but ${jsonl} missing"
        return 1
    fi
    _log_info "downloaded ${ds}"
    return 0
}

# -------------------------------------------------------------- model_safe

# Replace / with -- so HF ids become filesystem-safe directory names.
model_safe() {
    printf '%s' "${1//\//--}"
}

# Same sanitisation for quant tags (e.g. "Q4_K_M" -> "Q4_K_M", but reject
# anything containing characters that would confuse the filesystem or the
# -hf <repo>:<quant> resolver. Quant tags are short and known, so any
# slash or whitespace is almost certainly a typo.)
quant_safe() {
    local raw="$1"
    if [[ ! "$raw" =~ ^[A-Za-z0-9._-]+$ ]]; then
        _die "quant_safe: invalid quant tag '$raw' (allowed chars: A-Za-z0-9._-)"
    fi
    printf '%s' "${raw//\//--}"
}

# Returns 0 only if the JSON file exists and Python can parse it.
json_is_valid() {
    local path="$1"
    [[ -f "$path" ]] || return 1
    python3 - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path) as f:
        json.load(f)
except Exception:
    raise SystemExit(1)
PY
}

# ------------------------------------------------------------- run_eval

# Runs one (model, quant, dataset) cell. Sets STATUS[model|quant|ds] to
# OK / SKIPPED / FAILED and prints the log path in the summary. The
# `quant` arg is forwarded to the C++ binary as -hf <repo>:<quant>, which
# llama.cpp's resolver handles uniformly across flat and subdir layouts.
run_eval() {
    local model="$1"
    local quant="$2"
    local ds="$3"
    local safe;       safe="$(model_safe "$model")"
    local quant_safe; quant_safe="$(quant_safe "$quant")"
    local out_dir="${RESULTS_DIR}/${safe}/${quant_safe}/moe-${ds}"
    local log="${out_dir}/run.log"
    local plot_log="${out_dir}/plot.log"
    local routing_log="${out_dir}/routing.log"
    local json="${out_dir}/expert_counts.json"
    local binary; binary="$(binary_for "$ds")"
    local status_key="${model}|${quant}|${ds}"

    mkdir -p "$out_dir"

    if [[ $REDOWNLOAD -eq 0 && -f "$json" ]]; then
        if json_is_valid "$json"; then
        _log_info "(model=${model}, quant=${quant}, ds=${ds}) -> SKIPPED (${json} exists)"
        STATUS["${status_key}"]="SKIPPED"
        # The C++ step was skipped, but the derived artifacts may still
        # be missing on first run (or stale from a previous tool version).
        # Always (re)run the routing-graph step unless explicitly disabled,
        # since it's cheap and idempotent. We DON'T re-render heatmaps here
        # because that's the pre-existing convention.
        _render_routing_graph "$model" "$quant" "$ds" "$json" "$routing_log"
        return 0
        fi
        _log_warn "(model=${model}, quant=${quant}, ds=${ds}) existing ${json} is invalid; regenerating"
    fi

    _log_info "(model=${model}, quant=${quant}, ds=${ds}) running ${binary##*/} ..."

    # Build argv tokens for the dataset-specific flags. We use `read -r -a`
    # to split on whitespace so the output of config_flags_for /
    # dataset_input_flags_for flows into separate argv entries.
    local -a CF_FLAGS
    IFS=' ' read -r -a CF_FLAGS <<< "$(config_flags_for "$ds")"
    local -a DS_FLAGS
    IFS=' ' read -r -a DS_FLAGS <<< "$(dataset_input_flags_for "$ds")"

    # Compose C++ binary argv. --numa / -ngl / --split-mode /
    # --tensor-split are configurable from the env (NUMA_MODE / NGL /
    # SPLIT_MODE / EXTRA_TENSOR_SPLIT) so the spartan sbatch can tune
    # them per partition. We always forward them (even with defaults)
    # because llama.cpp's argument parser requires explicit values for
    # --split-mode / --tensor-split to take effect; the defaults below
    # are equivalent to "no-op" so it's safe to pass them unconditionally.
    local -a BIN_FLAGS=( -hf "${model}:${quant}" "-ngl" "${NGL}" )
    if [[ -n "${NUMA_MODE}" ]]; then
        BIN_FLAGS+=( --numa "${NUMA_MODE}" )
    fi
    if [[ -n "${SPLIT_MODE}" ]]; then
        BIN_FLAGS+=( --split-mode "${SPLIT_MODE}" )
    fi
    # Auto-derive --tensor-split when it's unset AND we're in a
    # multi-GPU split mode. llama.cpp's own default is "auto-balance"
    # which puts almost everything on GPU 0 for non-row/tensor splits;
    # we make the explicit uniform split so all allocated GPUs get
    # a fair share of layers.
    if [[ -n "${EXTRA_TENSOR_SPLIT}" ]]; then
        BIN_FLAGS+=( --tensor-split "${EXTRA_TENSOR_SPLIT}" )
    elif [[ "${SPLIT_MODE}" == "layer" || "${SPLIT_MODE}" == "row" ]]; then
        # We don't know n_gpu until the binary starts, so just pass a
        # reasonable 2-way split (most common HPC config). The binary
        # ignores extra fractions beyond n_gpu, so 50,50 is safe for
        # both 2-GPU and 4-GPU boxes. Users on a 1-GPU box should set
        # SPLIT_MODE=none (which skips this branch entirely).
        BIN_FLAGS+=( --tensor-split "50,50" )
    fi

    # Embedding mode toggle. Default off (no embeddings, --no-embeddings
    # to the eval binary) to avoid the CUDA illegal-memory-access bug on
    # long multilingual prefills; the cb_eval hook still captures routing
    # counts regardless. See examples/eval-moe-*/eval-moe-*.cpp for the
    # per-binary rationale.
    if [[ "${EMBD}" -eq 1 ]]; then
        BIN_FLAGS+=( --embeddings )
    else
        BIN_FLAGS+=( --no-embeddings )
    fi

    if "${binary}" "${BIN_FLAGS[@]}" \
            "${CF_FLAGS[@]}" "${DS_FLAGS[@]}" \
            -o "$json" \
            >"$log" 2>&1; then
        : # success
    else
        local rc=$?
        _log_err "(model=${model}, quant=${quant}, ds=${ds}) FAILED (rc=${rc}); log: ${log}"
        STATUS["${status_key}"]="FAILED|${log}"
        return 0
    fi

    if [[ -f "$json" ]]; then
        _log_info "(model=${model}, quant=${quant}, ds=${ds}) -> OK (json=${json})"
        STATUS["${status_key}"]="OK"
    else
        _log_warn "(model=${model}, quant=${quant}, ds=${ds}) exit=0 but ${json} missing; FAILED"
        STATUS["${status_key}"]="FAILED|${log}"
        return 0
    fi

    if [[ $SKIP_PLOTS -eq 0 ]]; then
        local plotter; plotter="$(plotter_for "$ds")"
        _log_info "(model=${model}, quant=${quant}, ds=${ds}) rendering heatmaps ..."
        if ! python3 "$plotter" -i "$json" \
                >"$plot_log" 2>&1; then
            _log_warn "(model=${model}, quant=${quant}, ds=${ds}) plot failed; log: ${plot_log}"
        fi
    fi

    if [[ $SKIP_ROUTING_GRAPHS -eq 0 ]]; then
        _render_routing_graph "$model" "$quant" "$ds" "$json" "$routing_log"
    fi
}

# Helper used by both the SKIPPED and the freshly-evaluated paths in
# run_eval(). Renders the routing graph from an existing expert_counts.json.
# No-op if --skip-routing-graphs was set. Writes to $routing_log; caller
# is responsible for declaring that variable.
_render_routing_graph() {
    local model="$1" quant="$2" ds="$3" json="$4" routing_log="$5"
    [[ $SKIP_ROUTING_GRAPHS -eq 0 ]] || return 0
    local rg; rg="$(routing_graph_for "$ds")"
    _log_info "(model=${model}, quant=${quant}, ds=${ds}) rendering routing graph ..."
    local -a RG_FLAGS=( -i "$json" )
    if [[ -n "${LINE_SCALE}" ]]; then
        RG_FLAGS+=( --line-scale "${LINE_SCALE}" )
    fi
    if ! python3 "$rg" "${RG_FLAGS[@]}" \
            >"$routing_log" 2>&1; then
        _log_warn "(model=${model}, quant=${quant}, ds=${ds}) routing graph failed; log: ${routing_log}"
    fi
}

# ------------------------------------------------------------ summary

print_summary() {
    echo
    echo "=== summary ==="
    printf '%-72s  %-10s  %s\n' "model | quant | dataset" "status" "log_path"
    printf '%-72s  %-10s  %s\n' "------------------------------------------------------------------------" "------" "--------"
    local m q ds safe qsafe key status log
    for m in "${MODELS[@]}"; do
        safe="$(model_safe "$m")"
        for q in "${QUANTS[@]}"; do
            qsafe="$(quant_safe "$q")"
            for ds in "${DATASETS[@]}"; do
                key="${m}|${q}|${ds}"
                status="${STATUS[$key]:-MISSING}"
                case "$status" in
                    OK)        log="${RESULTS_DIR}/${safe}/${qsafe}/moe-${ds}/run.log" ;;
                    SKIPPED)   log="${RESULTS_DIR}/${safe}/${qsafe}/moe-${ds}/expert_counts.json" ;;
                    FAILED*)   log="${status#FAILED|}" ;;
                    *)         log="(no record)" ;;
                esac
                printf '%-72s  %-10s  %s\n' "${key}" "${status%%|*}" "${log}"
            done
        done
    done
}

# ------------------------------------------------------------------ main

main() {
    parse_args "$@"
    if [[ $PRINT_USAGE -eq 1 ]]; then
        print_usage
        return 0
    fi
    validate_choices

    DATASETS_DIR="${DATASETS_DIR:-${BUILD_DIR}/datasets}"
    RESULTS_DIR="${RESULTS_DIR:-${BUILD_DIR}/results}"

    mkdir -p "${DATASETS_DIR}" "${RESULTS_DIR}"

    echo "============================================================"
    _log_info "REPO_ROOT      = ${REPO_ROOT}"
    _log_info "BUILD_DIR      = ${BUILD_DIR}"
    _log_info "DATASETS_DIR   = ${DATASETS_DIR}"
    _log_info "RESULTS_DIR    = ${RESULTS_DIR}"
    _log_info "MODELS (${#MODELS[@]})       = ${MODELS[*]}"
    _log_info "DATASETS (${#DATASETS[@]})     = ${DATASETS[*]}"
    _log_info "QUANTS (${#QUANTS[@]})       = ${QUANTS[*]}"
    _log_info "REBUILD=${REBUILD}, REDOWNLOAD=${REDOWNLOAD}, SKIP_PLOTS=${SKIP_PLOTS}, SKIP_ROUTING_GRAPHS=${SKIP_ROUTING_GRAPHS}, USE_CUDA=${USE_CUDA}, ROUTING_GRAPH_LINE_SCALE=${LINE_SCALE:-<default 5e-7>}, NUMA_MODE=${NUMA_MODE}, NGL=${NGL}, SPLIT_MODE=${SPLIT_MODE}, EXTRA_TENSOR_SPLIT=${EXTRA_TENSOR_SPLIT:-<unset>}, EMBD=${EMBD}"
    echo "============================================================"

    echo "============================================================"
    _log_info "build step"
    echo "============================================================"
    ensure_build || _die "build failed"

    echo "============================================================"
    _log_info "dataset step (idempotent)"
    echo "============================================================"
    local ds
    for ds in "${DATASETS[@]}"; do
        prepare_dataset "$ds" || _log_warn "prepare_dataset ${ds} had errors; continuing"
    done

    echo "============================================================"
    _log_info "eval step (model x quant x dataset)"
    echo "============================================================"
    local m q
    for m in "${MODELS[@]}"; do
        for q in "${QUANTS[@]}"; do
            echo "------------------------------------------------------------"
            _log_info "model: ${m}  quant: ${q}"
            echo "------------------------------------------------------------"
            for ds in "${DATASETS[@]}"; do
                run_eval "$m" "$q" "$ds"
            done
        done
    done

    echo "results root: ${RESULTS_DIR}"
}

main "$@"