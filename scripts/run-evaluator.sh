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
USE_CUDA=0
PRINT_USAGE=0

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
  --cuda                        enable CUDA build (passes -DGGML_CUDA=ON to cmake)
  --no-cuda                     disable CUDA build (default)
  -h, --help                    show this message and exit

Environment:
  EXTRA_CMAKE_FLAGS="..."        extra arguments forwarded to cmake configure
  CUDA_ARCHITECTURES="80;89"     forwarded as -DCMAKE_CUDA_ARCHITECTURES=<...>
                                 (only used when --cuda is set)

The hardcoded MODELS list is at the top of the script (DEFAULT_MODELS).
The default quantization is Q4_K_M (see QUANTS= at the top of the script).

Results tree: results/<model_safe>/<quant_safe>/moe-<dataset>/
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
        local -a CMAKE_FLAGS=( -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=OFF )
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
    local json="${out_dir}/expert_counts.json"
    local binary; binary="$(binary_for "$ds")"
    local status_key="${model}|${quant}|${ds}"

    mkdir -p "$out_dir"

    if [[ $REDOWNLOAD -eq 0 && -f "$json" ]]; then
        if json_is_valid "$json"; then
        _log_info "(model=${model}, quant=${quant}, ds=${ds}) -> SKIPPED (${json} exists)"
        STATUS["${status_key}"]="SKIPPED"
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

    if "${binary}" -hf "${model}:${quant}" -ngl 999 --numa distribute \
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
    _log_info "REBUILD=${REBUILD}, REDOWNLOAD=${REDOWNLOAD}, SKIP_PLOTS=${SKIP_PLOTS}, USE_CUDA=${USE_CUDA}"
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