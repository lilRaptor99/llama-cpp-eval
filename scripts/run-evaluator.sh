    print_summary
#!/usr/bin/env bash
#
# scripts/run-evaluator.sh - orchestrate the 4 MoE-routing evals across a
# hardcoded list of HF MoE GGUF repos.
#
# Behaviour:
#  - Downloads each of {MMLU, PopQA, BBH, INCLUDE} ONCE into a shared
#    build/datasets/<name>/ directory (idempotent - skipped if the .jsonl
#    already exists, or pass --redownload to force a refresh).
#  - Builds the four llama-eval-moe-* binaries if any are missing
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
    #"unsloth/gpt-oss-120b-GGUF"
)

# --------------------------------------------------------- defaults / state

BUILD_DIR="${REPO_ROOT}/build"
DATASETS_DIR=""
RESULTS_DIR=""
MODELS=("${DEFAULT_MODELS[@]}")
DATASETS=(mmlu popqa bigbench include)
REBUILD=0
REDOWNLOAD=0
SKIP_PLOTS=0
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
        include)  echo "${REPO_ROOT}/examples/eval-moe-include/heatmap_from_cpp.py" ;;
        *) _die "plotter_for: unknown dataset '$1'" ;;
    esac
}

# Dataset-specific C++ flags to control eval size + few-shot + decoding.
# Tweak per hardware budget - the values match the per-dataset README
# "Quick smoke test" recommendations.
config_flags_for() {
    case "$1" in
        mmlu)     printf -- '--questions-per-subject 10 --n-shots 5' ;;
        popqa)    printf -- '--questions-per-prop 10 --gen-tokens 16' ;;
        bigbench) printf -- '--questions-per-task 5 --gen-tokens 128' ;;
        include)  printf -- '--questions-per-langdom 5 --n-shots 5 --gen-tokens 16' ;;
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
  --datasets <ds> [<ds> ...]    restrict to subset of {mmlu,popqa,bigbench,include}
  --build-dir <path>            llama.cpp build dir   (default: <repo>/build)
  --datasets-dir <path>         shared dataset cache  (default: <build-dir>/datasets)
  --results-dir <path>          per-model results tree (default: <build-dir>/results)
  --rebuild                     force cmake --build even if binaries exist
  --redownload                  force re-download of every dataset
  --skip-plots                  skip heatmap_from_cpp.py invocation
  -h, --help                    show this message and exit

The hardcoded MODELS list is at the top of the script (DEFAULT_MODELS).
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
            --build-dir)        BUILD_DIR="$2"; shift 2 ;;
            --datasets-dir)     DATASETS_DIR="$2"; shift 2 ;;
            --results-dir)      RESULTS_DIR="$2"; shift 2 ;;
            --rebuild)          REBUILD=1; shift ;;
            --redownload)       REDOWNLOAD=1; shift ;;
            --skip-plots)       SKIP_PLOTS=1; shift ;;
            -h|--help)          PRINT_USAGE=1; shift ;;
            *)
                _die "unknown argument: $1 (use --help)"
                ;;
        esac
    done
}

validate_choices() {
    local -a ALL_DS=(mmlu popqa bigbench include)
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
        _log_info "running cmake configure + build ..."
        ( cd "${REPO_ROOT}" && \
          cmake -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release && \
          cmake --build "${BUILD_DIR}" \
              --target llama-eval-moe-mmlu llama-eval-moe-popqa \
                       llama-eval-moe-bigbench llama-eval-moe-include -j
        ) || _die "cmake build failed"
    else
        _log_info "all 4 binaries present in ${BUILD_DIR}/bin, skipping rebuild"
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

# ------------------------------------------------------------- run_eval

# Runs one (model, dataset) cell. Sets STATUS[model|ds] to OK / SKIPPED /
# FAILED and prints the log path in the summary.
run_eval() {
    local model="$1"
    local ds="$2"
    local safe; safe="$(model_safe "$model")"
    local out_dir="${RESULTS_DIR}/${safe}/moe-${ds}"
    local log="${out_dir}/run.log"
    local plot_log="${out_dir}/plot.log"
    local json="${out_dir}/expert_counts.json"
    local binary; binary="$(binary_for "$ds")"

    mkdir -p "$out_dir"

    if [[ -f "$json" && $REDOWNLOAD -eq 0 ]]; then
        _log_info "(model=${model}, ds=${ds}) -> SKIPPED (${json} exists)"
        STATUS["${model}|${ds}"]="SKIPPED"
        return 0
    fi

    _log_info "(model=${model}, ds=${ds}) running ${binary##*/} ..."

    # Build argv tokens for the dataset-specific flags. We use `read -r -a`
    # to split on whitespace so the output of config_flags_for /
    # dataset_input_flags_for flows into separate argv entries.
    local -a CF_FLAGS
    IFS=' ' read -r -a CF_FLAGS <<< "$(config_flags_for "$ds")"
    local -a DS_FLAGS
    IFS=' ' read -r -a DS_FLAGS <<< "$(dataset_input_flags_for "$ds")"

    if "${binary}" -hf "${model}" -ngl 999 --numa distribute \
            "${CF_FLAGS[@]}" "${DS_FLAGS[@]}" \
            -o "$json" \
            >"$log" 2>&1; then
        : # success
    else
        local rc=$?
        _log_err "(model=${model}, ds=${ds}) FAILED (rc=${rc}); log: ${log}"
        STATUS["${model}|${ds}"]="FAILED|${log}"
        return 0
    fi

    if [[ -f "$json" ]]; then
        _log_info "(model=${model}, ds=${ds}) -> OK (json=${json})"
        STATUS["${model}|${ds}"]="OK"
    else
        _log_warn "(model=${model}, ds=${ds}) exit=0 but ${json} missing; FAILED"
        STATUS["${model}|${ds}"]="FAILED|${log}"
        return 0
    fi

    if [[ $SKIP_PLOTS -eq 0 ]]; then
        local plotter; plotter="$(plotter_for "$ds")"
        _log_info "(model=${model}, ds=${ds}) rendering heatmaps ..."
        if ! python3 "$plotter" -i "$json" \
                >"$plot_log" 2>&1; then
            _log_warn "(model=${model}, ds=${ds}) plot failed; log: ${plot_log}"
        fi
    fi
}

# ------------------------------------------------------------ summary

print_summary() {
    echo
    echo "=== summary ==="
    printf '%-58s  %-10s  %s\n' "model | dataset" "status" "log_path"
    printf '%-58s  %-10s  %s\n' "--------------------------------------------------------" "------" "--------"
    local m ds key status log
    for m in "${MODELS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            key="${m}|${ds}"
            status="${STATUS[$key]:-MISSING}"
            case "$status" in
                OK)        log="${RESULTS_DIR}/$(model_safe "$m")/moe-${ds}/run.log" ;;
                SKIPPED)   log="${RESULTS_DIR}/$(model_safe "$m")/moe-${ds}/expert_counts.json" ;;
                FAILED*)   log="${status#FAILED|}" ;;
                *)         log="(no record)" ;;
            esac
            printf '%-58s  %-10s  %s\n' "${key}" "${status%%|*}" "${log}"
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
    _log_info "REBUILD=${REBUILD}, REDOWNLOAD=${REDOWNLOAD}, SKIP_PLOTS=${SKIP_PLOTS}"
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
    _log_info "eval step (model x dataset)"
    echo "============================================================"
    local m
    for m in "${MODELS[@]}"; do
        echo "------------------------------------------------------------"
        _log_info "model: ${m}"
        echo "------------------------------------------------------------"
        for ds in "${DATASETS[@]}"; do
            run_eval "$m" "$ds"
        done
    done

    echo "results root: ${RESULTS_DIR}"
}

main "$@"