#!/usr/bin/env bash
#
# spartan/download-datasets.sh - login-node pre-download for the 5 MoE
# evaluation datasets used by `scripts/run-evaluator.sh`.
#
# Why this exists: the GPU compute nodes on Spartan are firewalled off
# from the public internet for outbound HTTPS, AND they don't ship the
# `datasets` Python package. Both are required by the per-dataset
# `download_<ds>.py` scripts under examples/. Pre-running them on the
# login node (which has internet + an easy `pip install --user`) means
# the GPU job starts with every dataset already on disk and never tries
# to fetch one mid-eval.
#
# IMPORTANT: keep DEFAULT_DATASETS in sync with
# scripts/run-evaluator.sh. They intentionally duplicate rather than
# source because `run-evaluator.sh` defines its array inside the script
# body; this script must also work standalone on the login node.
#
# Behaviour:
#   - Resolves SCRATCH_BASE the same way the .sbatch does.
#   - For each dataset, runs examples/eval-moe-<ds>/download_<ds>.py
#     --outdir <SCRATCH_BASE>/datasets/moe-<ds>. The downloader emits
#     <ds>.jsonl plus a sidecar (subjects/tasks/etc).
#   - Idempotent: re-running skips datasets whose <ds>.jsonl already
#     exists and parses as valid JSONL.
#   - Logs everything; the GPU job's prepare_dataset() phase becomes a
#     no-op once this script has populated the cache.

set -uo pipefail

# ---------------------------------------------------------------- paths

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ------------------------------------------------------------- dataset list

# IMPORTANT: keep in sync with scripts/run-evaluator.sh DATASETS
readonly DEFAULT_DATASETS=(mmlu popqa bigbench humaneval include)

# Map short name -> path of the python downloader (relative to REPO_ROOT).
# MUST match the table in scripts/run-evaluator.sh::downloader_for().
readonly -A DOWNLOADER_FOR=(
    [mmlu]="examples/eval-moe-mmlu/download_mmlu.py"
    [popqa]="examples/eval-moe-popqa/download_popqa.py"
    [bigbench]="examples/eval-moe-bigbench/download_bigbench.py"
    [humaneval]="examples/eval-moe-humaneval/download_humaneval.py"
    [include]="examples/eval-moe-include/download_include.py"
)

# Per-dataset output JSONL filename (matches
# scripts/run-evaluator.sh::dataset_jsonl_for()).
readonly -A JSONL_FOR=(
    [mmlu]="mmlu.jsonl"
    [popqa]="popqa.jsonl"
    [bigbench]="bigbench.jsonl"
    [humaneval]="humaneval.jsonl"
    [include]="include.jsonl"
)

# --------------------------------------------------------- defaults / state

SCRATCH_BASE=""
DATASETS=("${DEFAULT_DATASETS[@]}")
ASSUME_YES=0
LIST_ONLY=0
PRINT_USAGE=0

# ----------------------------------------------------------- logging funcs

_log()      { printf '[%s] %s\n' "${1}" "${*:2}"; }
_log_info() { _log "info"  "$@"; }
_log_warn() { _log "warn"  "$@"; }
_log_err()  { _log "error" "$@" >&2; }
_die()      { _log_err "$@"; exit 1; }

# ----------------------------------------------------------- path resolution

# Resolves SCRATCH_BASE exactly the same way the SLURM job does so the
# pre-download lands in the same datasets/ directory the eval reads from.
# Override with --scratch-base for re-runs against a different scratch
# volume.
resolve_scratch_base() {
    if [[ -n "$SCRATCH_BASE" ]]; then
        printf '%s' "$SCRATCH_BASE"
        return
    fi
    if [[ -n "${SCRATCH:-}" ]]; then
        printf '%s/llama-cpp-eval' "${SCRATCH%/}"
        return
    fi
    printf '%s/llama-cpp-eval' "/data/gpfs/projects/uom00014"
}

# ----------------------------------------------------- python env preflight

# The downloaders all `import datasets` (HF datasets library). On the
# Spartan login node the system python usually doesn't have it; we tell
# the user once, install to --user, and proceed.
ensure_python_deps() {
    if python3 -c 'import datasets' >/dev/null 2>&1; then
        _log_info "datasets package already importable"
        return 0
    fi
    _log_warn "the 'datasets' python package is missing - installing to ~/.local now"
    if ! python3 -m pip install --user --quiet datasets; then
        _die "pip install --user datasets failed. Check your network or try \`python3 -m pip install --user --break-system-packages datasets\` if PEP 668 is in play."
    fi
    # Refresh PATH so subsequent python3 invocations see ~/.local.
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! python3 -c 'import datasets' >/dev/null 2>&1; then
        _die "pip install --user datasets succeeded but 'import datasets' still fails. Check PYTHONUSERBASE / ~/.local/lib/python*/site-packages exists."
    fi
    _log_info "datasets package installed"
}

# ----------------------------------------------------- per-dataset download

# Returns 0 if <outdir>/<jsonl> exists and is non-empty + parseable.
# We only need to peek the first JSON object - the downloaders always
# write one record per line.
dataset_is_cached() {
    local jsonl="$1"
    [[ -s "$jsonl" ]] || return 1
    python3 - "$jsonl" <<'PY' 2>/dev/null
import json
import sys
path = sys.argv[1]
try:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            json.loads(line)
            break
        else:
            raise SystemExit(1)
except Exception:
    raise SystemExit(1)
PY
}

# Print a short description of what each dataset produces + approx size.
# Used by --list and the pre-flight summary.
dataset_description() {
    local ds="$1"
    case "$ds" in
        mmlu)      echo "cais/mmlu (57 subjects, dev+test splits) - ~50 MB" ;;
        popqa)     echo "akariasai/PopQA (English factoid QA, subsampled) - ~5 MB" ;;
        bigbench)  echo "maveriq/bigbenchhard (23 tasks, bbh subset) - ~30 MB" ;;
        humaneval) echo "openai/openai_humaneval (164 programming problems) - ~2 MB" ;;
        include)   echo "CohereForAI/include-base-44 (44 langdomains) - ~15 MB" ;;
        *)         echo "?" ;;
    esac
}

download_one_dataset() {
    local ds="$1" outdir="$2"
    local script="${REPO_ROOT}/${DOWNLOADER_FOR[$ds]:?}"
    local jsonl="${outdir}/${JSONL_FOR[$ds]:?}"

    if [[ ! -f "$script" ]]; then
        _log_err "downloader for '${ds}' missing: ${script}"
        return 2
    fi
    mkdir -p "$outdir"

    if dataset_is_cached "$jsonl"; then
        _log_info "already prepared (${jsonl} exists), skip"
        return 0
    fi

    _log_info "downloading ${ds} -> ${outdir} ..."
    if ! python3 "$script" --outdir "$outdir" >"${outdir}.download.log" 2>&1; then
        _log_err "downloader for ${ds} failed; see ${outdir}.download.log"
        return 1
    fi
    if ! dataset_is_cached "$jsonl"; then
        _log_err "downloader for ${ds} finished but ${jsonl} missing or empty"
        return 1
    fi
    _log_info "downloaded ${ds} (${jsonl})"
    return 0
}

# ----------------------------------------------------------- CLI parsing

print_usage() {
    cat <<'EOF'
Usage: bash spartan/download-datasets.sh [options]

Pre-downloads the 5 MoE-eval datasets from the Spartan login node into
$SCRATCH_BASE/datasets/ so the GPU job's prepare_dataset() phase becomes
a no-op. Run this once (or whenever you add a new dataset).

Options:
  --datasets <list>   space-separated subset of {mmlu,popqa,bigbench,humaneval,include}
                      (default: all 5)
  --scratch-base DIR  override SCRATCH_BASE (default: /data/gpfs/projects/uom00014/llama-cpp-eval,
                      or ${SCRATCH}/llama-cpp-eval if $SCRATCH is set)
  --list              show what each dataset is + skip download
  --yes               assume yes to any confirmation prompts
  -h, --help          show this help

Examples:
  # Default: all 5 datasets into the canonical scratch path.
  bash spartan/download-datasets.sh

  # Just one dataset (useful for the smoke-test job).
  bash spartan/download-datasets.sh --datasets mmlu

  # Different scratch volume (e.g. a project-specific one).
  bash spartan/download-datasets.sh --scratch-base /data/gpfs/projects/uom00014/llama-cpp-eval

Why this exists:
  The Spartan GPU compute nodes are firewalled off from the public
  internet, AND they don't ship the `datasets` Python package. The
  matching C++ binaries all assume <DATASETS_DIR>/moe-<ds>/<ds>.jsonl
  is already on disk. This script makes that assumption true by
  populating the cache on the login node first.

Prerequisite:
  pip install --user huggingface_hub datasets   (run once on the login node)
EOF
}

parse_args() {
    while (( $# > 0 )); do
        case "$1" in
            --datasets)
                shift
                [[ $# -gt 0 ]] || _die "--datasets requires at least one name"
                DATASETS=()
                for d in "$@"; do
                    [[ "$d" == "--"* || "$d" == "-"* ]] && break
                    if [[ -z "${DOWNLOADER_FOR[$d]:-}" ]]; then
                        _die "--datasets: unknown dataset '$d' (allowed: ${!DOWNLOADER_FOR[*]})"
                    fi
                    DATASETS+=("$d")
                    shift
                done
                [[ ${#DATASETS[@]} -gt 0 ]] || _die "--datasets requires at least one name"
                ;;
            --scratch-base) SCRATCH_BASE="$2"; shift 2 ;;
            --list)         LIST_ONLY=1; shift ;;
            --yes|-y)       ASSUME_YES=1; shift ;;
            -h|--help)      PRINT_USAGE=1; shift ;;
            *) _die "unknown argument: $1 (try --help)" ;;
        esac
    done
}

# ------------------------------------------------------------------ main

main() {
    parse_args "$@"
    if [[ $PRINT_USAGE -eq 1 ]]; then
        print_usage
        return 0
    fi

    SCRATCH_BASE="$(resolve_scratch_base)"
    local datasets_dir="${SCRATCH_BASE}/datasets"

    echo "============================================================"
    echo "[$(date -Iseconds)] download-datasets starting"
    echo "[$(date -Iseconds)] REPO_ROOT     = ${REPO_ROOT}"
    echo "[$(date -Iseconds)] SCRATCH_BASE  = ${SCRATCH_BASE}"
    echo "[$(date -Iseconds)] DATASETS_DIR  = ${datasets_dir}"
    echo "[$(date -Iseconds)] DATASETS      = ${DATASETS[*]}"
    echo "============================================================"

    if [[ ! -d "${REPO_ROOT}/examples" ]]; then
        _die "REPO_ROOT=${REPO_ROOT} does not look like a llama-cpp-eval checkout (no examples/ dir)"
    fi

    if [[ $LIST_ONLY -eq 1 ]]; then
        for ds in "${DATASETS[@]}"; do
            printf '%-10s  %s\n' "$ds" "$(dataset_description "$ds")"
        done
        return 0
    fi

    # Smoke test: can we even reach pypi / huggingface from here?
    if ! python3 -c 'import urllib.request, sys; \
        urllib.request.urlopen("https://huggingface.co/api/models/cais/mmlu", timeout=10).read()' \
            >/dev/null 2>&1; then
        _die "can't reach huggingface.co from this host. Are you on the Spartan login node (not a compute node)?"
    fi

    ensure_python_deps

    mkdir -p "${datasets_dir}"

    # Summary table before we start, so the user sees the full plan.
    echo
    _log_info "plan:"
    for ds in "${DATASETS[@]}"; do
        printf '  %-10s  %s\n' "$ds" "$(dataset_description "$ds")"
    done
    echo

    local rc=0
    local failed=()
    for ds in "${DATASETS[@]}"; do
        if ! download_one_dataset "$ds" "${datasets_dir}/moe-${ds}"; then
            failed+=("$ds")
            rc=1
        fi
    done

    echo
    echo "============================================================"
    if [[ $rc -eq 0 ]]; then
        _log_info "all ${#DATASETS[@]} dataset(s) ready under ${datasets_dir}"
    else
        _log_err "FAILED for ${#failed[@]} dataset(s): ${failed[*]}"
        _log_err "see ${datasets_dir}/moe-*.download.log for each failure"
    fi
    echo "============================================================"
    return $rc
}

main "$@"