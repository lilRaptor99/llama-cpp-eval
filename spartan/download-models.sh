#!/usr/bin/env bash
#
# spartan/download-models.sh - login-node pre-download for the 4 hardcoded
# HF MoE GGUF repos used by `scripts/run-evaluator.sh`.
#
# Why this exists: the C++ eval binary's `-hf <repo>` resolves files via the
# standard HF hub cache layout ($HF_HOME/hub/models--<org>--<name>/snapshots/<rev>/...).
# Pre-seeding that cache from the login node (where HF egress is unrestricted
# and bandwidth is unmetered) means the GPU job starts warm and never has to
# pull a 60-240 GB GGUF during compute-quota hours.
#
# IMPORTANT: keep DEFAULT_MODELS + DEFAULT_QUANTS in sync with
# scripts/run-evaluator.sh. They intentionally duplicate rather than source
# because `run-evaluator.sh` defines its arrays inside the script body; this
# script must also work standalone on the login node where the eval script
# may not have been read into the shell.
#
# Behaviour:
#   - Resolves SCRATCH_BASE the same way the .sbatch does.
#   - For each (model, quant), derives an `allow_patterns` regex that
#     matches both flat (`<name>-Q4_K_M.gguf`) and subdirectory
#     (`Q4_K_M/Q4_K_M-00001-of-00003.gguf`) layouts.
#   - Estimates total disk usage and (unless --yes) asks for confirmation.
#   - Calls huggingface_hub.snapshot_download with the derived patterns.
#   - Idempotent: re-running skips already-cached blobs.

set -uo pipefail

# ---------------------------------------------------------------- paths

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------------------------- model list

# IMPORTANT: keep in sync with scripts/run-evaluator.sh DEFAULT_MODELS
readonly DEFAULT_MODELS=(
    "allenai/OLMoE-1B-7B-0125-Instruct-GGUF"
    "LiteLLMs/Mixtral-8x22B-Instruct-v0.1-GGUF"
    "mradermacher/deepseek-moe-16b-chat-i1-GGUF"
    "unsloth/gpt-oss-120b-GGUF"
)

# IMPORTANT: keep in sync with scripts/run-evaluator.sh default QUANTS
readonly DEFAULT_QUANTS=("Q4_K_M")

# --------------------------------------------------------- defaults / state

SCRATCH_BASE=""
MODELS=("${DEFAULT_MODELS[@]}")
QUANTS=("${DEFAULT_QUANTS[@]}")
PATTERN_OVERRIDE=""
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
# pre-download lands in the same cache the eval reads from. Override
# with --scratch-base for re-runs against a different scratch volume.
resolve_scratch_base() {
    if [[ -n "$SCRATCH_BASE" ]]; then
        printf '%s' "$SCRATCH_BASE"
        return
    fi
    if [[ -n "${SCRATCH:-}" ]]; then
        printf '%s/llama-cpp-eval' "${SCRATCH%/}"
        return
    fi
    printf "%s/llama-cpp-eval" "/data/scratch/projects/uom00014"
}

# ----------------------------------------------------------- file enumeration

# Enumerate every .gguf filename in `repo_id` (paths are repo-relative and
# may include subdirectory prefixes for split-GGUFs). Echoes one path per
# line. Uses the HF Hub HTTP API so we don't need a local clone.
list_repo_gguf_files() {
    local repo_id="$1"
    python3 - "$repo_id" <<'PY'
import json
import sys
import urllib.error
import urllib.request

repo_id = sys.argv[1]
url = f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true"
try:
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
except urllib.error.HTTPError as e:
    sys.stderr.write(f"[error] {repo_id}: HTTP {e.code} {e.reason}\n")
    raise SystemExit(2)
except urllib.error.URLError as e:
    sys.stderr.write(f"[error] {repo_id}: {e.reason}\n")
    raise SystemExit(2)
for entry in data:
    if not isinstance(entry, dict):
        continue
    path = entry.get("path") or entry.get("rfilename") or ""
    if path.endswith(".gguf"):
        print(path)
PY
}

# Build an `allow_patterns` regex that matches:
#   <repo_root>/<quant>[.-]*.gguf                 (flat: OLMoE-...-Q4_K_M.gguf)
#   <repo_root>/Q4_K_M/<quant>[-_0-9.]*.gguf      (subdir split: Q4_K_M/Q4_K_M-00001-of-00003.gguf)
# Quant chars are escaped so literal dots / hyphens in tags don't widen.
# Output is a single line (one regex); empty line = no match.
quant_glob_regex() {
    local quant="$1"
    # Escape regex specials that could appear in a quant tag.
    local escaped
    escaped=$(python3 -c 'import re,sys; print(re.escape(sys.argv[1]))' "$quant")
    printf '(^|/)(%s)([./-]|$)' "$escaped"
}

# Per (model, quant): produce a python list of `allow_patterns` strings
# matching the actual filenames. Returns 0 if any match, 1 if none.
derive_allow_patterns() {
    local model="$1" quant="$2" repo_file="$3"
    python3 - "$model" "$quant" "$repo_file" <<'PY'
import re, sys

model, quant, repo_file = sys.argv[1], sys.argv[2], sys.argv[3]
files = [ln.strip() for ln in open(repo_file) if ln.strip()]

# Quant tag is a substring of the filename / path; match either:
#   "<quant>.gguf" or "<quant>-*.gguf" or "<quant>/.../*.gguf" (subdir split)
# Use a case-insensitive substring search so Q4_K_M and q4_k_m both work.
ql = quant.lower()
hits = []
for path in files:
    p = path.lower()
    if (f"{ql}.gguf" in p
            or f"{ql}-" in p
            or f"/{ql}/" in p
            or f"/{ql}." in p):
        hits.append(path)

if not hits:
    print("", end="")  # signal "no match" via empty stdout
    sys.exit(1)

# Emit one pattern per line (huggingface_hub accept_patterns wants globs).
# Convert the path to a glob: each part with a split suffix gets replaced
# by '*-of-*.gguf'. This handles both single-file and split-GGUFs uniformly.
for path in hits:
    # If the filename contains '-of-' it's a split file -> match the family.
    if "-of-" in path:
        head = path.rsplit("-", 3)[0]  # e.g. "Q4_K_M/Q4_K_M-00001"
        print(f"{head}*-of-*.gguf")
    else:
        # Single-file: glob the whole path (escape any regex chars later).
        print(path)
PY
}

# Estimate total bytes for a (model, quant) pair by summing the size of
# every matching file. Echoes "size_bytes pattern_count".
estimate_size() {
    local model="$1" quant="$2" repo_file="$3"
    python3 - "$model" "$quant" "$repo_file" <<'PY'
import json, sys, urllib.request, urllib.error

model, quant, repo_file = sys.argv[1], sys.argv[2], sys.argv[3]
files = [ln.strip() for ln in open(repo_file) if ln.strip()]

ql = quant.lower()
hits = []
for path in files:
    p = path.lower()
    if (f"{ql}.gguf" in p
            or f"{ql}-" in p
            or f"/{ql}/" in p
            or f"/{ql}." in p):
        hits.append(path)

if not hits:
    print("0 0")
    sys.exit(0)

# Fetch sizes via the same API (each entry has 'size').
url = f"https://huggingface.co/api/models/{model}/tree/main?recursive=true"
total = 0
hit_names = set(hits)
try:
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
except (urllib.error.HTTPError, urllib.error.URLError):
    pass
for entry in data:
    if not isinstance(entry, dict):
        continue
    path = entry.get("path") or entry.get("rfilename") or ""
    if path in hit_names:
        total += int(entry.get("size", 0) or 0)

print(f"{total} {len(hits)}")
PY
}

# ---------------------------------------------------------------- download

# Call huggingface_hub.snapshot_download for one (model, patterns_set) cell.
# `patterns_set` is a newline-delimited list of allow_patterns globs.
download_one() {
    local model="$1" patterns_set="$2"
    HF_HUB_OFFLINE=0 python3 - "$model" "$patterns_set" <<'PY'
import os, sys
from huggingface_hub import snapshot_download

model = sys.argv[1]
patterns_blob = sys.argv[2]
allow_patterns = [ln.strip() for ln in patterns_blob.splitlines() if ln.strip()]
cache_dir = os.environ["HUGGINGFACE_HUB_CACHE"]

print(f"  downloading {len(allow_patterns)} pattern(s) -> {cache_dir}", flush=True)
snapshot_download(
    repo_id=model,
    allow_patterns=allow_patterns,
    cache_dir=cache_dir,
    tqdm_class=None,  # cleaner log output
)
print("  done", flush=True)
PY
}

# -------------------------------------------------------------- CLI parsing

print_usage() {
    cat <<'EOF'
Usage: bash spartan/download-models.sh [options]

Pre-downloads GGUF files into the HF cache used by spartan/llama-moe-eval.sbatch.

Options:
  --models <id> [<id> ...]    restrict to subset of hardcoded MODELS
  --quant <tag> [<tag> ...]    quantizations to pre-download (default: Q4_K_M)
  --pattern <glob>             override the auto-derived allow_patterns
                               (applied to every (model, quant) cell)
  --scratch-base <path>        override SCRATCH_BASE
                               (default: ${SCRATCH:-/data/scratch/projects/uom00014}/llama-cpp-eval)
  --list-quants <model>        print the available .gguf quant tags for one
                               model and exit (does not download anything)
  --yes                        skip the disk-usage confirmation prompt
  -h, --help                   show this message and exit

Environment:
  HF_TOKEN                     optional; passed through to huggingface_hub
                               for gated repos. All DEFAULT_MODELS are public.

The hardcoded MODELS list is at the top of the script.
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
            --quant)
                shift; QUANTS=()
                while [[ $# -gt 0 && "$1" != --* && "$1" != -* ]]; do
                    QUANTS+=("$1"); shift
                done
                [[ ${#QUANTS[@]} -gt 0 ]] || _die "--quant requires at least one tag"
                ;;
            --pattern)        PATTERN_OVERRIDE="$2"; shift 2 ;;
            --scratch-base)   SCRATCH_BASE="$2"; shift 2 ;;
            --yes)            ASSUME_YES=1; shift ;;
            --list-quants)
                shift; list_quants_for "$1"; shift ;;
            -h|--help)        PRINT_USAGE=1; shift ;;
            *)
                _die "unknown argument: $1 (use --help)"
                ;;
        esac
    done
}

validate_choices() {
    local q
    for q in "${QUANTS[@]}"; do
        if [[ ! "$q" =~ ^[A-Za-z0-9._-]+$ ]]; then
            _die "invalid --quant '$q' (allowed chars: A-Za-z0-9._-)"
        fi
    done
}

# Print every .gguf filename for `model` so the user can see the available
# quant tags without downloading anything. We then derive a list of unique
# quant-looking tokens by splitting on '.gguf' / '-' / '_'.
list_quants_for() {
    local model="$1"
    local files
    files="$(list_repo_gguf_files "$model")" || _die "could not list files for $model"
    python3 - "$files" <<'PY'
import re, sys
files = sys.argv[1].splitlines()
known = (
    "Q2_K", "Q2_K_S", "Q2_K_L",
    "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
    "Q6_K",
    "Q8_0",
    "F16", "F32", "BF16", "FP16", "FP32",
    "IQ1_S", "IQ1_M",
    "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XS", "IQ3_S", "IQ3_M",
    "IQ4_XS", "IQ4_NL",
    "MXFP4",
)
seen = set()
for f in files:
    # Strip split-file suffix: gpt-oss-120b-Q4_K_M-00001-of-00002.gguf ->
    # gpt-oss-120b-Q4_K_M. Then strip .gguf, then take the trailing
    # quant-looking token.
    f2 = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", f)
    f2 = f2.replace(".gguf", "")
    parts = f2.replace("/", "-").split("-")
    for p in parts:
        if p in known:
            seen.add(p)
for q in sorted(seen):
    print(q)
PY
    exit 0
}

# ----------------------------------------------------------------- main

main() {
    parse_args "$@"
    if [[ $PRINT_USAGE -eq 1 ]]; then
        print_usage
        return 0
    fi
    validate_choices

    local scratch_base; scratch_base="$(resolve_scratch_base)"
    local hf_cache="${scratch_base}/hf_cache"
    mkdir -p "$hf_cache"

    export HUGGINGFACE_HUB_CACHE="$hf_cache"
    # Belt-and-braces: also export HF_HOME for tools that look at the
    # legacy var. C++ reads both (see common/hf-cache.cpp).
    export HF_HOME="$hf_cache"

    echo "============================================================"
    _log_info "SCRATCH_BASE  = ${scratch_base}"
    _log_info "HF_HOME       = ${hf_cache}"
    _log_info "MODELS (${#MODELS[@]}) = ${MODELS[*]}"
    _log_info "QUANTS (${#QUANTS[@]}) = ${QUANTS[*]}"
    if [[ -n "$PATTERN_OVERRIDE" ]]; then
        _log_info "PATTERN_OVERRIDE = ${PATTERN_OVERRIDE}"
    fi
    echo "============================================================"

    # ----- Phase 1: enumerate each repo once and derive per-cell plans
    local total_bytes=0
    local -a cell_plans=()

    local m q
    for m in "${MODELS[@]}"; do
        _log_info "enumerating ${m} ..."
        local repo_file
        repo_file="$(mktemp)"
        # shellcheck disable=SC2064  # we want $repo_file captured NOW
        trap "rm -f '$repo_file'" RETURN
        if ! list_repo_gguf_files "$m" >"$repo_file"; then
            _log_warn "could not enumerate ${m}; skipping"
            rm -f "$repo_file"
            continue
        fi
        local n_files
        n_files=$(wc -l <"$repo_file" | tr -d ' ')

        for q in "${QUANTS[@]}"; do
            local sz pc
            read -r sz pc < <(estimate_size "$m" "$q" "$repo_file")
            if [[ "$pc" -eq 0 ]]; then
                _log_warn "  ${m} :: ${q}: no matching files in repo; skipping"
                continue
            fi
            local patterns
            patterns="$(derive_allow_patterns "$m" "$q" "$repo_file")" || {
                _log_warn "  ${m} :: ${q}: pattern derivation failed; skipping"
                continue
            }
            cell_plans+=( "${m}|${q}|${sz}|${pc}|${patterns}" )
            total_bytes=$(( total_bytes + sz ))
            local sz_human
            sz_human=$(python3 -c "import sys; n=int(sys.argv[1]); print(f'{n/1024**3:.1f} GB' if n else '0 GB')" "$sz")
            _log_info "  ${m} :: ${q}: ${sz_human} (${pc} file(s))"
        done
        rm -f "$repo_file"
    done

    if [[ ${#cell_plans[@]} -eq 0 ]]; then
        _die "no (model, quant) cells matched anything in any repo; nothing to do"
    fi

    local total_human
    total_human=$(python3 -c "import sys; n=int(sys.argv[1]); print(f'{n/1024**3:.1f} GB' if n else '0 GB')" "$total_bytes")
    echo "============================================================"
    _log_info "total download plan: ${total_human} across ${#cell_plans[@]} cell(s)"
    echo "============================================================"

    if [[ $ASSUME_YES -eq 0 ]]; then
        printf "proceed? [y/N] "
        read -r reply || reply=""
        case "${reply,,}" in
            y|yes) : ;;
            *) _log_info "aborted by user"; exit 0 ;;
        esac
    fi

    # ----- Phase 2: download
    local cell m q sz pc patterns
    for cell in "${cell_plans[@]}"; do
        IFS='|' read -r m q sz pc patterns <<< "$cell"
        _log_info "downloading ${m} :: ${q} ..."
        if [[ -n "$PATTERN_OVERRIDE" ]]; then
            download_one "$m" "$PATTERN_OVERRIDE" || _log_warn "${m} :: ${q} download failed; continuing"
        else
            download_one "$m" "$patterns" || _log_warn "${m} :: ${q} download failed; continuing"
        fi
    done

    # ----- Phase 3: summary
    echo "============================================================"
    _log_info "cache footprint per repo"
    echo "============================================================"
    local repo
    for repo in "${MODELS[@]}"; do
        local folder="${hf_cache}/hub/models--${repo//\//--}"
        if [[ -d "$folder" ]]; then
            local size
            size=$(du -sh "$folder" 2>/dev/null | awk '{print $1}')
            printf '  %-60s  %s\n' "$repo" "${size:-?}"
        fi
    done

    echo "============================================================"
    _log_info "cache root: ${hf_cache}"
    _log_info "ready for spartan/llama-moe-eval.sbatch"
    echo "============================================================"
}

main "$@"
