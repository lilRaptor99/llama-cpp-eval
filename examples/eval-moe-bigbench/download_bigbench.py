#!/usr/bin/env python3
# type: ignore

"""Download the maveriq/bigbenchhard dataset and emit a consolidated JSONL
plus a tasks list, ready to be consumed by llama-eval-moe-bigbench.

`maveriq/bigbenchhard` exposes 27 configurations from Suzgun et al. (2022)
"Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them".
Three canonical BBH task families are split by object count, so the HF
dataset exposes 27 configs where the paper calls them 23 tasks:

    - logical_deduction_three_objects / _five_objects / _seven_objects
    - tracking_shuffled_objects_three_objects / _five_objects / _seven_objects

Each row from `load_dataset(..., split="train")` exposes just two fields:

    input:   str  - the prompt / question
    target:  str  - the gold answer (heterogeneous: choice letter, boolean,
                     integer, sequence, free-form ordered list, ...)

We keep both fields as-is and assign a stable per-task `index` so that
downstream tooling can reference examples deterministically. We do NOT
introduce a held-out split: the dataset has only `train`, and the BBH
benchmark is evaluated zero-shot over that same data.

Outputs (default under build/moe-bigbench/, override with --outdir):
  - bigbench.jsonl   one row per question:
                     {"task", "index", "input", "target"}
  - tasks.txt        one task config per line (canonical 27-config order)
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path


# Canonical 27 maveriq/bigbenchhard configurations in alphabetical order.
# This matches the order Hugging Face exposes via `builder_configs` and the
# dataset's README. Treat it as authoritative - both the downloader and the
# C++ evaluator walk this list to keep JSONL, tasks.txt, and routing
# matrices in a stable order.
BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

# Canonical upstream location for the per-task JSON files served by the
# Hugging Face dataset builder. We fall back to direct HTTPS download when
# `datasets>=4` refuses to execute the repo's generator script.
BBH_UPSTREAM_URL = (
    "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{task}.json"
)


def _fetch_via_github(task: str, dest: Path) -> Path:
    """Download one task's BBH JSON from the canonical upstream GitHub URL.

    Returns the local path on success; raises OSError on failure so the
    caller can record the task as skipped.
    """
    url = BBH_UPSTREAM_URL.format(task=task)
    out = dest / f"{task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    if not data:
        raise OSError(f"empty response from {url}")
    out.write_bytes(data)
    return out


def _load_one_task(task: str, cache_dir: Path):
    """Yield (input, target) tuples for a single BBH task.

    Tries the Hugging Face dataset API first; on the modern 'no dataset
    scripts' failure, falls back to direct GitHub download of the
    canonical JSON. Returns the rows list (or raises on total failure).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("maveriq/bigbenchhard", task, split="train")
        return list(ds), "huggingface"
    except Exception as hf_err:
        # Modern `datasets` (>=4) refuses to execute repo scripts. Fall back
        # to the canonical upstream JSON that the builder script would have
        # downloaded anyway. This avoids any new dependency on `requests`
        # by using urllib.request from the stdlib.
        try:
            path = _fetch_via_github(task, cache_dir)
        except Exception as http_err:
            raise RuntimeError(
                f"both HF ({hf_err}) and GitHub fallback ({http_err}) failed"
            )
        with open(path) as f:
            payload = json.load(f)
        rows = payload.get("examples")
        if not isinstance(rows, list):
            raise RuntimeError(f"upstream JSON for {task} has no 'examples' list")
        out = [{"input": r["input"], "target": r["target"]} for r in rows]
        return out, "github"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parents[2] / "build" / "moe-bigbench"),
        help="output directory for bigbench.jsonl and tasks.txt",
    )
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=(
            "restrict to these task configs only (default: all 27). "
            "Use exact config names from maveriq/bigbenchhard, e.g. "
            "boolean_expressions, dyck_languages, ..."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="keep at most N rows per task (0 = no limit).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="seed reserved for future subsampling variants (unused today).",
    )
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.tasks is None:
        tasks = list(BBH_TASKS)
    else:
        unknown = [t for t in args.tasks if t not in BBH_TASKS]
        if unknown:
            print(
                f"[bigbench] error: unknown task(s): {', '.join(unknown)}",
                file=sys.stderr,
            )
            print(
                "[bigbench] valid configs are:\n  "
                + "\n  ".join(BBH_TASKS),
                file=sys.stderr,
            )
            return 1
        # Preserve user-requested ordering; it is what downstream tooling
        # will iterate over.
        tasks = list(args.tasks)

    if args.limit < 0:
        print(
            f"[bigbench] error: --limit must be >= 0 (got {args.limit})",
            file=sys.stderr,
        )
        return 1

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "error: the 'datasets' package is required.\n"
            "  pip install -r requirements/requirements-server-bench.txt",
            file=sys.stderr,
        )
        return 1

    print(
        f"[bigbench] loading maveriq/bigbenchhard ({len(tasks)} task(s)) into {outdir}",
        flush=True,
    )

    out_jsonl = outdir / "bigbench.jsonl"
    cache_dir = outdir / "_cache"
    tasks_seen: list[str] = []
    n_kept = 0
    n_skipped = 0
    failures: list[str] = []
    sources: dict[str, str] = {}

    with out_jsonl.open("w") as f:
        for task in tasks:
            try:
                rows, source = _load_one_task(task, cache_dir)
            except Exception as e:
                msg = f"[bigbench] skipping {task}: {e}"
                print(msg, file=sys.stderr, flush=True)
                failures.append(task)
                continue
            sources[task] = source

            kept_this_task = 0
            skipped_this_task = 0
            n_rows = len(rows)
            # The upstream JSON is already in canonical order, so a simple
            # prefix truncation implements `--limit` deterministically.
            for i, row in enumerate(rows):
                if args.limit > 0 and i >= args.limit:
                    break
                inp = row.get("input")
                tgt = row.get("target")
                if inp is None or tgt is None:
                    skipped_this_task += 1
                    continue
                inp = str(inp).strip()
                tgt = str(tgt).strip()
                if not inp or not tgt:
                    skipped_this_task += 1
                    continue
                obj = {
                    "task":   task,
                    "index":  i,
                    "input":  inp,
                    "target": tgt,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept_this_task += 1
                n_kept += 1
            n_skipped += skipped_this_task
            if task not in tasks_seen:
                tasks_seen.append(task)
            print(
                f"[bigbench]   {task}: kept {kept_this_task} / "
                f"{n_rows} rows via {source} (skipped {skipped_this_task})",
                flush=True,
            )

    (outdir / "tasks.txt").write_text("\n".join(tasks_seen) + "\n")

    # Record per-task provenance so users know whether the HF builder ran
    # or the GitHub fallback handled this task.
    (outdir / "sources.json").write_text(
        json.dumps({"dataset": "maveriq/bigbenchhard", "split": "train",
                    "sources": sources}, indent=2) + "\n"
    )

    print(
        f"[bigbench] wrote {n_kept} rows to {out_jsonl} (skipped {n_skipped})"
    )
    print(
        f"[bigbench] wrote {len(tasks_seen)} task(s) to {outdir / 'tasks.txt'}"
    )
    print(
        f"[bigbench] wrote source map to {outdir / 'sources.json'}"
    )
    if failures:
        print(
            f"[bigbench] failed to load {len(failures)} task(s): "
            f"{', '.join(failures)}",
            file=sys.stderr,
        )
        return 1
    if n_kept == 0:
        print("[bigbench] error: no rows emitted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
