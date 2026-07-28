#!/usr/bin/env python3
# type: ignore

"""Download the openai/openai_humaneval dataset and emit a consolidated JSONL
plus a tasks list, ready to be consumed by llama-eval-moe-humaneval.

`openai/openai_humaneval` ships a single `test` split (no train/dev/val,
no leakage guard; the dataset was hand-curated by OpenAI). Each row exposes:

    task_id, prompt, canonical_solution, test, entry_point

We persist all five fields as-is in the JSONL: the C++ binary only reads
`task_id`, `prompt`, and `entry_point` for routing capture, but downstream
scoring tools (or pass@1 re-evaluators) may want the canonical solution or
the `check(candidate)` test. All fields are kept verbatim.

Outputs (default under build/moe-humaneval/, override with --outdir):
  - humaneval.jsonl    one row per problem:
                       {"task_id", "prompt", "canonical_solution",
                        "test", "entry_point"}
  - tasks.txt          one task_id per line, in upstream order

`humaneval.jsonl` is consumed by the C++ evaluator. `tasks.txt` is the
authoritative list of `task_id`s in the order they will be evaluated, so
both files must stay in lock-step (we write them deterministically together).
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parents[2] / "build" / "moe-humaneval"),
        help="output directory for humaneval.jsonl and tasks.txt",
    )
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=(
            "restrict to these task_id values only (default: all 164). "
            "Use exact upstream ids, e.g. 'HumanEval/0' or 'test/0' "
            "depending on the revision."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="keep at most N rows total (0 = no limit).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed reserved for future subsampling variants (unused today).",
    )
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.limit < 0:
        print(
            f"[humaneval] error: --limit must be >= 0 (got {args.limit})",
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

    print(f"[humaneval] loading openai/openai_humaneval into {outdir}", flush=True)
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception as e:
        print(f"[humaneval] failed to load dataset: {e}", file=sys.stderr)
        return 1

    print(f"[humaneval]   total rows: {len(ds)}", flush=True)

    # --tasks filter
    allow = set(args.tasks) if args.tasks else None

    # Always write rows in upstream order. HumanEval ships a single canonical
    # `test` split with rows already in a stable order, so a simple
    # prefix-truncation implements `--limit` deterministically without any
    # extra seeding or shuffling.
    out_jsonl = outdir / "humaneval.jsonl"
    tasks_seen: list[str] = []
    n_kept = 0
    n_skipped = 0
    n_filtered = 0

    with out_jsonl.open("w") as f:
        for i, row in enumerate(ds):
            if args.limit > 0 and i >= args.limit:
                break

            task_id = str(row.get("task_id") or "")
            if allow and task_id not in allow:
                n_filtered += 1
                continue

            prompt = row.get("prompt")
            canonical_solution = row.get("canonical_solution")
            test = row.get("test")
            entry_point = row.get("entry_point")
            if prompt is None or entry_point is None or not task_id:
                n_skipped += 1
                continue

            obj = {
                "task_id":            task_id,
                "prompt":             str(prompt),
                "canonical_solution": str(canonical_solution) if canonical_solution is not None else "",
                "test":               str(test) if test is not None else "",
                "entry_point":        str(entry_point),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            tasks_seen.append(task_id)
            n_kept += 1

    (outdir / "tasks.txt").write_text("\n".join(tasks_seen) + "\n")

    print(
        f"[humaneval] wrote {n_kept} rows to {out_jsonl} "
        f"(filtered {n_filtered}, skipped {n_skipped})"
    )
    print(
        f"[humaneval] wrote {len(tasks_seen)} task_id(s) to "
        f"{outdir / 'tasks.txt'}"
    )
    if n_kept == 0:
        print("[humaneval] error: no rows emitted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())