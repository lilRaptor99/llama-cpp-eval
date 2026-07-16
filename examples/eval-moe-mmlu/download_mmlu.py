#!/usr/bin/env python3
# type: ignore

"""Download the cais/mmlu dataset (per-subject) and emit a consolidated JSONL
plus a subjects list, ready to be consumed by llama-eval-moe-mmlu.

Outputs (default under build/moe-mmlu/, override with --outdir):
  - mmlu.jsonl     one row per question:
                   {"split": "dev"|"test", "subject": "...", "question": "...",
                    "choices": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": 0..3}
  - subjects.txt   one subject per line (alphabetical, matches MMLU canonical order)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Canonical 57 MMLU subjects in alphabetical order (matches cais/mmlu configs).
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def letter(i: int) -> str:
    return chr(ord("A") + i)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parents[2] / "build" / "moe-mmlu"),
        help="output directory for mmlu.jsonl and subjects.txt",
    )
    p.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help=f"override the 57-subject list (default: all of cais/mmlu). Use subject names from: {', '.join(MMLU_SUBJECTS[:3])}, ...",
    )
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects if args.subjects else MMLU_SUBJECTS
    print(f"[mmlu] downloading {len(subjects)} subject(s) into {outdir}", flush=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "error: the 'datasets' package is required.\n"
            "  pip install -r requirements/requirements-server-bench.txt",
            file=sys.stderr,
        )
        return 1

    out_jsonl = outdir / "mmlu.jsonl"
    n_dev = n_test = 0
    with out_jsonl.open("w") as f:
        for subj in subjects:
            try:
                ds = load_dataset("cais/mmlu", subj)
            except Exception as e:
                print(f"[mmlu] skipping {subj}: {e}", file=sys.stderr, flush=True)
                continue

            for split_name in ("dev", "test"):
                if split_name not in ds:
                    continue
                for row in ds[split_name]:
                    choices = list(row["choices"])
                    labeled = [f"{letter(i)}. {c}" for i, c in enumerate(choices)]
                    obj = {
                        "split": split_name,
                        "subject": subj,
                        "question": row["question"],
                        "choices": labeled,
                        "answer": int(row["answer"]),
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    if split_name == "dev":
                        n_dev += 1
                    else:
                        n_test += 1
            print(f"[mmlu]   {subj}: {len(ds['dev'])} dev / {len(ds['test'])} test", flush=True)

    (outdir / "subjects.txt").write_text("\n".join(subjects) + "\n")

    print(f"[mmlu] wrote {n_dev} dev + {n_test} test rows to {out_jsonl}")
    print(f"[mmlu] wrote subjects list to {outdir / 'subjects.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
