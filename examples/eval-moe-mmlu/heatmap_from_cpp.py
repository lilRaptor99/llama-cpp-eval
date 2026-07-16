#!/usr/bin/env python3
# type: ignore

"""Render OLMoE expert-routing heatmaps from the C++ `llama-eval-moe-mmlu` JSON.

Reads `expert_counts.json` produced by `llama-eval-moe-mmlu` and writes:
  - `routing_heatmap.png`                  layer x expert overview
  - `routing_heatmap_by_subject.png`       57 subjects x 1024 cells (log1p)
  - `routing_heatmap_by_category.png`      6 categories x 1024 cells + per-layer-pair breakdown (log1p)
  - `routing_heatmap_by_category_normalized.png`  same with rows normalized to 1
  - `counts_total.json`                    aggregated [L, E] matrix (sum across subjects)
  - `metadata.json`                        pass-through + computed totals

Schema of the input JSON (`expert_counts.json`):
  {
    "model": "<hf model id>",
    "model_arch": {"name": "olmoe", "n_layer": 16, "n_expert": 64, "n_expert_used": 8},
    "config":  {"questions_per_subject": 50, "n_shot": 5, "few_shot_pool": ..., "prompt_format": ...},
    "totals":  {"subjects_run": 57, "questions_total": 2850, "tokens_total": 1862535},
    "subjects": {
       "<subject>": {
         "questions": 50,
         "n_tokens": 19460,
         "layer_expert_counts": [[int, int, ...] * 64] * 16  // [L, E] int64
       },
       ...
    }
  }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


TOP_K = 8
SUBJECT_CATEGORIES: dict[str, list[str]] = {
    "STEM": [
        "abstract_algebra", "astronomy", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "world_religions",
    ],
    "Medicine & Health": [
        "anatomy", "clinical_knowledge", "college_biology", "college_medicine",
        "high_school_biology", "human_aging", "human_sexuality",
        "medical_genetics", "nutrition", "professional_medicine", "virology",
    ],
    "Business & Law": [
        "business_ethics", "management", "marketing", "professional_accounting",
        "professional_law",
    ],
    "Other": [
        "global_facts", "miscellaneous",
    ],
}


# --------------------------------------------------------------------------- I/O

def _load_cpp_json(path: Path) -> tuple[dict, np.ndarray, dict[str, np.ndarray], int, dict]:
    """Parse the C++ `expert_counts.json` file.

    Returns (arch, counts_total, subject_counts, total_tokens, metadata).
    """
    with open(path) as f:
        data = json.load(f)

    arch = data.get("model_arch", {})
    L = int(arch.get("n_layer", 16))
    E = int(arch.get("n_expert", 64))

    subjects_raw = data.get("subjects", {})
    if not subjects_raw:
        raise SystemExit(f"[error] no subjects found in {path}")

    subject_counts: dict[str, np.ndarray] = {}
    for subj, body in subjects_raw.items():
        mat = np.asarray(body["layer_expert_counts"], dtype=np.int64)
        if mat.shape != (L, E):
            raise SystemExit(
                f"[error] subject '{subj}' has shape {mat.shape}, expected ({L}, {E})"
            )
        subject_counts[subj] = mat

    counts_total = np.zeros((L, E), dtype=np.int64)
    for mat in subject_counts.values():
        counts_total += mat

    total_tokens = sum(int(v.get("n_tokens", 0)) for v in subjects_raw.values())

    metadata = {
        "model": data.get("model"),
        "model_arch": arch,
        "config": data.get("config", {}),
        "totals": data.get("totals", {}),
        "total_tokens_seen": total_tokens,
        "expected_top_k": TOP_K,
    }
    return arch, counts_total, subject_counts, total_tokens, metadata


def _save_metadata(metadata: dict, counts_total: np.ndarray, path: Path) -> None:
    """Persist a Python-style metadata.json + counts_total.json for downstream tools."""
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(path.parent / "counts_total.json", "w") as f:
        json.dump(counts_total.tolist(), f)


# --------------------------------------------------------------- heatmap renderers

def save_overview_heatmap(
    counts: np.ndarray,
    total_tokens: int,
    path: Path,
    *,
    colormap: str = "viridis",
    dpi: int = 120,
) -> None:
    """Layer x expert heatmap of per-token activation rate."""
    L, E = counts.shape
    per_token = counts.astype(np.float64) / max(total_tokens, 1)

    fig, ax = plt.subplots(figsize=(max(14, E * 0.32), max(4, L * 0.5)))
    im = ax.imshow(per_token, aspect="auto", cmap=colormap)
    ax.set_xlabel("expert id")
    ax.set_ylabel("layer")
    ax.set_title(
        f"OLMoE per-token activation rate (top-{TOP_K} of {E}); "
        f"{total_tokens:,} tokens"
    )
    plt.colorbar(
        im, ax=ax,
        label=f"selections / token (uniform = {TOP_K / E:.4f})",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_subject_heatmap(
    subject_counts: dict[str, np.ndarray],
    subjects: list[str],
    path: Path,
    scale: str = "log",
    *,
    colormap: str = "viridis",
    dpi: int = 120,
) -> bool:
    """Subject x 1024-cell heatmap."""
    if scale not in ("linear", "log"):
        raise ValueError(f"scale must be 'linear' or 'log'; got {scale!r}")

    rows = []
    labels = []
    for s in subjects:
        mat = subject_counts.get(s)
        if mat is None or mat.size == 0 or mat.sum() <= 0:
            continue
        rows.append(mat.flatten())
        labels.append(s)
    if not rows:
        return False

    M = np.stack(rows)
    L, E = next(iter(subject_counts.values())).shape

    M_display = np.log1p(M) if scale == "log" else M.astype(np.float64)
    cbar_label = "log1p(count)" if scale == "log" else "count"

    fig, ax = plt.subplots(figsize=(30, max(8, 0.3 * len(labels))))
    im = ax.imshow(M_display, aspect="auto", cmap=colormap)
    ax.set_xlabel(f"expert cell (layer-major: {E} experts x {L} layers)")
    ax.set_ylabel("subject")
    ax.set_title(
        f"OLMoE expert activations by MMLU subject  -  "
        f"{M.shape[0]} subjects x {M.shape[1]} cells (scale={scale})"
    )
    plt.colorbar(im, ax=ax, label=cbar_label)

    layer_starts = [layer_idx * E for layer_idx in range(L)]
    ax.set_xticks(layer_starts)
    ax.set_xticklabels([f"L{l}" for l in range(L)], rotation=0, fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _aggregate_counts_by_category(
    subject_counts: dict[str, np.ndarray],
    category_map: dict[str, list[str]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Sum per-subject [L, E] matrices within each category.

    Returns (labels, flat, threeD):
      - labels:    category names that survived (non-empty after summing)
      - flat:      [num_categories, L*E] cell-major counts
      - threeD:    [num_categories, L, E] raw counts (for split heatmaps)
    """
    sample = next(iter(subject_counts.values()))
    L, E = sample.shape

    labels: list[str] = []
    flat_rows: list[np.ndarray] = []
    three_d_rows: list[np.ndarray] = []

    covered: set[str] = set()
    for category, members in category_map.items():
        summed = np.zeros((L, E), dtype=np.int64)
        for s in members:
            covered.add(s)
            mat = subject_counts.get(s)
            if mat is None or mat.size == 0:
                continue
            summed += mat.astype(np.int64, copy=False)
        if summed.sum() <= 0:
            continue
        labels.append(category)
        flat_rows.append(summed.reshape(-1))
        three_d_rows.append(summed)

    unmapped = sorted(set(subject_counts.keys()) - covered)
    for s in unmapped:
        print(f"[warn] subject '{s}' is not assigned to any category; ignoring", file=sys.stderr)

    flat = np.stack(flat_rows) if flat_rows else np.zeros((0, L * E), dtype=np.int64)
    three_d = np.stack(three_d_rows) if three_d_rows else np.zeros((0, L, E), dtype=np.int64)
    return labels, flat, three_d


def _draw_category_heatmap(
    M_3d_disp: np.ndarray,
    labels: list[str],
    L: int,
    E: int,
    cbar_label: str,
    title_prefix: str,
    path: Path,
    *,
    colormap: str = "viridis",
    dpi: int = 120,
    layers_per_subplot: int = 2,
) -> bool:
    """Shared renderer for category heatmaps with overall + per-layer-pair panels."""
    if not labels:
        return False

    M_flat_disp = M_3d_disp.reshape(M_3d_disp.shape[0], -1)

    n_groups = L // layers_per_subplot
    if n_groups == 0 or L % layers_per_subplot != 0:
        raise ValueError(
            f"num_layers={L} not divisible by layers_per_subplot={layers_per_subplot}"
        )
    layer_pairs = [
        (i * layers_per_subplot, (i + 1) * layers_per_subplot - 1)
        for i in range(n_groups)
    ]

    n_cols = 4
    n_subplot_rows = (n_groups + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(32, 8 + 6 * n_subplot_rows))
    height_ratios = [3] + [1.5] * n_subplot_rows
    gs = fig.add_gridspec(
        1 + n_subplot_rows, n_cols,
        height_ratios=height_ratios,
        hspace=0.5, wspace=0.3,
    )

    ax_overall = fig.add_subplot(gs[0, :])
    im_overall = ax_overall.imshow(M_flat_disp, aspect="auto", cmap=colormap)
    ax_overall.set_xlabel(f"expert cell (layer-major: {E} experts x {L} layers)")
    ax_overall.set_ylabel("category")
    ax_overall.set_title(
        f"All layers (overall)  -  {len(labels)} categories x "
        f"{M_flat_disp.shape[1]} cells"
    )
    plt.colorbar(im_overall, ax=ax_overall, label=cbar_label)

    layer_starts = [layer_idx * E for layer_idx in range(L)]
    ax_overall.set_xticks(layer_starts)
    ax_overall.set_xticklabels(
        [f"L{l}" for l in range(L)], rotation=0, fontsize=8
    )
    ax_overall.set_yticks(range(len(labels)))
    ax_overall.set_yticklabels(labels, fontsize=9)

    for i, (lo, hi) in enumerate(layer_pairs):
        row = 1 + i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])

        M_pair = M_3d_disp[:, lo:hi + 1, :].reshape(M_3d_disp.shape[0], -1)
        im = ax.imshow(M_pair, aspect="auto", cmap=colormap)
        ax.set_title(f"Layers {lo}-{hi}")
        ax.set_xlabel("expert cell")
        ax.set_ylabel("category")
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.05, pad=0.04)

        n_pair_layers = hi - lo + 1
        pair_layer_starts = [j * E for j in range(n_pair_layers)]
        ax.set_xticks(pair_layer_starts)
        ax.set_xticklabels(
            [f"L{lo + j}" for j in range(n_pair_layers)],
            rotation=0, fontsize=7,
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)

    fig.suptitle(title_prefix, fontsize=14)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def save_category_heatmap(
    subject_counts: dict[str, np.ndarray],
    category_map: dict[str, list[str]],
    path: Path,
    scale: str = "log",
    *,
    colormap: str = "viridis",
    dpi: int = 120,
) -> bool:
    """Raw category heatmap (overall + per-layer-pair breakdown)."""
    labels, _flat, M_3d = _aggregate_counts_by_category(subject_counts, category_map)
    if not labels:
        return False

    L, E = M_3d.shape[1:]
    M_3d_disp = np.log1p(M_3d) if scale == "log" else M_3d.astype(np.float64)
    cbar_label = "log1p(count)" if scale == "log" else "count"

    return _draw_category_heatmap(
        M_3d_disp, labels, L, E, cbar_label,
        title_prefix=(
            f"OLMoE expert activations by MMLU subject category  -  "
            f"{len(labels)} categories (overall + per-layer-pair breakdown, scale={scale})"
        ),
        path=path, colormap=colormap, dpi=dpi,
    )


def save_category_heatmap_normalized(
    subject_counts: dict[str, np.ndarray],
    category_map: dict[str, list[str]],
    path: Path,
    *,
    colormap: str = "viridis",
    dpi: int = 120,
) -> bool:
    """Row-normalized category heatmap (overall + per-layer-pair breakdown)."""
    labels, flat, M_3d = _aggregate_counts_by_category(subject_counts, category_map)
    if not labels:
        return False

    L, E = M_3d.shape[1:]
    row_sums = flat.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1, row_sums)
    flat_norm = flat.astype(np.float64) / safe
    M_3d_norm = flat_norm.reshape(flat_norm.shape[0], L, E)

    return _draw_category_heatmap(
        M_3d_norm, labels, L, E,
        cbar_label="fraction of category's selections",
        title_prefix=(
            "OLMoE expert activations by MMLU subject category "
            "(row-normalized)  -  overall + per-layer-pair breakdown"
        ),
        path=path, colormap=colormap, dpi=dpi,
    )


# ----------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Path to expert_counts.json produced by llama-eval-moe-mmlu.",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Where to write heatmaps and JSON. Defaults to the input file's directory.",
    )
    parser.add_argument(
        "--heatmap",
        choices=("overview", "subjects", "categories", "normalized", "all"),
        default="all",
        help="Which heatmap(s) to produce.",
    )
    parser.add_argument(
        "--scale", choices=("linear", "log"), default="log",
        help="Color scale for the subject / category heatmaps (raw counts or log1p).",
    )
    parser.add_argument("--colormap", type=str, default="viridis",
                        help="Matplotlib colormap name.")
    parser.add_argument("--dpi", type=int, default=120,
                        help="Output PNG DPI.")
    args = parser.parse_args()

    input_path: Path = args.input
    output_dir: Path = args.output_dir if args.output_dir is not None else input_path.parent

    print(f"[load] input = {input_path.resolve()}")
    arch, counts_total, subject_counts, total_tokens, metadata = _load_cpp_json(input_path)
    L = int(arch.get("n_layer", counts_total.shape[0]))
    E = int(arch.get("n_expert", counts_total.shape[1]))
    subjects = sorted(subject_counts.keys())
    print(
        f"[load] arch={arch.get('name')} L={L} E={E}, "
        f"subjects={len(subjects)}, total_tokens={total_tokens:,}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_metadata(metadata, counts_total, output_dir / "metadata.json")
    print(f"[save] {output_dir.resolve() / 'metadata.json'}")
    print(f"[save] {output_dir.resolve() / 'counts_total.json'}")

    if args.heatmap in ("overview", "all"):
        p = output_dir / "routing_heatmap.png"
        save_overview_heatmap(
            counts_total, total_tokens, p,
            colormap=args.colormap, dpi=args.dpi,
        )
        print(f"[save] {p.resolve()}")

    if args.heatmap in ("subjects", "all"):
        p = output_dir / "routing_heatmap_by_subject.png"
        wrote = save_subject_heatmap(
            subject_counts, subjects, p, args.scale,
            colormap=args.colormap, dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()} (scale={args.scale})")
        else:
            print(f"[skip] no subjects with activations; {p.name} not written")

    if args.heatmap in ("categories", "all"):
        p = output_dir / "routing_heatmap_by_category.png"
        wrote = save_category_heatmap(
            subject_counts, SUBJECT_CATEGORIES, p, args.scale,
            colormap=args.colormap, dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()} (scale={args.scale})")
        else:
            print(f"[skip] no categories with activations; {p.name} not written")

    if args.heatmap in ("normalized", "all"):
        p = output_dir / "routing_heatmap_by_category_normalized.png"
        wrote = save_category_heatmap_normalized(
            subject_counts, SUBJECT_CATEGORIES, p,
            colormap=args.colormap, dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()} (row-normalized)")
        else:
            print(f"[skip] no categories with activations; {p.name} not written")

    print(f"[done] output_dir = {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())