#!/usr/bin/env python3
# type: ignore

"""Render MoE expert-routing heatmaps from the C++ `llama-eval-moe-include`
JSON.

Reads `expert_counts.json` produced by `llama-eval-moe-include` and writes:
  - `routing_heatmap.png`                          layer x expert overview
  - `routing_heatmap_by_language.png`              ~44 languages x (L*E) cells (log1p)
  - `routing_heatmap_by_language_normalized.png`   ~44 languages x (L*E) cells (row-normalized)
  - `routing_heatmap_by_domain.png`                ~11 domains x (L*E) cells + per-layer-pair
  - `routing_heatmap_by_langdom.png`               ~484 keys x (L*E) cells (log1p)
  - `routing_heatmap_by_langdom_normalized.png`    ~484 keys x (L*E) cells (row-normalized)
  - `accuracy_by_langdom.png`                      per-(lang,dom) substring-match accuracy
  - `counts_total.json`                            aggregated [L, E] matrix (sum across langdoms)
  - `metadata.json`                                pass-through + computed totals

Schema of the input JSON (`expert_counts.json`):
  {
    "model": "<hf model id>",
    "model_arch": {"name": "olmoe", "n_layer": 16, "n_expert": 64, "n_expert_used": 8},
    "config":  {"questions_per_langdom": 5, "n_shot": 5, "gen_tokens": 16,
                "prompt_format": "few_shot_inlang_5shot",
                "match_metric": "substring_normalized_first_letter"},
    "dataset": "CohereLabs/include-base-44",
    "totals":  {"langdoms_run": 200, "languages_run": 38,
                "questions_total": 1000, "tokens_total_prefill": ...,
                "tokens_total_generated": ..., "correct": 412, "accuracy": 0.412},
    "by_langdom": {
       "<language>::<domain>": {
         "language": "<language>", "domain": "<domain>",
         "questions": 5, "n_tokens_prefill": ..., "n_tokens_generated": ...,
         "n_correct": 3, "match_rate": 0.6,
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
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


TOP_K = 8


# --------------------------------------------------------------------------- I/O

def _load_cpp_json(path: Path) -> tuple[dict, np.ndarray, dict[str, np.ndarray],
                                         dict[str, np.ndarray], dict[str, np.ndarray],
                                         int, dict]:
    """Parse the C++ `expert_counts.json` file.

    Returns (arch, counts_total, langdom_counts, language_counts, domain_counts,
             total_tokens, metadata).
    """
    with open(path) as f:
        data = json.load(f)

    arch = data.get("model_arch", {})
    L = int(arch.get("n_layer", 16))
    E = int(arch.get("n_expert", 64))

    raw = data.get("by_langdom", {})
    if not raw:
        raise SystemExit(f"[error] no by_langdom entries found in {path}")

    langdom_counts: dict[str, np.ndarray] = {}
    for key, body in raw.items():
        mat = np.asarray(body["layer_expert_counts"], dtype=np.int64)
        if mat.shape != (L, E):
            raise SystemExit(
                f"[error] langdom '{key}' has shape {mat.shape}, expected ({L}, {E})"
            )
        langdom_counts[key] = mat

    # Aggregate per-language (sum across domains within a language) and
    # per-domain (sum across languages within a domain). Matrices are
    # summed on the fly — these aggregations are cheap ([L, E] int64 add).
    language_counts: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((L, E), dtype=np.int64)
    )
    domain_counts: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((L, E), dtype=np.int64)
    )
    for key, mat in langdom_counts.items():
        language = str(raw[key].get("language", key.split("::", 1)[0]))
        domain   = str(raw[key].get("domain",   key.split("::", 1)[1] if "::" in key else "Unknown"))
        language_counts[language] += mat
        domain_counts[domain]     += mat

    counts_total = np.zeros((L, E), dtype=np.int64)
    for mat in langdom_counts.values():
        counts_total += mat

    total_tokens = sum(
        int(v.get("n_tokens_prefill", 0)) + int(v.get("n_tokens_generated", 0))
        for v in raw.values()
    )

    metadata = {
        "model": data.get("model"),
        "model_arch": arch,
        "config": data.get("config", {}),
        "dataset": data.get("dataset", ""),
        "totals": data.get("totals", {}),
        "total_tokens_seen": total_tokens,
        "expected_top_k": TOP_K,
    }
    return (arch, counts_total, langdom_counts,
            dict(language_counts), dict(domain_counts),
            total_tokens, metadata)


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
        f"INCLUDE MoE per-token activation rate (top-{TOP_K} of {E}); "
        f"{total_tokens:,} tokens"
    )
    plt.colorbar(
        im, ax=ax,
        label=f"selections / token (uniform = {TOP_K / E:.4f})",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_language_heatmap(
    language_counts: dict[str, np.ndarray],
    languages: list[str],
    path: Path,
    scale: str = "log",
    *,
    colormap: str = "viridis",
    dpi: int = 120,
) -> bool:
    """Language x (L*E) cells heatmap (sum across domains per language)."""
    if scale not in ("linear", "log"):
        raise ValueError(f"scale must be 'linear' or 'log'; got {scale!r}")

    rows:    list[np.ndarray] = []
    labels:  list[str]       = []
    for lang in languages:
        mat = language_counts.get(lang)
        if mat is None or mat.size == 0 or mat.sum() <= 0:
            continue
        rows.append(mat.flatten())
        labels.append(lang)
    if not rows:
        return False

    M = np.stack(rows)
    L, E = next(iter(language_counts.values())).shape

    M_display = np.log1p(M) if scale == "log" else M.astype(np.float64)
    cbar_label = "log1p(count)" if scale == "log" else "count"

    fig, ax = plt.subplots(figsize=(30, max(8, 0.35 * len(labels))))
    im = ax.imshow(M_display, aspect="auto", cmap=colormap)
    ax.set_xlabel(f"expert cell (layer-major: {E} experts x {L} layers)")
    ax.set_ylabel("language")
    ax.set_title(
        f"MoE expert activations by INCLUDE language  -  "
        f"{M.shape[0]} languages x {M.shape[1]} cells (scale={scale})"
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


def save_domain_heatmap(
    domain_counts: dict[str, np.ndarray],
    domains: list[str],
    path: Path,
    scale: str = "log",
    *,
    colormap: str = "viridis",
    dpi: int = 120,
) -> bool:
    """Domain x (L*E) heatmap with per-layer-pair breakdown panels.

    Mirrors the MMLU subject-category renderer.
    """
    if scale not in ("linear", "log"):
        raise ValueError(f"scale must be 'linear' or 'log'; got {scale!r}")

    rows:    list[np.ndarray] = []
    labels:  list[str]       = []
    for dom in domains:
        mat = domain_counts.get(dom)
        if mat is None or mat.size == 0 or mat.sum() <= 0:
            continue
        rows.append(mat.flatten())
        labels.append(dom)
    if not rows:
        return False

    M_flat = np.stack(rows)
    sample = next(iter(domain_counts.values()))
    L, E = sample.shape

    M_3d = np.stack([
        np.asarray(domain_counts[d], dtype=np.int64).reshape(L, E)
        for d in labels
    ])
    M_3d_disp = np.log1p(M_3d) if scale == "log" else M_3d.astype(np.float64)
    cbar_label = "log1p(count)" if scale == "log" else "count"

    return _draw_category_heatmap(
        M_3d_disp, labels, L, E, cbar_label,
        title_prefix=(
            f"MoE expert activations by INCLUDE domain  -  "
            f"{len(labels)} domains x {L * E} cells (overall + per-layer-pair, scale={scale})"
        ),
        path=path, colormap=colormap, dpi=dpi,
    )


def save_langdom_heatmap(
    langdom_counts: dict[str, np.ndarray],
    langdom_keys: list[str],
    path: Path,
    scale: str = "log",
    *,
    colormap: str = "viridis",
    dpi: int = 120,
) -> bool:
    """~484 (language, domain) keys x (L*E) cells heatmap (log1p).

    Wide-but-short figure; no per-layer-pair breakdown (too tall).
    """
    if scale not in ("linear", "log"):
        raise ValueError(f"scale must be 'linear' or 'log'; got {scale!r}")

    rows:   list[np.ndarray] = []
    labels: list[str]       = []
    for key in langdom_keys:
        mat = langdom_counts.get(key)
        if mat is None or mat.size == 0 or mat.sum() <= 0:
            continue
        rows.append(mat.flatten())
        labels.append(key)
    if not rows:
        return False

    M = np.stack(rows)
    L, E = next(iter(langdom_counts.values())).shape
    M_display = np.log1p(M) if scale == "log" else M.astype(np.float64)
    cbar_label = "log1p(count)" if scale == "log" else "count"

    fig, ax = plt.subplots(figsize=(30, max(10, 0.15 * len(labels))))
    im = ax.imshow(M_display, aspect="auto", cmap=colormap)
    ax.set_xlabel(f"expert cell (layer-major: {E} experts x {L} layers)")
    ax.set_ylabel("(language, domain) bucket")
    ax.set_title(
        f"MoE expert activations by INCLUDE (language, domain)  -  "
        f"{M.shape[0]} buckets x {M.shape[1]} cells (scale={scale})"
    )
    plt.colorbar(im, ax=ax, label=cbar_label)

    layer_starts = [layer_idx * E for layer_idx in range(L)]
    ax.set_xticks(layer_starts)
    ax.set_xticklabels([f"L{l}" for l in range(L)], rotation=0, fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def save_langdom_heatmap_normalized(
    langdom_counts: dict[str, np.ndarray],
    langdom_keys: list[str],
    path: Path,
    *,
    colormap: str = "viridis",
    dpi: int = 120,
) -> bool:
    """Row-normalized langdom heatmap (overall flat; no per-layer-pair)."""
    rows:   list[np.ndarray] = []
    labels: list[str]       = []
    for key in langdom_keys:
        mat = langdom_counts.get(key)
        if mat is None or mat.size == 0 or mat.sum() <= 0:
            continue
        rows.append(mat.flatten().astype(np.float64))
        labels.append(key)
    if not rows:
        return False

    M = np.stack(rows)
    row_sums = M.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1, row_sums)
    M_norm = M / safe
    L, E = next(iter(langdom_counts.values())).shape

    fig, ax = plt.subplots(figsize=(30, max(10, 0.15 * len(labels))))
    im = ax.imshow(M_norm, aspect="auto", cmap=colormap)
    ax.set_xlabel(f"expert cell (layer-major: {E} experts x {L} layers)")
    ax.set_ylabel("(language, domain) bucket")
    ax.set_title(
        f"MoE expert activations by INCLUDE (language, domain) (row-normalized)  -  "
        f"{M.shape[0]} buckets x {M.shape[1]} cells"
    )
    plt.colorbar(im, ax=ax, label="fraction of bucket's selections")

    layer_starts = [layer_idx * E for layer_idx in range(L)]
    ax.set_xticks(layer_starts)
    ax.set_xticklabels([f"L{l}" for l in range(L)], rotation=0, fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def save_language_heatmap_normalized(
    language_counts: dict[str, np.ndarray],
    languages: list[str],
    path: Path,
    *,
    colormap: str = "viridis",
    dpi: int = 120,
) -> bool:
    """Language x (L*E) cells heatmap, row-normalized across all domains.

    Each row (language) sums to 1, so the colour scale surfaces the
    per-language expert-mix shape independent of token volume. The
    per-language matrix is built once in `_load_cpp_json` by summing the
    per-(lang, dom) matrices across all domains within each language.
    """
    rows:   list[np.ndarray] = []
    labels: list[str]       = []
    for lang in languages:
        mat = language_counts.get(lang)
        if mat is None or mat.size == 0 or mat.sum() <= 0:
            continue
        rows.append(mat.flatten().astype(np.float64))
        labels.append(lang)
    if not rows:
        return False

    M = np.stack(rows)
    row_sums = M.sum(axis=1, keepdims=True)
    safe     = np.where(row_sums == 0, 1, row_sums)
    M_norm   = M / safe
    L, E     = next(iter(language_counts.values())).shape

    fig, ax = plt.subplots(figsize=(30, max(8, 0.35 * len(labels))))
    im = ax.imshow(M_norm, aspect="auto", cmap=colormap)
    ax.set_xlabel(f"expert cell (layer-major: {E} experts x {L} layers)")
    ax.set_ylabel("language")
    ax.set_title(
        f"MoE expert activations by INCLUDE language (row-normalized)  -  "
        f"{M.shape[0]} languages x {M.shape[1]} cells"
    )
    plt.colorbar(im, ax=ax, label="fraction of language's selections")

    layer_starts = [layer_idx * E for layer_idx in range(L)]
    ax.set_xticks(layer_starts)
    ax.set_xticklabels([f"L{l}" for l in range(L)], rotation=0, fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def save_accuracy_chart(
    langdom_keys: list[str],
    raw_by_langdom: dict,
    path: Path,
    *,
    dpi: int = 120,
) -> bool:
    """Horizontal bar chart of match_rate per (lang, dom), sorted desc."""
    rows = []
    for key in langdom_keys:
        body = raw_by_langdom.get(key)
        if body is None:
            continue
        n_q = int(body.get("questions", 0))
        if n_q <= 0:
            continue
        n_c = int(body.get("n_correct", 0))
        mr  = float(body.get("match_rate", 0.0))
        rows.append((key, n_q, n_c, mr))
    if not rows:
        return False

    rows.sort(key=lambda t: t[3], reverse=True)
    labels = [r[0] for r in rows]
    rates  = [r[3] for r in rows]
    n_qs   = [r[1] for r in rows]
    n_ok   = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(12, max(6, 0.18 * len(labels))))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, rates, color="#3a7ca5", edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("match_rate (substring, first non-empty line, normalized)")
    ax.set_xlim(0.0, 1.0)
    overall_correct = sum(int(b.get("n_correct", 0)) for b in raw_by_langdom.values())
    overall_total   = sum(int(b.get("questions", 0)) for b in raw_by_langdom.values())
    overall_acc     = overall_correct / max(1, overall_total)
    ax.set_title(
        f"INCLUDE match_rate by (language, domain)  -  "
        f"{len(labels)} buckets, overall accuracy = {overall_acc:.3f}"
        f" ({overall_correct}/{overall_total})"
    )
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    for bar, q, k in zip(bars, n_qs, n_ok):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{k}/{q}",
            va="center", fontsize=6,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


# Reused from MMLU heatmap_from_cpp.py (verbatim). Renders an overall panel
# plus per-layer-pair subplots using a shared grid spec.
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
    ax_overall.set_ylabel("domain")
    ax_overall.set_title(
        f"All layers (overall)  -  {len(labels)} domains x "
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
        ax.set_ylabel("domain")
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


# ----------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Path to expert_counts.json produced by llama-eval-moe-include.",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Where to write heatmaps and JSON. Defaults to the input file's directory.",
    )
    parser.add_argument(
        "--heatmap",
        choices=("overview", "languages", "languages-norm", "domains",
                 "langdoms", "normalized", "accuracy", "all"),
        default="all",
        help="Which plot(s) to produce.",
    )
    parser.add_argument(
        "--scale", choices=("linear", "log"), default="log",
        help="Color scale for the by-* heatmaps (raw counts or log1p).",
    )
    parser.add_argument("--colormap", type=str, default="viridis",
                        help="Matplotlib colormap name.")
    parser.add_argument("--dpi", type=int, default=120,
                        help="Output PNG DPI.")
    args = parser.parse_args()

    input_path: Path = args.input
    output_dir: Path = args.output_dir if args.output_dir is not None else input_path.parent

    print(f"[load] input = {input_path.resolve()}")
    (arch, counts_total, langdom_counts,
     language_counts, domain_counts,
     total_tokens, metadata) = _load_cpp_json(input_path)
    L = int(arch.get("n_layer", counts_total.shape[0]))
    E = int(arch.get("n_expert", counts_total.shape[1]))
    # Preserve original `by_langdom` ordering (sorted by source JSON) for viz.
    with open(input_path) as f:
        _raw = json.load(f)
    langdom_keys = list(_raw.get("by_langdom", {}).keys())
    languages    = sorted(language_counts.keys())
    domains      = sorted(domain_counts.keys())
    print(
        f"[load] arch={arch.get('name')} L={L} E={E}, "
        f"langdoms={len(langdom_keys)}, languages={len(languages)}, "
        f"domains={len(domains)}, total_tokens={total_tokens:,}"
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

    if args.heatmap in ("languages", "all"):
        p = output_dir / "routing_heatmap_by_language.png"
        wrote = save_language_heatmap(
            language_counts, languages, p, args.scale,
            colormap=args.colormap, dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()} (scale={args.scale})")
        else:
            print(f"[skip] no languages with activations; {p.name} not written")

    if args.heatmap in ("languages-norm", "all"):
        p = output_dir / "routing_heatmap_by_language_normalized.png"
        wrote = save_language_heatmap_normalized(
            language_counts, languages, p,
            colormap=args.colormap, dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()} (row-normalized)")
        else:
            print(f"[skip] no languages with activations; {p.name} not written")

    if args.heatmap in ("domains", "all"):
        p = output_dir / "routing_heatmap_by_domain.png"
        wrote = save_domain_heatmap(
            domain_counts, domains, p, args.scale,
            colormap=args.colormap, dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()} (scale={args.scale})")
        else:
            print(f"[skip] no domains with activations; {p.name} not written")

    if args.heatmap in ("langdoms", "all"):
        p = output_dir / "routing_heatmap_by_langdom.png"
        wrote = save_langdom_heatmap(
            langdom_counts, langdom_keys, p, args.scale,
            colormap=args.colormap, dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()} (scale={args.scale})")
        else:
            print(f"[skip] no langdoms with activations; {p.name} not written")

    if args.heatmap in ("normalized", "all"):
        p = output_dir / "routing_heatmap_by_langdom_normalized.png"
        wrote = save_langdom_heatmap_normalized(
            langdom_counts, langdom_keys, p,
            colormap=args.colormap, dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()} (row-normalized)")
        else:
            print(f"[skip] no langdoms with activations; {p.name} not written")

    if args.heatmap in ("accuracy", "all"):
        p = output_dir / "accuracy_by_langdom.png"
        wrote = save_accuracy_chart(
            langdom_keys, _raw.get("by_langdom", {}), p,
            dpi=args.dpi,
        )
        if wrote:
            print(f"[save] {p.resolve()}")
        else:
            print(f"[skip] no langdoms with completions; {p.name} not written")

    print(f"[done] output_dir = {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
