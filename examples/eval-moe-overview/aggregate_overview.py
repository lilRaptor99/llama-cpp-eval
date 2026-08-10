#!/usr/bin/env python3
# type: ignore

"""Cross-dataset aggregator for MoE expert-routing statistics.

Walks every `expert_counts.json` produced by the per-dataset
`llama-eval-moe-*` C++ binaries under `<results-dir>/<model>/moe-*/`,
sums the [n_layer, n_expert] count matrices across all datasets for
each model, and writes one overview set of artifacts per model under
`<results-dir>/<model>/overall/`.

Per model, the script writes:
  - routing_heatmap_overview.png             L x E selections/token heatmap
  - routing_heatmap_overview_highlighted.png same heatmap with top-K cells outlined
  - top_experts_bars.png                     per-layer bar chart of top-K counts
  - top_experts.json                         per-layer top-K list + statistics
  - counts_total_overview.json               raw aggregated L x E matrix (sum of counts)
  - metadata_overview.json                   model + arch + per-dataset token split

Aggregation: sum raw counts across datasets, divide by total tokens
(prefill + generated; mmlu uses `subjects.<subj>.n_tokens`) to get
selections/token. This matches the existing per-dataset
`save_overview_heatmap` convention.

Top-K: ceil(n_expert * --top-k-fraction) experts per layer, ranked by
aggregated count desc. Default fraction is 0.125 (so 16/8/1/8 experts
per layer for gpt-oss / OLMoE / Mixtral / deepseek).

The script is standalone (no shared library import) to match the
convention of every existing per-dataset `heatmap_from_cpp.py`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# --------------------------------------------------------------------------- I/O

# Top-level keys used by each per-dataset binary to hold the per-row
# records that carry `layer_expert_counts`. Probed in this order.
_RECORD_KEYS: tuple[str, ...] = ("tasks", "subjects", "props", "by_langdom")


def _load_one(path: Path, default_L: int = 16, default_E: int = 64,
              default_k: int = 8) -> dict[str, Any]:
    """Parse one `expert_counts.json` and return per-file summary.

    Returns a dict with keys:
      arch             : dict (name, n_layer, n_expert, n_expert_used)
      counts_total     : np.ndarray [L, E] int64 (sum across all rows)
      total_tokens     : int (prefill + generated summed across rows; mmlu uses n_tokens)
      n_rows           : int (number of tasks / subjects / props / langdoms)
      dataset_label    : str (moe-<humaneval|bigbench|mmlu|popqa|include>, derived from path)
      per_dataset_meta : dict (tokens, n_rows, n_questions)
    """
    with open(path) as f:
        data = json.load(f)

    arch = data.get("model_arch", {}) or {}
    L = int(arch.get("n_layer", default_L))
    E = int(arch.get("n_expert", default_E))
    k = int(arch.get("n_expert_used", default_k))

    record_key = None
    for key in _RECORD_KEYS:
        if key in data and isinstance(data[key], dict) and data[key]:
            record_key = key
            break
    if record_key is None:
        raise SystemExit(
            f"[error] {path}: no tasks/subjects/props/by_langdom entries found"
        )

    raw = data[record_key]

    counts_total = np.zeros((L, E), dtype=np.int64)
    total_tokens = 0
    n_questions = 0
    for _row_key, body in raw.items():
        mat = body.get("layer_expert_counts")
        if mat is None:
            continue
        arr = np.asarray(mat, dtype=np.int64)
        if arr.shape != (L, E):
            raise SystemExit(
                f"[error] {path}: row shape {arr.shape}, expected ({L}, {E})"
            )
        counts_total += arr

        # Token accounting differs by dataset:
        #   humaneval / bigbench / popqa / include : prefill + generated
        #   mmlu                                   : n_tokens
        if "n_tokens" in body and "n_tokens_prefill" not in body:
            total_tokens += int(body.get("n_tokens", 0))
        else:
            total_tokens += int(body.get("n_tokens_prefill", 0)) + int(
                body.get("n_tokens_generated", 0)
            )
        # n_questions / n_correct / match_rate appear in different datasets.
        n_questions += int(body.get("questions", 0))

    dataset_label = path.parent.name  # e.g. "moe-humaneval"
    return {
        "arch": {
            "name": arch.get("name", "unknown"),
            "n_layer": L,
            "n_expert": E,
            "n_expert_used": k,
        },
        "counts_total": counts_total,
        "total_tokens": total_tokens,
        "n_rows": len(raw),
        "n_questions": n_questions,
        "dataset_label": dataset_label,
        "model_id": data.get("model"),
        "per_dataset_meta": {
            "dataset": dataset_label,
            "tokens": int(total_tokens),
            "n_rows": len(raw),
            "n_questions": int(n_questions),
        },
    }


def _discover_results(results_dir: Path, only_models: list[str] | None,
                      ) -> list[tuple[Path, list[Path]]]:
    """Find all `<model>/moe-*/expert_counts.json` under results_dir.

    Returns a list of (model_dir, [json_paths]) sorted by model name.
    Models without any usable JSON files are dropped from the output.
    """
    found: list[tuple[Path, list[Path]]] = []
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if only_models and model_dir.name not in only_models:
            continue
        json_paths = sorted(model_dir.glob("moe-*/expert_counts.json"))
        if json_paths:
            found.append((model_dir, json_paths))
    return found


# ----------------------------------------------------------- top-K computation

def compute_top_k(counts: np.ndarray, top_k_count: int,
                  ) -> list[list[dict[str, Any]]]:
    """For each layer, return the top-K experts ranked by count desc.

    Output shape: [n_layer][top_k_count] of dicts with keys:
      expert_id, rank, count, fraction_of_layer_total, cumulative_share
    """
    L, E = counts.shape
    out: list[list[dict[str, Any]]] = []
    for layer in range(L):
        row = counts[layer].astype(np.int64)
        layer_total = int(row.sum())
        # np.argsort is not stable; partition-then-sort the top-K is faster and stable for ties.
        if top_k_count >= E:
            order = np.argsort(-row, kind="stable")
        else:
            # argpartition gives an unordered set of the top-K indices.
            top_unsorted = np.argpartition(-row, top_k_count - 1)[:top_k_count]
            order = top_unsorted[np.argsort(-row[top_unsorted], kind="stable")]
        # `order` is now top-K indices in descending-count order.
        cumulative = 0
        per_layer: list[dict[str, Any]] = []
        for rank, eid in enumerate(order[:top_k_count], start=1):
            count = int(row[eid])
            cumulative += count
            per_layer.append({
                "expert_id": int(eid),
                "rank": rank,
                "count": count,
                "fraction_of_layer_total": (count / layer_total) if layer_total else 0.0,
                "cumulative_share": (cumulative / layer_total) if layer_total else 0.0,
            })
        out.append(per_layer)
    return out


def compute_per_dataset_top_k(per_dataset_counts: dict[str, np.ndarray],
                              top_k_count: int) -> dict[str, list[list[int]]]:
    """For each dataset and each layer, return the top-K expert IDs (no stats).

    Output: {dataset_name: [[expert_ids_per_layer] * top_k_count] * n_layer}
    """
    out: dict[str, list[list[int]]] = {}
    for ds_name, counts in per_dataset_counts.items():
        L, E = counts.shape
        per_layer: list[list[int]] = []
        for layer in range(L):
            row = counts[layer].astype(np.int64)
            if top_k_count >= E:
                order = np.argsort(-row, kind="stable")
            else:
                top_unsorted = np.argpartition(-row, top_k_count - 1)[:top_k_count]
                order = top_unsorted[np.argsort(-row[top_unsorted], kind="stable")]
            per_layer.append([int(e) for e in order[:top_k_count]])
        out[ds_name] = per_layer
    return out


# ----------------------------------------------------------- heatmap renderers

def save_overview_heatmap(counts: np.ndarray, total_tokens: int,
                          model_id: str, arch: dict[str, Any],
                          top_k: int, top_k_count: int,
                          path: Path, *, colormap: str = "viridis",
                          dpi: int = 120) -> None:
    """Layer x expert heatmap of per-token activation rate."""
    L, E = counts.shape
    per_token = counts.astype(np.float64) / max(total_tokens, 1)

    fig, ax = plt.subplots(figsize=(max(14, E * 0.32), max(4, L * 0.5)))
    im = ax.imshow(per_token, aspect="auto", cmap=colormap)
    ax.set_xlabel("expert id")
    ax.set_ylabel("layer")
    ax.set_title(
        f"{model_id}  -  per-token MoE activation rate (top-{top_k} of {E})\n"
        f"aggregated across all datasets; {total_tokens:,} tokens; "
        f"top {top_k_count} highlighted in next plot"
    )
    plt.colorbar(
        im, ax=ax,
        label=f"selections / token (uniform = {top_k / E:.4f})",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_highlighted_heatmap(counts: np.ndarray, total_tokens: int,
                              model_id: str, arch: dict[str, Any],
                              top_k: int, top_k_count: int,
                              top_k_per_layer: list[list[dict[str, Any]]],
                              path: Path, *, colormap: str = "viridis",
                              dpi: int = 120) -> None:
    """Same overview heatmap with each top-K cell outlined in red."""
    L, E = counts.shape
    per_token = counts.astype(np.float64) / max(total_tokens, 1)

    fig, ax = plt.subplots(figsize=(max(14, E * 0.32), max(4, L * 0.5)))
    im = ax.imshow(per_token, aspect="auto", cmap=colormap)
    ax.set_xlabel("expert id")
    ax.set_ylabel("layer")
    ax.set_title(
        f"{model_id}  -  per-token MoE activation rate (top-{top_k} of {E})\n"
        f"top {top_k_count} experts per layer outlined in red; "
        f"{total_tokens:,} tokens aggregated"
    )
    plt.colorbar(
        im, ax=ax,
        label=f"selections / token (uniform = {top_k / E:.4f})",
    )

    # Overlay red rectangles on the top-K cells. imshow maps each cell
    # to the rectangle [col-0.5, col+0.5] x [row-0.5, row+0.5].
    for layer, layer_top in enumerate(top_k_per_layer):
        for entry in layer_top:
            eid = entry["expert_id"]
            rect = Rectangle(
                (eid - 0.5, layer - 0.5), 1.0, 1.0,
                fill=False, edgecolor="red", linewidth=1.5,
            )
            ax.add_patch(rect)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_top_k_bar_chart(counts: np.ndarray,
                         top_k_per_layer: list[list[dict[str, Any]]],
                         model_id: str, arch: dict[str, Any],
                         top_k_count: int,
                         path: Path, *, dpi: int = 120) -> None:
    """One vertical panel per layer: top-K activation fractions (descending).

    Y-axis is `fraction_of_layer_total` (the same value that `top_experts.json`
    persists per expert), so each layer shows the within-layer distribution
    of its top-K experts on a comparable 0..1 scale. The leading bar's
    fraction-of-layer_total is annotated next to the title.
    """
    L = len(top_k_per_layer)

    # Layout: rows = layers; columns = 1 (we use a tall figure).
    # Per-layer panel height scales with top-K so wide-bar models (large
    # top_k_count) don't get squeezed and narrow-bar models (e.g. Mixtral
    # top_k=1) don't waste pixels.
    cell_w = 6.5
    cell_h = max(0.35, min(1.0, 0.06 * top_k_count + 0.25))
    fig_h = max(4.0, L * cell_h)
    fig_w = cell_w
    fig, axes = plt.subplots(L, 1, figsize=(fig_w, fig_h), squeeze=False)

    bar_color = "#4477aa"
    text_color = "#222222"

    for layer in range(L):
        ax = axes[layer][0]
        layer_top = top_k_per_layer[layer]
        eids = [e["expert_id"] for e in layer_top]
        fracs = [e["fraction_of_layer_total"] for e in layer_top]
        x = np.arange(len(eids))
        ax.bar(x, fracs, color=bar_color, edgecolor="black", linewidth=0.4)
        # Annotate bars with expert ids and fractions.
        for xi, eid, frac in zip(x, eids, fracs):
            ax.text(xi, frac, f"e{eid}\n{frac:.4f}", ha="center", va="bottom",
                    fontsize=5, color=text_color, rotation=0)
        top_frac = fracs[0] if fracs else 0.0
        # Sum of all top-K fractions (what the highlighted bars cover).
        total_top_share = sum(fracs)
        ax.set_title(
            f"layer {layer}  -  top {top_k_count} experts  "
            f"(top-1 = {top_frac:.4f}, top-{top_k_count} sum = {total_top_share:.4f} "
            f"of layer)",
            fontsize=7, loc="left",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(e) for e in eids], fontsize=6, rotation=0)
        ax.set_xlim(-0.5, max(len(eids) - 0.5, 0.5))
        ax.set_ylim(0, max(max(fracs) * 1.30, 0.001) if fracs else 1.0)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y", linestyle=":", alpha=0.3)
        ax.set_ylabel("fraction of layer", fontsize=6)

    axes[-1][0].set_xlabel("expert id", fontsize=8)
    fig.suptitle(
        f"{model_id}  -  top {top_k_count} experts per layer "
        f"(ceil(n_expert * --top-k-fraction) = {top_k_count}; "
        f"model top_k/n_expert = {arch['n_expert_used']}/{arch['n_expert']}; "
        f"counts aggregated across all datasets)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- main path

def process_model(model_dir: Path, json_paths: list[Path],
                  output_dir: Path, top_k_fraction: float,
                  include_per_dataset: bool, *, colormap: str,
                  dpi: int) -> dict[str, Any]:
    """Aggregate one model's datasets and write the overview artifacts.

    Returns a summary dict for stdout reporting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- aggregate
    per_dataset_counts: dict[str, np.ndarray] = {}
    per_dataset_meta: list[dict[str, Any]] = []
    model_id: str | None = None
    arch: dict[str, Any] | None = None
    counts_total = None
    total_tokens = 0
    n_datasets_used = 0
    n_datasets_skipped = 0
    n_questions_total = 0

    for jp in json_paths:
        try:
            loaded = _load_one(jp)
        except (SystemExit, json.JSONDecodeError, OSError) as e:
            print(f"[skip] {jp}: {e}")
            n_datasets_skipped += 1
            continue

        ds_name = loaded["dataset_label"]
        if model_id is None:
            model_id = loaded["model_id"]
            arch = loaded["arch"]
            counts_total = loaded["counts_total"].copy()
        else:
            # Sanity-check consistency across datasets for the same model.
            if (loaded["arch"]["n_layer"] != arch["n_layer"]
                    or loaded["arch"]["n_expert"] != arch["n_expert"]):
                print(
                    f"[warn] {jp}: arch shape mismatch "
                    f"(got {loaded['arch']}, expected {arch}); skipping"
                )
                n_datasets_skipped += 1
                continue
            if loaded["counts_total"].shape != counts_total.shape:
                print(f"[warn] {jp}: matrix shape mismatch; skipping")
                n_datasets_skipped += 1
                continue
            counts_total += loaded["counts_total"]

        per_dataset_counts[ds_name] = loaded["counts_total"]
        per_dataset_meta.append(loaded["per_dataset_meta"])
        total_tokens += loaded["total_tokens"]
        n_questions_total += loaded["n_questions"]
        n_datasets_used += 1
        print(
            f"[load] {jp.parent.name}: tokens={loaded['total_tokens']:,}, "
            f"rows={loaded['n_rows']}, questions={loaded['n_questions']}"
        )

    if counts_total is None or arch is None:
        raise SystemExit(f"[error] {model_dir.name}: no usable datasets found")

    L = arch["n_layer"]
    E = arch["n_expert"]
    top_k = arch["n_expert_used"]
    top_k_count = math.ceil(E * top_k_fraction)

    # ----------------------------------------------------------------- compute
    top_k_per_layer = compute_top_k(counts_total, top_k_count)

    # Per-layer totals (used both for sanity checks and JSON metadata).
    per_layer_totals = [int(counts_total[layer].sum()) for layer in range(L)]

    # Quick sanity check: top-K should capture a non-trivial share.
    uniform_share = top_k_count / E
    per_layer_top_share: list[tuple[int, float]] = []  # (layer, share)
    for layer, layer_top in enumerate(top_k_per_layer):
        layer_total = int(counts_total[layer].sum())
        if layer_total <= 0:
            continue  # empty layer -- nothing to compare against
        share = sum(e["count"] for e in layer_top) / layer_total
        per_layer_top_share.append((layer, share))
    non_empty_layers = len(per_layer_top_share)
    if non_empty_layers == 0:
        print(f"[warn] all {L} layers are empty for {model_id}")
    else:
        worst_layer, worst_share = min(per_layer_top_share, key=lambda x: x[1])
        if worst_share < uniform_share * 1.05:
            print(
                f"[warn] top-{top_k_count} share at layer {worst_layer} "
                f"(most uniform of {non_empty_layers} non-empty layers) is "
                f"{worst_share:.4f}, barely above uniform {uniform_share:.4f} "
                f"-- the count distribution may be near-flat"
            )

    # ----------------------------------------------------------------- persist
    # 1. Overview heatmap (selections/token).
    p = output_dir / "routing_heatmap_overview.png"
    save_overview_heatmap(
        counts_total, total_tokens, model_id or model_dir.name, arch,
        top_k, top_k_count, p, colormap=colormap, dpi=dpi,
    )
    print(f"[save] {p}")

    # 2. Highlighted heatmap.
    p = output_dir / "routing_heatmap_overview_highlighted.png"
    save_highlighted_heatmap(
        counts_total, total_tokens, model_id or model_dir.name, arch,
        top_k, top_k_count, top_k_per_layer, p,
        colormap=colormap, dpi=dpi,
    )
    print(f"[save] {p}")

    # 3. Per-layer top-K bar chart.
    p = output_dir / "top_experts_bars.png"
    save_top_k_bar_chart(
        counts_total, top_k_per_layer,
        model_id or model_dir.name, arch, top_k_count,
        p, dpi=dpi,
    )
    print(f"[save] {p}")

    # 4. top_experts.json
    top_experts_payload: dict[str, Any] = {
        "model": model_id,
        "model_arch": arch,
        "top_k_fraction": top_k_fraction,
        "top_k_count": top_k_count,
        "uniform_share_per_layer": uniform_share,
        "aggregation": {
            "method": "sum_counts_divide_by_total_tokens",
            "datasets": per_dataset_meta,
        },
        "totals": {
            "datasets_used": n_datasets_used,
            "datasets_skipped": n_datasets_skipped,
            "tokens_total": int(total_tokens),
            "questions_total": int(n_questions_total),
        },
        "layers": [
            {
                "layer": layer,
                "layer_total": per_layer_totals[layer],
                "top_k": top_k_per_layer[layer],
            }
            for layer in range(L)
        ],
    }
    if include_per_dataset:
        top_experts_payload["per_dataset_layer_topk"] = compute_per_dataset_top_k(
            per_dataset_counts, top_k_count,
        )
    p = output_dir / "top_experts.json"
    with open(p, "w") as f:
        json.dump(top_experts_payload, f, indent=2)
    print(f"[save] {p}")

    # 5. counts_total_overview.json (raw aggregated LxE)
    p = output_dir / "counts_total_overview.json"
    with open(p, "w") as f:
        json.dump({
            "model": model_id,
            "model_arch": arch,
            "totals_tokens": int(total_tokens),
            "shape": [int(L), int(E)],
            "counts_total": counts_total.tolist(),
        }, f)
    print(f"[save] {p}")

    # 6. metadata_overview.json
    p = output_dir / "metadata_overview.json"
    meta = {
        "model": model_id,
        "model_arch": arch,
        "top_k_fraction": top_k_fraction,
        "top_k_count": top_k_count,
        "aggregation_method": "sum_counts_divide_by_total_tokens",
        "results_dir": str(model_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "datasets": per_dataset_meta,
        "totals": {
            "datasets_used": n_datasets_used,
            "datasets_skipped": n_datasets_skipped,
            "tokens_total": int(total_tokens),
            "questions_total": int(n_questions_total),
        },
    }
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] {p}")

    return {
        "model_dir": model_dir,
        "model_id": model_id,
        "arch": arch,
        "top_k_count": top_k_count,
        "tokens_total": total_tokens,
        "n_datasets_used": n_datasets_used,
        "n_datasets_skipped": n_datasets_skipped,
        "output_dir": output_dir,
    }


# --------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("build/results"),
        help="Root directory containing per-model subdirectories "
             "(each with moe-*/expert_counts.json).",
    )
    parser.add_argument(
        "--top-k-fraction", type=float, default=0.125,
        help="Fraction of experts per layer to highlight / report as top-K "
             "(default: 0.125).",
    )
    parser.add_argument(
        "--include-per-dataset", action="store_true",
        help="Also write per_dataset_layer_topk into top_experts.json.",
    )
    parser.add_argument(
        "--models", action="append", default=None,
        help="Restrict to a subset of model directory names (repeatable).",
    )
    parser.add_argument("--colormap", type=str, default="viridis",
                        help="Matplotlib colormap name.")
    parser.add_argument("--dpi", type=int, default=120,
                        help="Output PNG DPI.")
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        raise SystemExit(f"[error] --results-dir {results_dir} is not a directory")

    print(f"[load] results_dir = {results_dir.resolve()}")
    discovered = _discover_results(results_dir, args.models)
    if not discovered:
        raise SystemExit(f"[error] no <model>/moe-*/expert_counts.json found under {results_dir}")

    summaries: list[dict[str, Any]] = []
    for model_dir, json_paths in discovered:
        print(f"\n[model] {model_dir.name}: {len(json_paths)} dataset JSON(s)")
        output_dir = model_dir / "overall"
        summary = process_model(
            model_dir, json_paths, output_dir,
            top_k_fraction=args.top_k_fraction,
            include_per_dataset=args.include_per_dataset,
            colormap=args.colormap, dpi=args.dpi,
        )
        summaries.append(summary)

    # Stdout summary table.
    print("\n[summary]")
    cols = ("model", "arch", "LxE", "top_k", "datasets", "tokens")
    print("  ".join(f"{c:<28}" if c != "LxE" else f"{c:<10}" for c in cols))
    for s in summaries:
        arch = s["arch"]
        print(
            "  ".join([
                f"{(s['model_id'] or s['model_dir'].name):<28}",
                f"{arch['name']:<28}",
                f"{arch['n_layer']}x{arch['n_expert']:<6}",
                f"{s['top_k_count']:<6}",
                f"{s['n_datasets_used']:<8}",
                f"{s['tokens_total']:,}",
            ])
        )
    print(f"\n[done] wrote overview artifacts to <model>/overall/ for "
          f"{len(summaries)} model(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
