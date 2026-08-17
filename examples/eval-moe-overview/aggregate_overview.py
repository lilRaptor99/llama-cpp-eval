#!/usr/bin/env python3
# type: ignore

"""Cross-dataset aggregator for MoE expert-routing statistics.

Walks every `expert_counts.json` produced by the per-dataset
`llama-eval-moe-*` C++ binaries under the new model-quantization
directory structure:

    <results-dir>/<model>/<quant>/moe-*/expert_counts.json

For each `(model, quant)` cell it sums the [n_layer, n_expert] count
matrices across all datasets and writes one overview set of artifacts
under:

    <results-dir>/<model>/<quant>/overall/

A legacy (no-quant) layout is also tolerated for backward compatibility:

    <results-dir>/<model>/moe-*/expert_counts.json
        -> <results-dir>/<model>/overall/      (quant_name = "")

Per (model, quant) cell, the script writes:
  - routing_heatmap_overview.png             L x E circle grid, blue gradient,
                                             linear scale of raw counts, plus
                                             top-K adjacent-layer co-activation
                                             lines (mirrors eval-moe-mmlu
                                             routing_graph_from_cpp.py)
  - routing_heatmap_overview_highlighted.png same heatmap with red rings
                                             around top-K cells per layer
                                             (and the same co-activation lines)
  - top_experts_bars.png                     per-layer bar chart of top-K counts
  - top_experts.json                         per-layer top-K list + statistics
  - counts_total_overview.json               raw aggregated L x E matrix (sum of
                                             counts) + aggregated
                                             adjacent_pair_counts (when present
                                             in the per-dataset JSONs)
  - metadata_overview.json                   model + arch + quant + per-dataset
                                             token split + aggregate-block coverage

Aggregation: sum raw counts across datasets, divide by total tokens
(prefill + generated; mmlu uses `subjects.<subj>.n_tokens`) to get the
per-token rate that `top_k / n_expert` is compared against in the
metadata. The PNG heatmaps themselves use raw counts (no per-token
normalisation) to match the visual style of the routing-graph
heatmaps in `eval-moe-mmlu/routing_graph_from_cpp.py`.

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
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter


# --------------------------------------------------------------------------- constants

# Highlight ring colour copied from eval-moe-mmlu/routing_graph_from_cpp.py
# so the overview heatmap's top-K highlight matches the routing graph's.
# The overview heatmap uses a THINNER ring than the routing graph
# (routing_graph uses 4.0; here we use 1.5) because the overview circles
# are much smaller in data-units per expert, so 4.0 visually dominates them.
HIGHLIGHT_EDGE_COLOR = "#ff1744"  # red (matches routing_graph_from_cpp.py)
HIGHLIGHT_EDGE_WIDTH = 1.5
# Vertical spacing between layer rows in the circle-grid heatmap. Picked
# so that adjacent-row circles don't overlap (circle_radius = 0.40 of
# col_spacing = 1.0). Larger than 0.4 lets the cross-layer connection
# lines have a clearly visible vertical distance to traverse (the routing
# graph uses 2.5; we use 1.5 here because the overview grid is denser).
DEFAULT_OVERVIEW_COL_SPACING = 1.5
DEFAULT_OVERVIEW_ROW_SPACING = 2.5
# Default colormap for the circle-grid heatmap. The user explicitly asked
# for the same blue gradient as the routing graph, so it is no longer
# configurable via the CLI.
DEFAULT_OVERVIEW_COLORMAP = "Blues"
# Top-K co-activation pairs (adjacent-layer) to draw on the overview
# heatmap, mirroring the routing graph in eval-moe-mmlu. Default 64 is
# the routing graph's DEFAULT_TOP_K_PAIRS; --top-k-pairs 0 disables lines.
DEFAULT_OVERVIEW_TOP_K_PAIRS = 24
# Line colour and width-scale for the cross-layer connection lines, copied
# from the routing graph so the two plots have the same visual contract.
PAIR_LINE_COLOR = "#1f3a93"  # dark blue (matches routing_graph_from_cpp.py)
# line width = 0.1 + line_scale * raw_count. OLMoE-scale counts (~1e7)
# want ~1e-7; Mixtral-scale counts (~1e6) want ~1e-6. Default 5e-7 is
# the routing graph's default and works across the models in this repo.
DEFAULT_OVERVIEW_LINE_SCALE = 4e-7


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
      marginal         : np.ndarray [L, E] int64, or None (aggregate.marginal_expert_counts)
      adj              : np.ndarray [L-1, E, E] int64, or None (aggregate.adjacent_pair_counts)
      has_aggregate    : bool (True iff both marginal and adj were loaded)
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

    # ---- optional aggregate block (marginal + adjacent pair counts) ----
    # The aggregate block is emitted by the updated per-dataset C++ binaries
    # (with co-activation capture). Older binaries lack it; we tolerate
    # that and just leave the aggregate fields as None so the overview
    # can fall back to the count-only heatmap.
    marginal: np.ndarray | None = None
    adj: np.ndarray | None = None
    agg = data.get("aggregate")
    if isinstance(agg, dict):
        marg_raw = agg.get("marginal_expert_counts")
        if marg_raw is not None:
            marg_arr = np.asarray(marg_raw, dtype=np.int64)
            if marg_arr.shape == (L, E):
                marginal = marg_arr
        adj_raw = agg.get("adjacent_pair_counts", [])
        if isinstance(adj_raw, list) and len(adj_raw) == 0:
            adj_arr = np.zeros((max(L - 1, 0), E, E), dtype=np.int64)
        elif adj_raw:
            adj_arr = np.asarray(adj_raw, dtype=np.int64)
            if adj_arr.shape == (max(L - 1, 0), E, E):
                adj = adj_arr
    has_aggregate = (marginal is not None) and (adj is not None)

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
        "marginal": marginal,
        "adj": adj,
        "has_aggregate": has_aggregate,
        "per_dataset_meta": {
            "dataset": dataset_label,
            "tokens": int(total_tokens),
            "n_rows": len(raw),
            "n_questions": int(n_questions),
        },
    }


def _discover_results(results_dir: Path, only_models: list[str] | None,
                      ) -> list[tuple[Path, list[Path], str]]:
    """Find all `<model>/<quant>/moe-*/expert_counts.json` under results_dir.

    New structure (preferred):
        <results-dir>/<model>/<quant>/moe-*/expert_counts.json

    Legacy structure (still supported, no quant):
        <results-dir>/<model>/moe-*/expert_counts.json

    Returns a list of ``(model_dir, json_paths, quant_name)`` tuples sorted
    by model name then quant name. Models without any usable JSON files are
    dropped from the output. When the legacy layout is used, ``quant_name``
    is the empty string and ``model_dir`` is the cell directory itself.
    """
    found: list[tuple[Path, list[Path], str]] = []
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if only_models and model_dir.name not in only_models:
            continue

        # Try the new structure first: <model>/<quant>/moe-*/expert_counts.json.
        # The cell directory is the same as the quant directory in that case.
        per_cell: list[tuple[Path, str, list[Path]]] = []
        for sub in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            json_paths = sorted(sub.glob("moe-*/expert_counts.json"))
            if json_paths:
                per_cell.append((sub, sub.name, json_paths))

        if not per_cell:
            # Fall back to legacy: <model>/moe-*/expert_counts.json.
            json_paths = sorted(model_dir.glob("moe-*/expert_counts.json"))
            if json_paths:
                per_cell.append((model_dir, "", json_paths))

        for cell_dir, quant_name, json_paths in per_cell:
            found.append((model_dir, json_paths, quant_name))
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

def _colorbar_millions_formatter(x: float, pos: int) -> str:
    """Colorbar tick formatter that prints in millions (e.g. 40.0M)."""
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.1f}K"
    return f"{int(x)}"


def _filter_top_k_pairs(adj: np.ndarray, K: int) -> list[list[tuple[int, int, int]]]:
    """For each layer pair L, return the top-K (e_i, e_j, count) triples.

    Mirror of ``eval-moe-mmlu/routing_graph_from_cpp.py::filter_top_k_pairs``
    so the overview and the routing graph pick the same set of edges when
    given the same aggregate block and K. Zero-count entries are dropped.
    K is capped at E*E defensively.
    """
    if adj.shape[0] == 0:
        return []

    L_pairs, E, _ = adj.shape
    K_eff = min(K, E * E)

    out: list[list[tuple[int, int, int]]] = []
    for L_ in range(L_pairs):
        flat = adj[L_].reshape(-1)
        if K_eff >= flat.size:
            nz_idx = np.flatnonzero(flat)
            triples = [(int(idx // E), int(idx % E), int(flat[idx])) for idx in nz_idx]
            triples.sort(key=lambda t: t[2], reverse=True)
        else:
            top_idx = np.argpartition(flat, -K_eff)[-K_eff:]
            triples = [(int(idx // E), int(idx % E), int(flat[idx])) for idx in top_idx]
            triples.sort(key=lambda t: t[2], reverse=True)
        out.append(triples)
    return out


def _draw_circle_grid_overview(
    counts: np.ndarray,
    total_tokens: int,
    model_id: str,
    arch: dict[str, Any],
    top_k: int,
    top_k_count: int,
    top_k_per_layer: Optional[list[list[dict[str, Any]]]],
    path: Path,
    *,
    dpi: int = 120,
    colormap: str = DEFAULT_OVERVIEW_COLORMAP,
    quant: str = "",
    adj: Optional[np.ndarray] = None,
    top_k_pairs: int = DEFAULT_OVERVIEW_TOP_K_PAIRS,
    line_scale: float = DEFAULT_OVERVIEW_LINE_SCALE,
) -> None:
    """Render the overview heatmap as a circle grid (mirrors routing_graph).

    Layout: L rows (one per layer) × E columns (one per expert). Each cell
    is a circle whose fill is linearly scaled to the raw activation count
    (Blues colormap by default, no log scale). When ``top_k_per_layer`` is
    provided, a thick red ring is drawn around each top-K cell per layer
    (matching the highlight style used by the routing graph in
    ``eval-moe-mmlu/routing_graph_from_cpp.py``).

    When ``adj`` is provided (shape ``[L-1, E, E]``), top-K adjacent-layer
    co-activation pairs are drawn as lines connecting the two expert
    circles, exactly like the routing graph. Line thickness is
    proportional to the raw pair count (NOT normalised per layer pair),
    so absolute volume is comparable across the whole figure. Pass
    ``top_k_pairs=0`` to disable the lines.

    The figure axes are inverted so layer 0 sits at the top. Both axes
    are turned off; integer layer/expert tick labels are drawn manually
    next to the left edge and below the bottom row.
    """
    L, E = counts.shape

    # ---- grid geometry (decoupled col/row spacing so we can tune both) ----
    col_spacing = DEFAULT_OVERVIEW_COL_SPACING
    row_spacing = DEFAULT_OVERVIEW_ROW_SPACING
    circle_radius = col_spacing * 0.40

    # ---- figure sizing (auto-scaled to model shape) ----
    width = max(10.0, E * col_spacing * 0.35)
    height = max(5.0, L * row_spacing * 0.60)
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- linear normalisation for the circle fill (NO log scale) ----
    counts_f = counts.astype(np.float64)
    global_max = float(counts_f.max()) if counts_f.size > 0 else 1.0
    if global_max <= 0:
        global_max = 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=global_max)
    cmap = plt.get_cmap(colormap)

    # ---- compute (x, y) centres on a simple grid (L rows, E cols) ----
    pos = np.zeros((L, E, 2), dtype=np.float64)
    for layer in range(L):
        for e in range(E):
            pos[layer, e, 0] = e * col_spacing
            pos[layer, e, 1] = layer * row_spacing

    # ---- co-activation lines (drawn FIRST so circles sit on top) ----
    # Mirrors the routing graph in eval-moe-mmlu: top-K adjacent-layer
    # pairs by raw count, line width proportional to raw count.
    n_lines_drawn = 0
    if adj is not None and L >= 2 and top_k_pairs > 0:
        from matplotlib.lines import Line2D  # local import keeps header tidy
        filtered = _filter_top_k_pairs(adj, top_k_pairs)
        for layer in range(L - 1):
            for (e_i, e_j, count) in filtered[layer]:
                if count <= 0:
                    continue
                x1, y1 = pos[layer, e_i]
                x2, y2 = pos[layer + 1, e_j]
                lw = 0.1 + line_scale * count
                # Alpha scales with rank-ish: high counts get more opaque.
                alpha = min(0.85, 0.15 + 0.7 * math.tanh(count * line_scale * 5.0))
                ax.add_line(
                    Line2D(
                        [x1, x2], [y1, y2],
                        color=PAIR_LINE_COLOR,
                        linewidth=lw,
                        alpha=alpha,
                        solid_capstyle="round",
                        zorder=1,
                    )
                )
                n_lines_drawn += 1

    # ---- expert circles: per-expert activation fill ----
    edge_color = "#bdbdbd"  # light grey border so zero-count circles are visible
    for layer in range(L):
        for e in range(E):
            x, y = pos[layer, e]
            facecolor = cmap(norm(counts_f[layer, e]))
            circ = plt.Circle(
                (x, y), circle_radius,
                facecolor=facecolor,
                edgecolor=edge_color,
                linewidth=0.5,
                zorder=2,
            )
            ax.add_patch(circ)

    # ---- highlight rings: red rings around top-K cells per layer ----
    if top_k_per_layer is not None:
        for layer, layer_top in enumerate(top_k_per_layer):
            for entry in layer_top:
                eid = entry["expert_id"]
                x, y = pos[layer, eid]
                ring = plt.Circle(
                    (x, y), circle_radius * 1.08,
                    facecolor="none",
                    edgecolor=HIGHLIGHT_EDGE_COLOR,
                    linewidth=HIGHLIGHT_EDGE_WIDTH,
                    zorder=3,
                )
                ax.add_patch(ring)

    # ---- axis limits (with margin for layer/expert labels) ----
    if L > 0 and E > 0:
        x_min = -col_spacing * 1.5
        x_max = float(pos[:, :, 0].max()) + col_spacing * 1.5
        y_min = -row_spacing * 0.5
        y_max = float(pos[:, :, 1].max()) + row_spacing * 0.4
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # invert so L0 is at top

    # ---- layer labels (left, right-aligned) ----
    for layer in range(L):
        y = float(pos[layer, 0, 1])
        ax.text(
            -col_spacing * 0.4, y, f"{layer}",
            ha="right", va="center",
            fontsize=8, color="#333", family="monospace",
        )

    # ---- expert index labels (bottom, centred under each column) ----
    for e in range(E):
        x = float(pos[L - 1, e, 0])
        ax.text(
            x, -row_spacing * 0.55, f"{e}",
            ha="center", va="top",
            fontsize=8, color="#333", family="monospace",
        )

    # ---- axis labels ----
    if L > 0 and E > 0:
        ax.text(
            -col_spacing * 1.1, float(pos[:, :, 1].mean()),
            "Layer Index", rotation=90, ha="center", va="center",
            fontsize=10, color="#333",
        )
        ax.text(
            float(pos[:, :, 0].mean()), -row_spacing * 1.0,
            "Expert Index (within same layer)", ha="center", va="top",
            fontsize=10, color="#333",
        )

    # ---- colorbar for circle fill (raw counts, linear scale) ----
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Raw activation count (linear scale)", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(_colorbar_millions_formatter))

    # ---- title ----
    quant_str = f"  -  quant: {quant}" if quant else ""
    highlight_str = (
        f"\nRed ring = top {top_k_count} experts per layer"
        if top_k_per_layer is not None
        else f"\nTop {top_k_count} experts per layer shown in the highlighted variant"
    )
    lines_str = ""
    if adj is not None and L >= 2 and top_k_pairs > 0:
        lines_str = (
            f"\nDark blue lines = top {top_k_pairs} adjacent-layer "
            f"co-activations per layer pair ({n_lines_drawn:,} total)"
        )
    elif adj is not None and L >= 2 and top_k_pairs == 0:
        lines_str = "\nAdjacent-layer co-activation lines disabled (--top-k-pairs 0)"
    ax.set_title(
        f"{model_id}{quant_str}  -  MoE expert activation counts "
        f"(top-{top_k} of {E})\n"
        f"aggregated across all datasets; {total_tokens:,} tokens"
        f"{highlight_str}{lines_str}",
        fontsize=11, pad=14,
    )

    # ---- save ----
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_overview_heatmap(counts: np.ndarray, total_tokens: int,
                          model_id: str, arch: dict[str, Any],
                          top_k: int, top_k_count: int,
                          path: Path, *, dpi: int = 120,
                          quant: str = "",
                          adj: Optional[np.ndarray] = None,
                          top_k_pairs: int = DEFAULT_OVERVIEW_TOP_K_PAIRS,
                          line_scale: float = DEFAULT_OVERVIEW_LINE_SCALE,
                          ) -> None:
    """Overview heatmap: L x E circle grid, blue gradient, linear scale.

    Visual style mirrors the routing graph in
    ``eval-moe-mmlu/routing_graph_from_cpp.py``: one circle per expert on
    a decoupled grid, circle fill linearly scaled to the **raw** activation
    count (no per-token normalisation). No highlight rings here — see
    ``save_highlighted_heatmap`` for the top-K variant.

    When ``adj`` (shape ``[L-1, E, E]``) is provided, the top
    ``top_k_pairs`` adjacent-layer co-activation pairs are drawn as dark
    blue lines (line thickness proportional to raw count, same contract
    as the routing graph). Pass ``top_k_pairs=0`` to disable the lines.
    """
    _draw_circle_grid_overview(
        counts, total_tokens, model_id, arch, top_k, top_k_count,
        top_k_per_layer=None, path=path, dpi=dpi,
        quant=quant, adj=adj, top_k_pairs=top_k_pairs, line_scale=line_scale,
    )


def save_highlighted_heatmap(counts: np.ndarray, total_tokens: int,
                              model_id: str, arch: dict[str, Any],
                              top_k: int, top_k_count: int,
                              top_k_per_layer: list[list[dict[str, Any]]],
                              path: Path, *, dpi: int = 120,
                              quant: str = "",
                              adj: Optional[np.ndarray] = None,
                              top_k_pairs: int = DEFAULT_OVERVIEW_TOP_K_PAIRS,
                              line_scale: float = DEFAULT_OVERVIEW_LINE_SCALE,
                              ) -> None:
    """Same overview heatmap with red rings around top-K cells per layer.

    Identical to ``save_overview_heatmap`` except that each top-K cell
    (per layer) is overlaid with a thin red ring. The ring colour matches
    the highlight ring used by the routing graph in
    ``eval-moe-mmlu/routing_graph_from_cpp.py`` (width is reduced here
    so the ring does not visually dominate the smaller overview circles).

    When ``adj`` is provided, top-K adjacent-layer co-activation lines
    are drawn just like in the un-highlighted overview variant.
    """
    _draw_circle_grid_overview(
        counts, total_tokens, model_id, arch, top_k, top_k_count,
        top_k_per_layer=top_k_per_layer, path=path, dpi=dpi,
        quant=quant, adj=adj, top_k_pairs=top_k_pairs, line_scale=line_scale,
    )


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
                  include_per_dataset: bool, *, dpi: int,
                  quant: str = "",
                  top_k_pairs: int = DEFAULT_OVERVIEW_TOP_K_PAIRS,
                  line_scale: float = DEFAULT_OVERVIEW_LINE_SCALE,
                  ) -> dict[str, Any]:
    """Aggregate one (model, quant) cell's datasets and write the overview artifacts.

    Returns a summary dict for stdout reporting.

    ``top_k_pairs`` and ``line_scale`` control the adjacent-layer
    co-activation lines drawn on both overview heatmaps (mirroring the
    routing graph in ``eval-moe-mmlu``). Pass ``top_k_pairs=0`` to
    disable the lines.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- aggregate
    per_dataset_counts: dict[str, np.ndarray] = {}
    per_dataset_meta: list[dict[str, Any]] = []
    model_id: str | None = None
    arch: dict[str, Any] | None = None
    counts_total = None
    adj_total: np.ndarray | None = None
    n_datasets_with_agg = 0
    n_datasets_missing_agg = 0
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

        # Aggregate the adjacent-layer pair counts when present. Datasets
        # without the aggregate block (older binaries) are tolerated; we
        # only draw the overview lines when ALL used datasets have it.
        if loaded["has_aggregate"]:
            if adj_total is None:
                adj_total = loaded["adj"].astype(np.int64).copy()
            else:
                if loaded["adj"].shape != adj_total.shape:
                    print(
                        f"[warn] {jp}: adj shape {loaded['adj'].shape} "
                        f"!= expected {adj_total.shape}; "
                        f"skipping its pair-count contribution"
                    )
                else:
                    adj_total += loaded["adj"]
            n_datasets_with_agg += 1
        else:
            n_datasets_missing_agg += 1
            print(
                f"[note] {jp.parent.name}: no `aggregate` block; "
                f"its pair counts will NOT contribute to the overview lines"
            )

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
    # 1. Overview heatmap (circle grid, blue gradient, linear raw counts,
    #    plus top-K adjacent-layer co-activation lines when available).
    p = output_dir / "routing_heatmap_overview.png"
    save_overview_heatmap(
        counts_total, total_tokens, model_id or model_dir.name, arch,
        top_k, top_k_count, p, dpi=dpi, quant=quant,
        adj=adj_total, top_k_pairs=top_k_pairs, line_scale=line_scale,
    )
    print(f"[save] {p}")

    # 2. Highlighted heatmap (same circle grid + red rings on top-K per layer
    #    + the same co-activation lines).
    p = output_dir / "routing_heatmap_overview_highlighted.png"
    save_highlighted_heatmap(
        counts_total, total_tokens, model_id or model_dir.name, arch,
        top_k, top_k_count, top_k_per_layer, p, dpi=dpi, quant=quant,
        adj=adj_total, top_k_pairs=top_k_pairs, line_scale=line_scale,
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

    # 5. counts_total_overview.json (raw aggregated LxE + aggregated adj
    #    pair counts, when available). The adjacent pair-counts are
    #    persisted as a separate `adjacent_pair_counts` key with shape
    #    documented by the `shape` field of the same name, so downstream
    #    tools (and the routing-graph viewer) can consume them directly.
    p = output_dir / "counts_total_overview.json"
    payload: dict[str, Any] = {
        "model": model_id,
        "model_arch": arch,
        "totals_tokens": int(total_tokens),
        "shape": [int(L), int(E)],
        "counts_total": counts_total.tolist(),
    }
    if adj_total is not None:
        payload["adjacent_pair_counts"] = adj_total.tolist()
        payload["adjacent_pair_counts_shape"] = list(adj_total.shape)
        payload["adjacent_pair_counts_total"] = int(adj_total.sum())
    with open(p, "w") as f:
        json.dump(payload, f)
    print(f"[save] {p}")

    # 6. metadata_overview.json
    p = output_dir / "metadata_overview.json"
    meta = {
        "model": model_id,
        "model_arch": arch,
        "quant": quant,
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
            "datasets_with_aggregate": n_datasets_with_agg,
            "datasets_missing_aggregate": n_datasets_missing_agg,
        },
        "coactivation_lines": {
            "top_k_pairs": int(top_k_pairs),
            "line_scale": float(line_scale),
        },
    }
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] {p}")

    return {
        "model_dir": model_dir,
        "model_id": model_id,
        "quant": quant,
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
             "(each with <quant>/moe-*/expert_counts.json).",
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
    parser.add_argument(
        "--quants", action="append", default=None,
        help="Restrict to a subset of quantization directory names "
             "(repeatable). Default: process all quants found.",
    )
    parser.add_argument("--dpi", type=int, default=120,
                        help="Output PNG DPI.")
    parser.add_argument(
        "--top-k-pairs", type=int, default=DEFAULT_OVERVIEW_TOP_K_PAIRS,
        help=(
            "top-K adjacent-layer co-activation pairs to draw on the overview "
            "heatmaps (same role as --top-k in eval-moe-mmlu/routing_graph). "
            "Set to 0 to disable the cross-layer lines entirely. "
            f"(default: {DEFAULT_OVERVIEW_TOP_K_PAIRS})"
        ),
    )
    parser.add_argument(
        "--line-scale", type=float, default=DEFAULT_OVERVIEW_LINE_SCALE,
        help=(
            "line width = 0.1 + line_scale * raw_count for the cross-layer "
            "co-activation lines. Tune to match the magnitude of your pair "
            f"counts (default: {DEFAULT_OVERVIEW_LINE_SCALE:.0e}, matches "
            "eval-moe-mmlu/routing_graph_from_cpp.py)."
        ),
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        raise SystemExit(f"[error] --results-dir {results_dir} is not a directory")

    print(f"[load] results_dir = {results_dir.resolve()}")
    discovered = _discover_results(results_dir, args.models)
    if not discovered:
        raise SystemExit(
            f"[error] no <model>/<quant>/moe-*/expert_counts.json (or legacy "
            f"<model>/moe-*/expert_counts.json) found under {results_dir}"
        )

    # Apply --quants filter now (it's per-cell, not per-model).
    if args.quants:
        wanted = set(args.quants)
        discovered = [d for d in discovered if d[2] in wanted]
        if not discovered:
            raise SystemExit(
                f"[error] --quants={args.quants} matches no discovered cells"
            )

    summaries: list[dict[str, Any]] = []
    for model_dir, json_paths, quant in discovered:
        label = f"{model_dir.name}" + (f"/{quant}" if quant else " (legacy)")
        print(f"\n[model] {label}: {len(json_paths)} dataset JSON(s)")
        # Output directory lives inside the per-quantization directory when
        # the new structure is in use; under the model directory for legacy.
        output_dir = (
            (model_dir / quant / "overall") if quant
            else (model_dir / "overall")
        )
        summary = process_model(
            model_dir, json_paths, output_dir,
            top_k_fraction=args.top_k_fraction,
            include_per_dataset=args.include_per_dataset,
            dpi=args.dpi, quant=quant,
            top_k_pairs=args.top_k_pairs,
            line_scale=args.line_scale,
        )
        summaries.append(summary)

    # Stdout summary table.
    print("\n[summary]")
    cols = ("model", "quant", "arch", "LxE", "top_k", "datasets", "tokens")
    header_widths = {"LxE": 10, "top_k": 6, "datasets": 8}
    print("  ".join(
        f"{c:<{header_widths.get(c, 28)}}" for c in cols
    ))
    for s in summaries:
        arch = s["arch"]
        print(
            "  ".join([
                f"{(s['model_id'] or s['model_dir'].name):<28}",
                f"{(s['quant'] or '-'):<28}",
                f"{arch['name']:<28}",
                f"{arch['n_layer']}x{arch['n_expert']:<6}",
                f"{s['top_k_count']:<6}",
                f"{s['n_datasets_used']:<8}",
                f"{s['tokens_total']:,}",
            ])
        )
    n_models = len({s['model_dir'].name for s in summaries})
    n_quants = len({s['quant'] for s in summaries})
    print(f"\n[done] wrote overview artifacts for {len(summaries)} "
          f"(model, quant) cell(s) across {n_models} model(s) / "
          f"{n_quants} quant(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
