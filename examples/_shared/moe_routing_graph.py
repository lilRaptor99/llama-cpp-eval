#!/usr/bin/env python3
# type: ignore

"""Shared MoE routing-graph rendering library for the eval-moe-* tools.

Consumed by `examples/eval-moe-{mmlu,bigbench,humaneval,popqa,include}/routing_graph_from_cpp.py`.
Each per-dataset wrapper is a thin CLI that calls `run_main()` with its
dataset name.

Reads `expert_counts.json` produced by the updated C++ eval (which emits the
`aggregate` block: marginal firing counts + adjacent-layer pair counts).
Writes a single integrated PNG:

    routing_graph.png

Visual elements:
  - Each expert (L, e) is drawn as a circle on a simple grid, one row per
    MoE layer. Row spacing is decoupled from column spacing (rows are more
    widely spaced than columns) so the lines connecting adjacent-layer
    experts are clearly visible.
  - Circle fill = linearly-scaled marginal firing count (Blues colormap,
    white = 0, dark blue = max). No log scale.
  - Lines connect adjacent-layer (L, L+1) experts. For each layer pair only
    the top-K pairs by absolute count are drawn. Line thickness is
    proportional to the raw count (NOT normalised per layer pair), so the
    absolute volume of routing is comparable across the whole figure.
  - "Top co-activated" experts get a thick red ring. The selection mode is
    controlled by `--highlight-mode`:
      - `pair`        (default) — experts in BOTH the top 12.5%-by-marginal
        AND top 12.5%-by-outgoing-pair-sum for their layer.
      - `marginal`    — top 12.5% by marginal firing count only.
      - `pair-sum`    — top 12.5% by outgoing pair-sum only.

Schema required (from the updated C++ eval):
  {
    "model_arch": {"name": ..., "n_layer": L, "n_expert": E, "n_expert_used": k},
    "aggregate": {
      "marginal_expert_counts":   [[L, E] int64],
      "adjacent_pair_counts":    [[L-1, E, E] int64]
    }
  }
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import ScalarMappable


# ----- module-level defaults (overridable via CLI) -----
DEFAULT_COL_SPACING = 1.0     # horizontal distance between expert columns
DEFAULT_ROW_SPACING = 2.5     # vertical distance between layer rows
                                # (deliberately > col_spacing so cross-layer
                                #  lines are easy to follow)
DEFAULT_TOP_K_PAIRS = 64
DEFAULT_TOP_FRAC = 0.125      # 12.5% — highlight-ring fraction per layer
# Highlight ring: bright red contrasts sharply with the Blues circles and
# the dark-blue connection lines, so the highlighted experts stand out
# clearly even on dense plots.
HIGHLIGHT_EDGE_COLOR = "#ff1744"  # red
HIGHLIGHT_EDGE_WIDTH = 4.0
# Allowed values for --highlight-mode.
HIGHLIGHT_MODES = ("pair", "marginal", "pair-sum")
DEFAULT_HIGHLIGHT_MODE = "pair"
# line width = 0.1 + line_scale * raw_count. Tune via --line-scale to match
# the magnitude of your pair counts. OLMoE-scale counts (~1e7) want ~1e-7;
# Mixtral-scale counts (~1e6) want ~1e-6.
DEFAULT_LINE_WIDTH_SCALE = 5e-7


# ============================================================ JSON loading


def load_aggregate(path: Path) -> dict:
    """Load the C++ eval JSON and extract aggregate counts as numpy arrays.

    Returns a dict with keys: L, E, k, arch, marginal, adj.

    Aborts with sys.exit if the JSON lacks the `aggregate` block (i.e. was
    produced by an older eval-moe-* binary without the coactivation patch).
    """
    with open(path) as fh:
        d = json.load(fh)

    arch = d.get("model_arch")
    if not isinstance(arch, dict):
        sys.exit("error: input JSON missing `model_arch` block")

    agg = d.get("aggregate")
    if not isinstance(agg, dict):
        sys.exit(
            "error: input JSON missing `aggregate` block.\n"
            "       this tool requires the C++ `aggregate` block emitted by the\n"
            "       updated eval-moe-* binary (with co-activation capture).\n"
            "       re-run your eval with the rebuilt llama-eval-moe-<dataset>."
        )

    marginal = np.array(agg.get("marginal_expert_counts"), dtype=np.int64)
    adj_raw = agg.get("adjacent_pair_counts", [])
    # Accept [] as a valid empty adjacent array (happens when n_layer == 1).
    if isinstance(adj_raw, list) and len(adj_raw) == 0:
        adj = None  # We'll fix up the shape below once we know (L, E).
    else:
        adj = np.array(adj_raw, dtype=np.int64)

    if marginal.ndim != 2:
        sys.exit(f"error: aggregate.marginal_expert_counts expected [L, E], got shape {marginal.shape}")

    L, E = int(marginal.shape[0]), int(marginal.shape[1])
    if adj is None:
        adj = np.zeros((max(L - 1, 0), E, E), dtype=np.int64)
    elif adj.ndim != 3:
        sys.exit(f"error: aggregate.adjacent_pair_counts expected [L-1, E, E], got shape {adj.shape}")

    if adj.shape[0] != max(L - 1, 0):
        sys.exit(f"error: adjacent_pair_counts L-axis ({adj.shape[0]}) != n_layer - 1 ({L - 1})")
    if adj.shape[1] != E or adj.shape[2] != E:
        sys.exit(f"error: adjacent_pair_counts E-axes ({adj.shape[1:]}) != n_expert ({E})")

    k = int(arch.get("n_expert_used", 0))
    # Per-layer minimum n_tokens observed across all decode calls (optional,
    # absent in JSONs produced by older eval-moe-* binaries). Useful for
    # prefill-only benchmarks (mmlu) where the last-layer ggml_get_rows
    # collapse shrinks the layer from a few hundred prefill tokens down to
    # 1. For GENERATIVE benchmarks (popqa/humaneval/bigbench) the min is
    # 1 for ALL layers because every generated step is n_tokens=1, so the
    # layer_min field alone can NOT tell the last layer from any other. We
    # use it here only as supporting evidence; the primary collapse signal
    # is the per-layer MARGINAL TOTAL (see _compute_collapse_note below).
    layer_min = agg.get("layer_min_topk_n_tokens")
    if layer_min is not None:
        layer_min = list(layer_min)
        if len(layer_min) != L:
            layer_min = None  # shape mismatch — ignore
    marginal_sum = np.asarray(marginal, dtype=np.int64).sum(axis=1).tolist()
    collapse_note = _compute_collapse_note(marginal_sum, layer_min)
    return {
        "L": L,
        "E": E,
        "k": k,
        "arch": arch.get("name", "unknown"),
        "marginal": marginal,
        "adj": adj,
        "layer_min_topk_n_tokens": layer_min,
        "collapse_note": collapse_note,
    }


def _compute_collapse_note(
    marginal_per_layer_sum: list[int],
    layer_min: list[int] | None = None,
) -> str:
    """Build a one-line title annotation listing collapsed / dense layers.

    The PRIMARY signal is the per-layer marginal total (`marginal_per_layer_sum`):
    a layer whose marginal is significantly smaller than the median across
    layers (default threshold: 50%) was evaluated with substantially fewer
    decode positions, almost always because of the universal pattern

        if (il == n_layer - 1 && inp_out_ids) cur = ggml_get_rows(cur, inp_out_ids);

    in every model file (olmoe, deepseek2, openai-moe, llama, ...). That
    shrinks the last layer to inp_out_ids.size() tokens per decode, so its
    marginal scales linearly with the number of decodes (Q) while every
    other layer scales linearly with prefill_length × k × Q (or, for
    generative benchmarks, prefill_length × k × Q + decode_steps × k × Q).

    `layer_min` (from `aggregate.layer_min_topk_n_tokens`) is used only as
    supporting evidence. For prefill-only benchmarks it shows the same
    collapse signature (1 vs ~600). For generative benchmarks it's always
    1 for every layer, so by itself it cannot distinguish collapsed from
    normal layers; the marginal-derived signal is then the source of truth.

    The returned string is multi-line (a "\\nNote: ..." block) ready to be
    appended to the matplotlib title.
    """
    if not marginal_per_layer_sum or len(marginal_per_layer_sum) == 0:
        return ""

    L = len(marginal_per_layer_sum)

    # Use the median (not the max) as the reference so a single collapsed
    # layer doesn't drag down the threshold. Zero marginals excluded.
    non_zero = [v for v in marginal_per_layer_sum if v > 0]
    if not non_zero:
        return ""  # all layers have zero marginal — nothing useful to report
    sorted_nz = sorted(non_zero)
    median = sorted_nz[len(sorted_nz) // 2]
    if median <= 0:
        return ""

    THRESHOLD_FRAC = 0.5
    threshold = max(1, int(median * THRESHOLD_FRAC))

    dense_layers: list[int] = []   # marginal == 0 (no MoE; cb never fired)
    collapsed: list[tuple[int, float]] = []  # marginal > 0 but < threshold

    for i, v in enumerate(marginal_per_layer_sum):
        if v == 0:
            dense_layers.append(i)
        elif v < threshold:
            collapsed.append((i, v / median))

    if not dense_layers and not collapsed:
        return ""  # all layers healthy
    if len(dense_layers) == L:
        return ""  # everything dense — no MoE evaluation happened

    parts: list[str] = []
    if dense_layers:
        if len(dense_layers) <= 4:
            parts.append("dense (no MoE): " + ", ".join(str(i) for i in dense_layers))
        else:
            parts.append(f"dense (no MoE): {len(dense_layers)} layers")

    if collapsed:
        # Sort by layer index for stable output.
        collapsed.sort(key=lambda x: x[0])
        if len(collapsed) <= 4:
            collapsed_str = ", ".join(
                f"L{i} ({ratio * 100:.1f}% of median)"
                for i, ratio in collapsed
            )
            parts.append(f"collapsed: {collapsed_str}")
        else:
            parts.append(
                f"collapsed: {len(collapsed)} layers "
                f"(smallest {min(r for _, r in collapsed) * 100:.1f}% of median)"
            )

    # Add a hint about WHY when the collapsed layers are clustered at the
    # end (typical ggml_get_rows last-layer-collapse fingerprint).
    if collapsed:
        collapsed_indices = [i for i, _ in collapsed]
        n_at_end = sum(1 for i in collapsed_indices if i == L - 1 or i == L - 2)
        hint = ""
        if n_at_end and n_at_end == len(collapsed_indices):
            hint = " (typically the last layer — ggml inp_out_ids shrinks it to n_outputs tokens)"
        elif len(collapsed_indices) >= 2 and all(
            i >= L - len(collapsed_indices) - 1 for i in collapsed_indices
        ):
            hint = " (likely last-layer ggml inp_out_ids collapse)"
        parts[-1] = parts[-1] + hint

    return "\nNote: " + "; ".join(parts)


# ============================================================ simple grid layout


def grid_positions(L: int, E: int, col_spacing: float, row_spacing: float) -> np.ndarray:
    """Return (L, E, 2) float array of (x, y) centres on a decoupled grid.

    x(L, e) = e * col_spacing
    y(L, e) = L * row_spacing   (caller inverts the y-axis so L0 is at the top)

    row_spacing is intentionally larger than col_spacing so the lines
    connecting adjacent-layer (L, L+1) experts have a clearly visible
    vertical distance to traverse.
    """
    if L == 0 or E == 0:
        return np.zeros((L, E, 2), dtype=np.float64)
    pos = np.zeros((L, E, 2), dtype=np.float64)
    for L_ in range(L):
        for e in range(E):
            pos[L_, e, 0] = e * col_spacing
            pos[L_, e, 1] = L_ * row_spacing
    return pos


# ============================================================ highlight set


def compute_highlight_set(
    marginal: np.ndarray,
    adj: np.ndarray,
    top_frac: float,
    mode: str = DEFAULT_HIGHLIGHT_MODE,
) -> list[set[int]]:
    """For each layer L, return the set of experts to highlight.

    `mode` selects how the highlight set is computed:
      - "pair"      — experts in BOTH the top top_frac-by-marginal AND the
        top top_frac-by-outgoing-pair-sum for the layer (intersection).
      - "marginal"  — top top_frac-by-marginal firing count only.
      - "pair-sum"  — top top_frac-by-outgoing pair-sum only.

    For the last layer (no outgoing adj entries), all three modes fall back
    to top-by-marginal so the highlight is still drawn.
    """
    if mode not in HIGHLIGHT_MODES:
        sys.exit(
            f"error: invalid --highlight-mode '{mode}'; "
            f"allowed: {', '.join(HIGHLIGHT_MODES)}"
        )

    L, E = marginal.shape
    k_each = max(1, int(math.ceil(E * top_frac)))

    # Top-k by marginal per layer (precomputed once; used by all modes and
    # by the last-layer fallback).
    top_marg = [set(np.argpartition(marginal[L_], -k_each)[-k_each:].tolist()) for L_ in range(L)]

    # For layer L < L-1, top-k by outgoing pair-sum adj[L][e, :].sum().
    # Last layer: no outgoing pairs -> use top_marg so highlight is non-empty.
    top_pair_per_layer: list[set[int]] = []
    for L_ in range(L):
        if L_ < adj.shape[0]:
            outgoing = adj[L_].sum(axis=1)
            top_pair_per_layer.append(
                set(np.argpartition(outgoing, -k_each)[-k_each:].tolist())
            )
        else:
            top_pair_per_layer.append(top_marg[L_])

    if mode == "marginal":
        return top_marg
    if mode == "pair-sum":
        return top_pair_per_layer
    # mode == "pair"  (intersection; default)
    return [top_marg[L_] & top_pair_per_layer[L_] for L_ in range(L)]


# ============================================================ top-K pair filter


def filter_top_k_pairs(adj: np.ndarray, K: int) -> list[list[tuple[int, int, int]]]:
    """For each layer pair L, return the top-K (e_i, e_j, count) triples.

    Drops zero-count entries. K is capped at E*E defensively.
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


# ============================================================ drawing


def millions_formatter(x: float, pos: int) -> str:
    """Colorbar tick formatter that prints in millions (e.g. 40.0M)."""
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.1f}K"
    return f"{int(x)}"


def draw_routing_graph(
    marginal: np.ndarray,
    adj: np.ndarray,
    highlight: list[set[int]],
    K: int,
    col_spacing: float,
    row_spacing: float,
    line_scale: float,
    path: Path,
    title_prefix: str = "Integrated routing graph",
    collapse_note: str = "",
) -> None:
    """Render the integrated routing graph to `path`.

    Layout: simple grid (one row per layer, E columns per row) with row
    spacing > col spacing so the lines connecting adjacent-layer experts
    have a clearly visible vertical distance to traverse. Circles are filled
    with linearly-scaled marginal firing count (Blues, no log scale).
    "Top co-activated" experts (in `highlight`) get a thick red ring.
    Lines connect top-K adjacent-layer (L, L+1) pairs by raw count; line
    thickness is proportional to raw count (no per-layer normalisation).
    A colourbar on the right encodes the marginal firing rate.

    `title_prefix` lets per-dataset wrappers add context (e.g. "(per-task)")
    to the auto-generated title without duplicating the rest of the title.

    `collapse_note` is an optional multi-line text annotation appended to
    the title. It typically lists layers that were evaluated with
    substantially fewer tokens than the dataset's typical prefill (e.g.
    the universal `ggml_get_rows(cur, inp_out_ids)` collapse on the last
    layer of every MoE model, OR dense leading layers that emit no
    `ffn_moe_topk-<il>` tensor at all). Empty string = no annotation.
    """
    L, E = marginal.shape
    pos = grid_positions(L, E, col_spacing, row_spacing)

    # ---- figure sizing (auto-scaled to model shape) ----
    width = max(8.0, E * col_spacing)
    height = max(6.0, L * row_spacing * 0.9)
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- linear normalisation for the circle fill (NO log scale) ----
    marg_f = marginal.astype(np.float64)
    global_max = float(marg_f.max()) if marg_f.size > 0 else 1.0
    if global_max <= 0:
        global_max = 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=global_max)
    cmap = plt.get_cmap("Blues")

    # ---- expert circles: per-expert activation fill ----
    circle_radius = col_spacing * 0.4
    for L_ in range(L):
        for e in range(E):
            x, y = pos[L_, e]
            facecolor = cmap(norm(marg_f[L_, e]))
            circ = plt.Circle(
                (x, y), circle_radius,
                facecolor=facecolor,
                edgecolor="#bdbdbd",  # light grey border so zero-count circles are still visible
                linewidth=0.5,
                zorder=2,
            )
            ax.add_patch(circ)

    # ---- highlight rings: redraw on top, red outline ----
    for L_ in range(L):
        for e in highlight[L_]:
            x, y = pos[L_, e]
            ring = plt.Circle(
                (x, y), circle_radius * 1.08,
                facecolor="none",
                edgecolor=HIGHLIGHT_EDGE_COLOR,
                linewidth=HIGHLIGHT_EDGE_WIDTH,
                zorder=3,
            )
            ax.add_patch(ring)

    # ---- lines: top-K co-activation pairs per adjacent layer pair ----
    # Thickness is proportional to the RAW count (NOT normalised per layer
    # pair), so absolute volume is comparable across the whole figure.
    if L >= 2:
        filtered = filter_top_k_pairs(adj, K)
        for L_ in range(L - 1):
            for (e_i, e_j, count) in filtered[L_]:
                if count <= 0:
                    continue
                x1, y1 = pos[L_, e_i]
                x2, y2 = pos[L_ + 1, e_j]
                lw = 0.1 + line_scale * count
                # Alpha scales with rank-ish: high counts get more opaque.
                alpha = min(0.85, 0.15 + 0.7 * math.tanh(count * line_scale * 5.0))
                ax.add_line(
                    Line2D(
                        [x1, x2], [y1, y2],
                        color="#1f3a93",  # dark blue, matches Blues family
                        linewidth=lw,
                        alpha=alpha,
                        solid_capstyle="round",
                        zorder=1,
                    )
                )

    # ---- axis limits (with margin for layer/expert labels) ----
    if L > 0 and E > 0:
        x_min = -col_spacing * 1.5
        x_max = float(pos[:, :, 0].max()) + col_spacing * 1.5
        y_min = -row_spacing * 0.5
        y_max = float(pos[:, :, 1].max()) + row_spacing * 0.4
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # invert so L0 is at top

    # ---- layer labels (left, right-aligned) ----
    for L_ in range(L):
        y = float(pos[L_, 0, 1])
        ax.text(
            -col_spacing * 0.4, y, f"{L_}",
            ha="right", va="center",
            fontsize=9, color="#333", family="monospace",
        )

    # ---- expert index labels (bottom, centred under each column) ----
    for e in range(E):
        x = float(pos[L - 1, e, 0])
        ax.text(
            x, -row_spacing * 0.55, f"{e}",
            ha="center", va="top",
            fontsize=9, color="#333", family="monospace",
        )

    # ---- axis labels ----
    if L > 0 and E > 0:
        ax.text(
            -col_spacing * 1.1, float(pos[:, :, 1].mean()),
            "Layer Index", rotation=90, ha="center", va="center",
            fontsize=11, color="#333",
        )
        ax.text(
            float(pos[:, :, 0].mean()), -row_spacing * 1.0,
            "Expert Index (within same layer)", ha="center", va="top",
            fontsize=11, color="#333",
        )

    # ---- colorbar for circle fill ----
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Token Processing Load", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

    # ---- title ----
    main_title = (
        f"{title_prefix} ({L} layers × {E} experts, "
        f"top {K} adjacent-layer pairs drawn)"
    )
    if collapse_note:
        # Stack the collapse note under the main title so it stays attached
        # to the figure but doesn't blow up the title font size. matplotlib
        # inserts \n as a real newline in set_title() output.
        main_title = main_title + collapse_note
    ax.set_title(main_title, fontsize=11, pad=14)

    # ---- save ----
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ============================================================ CLI glue


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Build the standard argparse for all per-dataset routing-graph wrappers."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-i", "--input", required=True, type=Path,
        help="path to expert_counts.json (must have an `aggregate` block)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output PNG path (default: <input_dir>/routing_graph.png)",
    )
    parser.add_argument(
        "--col-spacing", type=float, default=DEFAULT_COL_SPACING,
        help=(
            "horizontal distance between expert columns in data units "
            f"(default: {DEFAULT_COL_SPACING})"
        ),
    )
    parser.add_argument(
        "--row-spacing", type=float, default=DEFAULT_ROW_SPACING,
        help=(
            "vertical distance between layer rows in data units. "
            "Increase to make the cross-layer connection lines more visible. "
            f"(default: {DEFAULT_ROW_SPACING})"
        ),
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K_PAIRS,
        help=f"top-K adjacent-layer pairs to draw per layer pair (default: {DEFAULT_TOP_K_PAIRS})",
    )
    parser.add_argument(
        "--top-frac", type=float, default=DEFAULT_TOP_FRAC,
        help=(
            "fraction of experts per layer to highlight with a red ring "
            "(see --highlight-mode for selection rule; default: "
            f"{DEFAULT_TOP_FRAC}). Set to 0 to disable highlighting."
        ),
    )
    parser.add_argument(
        "--highlight-mode", type=str, default=DEFAULT_HIGHLIGHT_MODE,
        choices=HIGHLIGHT_MODES,
        help=(
            "selection rule for the highlight ring. "
            "'pair' (default): intersection of top top_frac-by-marginal "
            "AND top top_frac-by-outgoing-pair-sum (top co-activated). "
            "'marginal': top top_frac-by-marginal only. "
            "'pair-sum': top top_frac-by-outgoing-pair-sum only."
        ),
    )
    parser.add_argument(
        "--line-scale", type=float, default=DEFAULT_LINE_WIDTH_SCALE,
        help=(
            "line width = 0.1 + line_scale * raw_count. "
            f"Tune to match the magnitude of your counts (default: {DEFAULT_LINE_WIDTH_SCALE:.0e})"
        ),
    )
    return parser


def run_main(dataset_label: str, args: argparse.Namespace | None = None) -> None:
    """Run the standard routing-graph pipeline for any eval-moe-* dataset.

    `dataset_label` is shown in the banner (e.g. "mmlu", "popqa"). If
    `args` is None, parse from `sys.argv`.
    """
    if args is None:
        parser = build_arg_parser(
            description=(
                f"Render the integrated routing-graph PNG for the {dataset_label} eval. "
                "Consumes the aggregate block (marginal + adjacent pair counts)."
            )
        )
        args = parser.parse_args()

    data = load_aggregate(args.input)
    L, E, k, arch = data["L"], data["E"], data["k"], data["arch"]
    print(f"[load] dataset={dataset_label} input = {args.input}")
    print(f"[load] arch={arch} L={L} E={E}, k={k}")

    if L < 2:
        print(f"[warn] only {L} layer(s); no adjacent-layer pairs to draw")

    out_path = args.output
    if out_path is None:
        out_path = args.input.parent / "routing_graph.png"

    if args.top_frac > 0:
        highlight = compute_highlight_set(
            data["marginal"], data["adj"], args.top_frac, args.highlight_mode
        )
        n_hl = sum(len(s) for s in highlight)
        mode_label = {
            "pair":     "marginal ∩ pair-sum (co-activated)",
            "marginal": "marginal only (top-firing)",
            "pair-sum": "pair-sum only (top outgoing pairs)",
        }[args.highlight_mode]
        print(
            f"[load] highlight experts (top {args.top_frac:.3f} by {mode_label}): "
            f"{n_hl} total across {L} layers"
        )
    else:
        highlight = [set() for _ in range(L)]
        print(f"[load] highlight disabled (--top-frac 0)")

    # Per-layer token-collapse annotation (e.g. ggml_get_rows last-layer
    # collapse). Empty when the source JSON doesn't carry the field (older
    # eval-moe-* binaries) or when no layer is significantly smaller than
    # the dataset's typical prefill length.
    collapse_note = data.get("collapse_note", "")
    if collapse_note:
        # Surface the same note on stdout so headless users / CI logs can
        # see it without opening the PNG.
        print(collapse_note.strip().replace("\n", "  "))

    print(
        f"[draw] top-K={args.top_k} pairs/layer-pair, "
        f"col-spacing={args.col_spacing}, row-spacing={args.row_spacing}, "
        f"line-scale={args.line_scale:.2e}"
    )
    draw_routing_graph(
        marginal=data["marginal"],
        adj=data["adj"],
        highlight=highlight,
        K=args.top_k,
        col_spacing=args.col_spacing,
        row_spacing=args.row_spacing,
        line_scale=args.line_scale,
        path=out_path,
        title_prefix=f"Integrated routing graph ({dataset_label})",
        collapse_note=collapse_note,
    )
    print(f"[save] {out_path}")
    print("[done]")