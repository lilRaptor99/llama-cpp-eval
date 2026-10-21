#!/usr/bin/env python3
# type: ignore

"""Shared CSV / Markdown emitters for the pairwise routing-similarity tables.

Two sibling figures render the SAME matrices this module tabulates:

  * `cross_model_comparison.py`       -> cross-DATASET (models x models)
  * `cross_quantization_comparison.py` -> cross-QUANT  (quants x quants)

Each draws a 2 x N figure whose cells are layer-averaged pairwise
Jaccard / Jensen-Shannon divergence values. A printed paper cannot
ship a heatmap as the only record of those numbers, so the two table
CLIs (`cross_dataset_table.py`, `cross_quant_matrix_table.py`) feed
their matrices through this module to emit:

  *.csv   long-form, one row per (model, quant_ref, metric, unit_i,
          unit_j) cell, at full float64 precision. This is the
          machine-readable source of truth.
  *.md    one upper-triangular Markdown table per (block, metric),
          renderable in GitHub / a PR. Cells below the adaptive
          threshold switch to scientific notation so the ~1e-4
          cross-quant JSD cells do not all collapse to `0.000`.

Deliberately NOT emitted: `.tex` and `.txt`. The paper pulls the
Markdown tables through its own LaTeX pipeline, and a fixed-width
terminal view adds nothing over the CSV.

CSV precision contract
----------------------
Values are written with `repr(float)`, which in Python 3 yields the
shortest decimal string that round-trips to the identical float64
(`repr(2.4e-4)` -> `'0.00024'`, `repr(0.1)` -> `'0.1'`). Full
precision, no noise digits, byte-stable across runs.

Markdown precision contract
---------------------------
`value` is always the layer-mean the heatmap drew. `layer_std` is the
population standard deviation (ddof=0) of the SAME per-layer quantity
across the L layers, so `value ± layer_std` tells the reader how
stable that cell's agreement is with depth. The Markdown shows both
inline, matching the CSV columns one-for-one.

Notation switches to scientific (`2.4e-04`) per (block, metric) when
the largest off-diagonal magnitude in that matrix falls below
`SCI_THRESHOLD` (1e-2) — chosen so cross-quant JSD blocks, whose
largest off-diagonal is typically ~1e-4 to 1e-2, stay legible instead
of rendering as a wall of `0.000`. Jaccard blocks never trip it.

Diagonal and lower-triangle cells render as `—` (em dash): the
matrices are symmetric and the diagonal is tautological (Jaccard
exactly 1.0, JSD exactly 0.0). The CSV still carries every cell so
downstream tooling gets the full square matrix.

Importable helpers (no CLI here): `PairwiseBlock`, `write_csv`,
`write_markdown`, `verify_block`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# A matrix whose largest off-diagonal cell is below this magnitude is
# rendered in scientific notation instead of fixed decimals.
SCI_THRESHOLD: float = 1e-2

# Fixed-decimal digits used when a matrix stays above SCI_THRESHOLD.
# 3 works for both metrics: Jaccard reads `0.873`, JSD reads `0.087`.
FIXED_DECIMALS: int = 3

# Significant digits used in scientific mode. `.1e` == 2 sig figs,
# which matches the precision the heatmap cell text conveys.
SCI_SIGFIGS: int = 1

# Placeholder for "symmetric or diagonal, deliberately omitted".
DASH: str = "—"

# Metric keys -> (display name, human-readable definition). Used for
# the Markdown sub-headers and the summary table.
METRIC_TITLES: dict[str, str] = {
    "jaccard": "Jaccard similarity",
    "jsd": "JS divergence",
}


# ------------------------------------------------------------------- data

@dataclass
class PairwiseBlock:
    """One square symmetric matrix block: a single model at a single
    quantisation level, tabulated across its datasets or its quants.

    `mean`/`std` pairs are the layer-averaged value and the
    across-layer population std, both `(N, N)`, produced by the
    `_layer_averaged_*_with_std` helpers in `paper_heatmaps.py`.
    """
    model: str                  # short publishable name
    model_safe: str             # on-disk results directory id
    quant_ref: str              # owning quant, "" for the cross-quant view
    unit_kind: str              # "dataset" | "quant" — what labels index the matrix
    labels: list[str]           # axis labels, in figure order
    arch: dict[str, Any]        # name / n_layer / n_expert / n_expert_used
    jaccard_mean: np.ndarray
    jaccard_std: np.ndarray
    jsd_mean: np.ndarray
    jsd_std: np.ndarray

    @property
    def n_layers(self) -> int:
        return int(self.arch["n_layer"])

    @property
    def n_expert(self) -> int:
        return int(self.arch["n_expert"])

    @property
    def k_model(self) -> int:
        return int(self.arch["n_expert_used"])

    def metric(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return `(mean, std)` for `'jaccard'` or `'jsd'`."""
        if name == "jaccard":
            return self.jaccard_mean, self.jaccard_std
        if name == "jsd":
            return self.jsd_mean, self.jsd_std
        raise KeyError(name)


# -------------------------------------------------------------- formatting

def format_cell(value: float, std: float, *, sci: bool) -> str:
    """Render one Markdown cell as `value ± std`.

    `sci=True` switches both numbers to scientific notation; the
    standard deviation uses the same magnitude as the value so the
    reader can see at a glance whether a cell is tight or noisy
    (e.g. `2.4e-04 ± 1.1e-05` vs `2.4e-04 ± 8.0e-03`).
    """
    if sci:
        return (f"{value:.{SCI_SIGFIGS}e} ± "
                f"{std:.{SCI_SIGFIGS}e}")
    return f"{value:.{FIXED_DECIMALS}f} ± {std:.{FIXED_DECIMALS}f}"


def _needs_sci(mat: np.ndarray, exclude: int | None = None) -> bool:
    """True when the largest off-diagonal magnitude is below the
    scientific-notation threshold.

    `exclude` drops an axis position (the random-router baseline) from
    the comparison. This matters: the baseline row is a real-routing
    vs. random-router comparison and lands ~3 orders of magnitude above
    the unit-vs-unit values, so including it would push the whole block
    into fixed decimals and collapse every meaningful cell to `0.000`.
    The baseline is a different kind of comparison on a different scale;
    it gets reported in the same notation but does not dictate it.

    Guarded against the degenerate all-zero / empty cases, which stay
    in fixed notation.
    """
    n = mat.shape[0]
    if n < 2:
        return False
    idx = [i for i in range(n) if i != exclude]
    pairs = [(i, j) for a, i in enumerate(idx) for j in idx[a + 1:]]
    if not pairs:
        return False
    m = max(abs(float(mat[i, j])) for i, j in pairs)
    return 0.0 < m < SCI_THRESHOLD


def _fmt_csv_float(v: float) -> str:
    """Full-precision, round-trippable float formatting for CSV."""
    return repr(float(v))


# ----------------------------------------------------------------- summary

def _offdiag_pairs(n: int, exclude: int | None) -> list[tuple[int, int]]:
    """Upper-triangular index pairs, optionally dropping every pair that
    touches the `exclude` axis position (the random-router baseline)."""
    idx = [i for i in range(n) if i != exclude]
    return [(i, j) for a, i in enumerate(idx) for j in idx[a + 1:]]


def _baseline_pairs(n: int, base: int) -> list[tuple[int, int]]:
    """Index pairs joining the baseline axis to every real unit."""
    return [(base, j) for j in range(n) if j != base]


def summary_rows(blocks: list[PairwiseBlock],
                 baseline_label: str) -> list[dict[str, Any]]:
    """One condensed row per block, for the single-column summary table.

    Reports the mean off-diagonal over the *real* units (baseline
    excluded, so the random router cannot drag the average down), plus
    the mean against the baseline alone — the latter is the most
    interpretable single number in the whole paper: how far real
    routing sits from a uniform random router on that model/quant.
    """
    rows: list[dict[str, Any]] = []
    for b in blocks:
        n = len(b.labels)
        base = (b.labels.index(baseline_label)
                if baseline_label in b.labels else None)
        pairs = _offdiag_pairs(n, base)
        entry: dict[str, Any] = {
            "model": b.model,
            "quant_ref": b.quant_ref,
            "units": n,
            "n_layers": b.n_layers,
        }
        for metric in ("jaccard", "jsd"):
            mean_mat, _ = b.metric(metric)
            entry[f"{metric}_real"] = (
                float(np.mean([mean_mat[i, j] for i, j in pairs]))
                if pairs else float("nan")
            )
            if base is None:
                entry[f"{metric}_baseline"] = float("nan")
            else:
                bp = _baseline_pairs(n, base)
                entry[f"{metric}_baseline"] = (
                    float(np.mean([mean_mat[i, j] for i, j in bp]))
                    if bp else float("nan")
                )
        rows.append(entry)
    return rows


# -------------------------------------------------------------------- CSV

CSV_HEADER: list[str] = [
    "model", "model_safe", "quant_ref", "unit_kind", "unit_i", "unit_j",
    "metric", "value", "layer_std", "n_layers", "n_expert", "k_model",
]


def write_csv(path: Path, blocks: list[PairwiseBlock]) -> Path:
    """Write the long-form CSV.

    Every cell of every matrix is emitted, diagonal and lower
    triangle included, so the file is a complete round-trippable
    record rather than a display view. `quant_ref` is empty for the
    cross-quant view, where the unit axis *is* the quantisation axis.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for b in blocks:
            for metric in ("jaccard", "jsd"):
                mean_mat, std_mat = b.metric(metric)
                n = mean_mat.shape[0]
                for i in range(n):
                    for j in range(n):
                        writer.writerow([
                            b.model, b.model_safe, b.quant_ref,
                            b.unit_kind, b.labels[i], b.labels[j],
                            metric,
                            _fmt_csv_float(mean_mat[i, j]),
                            _fmt_csv_float(std_mat[i, j]),
                            b.n_layers, b.n_expert, b.k_model,
                        ])
    return path


# --------------------------------------------------------------- Markdown

def _md_matrix_table(block: PairwiseBlock, metric: str,
                     baseline_index: int | None) -> list[str]:
    """One upper-triangular Markdown table for `(block, metric)`.

    `baseline_index` is the axis position of the random-router
    baseline, or None when the block has none. It is excluded from the
    notation decision but still rendered, so the reader keeps the
    baseline comparison on the same page as the rest.
    """
    mean_mat, std_mat = block.metric(metric)
    sci = _needs_sci(mean_mat, exclude=baseline_index)
    labels = block.labels
    n = len(labels)

    lines = [f"**{METRIC_TITLES[metric]}** "
             f"({'scientific notation' if sci else 'fixed decimals'}; "
             f"`{DASH}` = diagonal or symmetric duplicate)", ""]
    lines.append("| | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * n)
    for i in range(n):
        cells = [labels[i]]
        for j in range(n):
            if j <= i:
                cells.append(DASH)
            else:
                cells.append(format_cell(float(mean_mat[i, j]),
                                         float(std_mat[i, j]), sci=sci))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _md_summary_table(blocks: list[PairwiseBlock],
                      baseline_label: str) -> list[str]:
    """The condensed single-column summary table."""
    rows = summary_rows(blocks, baseline_label)
    lines = [
        "| Model | Quant | Units | L | "
        "mean Jaccard | mean JSD | mean Jaccard vs. baseline | "
        "mean JSD vs. baseline |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        def _f(v: float) -> str:
            return DASH if not np.isfinite(v) else f"{v:.3f}"

        lines.append(
            f"| {r['model']} | {r['quant_ref'] or DASH} | {r['units']} "
            f"| {r['n_layers']} | {_f(r['jaccard_real'])} "
            f"| {_f(r['jsd_real'])} | {_f(r['jaccard_baseline'])} "
            f"| {_f(r['jsd_baseline'])} |"
        )
    lines.append("")
    return lines


def write_markdown(path: Path, blocks: list[PairwiseBlock], *,
                   title: str, caption: str,
                   baseline_label: str) -> Path:
    """Write the human-readable Markdown report.

    Structure: title, caption, condensed summary, then one section per
    block with its metadata header and one matrix per metric.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"# {title}", "", caption, ""]

    lines += ["## Summary", "",
              "Mean off-diagonal values. The `mean Jaccard` / `mean JSD` "
              "columns exclude the random-router baseline "
              f"(`{baseline_label}`) and average the real units pairwise; "
              f"the `vs. baseline` columns average only the "
              f"`{baseline_label}` row, i.e. real routing against a "
              "uniform random router.", ""]
    lines += _md_summary_table(blocks, baseline_label)

    lines += ["## Per-model matrices", ""]
    for b in blocks:
        heading = (f"### {b.model} @ {b.quant_ref}" if b.quant_ref
                   else f"### {b.model}")
        baseline_index = (b.labels.index(baseline_label)
                          if baseline_label in b.labels else None)
        lines += [
            heading,
            "",
            f"`L`={b.n_layers} layers, `E`={b.n_expert} experts, "
            f"`K_model`={b.k_model}; {b.unit_kind}s: "
            f"{', '.join(b.labels)}.",
            "",
        ]
        for metric in ("jaccard", "jsd"):
            lines += _md_matrix_table(b, metric, baseline_index)

    path.write_text("\n".join(lines) + "\n")
    return path


# ----------------------------------------------------------- verification

def verify_block(block: PairwiseBlock, *,
                 jaccard_ref: np.ndarray, jsd_ref: np.ndarray,
                 tol: float = 1e-12) -> list[str]:
    """Cross-check a block's `*_with_std` means against the std-free
    matrices the figure scripts actually draw.

    The `_with_std` helpers are used by the tables but not by the
    figures. If they ever drift — a different top-K tie-break, a
    changed summation order — the tables would silently report numbers
    the published heatmap does not show. `--self-check` recomputes the
    figure's own matrices and asserts agreement. Returns a list of
    human-readable failures (empty = clean).
    """
    problems: list[str] = []
    n = len(block.labels)
    tag = f"{block.model} ({block.quant_ref or 'all quants'})"

    for metric, ref in (("jaccard", jaccard_ref), ("jsd", jsd_ref)):
        mean_mat, std_mat = block.metric(metric)
        if mean_mat.shape != (n, n) or std_mat.shape != (n, n):
            problems.append(f"{tag}: {metric} matrices are not ({n}, {n})")
            continue
        if ref.shape != (n, n):
            problems.append(
                f"{tag}: {metric} reference matrix is {ref.shape}, "
                f"expected ({n}, {n})")
            continue
        if not np.allclose(mean_mat, ref, rtol=0.0, atol=tol):
            worst = float(np.abs(mean_mat - ref).max())
            problems.append(
                f"{tag}: {metric} mean matrix disagrees with the figure "
                f"matrix by up to {worst:.3e} (tolerance {tol:.0e})")
        if not np.allclose(mean_mat, mean_mat.T, rtol=0.0, atol=tol):
            problems.append(f"{tag}: {metric} mean matrix is not symmetric")
        if not np.allclose(std_mat, std_mat.T, rtol=0.0, atol=tol):
            problems.append(f"{tag}: {metric} std matrix is not symmetric")
        if (std_mat < 0.0).any():
            problems.append(f"{tag}: {metric} std matrix has negative "
                            "entries (population std cannot be negative)")

    # Diagonal invariants: top-K set of a dataset against itself is the
    # whole set, so Jaccard is exactly 1.0 and JSD exactly 0.0.
    if not np.allclose(np.diag(block.jaccard_mean), 1.0, rtol=0.0, atol=tol):
        problems.append(f"{tag}: jaccard diagonal is not all 1.0")
    if not np.allclose(np.diag(block.jsd_mean), 0.0, rtol=0.0, atol=tol):
        problems.append(f"{tag}: jsd diagonal is not all 0.0")

    return problems