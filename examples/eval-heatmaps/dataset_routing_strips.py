#!/usr/bin/env python3
# type: ignore

"""Generate one publishable routing-strip heatmap per dataset in a model-quant cell.

This is a focused companion to `paper_heatmaps.py`: instead of rendering a single
routing strip per model-quant cell, it renders one strip for every dataset file
inside that cell, using the same publication styling and raw-count scale as the
main paper figure.

Example:

    python3 examples/eval-heatmaps/dataset_routing_strips.py \
        --cell-dir build/results/model/Q4_K_M \
        --output-dir build/results/model/paper_dataset_routing_strips
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Import the shared rendering + recovery helpers from the paper script so all
# published heatmaps keep the same look and behaviour.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import paper_heatmaps  # noqa: E402


def _dataset_output_name(dataset_dir_name: str) -> str:
    """Return the publishable label used in the output filename.

    Known aliases map to their paper-ready label. Unknown dataset names keep the
    raw unprefixed directory stem so the output remains stable when a new dataset
    is added without requiring a code update.
    """
    known = {
        "moe-bigbench": "BIG-Bench Hard",
        "moe-humaneval": "HumanEval",
        "moe-include": "INCLUDE",
        "moe-mmlu": "MMLU",
        "moe-popqa": "PopQA",
        "random": "Uniform",
    }
    if dataset_dir_name in known:
        return known[dataset_dir_name]
    if dataset_dir_name.startswith("moe-"):
        return dataset_dir_name[len("moe-") :]
    return dataset_dir_name


_MODEL_PUBLISH_LABELS: dict[str, str] = {
    "allenai--OLMoE-1B-7B-0125-Instruct-GGUF": "OLMoE-1B-7B",
    "LiteLLMs--Mixtral-8x22B-Instruct-v0.1-GGUF": "Mixtral-8x22B",
    "mradermacher--deepseek-moe-16b-chat-GGUF": "DeepSeek-MoE-16B",
    "unsloth--Qwen3-30B-A3B-GGUF": "Qwen3-30B-A3B",
    "allenai--OLMoE-1B-7B-0924-Instruct": "OLMoE-1B-7B",
    "mistralai--Mixtral-8x22B-Instruct-v0.1": "Mixtral-8x22B",
    "deepseek-ai--deepseek-moe-16b-chat": "DeepSeek-MoE-16B",
    "Qwen--Qwen3-30B-A3B": "Qwen3-30B-A3B",
}


def _publish_model_label(model_dir_name: str) -> str:
    """Use the explicit cross-quantization HF-id -> publishable-name mapping."""
    return _MODEL_PUBLISH_LABELS.get(model_dir_name, model_dir_name)


def _dataset_info_for(json_path: Path, cell_dir: Path) -> dict[str, str]:
    """Return dataset metadata used by the output filename and summary output."""
    dataset_dir = json_path.parent.name
    model_name = cell_dir.parent.name
    quant_name = cell_dir.name
    dataset_label = _dataset_output_name(dataset_dir)
    return {
        "dataset_dir": dataset_dir,
        "model_name": model_name,
        "model_label": _publish_model_label(model_name),
        "quant_name": quant_name,
        "output_stem": f"{model_name}__{quant_name}__{dataset_label}__routing_strip",
    }


def _iter_dataset_jsons(cell_dir: Path) -> list[Path]:
    """Return a stable list of `expert_counts.json` dataset files in a cell."""
    if not cell_dir.is_dir():
        return []
    jsons: list[Path] = []
    for sub in sorted(cell_dir.iterdir(), key=lambda p: p.name):
        if not sub.is_dir():
            continue
        candidate = sub / "expert_counts.json"
        if candidate.is_file():
            jsons.append(candidate)
    return jsons


def _load_counts_for_dataset(json_path: Path) -> np.ndarray:
    """Load and validate the marginal count matrix for one dataset JSON."""
    data = paper_heatmaps._safe_json_load(json_path)
    if data is None:
        raise ValueError(f"unable to load JSON from {json_path}")
    arch_in = data.get("model_arch") or {}
    L = int(arch_in.get("n_layer", 0))
    E = int(arch_in.get("n_expert", 0))
    if L <= 0 or E <= 0:
        raise ValueError(f"malformed model_arch in {json_path}")
    marginal = (data.get("aggregate") or {}).get("marginal_expert_counts")
    if marginal is None:
        raise ValueError(f"missing aggregate.marginal_expert_counts in {json_path}")
    counts = np.asarray(marginal, dtype=np.int64)
    if counts.shape != (L, E):
        raise ValueError(f"shape {counts.shape} != ({L}, {E}) in {json_path}")
    return counts


def _render_dataset_routing_strip(
    counts: np.ndarray,
    *,
    width_in: float = 3.3,
    dpi: int = 300,
    cmap: str = "Blues",
    title: str | None = None,
) -> Any:
    """Render a single dataset routing-strip figure using paper_heatmaps styles."""
    counts_arr = np.asarray(counts, dtype=np.float64)
    fig = paper_heatmaps._render_paper_routing_strip(
        counts_arr,
        width_in=width_in,
        dpi=dpi,
        cmap=cmap,
    )
    ax = fig.axes[0]
    if title:
        ax.set_title(title, fontsize=9, pad=8)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cell-dir",
        type=Path,
        required=True,
        help="Single model-quant cell directory, e.g. .../model/Q4_K_M",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write dataset routing-strip PNGs. Defaults to the cell directory itself.",
    )
    parser.add_argument(
        "--width-in",
        type=float,
        default=3.3,
        help="Figure width in inches (default: 3.3).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG DPI for the output figures (default: 300).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="Blues",
        help="Matplotlib colormap used for the routing strip (default: Blues).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing PNG files.",
    )
    args = parser.parse_args()

    cell_dir = args.cell_dir.resolve()
    if not cell_dir.is_dir():
        raise SystemExit(f"[error] cell directory does not exist: {cell_dir}")

    output_dir = args.output_dir.resolve() if args.output_dir is not None else cell_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_jsons = _iter_dataset_jsons(cell_dir)
    if not dataset_jsons:
        print(f"[warn] no dataset expert_counts.json files under {cell_dir}")
        return 0

    paper_heatmaps._setup_mpl(args.dpi)

    rendered = 0
    for json_path in dataset_jsons:
        info = _dataset_info_for(json_path, cell_dir)
        out_path = output_dir / f"{info['output_stem']}.png"
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path} exists; pass --force to regenerate")
            continue

        try:
            counts = _load_counts_for_dataset(json_path)
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            print(f"[warn] {json_path}: {exc}; skipping")
            continue

        fig = _render_dataset_routing_strip(
            counts,
            width_in=args.width_in,
            dpi=args.dpi,
            cmap=args.cmap,
            title=f"{info['model_label']} — {_dataset_output_name(info['dataset_dir'])}",
        )
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        fig.clf()
        plt = paper_heatmaps.plt
        plt.close(fig)
        rendered += 1
        print(f"[save] {out_path}  (dataset={info['dataset_dir']})")

    print(f"[done] wrote {rendered} routing-strip figure(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
