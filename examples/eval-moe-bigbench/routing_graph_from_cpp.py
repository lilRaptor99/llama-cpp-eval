#!/usr/bin/env python3
# type: ignore

"""Per-dataset wrapper around `examples/_shared/moe_routing_graph.py` for
the BBH eval. Consumes the aggregate block from the
`llama-eval-moe-bigbench` output JSON.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the shared library importable. The eval-moe-* dirs are siblings of
# _shared under examples/, so we add examples/ to sys.path.
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from _shared import moe_routing_graph as _shared  # noqa: E402


def main() -> None:
    _shared.run_main("bigbench")


if __name__ == "__main__":
    main()