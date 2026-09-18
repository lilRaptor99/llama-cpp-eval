import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

HF_CACHE_DIR = Path("/mnt/data/huggingface")
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python scripts/download-models.py <hf_model_id> <quant>", file=sys.stderr)
        print("example: python scripts/download-models.py LiteLLMs/Mixtral-8x22B-Instruct-v0.1-GGUF Q4_K_M", file=sys.stderr)
        return 2

    model_id = sys.argv[1].strip()
    quant = sys.argv[2].strip()

    if not model_id:
        print("error: hf_model_id cannot be empty", file=sys.stderr)
        return 2
    if not quant:
        print("error: quant cannot be empty", file=sys.stderr)
        return 2

    # huggingface_hub allow_patterns uses shell-style globs (not regex).
    allow_patterns = [
        f"*{quant}*.gguf",
        f"*{quant}*.GGUF",
    ]

    print(f"  model: {model_id}", flush=True)
    print(f"  quant: {quant}", flush=True)
    print(f"  cache_dir: {HF_CACHE_DIR}", flush=True)
    print(f"  allow_patterns: {allow_patterns}", flush=True)
    snapshot_download(
        repo_id=model_id,
        allow_patterns=allow_patterns,
        cache_dir=str(HF_CACHE_DIR),
        tqdm_class=None,  # cleaner log output
    )
    print("  done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
