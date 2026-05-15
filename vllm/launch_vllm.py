"""
launch_vllm.py — start vLLM's OpenAI-compatible server with two LoRA adapters
preloaded.

The benchmark client picks adapters in round-robin fashion, so the server
must have both `llama3-toy-lora-0` and `llama3-toy-lora-1` registered as
named LoRA modules.
"""
import argparse
import os
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LORA0_DIR = REPO_ROOT / "llama3-toy-lora-0"
LORA1_DIR = REPO_ROOT / "llama3-toy-lora-1"

# Local Llama-3.1-8B snapshot. Avoids hitting the HF hub for the gated model.
DEFAULT_MODEL = (
    "/home/jiaxuan_chen/scratch/models--meta-llama--Meta-Llama-3.1-8B"
    "/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Base model (HF hub id or local path).")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-loras", type=int, default=2,
                    help="Max number of LoRAs vLLM batches concurrently.")
    ap.add_argument("--max-lora-rank", type=int, default=16,
                    help="Must be >= the rank stored in each adapter (16).")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--extra", default="",
                    help="Extra flags forwarded verbatim to `vllm serve`.")
    args = ap.parse_args()

    for p in (LORA0_DIR, LORA1_DIR):
        if not p.exists():
            sys.exit(f"LoRA adapter not found: {p}")

    lora_modules = [
        f"llama3-toy-lora-0={LORA0_DIR}",
        f"llama3-toy-lora-1={LORA1_DIR}",
    ]

    cmd = [
        "vllm", "serve", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--enable-lora",
        "--max-loras", str(args.max_loras),
        "--max-lora-rank", str(args.max_lora_rank),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--lora-modules", *lora_modules,
        "--no-enable-prefix-caching",
    ]
    if args.extra:
        cmd += shlex.split(args.extra)

    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
