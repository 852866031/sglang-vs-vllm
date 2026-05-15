"""
launch_mini_sglang.py — start the mini-sglang API server.

Mini-sglang has NO LoRA support, so this launches the base model only.
The benchmark client (run_benchmark.py) sends plain /generate requests
without any adapter selection.
"""
import argparse
import os
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Local Llama-3.1-8B snapshot — same path used by the other engines.
DEFAULT_MODEL = (
    "/home/jiaxuan_chen/scratch/models--meta-llama--Meta-Llama-3.1-8B"
    "/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Base model (HF id or local path).")
    ap.add_argument("--port", type=int, default=1919)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--memory-ratio", type=float, default=0.85,
                    help="Fraction of GPU memory used for KV cache.")
    ap.add_argument("--max-seq-len-override", type=int, default=4096)
    ap.add_argument("--max-running-requests", type=int, default=256)
    ap.add_argument("--extra", default="",
                    help="Extra flags forwarded verbatim to `python -m minisgl`.")
    args = ap.parse_args()

    cmd = [
        sys.executable, "-m", "minisgl",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--tp-size", str(args.tp_size),
        "--memory-ratio", str(args.memory_ratio),
        "--max-seq-len-override", str(args.max_seq_len_override),
        "--max-running-requests", str(args.max_running_requests),
    ]
    if args.extra:
        cmd += shlex.split(args.extra)

    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
