"""
launch_sglang.py — start an SGLang server with both LoRA adapters preloaded.

The benchmark client picks adapters in round-robin fashion (per row_id), so
both `llama3-toy-lora-0` and `llama3-toy-lora-1` are registered as named
LoRA paths on the server.
"""
import argparse
import os
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LORA0_DIR = REPO_ROOT / "llama3-toy-lora-0"
LORA1_DIR = REPO_ROOT / "llama3-toy-lora-1"

# Local Llama-3.1-8B snapshot — same path used by the vLLM launcher.
DEFAULT_MODEL = (
    "/home/jiaxuan_chen/scratch/models--meta-llama--Meta-Llama-3.1-8B"
    "/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Base model (HF id or local path).")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-loras-per-batch", type=int, default=2,
                    help="Max LoRA adapters SGLang batches in one step.")
    ap.add_argument("--mem-fraction-static", type=float, default=0.85,
                    help="GPU memory fraction for KV cache / weights.")
    ap.add_argument("--context-length", type=int, default=4096)
    ap.add_argument("--disable-radix-cache", action="store_true", default=False,
                    help="Match vLLM's --no-enable-prefix-caching for fair comparison.")
    ap.add_argument("--max-running-requests", type=int, default=None,
                    help="Cap concurrent in-flight requests. SGLang's auto-tuner "
                         "can pick very high values from KV memory alone, which "
                         "inflates per-decode-step cost. Try 64-128 on this workload.")
    ap.add_argument("--lora-backend", default=None,
                    choices=["triton", "csgmv"],
                    help="LoRA kernel backend. Default (triton) is correct but "
                         "slower than csgmv (Punica/SGMV-class kernel).")
    ap.add_argument("--extra", default="",
                    help="Extra flags forwarded verbatim to sglang.launch_server.")
    args = ap.parse_args()

    for p in (LORA0_DIR, LORA1_DIR):
        if not p.exists():
            sys.exit(f"LoRA adapter not found: {p}")

    lora_paths = [
        f"llama3-toy-lora-0={LORA0_DIR}",
        f"llama3-toy-lora-1={LORA1_DIR}",
    ]

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--context-length", str(args.context_length),
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--enable-lora",
        "--max-loras-per-batch", str(args.max_loras_per_batch),
        "--lora-paths", *lora_paths,
    ]
    if args.disable_radix_cache:
        cmd.append("--disable-radix-cache")
    if args.max_running_requests is not None:
        cmd += ["--max-running-requests", str(args.max_running_requests)]
    if args.lora_backend is not None:
        cmd += ["--lora-backend", args.lora_backend]
    if args.extra:
        cmd += shlex.split(args.extra)

    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
