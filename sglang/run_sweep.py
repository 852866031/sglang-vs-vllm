#!/usr/bin/env python3
"""
run_sweep.py — execute the full SGLang variant ladder in one shot.

Skips any variant whose output CSV (../outputs/sglang_<tag>.csv) already
exists, so a re-run only fills in the gaps. Pass --force to re-run every
variant. Pass --tags A,B,C to restrict the sweep to specific entries.

The variant list mirrors section 4 of README.md.
"""
import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = REPO_ROOT / "outputs"


@dataclass
class Variant:
    tag: str
    launcher_args: str
    note: str


VARIANTS: List[Variant] = [
    # Incremental ladder — each step adds one flag on top of the previous.
    Variant(
        tag="default",
        launcher_args="",
        note="Baseline (no flags). csgmv + flashinfer already default.",
    ),
    Variant(
        tag="graph256",
        launcher_args="--extra '--cuda-graph-max-bs 256'",
        note="Bump CUDA-graph capture to cover the auto-picked running batch.",
    ),
    Variant(
        tag="graph256_nochunk",
        launcher_args="--extra '--cuda-graph-max-bs 256 --chunked-prefill-size -1'",
        note="Add: disable chunked prefill (no-op for 1-30 token prompts).",
    ),
    Variant(
        tag="graph256_nochunk_noradix",
        launcher_args=(
            "--disable-radix-cache "
            "--extra '--cuda-graph-max-bs 256 --chunked-prefill-size -1'"
        ),
        note="Add: disable radix-cache. Matches vLLM's --no-enable-prefix-caching.",
    ),
]


def out_csv_for(tag: str) -> Path:
    return DEFAULT_OUT_DIR / f"sglang_{tag}.csv"


def run_variant(v: Variant, extra_run_args: List[str], dry_run: bool) -> int:
    cmd = [sys.executable, str(SCRIPT_DIR / "run_benchmark.py"),
           "--run_tag", v.tag]
    if v.launcher_args:
        cmd += ["--launcher_args", v.launcher_args]
    cmd += extra_run_args

    print(f"\n{'=' * 78}", flush=True)
    print(f"[sweep] variant: {v.tag}", flush=True)
    print(f"[sweep] note:    {v.note}", flush=True)
    print(f"[sweep] command: {' '.join(repr(c) if ' ' in c else c for c in cmd)}", flush=True)
    print('=' * 78, flush=True)
    if dry_run:
        print("[sweep] --dry_run: skipping execution", flush=True)
        return 0

    t0 = time.monotonic()
    rc = subprocess.call(cmd)
    dt = time.monotonic() - t0
    status = "✅ ok" if rc == 0 else f"❌ exit {rc}"
    print(f"[sweep] {v.tag}: {status} in {dt:.1f}s", flush=True)
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--force", action="store_true",
        help="Re-run every variant even if its output CSV already exists.",
    )
    ap.add_argument(
        "--tags", default=None,
        help="Comma-separated list of variant tags to run. Default: all. "
             "Example: --tags default,graph256",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="List variants and whether each one's output already exists, "
             "then exit without running anything.",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Print the command for each scheduled variant but don't execute.",
    )
    ap.add_argument(
        "--stop_on_error", action="store_true",
        help="Abort the sweep on the first failing variant (default: keep going).",
    )
    ap.add_argument(
        "--pass", dest="passthrough", nargs=argparse.REMAINDER, default=[],
        help="Anything after `--pass` is forwarded verbatim to every "
             "run_benchmark.py invocation (e.g. --pass --fold 10).",
    )
    args = ap.parse_args()

    if args.tags:
        keep = {t.strip() for t in args.tags.split(",") if t.strip()}
        unknown = keep - {v.tag for v in VARIANTS}
        if unknown:
            sys.exit(f"unknown tag(s): {sorted(unknown)}. "
                     f"available: {[v.tag for v in VARIANTS]}")
        variants = [v for v in VARIANTS if v.tag in keep]
    else:
        variants = list(VARIANTS)

    print(f"[sweep] {len(variants)} variant(s) selected", flush=True)
    if args.list:
        for v in variants:
            out = out_csv_for(v.tag)
            mark = "EXISTS" if out.exists() else "missing"
            print(f"  [{mark}] {v.tag:30s} -> {out}")
            print(f"           {v.note}")
        return

    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    ran = 0
    skipped = 0
    failed: List[str] = []
    for v in variants:
        out = out_csv_for(v.tag)
        if out.exists() and not args.force:
            print(f"[sweep] skip {v.tag}: {out} already exists "
                  f"(--force to override)", flush=True)
            skipped += 1
            continue
        rc = run_variant(v, args.passthrough, args.dry_run)
        if rc != 0:
            failed.append(v.tag)
            if args.stop_on_error:
                print(f"[sweep] --stop_on_error: aborting after {v.tag}", flush=True)
                break
        else:
            ran += 1

    print(f"\n[sweep] done: ran={ran}, skipped={skipped}, "
          f"failed={len(failed)} ({failed})", flush=True)
    if failed and not args.stop_on_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
