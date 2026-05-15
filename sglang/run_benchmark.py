#!/usr/bin/env python3
"""
run_benchmark.py — SGLang multi-LoRA timeline benchmark.

Mirrors vllm/run_benchmark.py:
  - Spawns SGLang's launch_server with both LoRA adapters preloaded.
  - Waits for /health to return 200.
  - Warmup: replay the first N timeline rows on a normalized schedule
    (results discarded).
  - Main: replay ../timeline.csv at its native timestamps, streaming
    /generate so TTFT and per-token gaps can be measured.
  - Writes ../outputs/sglang_results.csv with columns matching
    example_script/example_output.csv:
      idx, t_rel_s, latency_s, status, ttft_s, avg_tbt_s, worst_tbt_s

LoRA selection: round-robin by row_id (even -> lora-0, odd -> lora-1).
Per-request adapter selection uses SGLang's native /generate endpoint with
the `lora_path` field set to the adapter's registered name.
"""
import argparse
import asyncio
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_TIMELINE = REPO_ROOT / "timeline.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
LORA_NAMES = ["llama3-toy-lora-0", "llama3-toy-lora-1"]
DEFAULT_MODEL = (
    "/home/jiaxuan_chen/scratch/models--meta-llama--Meta-Llama-3.1-8B"
    "/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
)

_active_pbar: Optional["tqdm"] = None


def _emit(line: str) -> None:
    if _active_pbar is not None:
        tqdm.write(line, file=sys.stdout)
    else:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


# ----------------------------
# Timeline
# ----------------------------
@dataclass
class TimelineRow:
    timestamp_s: float
    prompt_length: int
    max_new_tokens: int
    row_id: int


def load_timeline_csv(path: str) -> List[TimelineRow]:
    rows: List[TimelineRow] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"timestamp_s", "prompt_length", "max_new_tokens"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Timeline CSV missing required columns: {sorted(missing)}")
        for i, r in enumerate(reader):
            rows.append(
                TimelineRow(
                    timestamp_s=float(r["timestamp_s"]),
                    prompt_length=int(float(r["prompt_length"])),
                    max_new_tokens=int(float(r["max_new_tokens"])),
                    row_id=i,
                )
            )
    rows.sort(key=lambda x: x.timestamp_s)
    return rows


def make_prompt_from_length(prompt_length: int) -> str:
    base = "Instruction:\n"
    tail = "\n### Response: "
    filler_needed = max(0, prompt_length - len(base) - len(tail))
    filler = ("hello " * 10000)[:filler_needed]
    prompt = base + filler + tail
    if len(prompt) < prompt_length:
        prompt += "x" * (prompt_length - len(prompt))
    return prompt[:prompt_length]


# ----------------------------
# Server health
# ----------------------------
async def try_health(session: aiohttp.ClientSession, server: str, timeout_s: float = 1.5) -> bool:
    """SGLang exposes /health (alive) and /health_generate (probe). Try
    /health first; fall back to the lighter /get_server_info."""
    base = server.rstrip("/")
    for endpoint in ("/health", "/get_server_info"):
        try:
            async with session.get(f"{base}{endpoint}", timeout=timeout_s) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            continue
    return False


async def wait_for_server(server: str, max_wait_s: float = 900.0, poll_period_s: float = 1.0) -> None:
    t0 = time.monotonic()
    async with aiohttp.ClientSession() as session:
        while True:
            if await try_health(session, server):
                _emit(f"[orchestrator] Server is up ✅ at {server}")
                return
            if time.monotonic() - t0 > max_wait_s:
                raise TimeoutError(f"Server didn't become healthy within {max_wait_s:.1f}s")
            await asyncio.sleep(poll_period_s)


# ----------------------------
# Request: SGLang native /generate with streaming
# ----------------------------
async def send_one_streaming(
    session: aiohttp.ClientSession,
    server: str,
    idx: int,
    t_rel: float,
    prompt: str,
    lora_name: str,
    max_new_tokens: int,
    request_timeout_s: float = 600.0,
) -> Tuple[int, float, float, str, Optional[float], Optional[float], Optional[float]]:
    """
    POST /generate with stream=True. SGLang's streaming format is SSE where
    each chunk's `text` field is the *cumulative* generated text. Inter-chunk
    arrival time gives TBT (one chunk per token at default stream_interval=1).
    """
    url = f"{server.rstrip('/')}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "lora_path": lora_name,
        "stream": True,
    }

    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    t_send = time.monotonic()
    ttft: Optional[float] = None
    last_token_t: Optional[float] = None
    prev_text_len = 0
    tbts: List[float] = []
    tokens_received = 0          # incremented per chunk where text grew
    saw_done = False
    finish_reason: Optional[str] = None

    try:
        async with session.post(url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.read()
                latency = time.monotonic() - t_send
                return (idx, t_rel, latency, f"error:http{resp.status}",
                        None, None, None, 0, 0, False, None)
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    saw_done = True
                    continue
                if not data:
                    continue
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                meta = obj.get("meta_info") or {}
                fr = meta.get("finish_reason")
                if isinstance(fr, dict):
                    fr = fr.get("type")
                if fr:
                    finish_reason = fr
                text = obj.get("text", "") or ""
                if len(text) <= prev_text_len:
                    continue
                now = time.monotonic()
                if ttft is None:
                    ttft = now - t_send
                else:
                    if last_token_t is not None:
                        tbts.append(now - last_token_t)
                last_token_t = now
                prev_text_len = len(text)
                tokens_received += 1

        latency = time.monotonic() - t_send
        avg_tbt = (sum(tbts) / len(tbts)) if tbts else None
        worst_tbt = max(tbts) if tbts else None
        return (idx, t_rel, latency, "ok", ttft, avg_tbt, worst_tbt,
                tokens_received, prev_text_len, saw_done, finish_reason)
    except Exception as e:
        latency = time.monotonic() - t_send
        return (idx, t_rel, latency, f"error:{type(e).__name__}",
                None, None, None, tokens_received, prev_text_len, saw_done, finish_reason)


def pick_lora(row_id: int) -> str:
    return LORA_NAMES[row_id % len(LORA_NAMES)]


# ----------------------------
# Warmup + main timeline runners
# ----------------------------
WARMUP_START_OFFSET_S = 1.0


async def run_warmup(
    server: str,
    warmup_rows: List[TimelineRow],
    stop_event: asyncio.Event,
    request_timeout_s: float = 600.0,
) -> None:
    if not warmup_rows:
        return
    base_ts = min(r.timestamp_s for r in warmup_rows)
    span = max(r.timestamp_s for r in warmup_rows) - base_ts
    _emit(f"[orchestrator] Warmup: {len(warmup_rows)} requests "
          f"(first at +{WARMUP_START_OFFSET_S:.2f}s, span {span:.2f}s)")

    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        t0 = time.monotonic()

        async def _run(row: TimelineRow) -> None:
            target = t0 + (row.timestamp_s - base_ts) + WARMUP_START_OFFSET_S
            delay = target - time.monotonic()
            if delay > 0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    pass
            if stop_event.is_set():
                return
            t_rel = time.monotonic() - t0
            await send_one_streaming(
                session=session, server=server, idx=row.row_id, t_rel=t_rel,
                prompt=make_prompt_from_length(row.prompt_length),
                lora_name=pick_lora(row.row_id),
                max_new_tokens=row.max_new_tokens,
            )

        await asyncio.gather(*(asyncio.create_task(_run(r)) for r in warmup_rows))
    _emit("[orchestrator] Warmup completed ✅")


async def run_timeline(
    server: str,
    timeline_rows: List[TimelineRow],
    stop_event: asyncio.Event,
    request_timeout_s: float = 600.0,
) -> List[Tuple[int, float, float, str, Optional[float], Optional[float], Optional[float]]]:
    if not timeline_rows:
        return []
    base_ts = min(r.timestamp_s for r in timeline_rows)
    total_duration = max(r.timestamp_s for r in timeline_rows) - base_ts

    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    results: List[Tuple[int, float, float, str, Optional[float], Optional[float], Optional[float]]] = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        t0 = time.monotonic()
        _emit(f"[orchestrator] Timeline start: {len(timeline_rows)} rows, "
              f"duration {total_duration:.2f}s")

        global _active_pbar
        pbar = tqdm(
            total=max(total_duration, 1e-3),
            desc="Timeline", unit="s",
            bar_format="{desc}: {percentage:5.1f}%|{bar}| {n:6.1f}/{total:.1f}s "
                       "[{elapsed}<{remaining}]",
            leave=True, dynamic_ncols=True, file=sys.stdout, mininterval=0.2,
        )
        _active_pbar = pbar

        async def _progress() -> None:
            last = 0.0
            while not stop_event.is_set():
                elapsed = min(time.monotonic() - t0, total_duration)
                if elapsed - last > 0:
                    pbar.update(elapsed - last)
                    last = elapsed
                if elapsed >= total_duration:
                    return
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=0.5)
                    return
                except asyncio.TimeoutError:
                    continue

        prog_task = asyncio.create_task(_progress())

        async def _run(row: TimelineRow) -> None:
            target = t0 + (row.timestamp_s - base_ts)
            delay = target - time.monotonic()
            if delay > 0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    pass
            if stop_event.is_set():
                return
            scheduled_t_rel = row.timestamp_s - base_ts
            t_rel = time.monotonic() - t0
            res = await send_one_streaming(
                session=session, server=server, idx=row.row_id, t_rel=t_rel,
                prompt=make_prompt_from_length(row.prompt_length),
                lora_name=pick_lora(row.row_id),
                max_new_tokens=row.max_new_tokens,
            )
            # res = (idx, t_rel, latency, status, ttft, avg_tbt, worst_tbt)
            # Splice in the scheduled-time column as the second element so the
            # plot can show requests on the timeline they were SUPPOSED to fire
            # at, not when the orchestrator actually got around to dispatching
            # them (which drifts under event-loop saturation).
            results.append((res[0], scheduled_t_rel) + res[1:])

        try:
            await asyncio.gather(*(asyncio.create_task(_run(r)) for r in timeline_rows))
        finally:
            prog_task.cancel()
            try:
                await prog_task
            except (asyncio.CancelledError, Exception):
                pass
            if not stop_event.is_set() and pbar.n < pbar.total:
                pbar.update(pbar.total - pbar.n)
            pbar.refresh()
            pbar.close()
            _active_pbar = None

    results.sort(key=lambda x: x[0])
    _emit("[orchestrator] Timeline completed ✅")
    return results


def _summarize_token_counts(results, timeline_rows) -> None:
    """Print a quick token-count summary so silent truncation can't make
    latency numbers look better than they should."""
    expected_by_idx = {r.row_id: r.max_new_tokens for r in timeline_rows}
    ok_rows = [row for row in results if row[4] == "ok"]
    if not ok_rows:
        _emit("[verify] no ok rows — skipping token-count summary")
        return
    tokens = sorted(row[8] for row in ok_rows)
    exact = sum(1 for row in ok_rows
                if row[8] == expected_by_idx.get(row[0], -1))
    saw_done = sum(1 for row in ok_rows if row[10])
    med = tokens[len(tokens) // 2]
    _emit(
        f"[verify] tokens_received over {len(tokens)} ok rows: "
        f"min={tokens[0]}, median={med}, max={tokens[-1]}; "
        f"exact-match with max_new_tokens: {exact}/{len(ok_rows)}; "
        f"saw_done terminator: {saw_done}/{len(ok_rows)}"
    )
    if exact < len(ok_rows):
        short = [
            (row[0], row[8], expected_by_idx.get(row[0]))
            for row in ok_rows
            if row[8] != expected_by_idx.get(row[0], -1)
        ][:5]
        _emit(
            f"[verify] WARNING: {len(ok_rows) - exact} requests did not "
            f"receive the requested max_new_tokens. Sample (idx, got, "
            f"expected): {short}"
        )


def write_results_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx", "scheduled_t_rel_s", "t_rel_s", "latency_s", "status",
            "ttft_s", "avg_tbt_s", "worst_tbt_s",
            "tokens_received", "gen_chars", "saw_done", "finish_reason",
        ])
        for row in rows:
            w.writerow(row)
    _emit(f"[orchestrator] Wrote results CSV: {path}")


# ----------------------------
# Process orchestration
# ----------------------------
def terminate_tree(p: subprocess.Popen, grace_s: float = 1.0) -> None:
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass
    t0 = time.monotonic()
    while time.monotonic() - t0 < grace_s:
        if p.poll() is not None:
            return
        time.sleep(0.05)
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline_csv", default=str(DEFAULT_TIMELINE))
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out_csv", default=None,
                    help="Explicit output CSV path. If unset, the path is "
                         "derived from --run_tag.")
    ap.add_argument("--run_tag", default="default",
                    help="Short tag identifying this SGLang config (e.g. "
                         "'default', 'cap96', 'csgmv', 'cap96_csgmv'). The "
                         "results land in outputs/sglang_<tag>.csv unless "
                         "--out_csv is set explicitly.")
    ap.add_argument("--launcher", default=str(SCRIPT_DIR / "launch_sglang.py"))
    ap.add_argument("--warmup_count", type=int, default=200)
    ap.add_argument("--warmup_duration_s", type=float, default=20.0)
    ap.add_argument("--warmup_rest_s", type=float, default=2.0)
    ap.add_argument("--fold", type=int, default=1,
                    help="Subsample: keep every N-th row by row_id.")
    ap.add_argument("--server_ready_timeout_s", type=float, default=900.0)
    ap.add_argument("--no-spawn", action="store_true",
                    help="Skip launching the server; talk to one already running.")
    ap.add_argument("--launcher_args", default="",
                    help="Extra args passed verbatim to launch_sglang.py.")
    args = ap.parse_args()

    timeline_rows = load_timeline_csv(args.timeline_csv)
    _emit(f"[orchestrator] Loaded {len(timeline_rows)} rows from {args.timeline_csv}")
    if args.fold > 1:
        before = len(timeline_rows)
        timeline_rows = [r for r in timeline_rows if r.row_id % args.fold == 0]
        _emit(f"[orchestrator] --fold {args.fold}: kept {len(timeline_rows)}/{before}")

    server = f"http://{args.host}:{args.port}"
    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        tag = args.run_tag.strip() or "default"
        out_csv = DEFAULT_OUTPUT_DIR / f"sglang_{tag}.csv"
    _emit(f"[orchestrator] results will be written to {out_csv}")

    p: Optional[subprocess.Popen] = None
    log_task: Optional[asyncio.Task] = None
    stop_event = asyncio.Event()

    try:
        if not args.no_spawn:
            cmd = [sys.executable, "-u", args.launcher,
                   "--port", str(args.port), "--host", args.host,
                   "--model", args.model]
            if args.launcher_args:
                cmd += shlex.split(args.launcher_args)
            _emit("[orchestrator] launching: " + " ".join(cmd))
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            p = subprocess.Popen(
                cmd, preexec_fn=os.setsid,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
            )

            async def pump() -> None:
                """Forward server output line-by-line. Splits on both '\\n' and
                '\\r' so server-side tqdm progress bars don't silently block."""
                assert p is not None and p.stdout is not None
                loop = asyncio.get_running_loop()
                stream = p.stdout
                buf = []

                def read_chunk() -> bytes:
                    return stream.read(256)

                while True:
                    raw = await loop.run_in_executor(None, read_chunk)
                    if not raw:
                        if buf:
                            _emit("[server] " + "".join(buf))
                        return
                    chunk = raw.decode("utf-8", errors="replace")
                    for ch in chunk:
                        if ch in ("\n", "\r"):
                            if buf:
                                _emit("[server] " + "".join(buf))
                                buf.clear()
                        else:
                            buf.append(ch)

            log_task = asyncio.create_task(pump())

        loop = asyncio.get_running_loop()

        def _on_sigint() -> None:
            if p is not None and p.poll() is None:
                terminate_tree(p, grace_s=0.5)
            stop_event.set()

        loop.add_signal_handler(signal.SIGINT, _on_sigint)

        waiter = asyncio.create_task(
            wait_for_server(server, max_wait_s=args.server_ready_timeout_s)
        )
        stopper = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait({waiter, stopper}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        if stop_event.is_set():
            return

        if timeline_rows and args.warmup_count > 0 and args.warmup_duration_s > 0:
            base = timeline_rows[0].timestamp_s
            by_count = min(args.warmup_count, len(timeline_rows))
            by_dur = sum(1 for r in timeline_rows if r.timestamp_s - base <= args.warmup_duration_s)
            warmup_n = min(by_count, by_dur)
        else:
            warmup_n = 0
        warmup_rows = timeline_rows[:warmup_n]
        _emit(f"[orchestrator] Warmup selection: count_cap={args.warmup_count}, "
              f"duration_cap={args.warmup_duration_s:.2f}s -> {len(warmup_rows)} rows")

        if warmup_rows:
            await run_warmup(server, warmup_rows, stop_event)
            if stop_event.is_set():
                return
            if args.warmup_rest_s > 0:
                _emit(f"[orchestrator] Resting {args.warmup_rest_s:.2f}s after warmup...")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=args.warmup_rest_s)
                    return
                except asyncio.TimeoutError:
                    pass

        results = await run_timeline(server, timeline_rows, stop_event)
        write_results_csv(out_csv, results)
        _summarize_token_counts(results, timeline_rows)

    except KeyboardInterrupt:
        if p is not None:
            terminate_tree(p, grace_s=0.5)
    finally:
        if p is not None and p.poll() is None:
            _emit("[orchestrator] shutting down server...")
            terminate_tree(p, grace_s=1.0)
        if log_task is not None:
            log_task.cancel()
            try:
                await asyncio.wait_for(log_task, timeout=1.0)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
