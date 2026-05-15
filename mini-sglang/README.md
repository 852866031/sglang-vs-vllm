# mini-sglang timeline benchmark

This directory drives the timeline workload from `../timeline.csv` against
the [mini-sglang](https://github.com/sgl-project/mini-sglang) reference
implementation (LMSYS's ~5k-LoC distillation of SGLang).

## ⚠️ Important: no LoRA support

mini-sglang is a teaching-grade reference engine and **does not implement
LoRA adapter serving**. There is no `--lora-paths` flag, no per-request
adapter selection, and no path to enable it without forking the engine.

The scripts here therefore run the timeline workload against the **base
Llama-3.1-8B model only**. This isolates mini-sglang's batching/scheduler
performance but is NOT directly comparable to the vLLM/SGLang runs, which
are exercising multi-LoRA batching. Treat the resulting numbers as
"base-model throughput / latency on the same request trace," not as a
multi-LoRA comparison.

If you want a true multi-LoRA mini-sglang number you'd need to fork the
engine — well out of scope here.

## 1. Create the env and install mini-sglang

mini-sglang ships as a source package and isn't on PyPI, so we clone it
and `pip install -e .` from inside a fresh conda env. Python 3.12 matches
the upstream README (3.10+ should also work).

```bash
# 1. New conda env.
conda create -n minisgl-bench python=3.12 -y
conda activate minisgl-bench

# 2. Clone and install mini-sglang. Stash it somewhere outside this repo
#    (the upstream package name is `minisgl`).
git clone https://github.com/sgl-project/mini-sglang.git ~/mini-sglang
cd ~/mini-sglang
pip install --upgrade pip
pip install -e .

# 3. Client deps for run_benchmark.py.
pip install aiohttp tqdm
```

Sanity-check that the server module is importable:

```bash
python -c "import minisgl; print(minisgl.__file__)"
python -m minisgl --help | head
```

The base model defaults to a local Llama-3.1-8B snapshot at

```
/home/jiaxuan_chen/scratch/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
```

so no HF login is needed. Pass `--model <path-or-hub-id>` to override.

## 2. Run the benchmark

```bash
conda activate minisgl-bench
cd /home/jiaxuan_chen/engine-compare/mini-sglang
python run_benchmark.py
```

What happens:

1. Spawns `launch_mini_sglang.py`, which runs
   `python -m minisgl --model <model> --host 127.0.0.1 --port 1919
   --dtype bfloat16 --tp-size 1 --memory-ratio 0.85
   --max-seq-len-override 4096 --max-running-requests 256`.
2. Polls `GET /v1/models` until it returns 200 (mini-sglang has no
   `/health` endpoint, so `/v1/models` is the cheapest readiness probe).
3. Runs the warmup phase (≤200 rows, capped at 20 s).
4. Replays `../timeline.csv` against the native `/generate` endpoint,
   streaming per-token deltas to measure TTFT and inter-token gaps.
5. Writes `../outputs/mini_sglang_results.csv` and tears the server down.

The output CSV columns match `../example_script/example_output.csv`:
`idx, t_rel_s, latency_s, status, ttft_s, avg_tbt_s, worst_tbt_s`.

## 3. Useful knobs

`run_benchmark.py`:

| flag | default | meaning |
| --- | --- | --- |
| `--port` | `1919` | server port (mini-sglang default) |
| `--out_csv` | `../outputs/mini_sglang_results.csv` | results path |
| `--timeline_csv` | `../timeline.csv` | request schedule |
| `--warmup_count` | `200` | max warmup rows |
| `--warmup_duration_s` | `20.0` | warmup time cap |
| `--warmup_rest_s` | `2.0` | pause after warmup |
| `--fold N` | `1` | subsample: keep every N-th row |
| `--no-spawn` | off | talk to an already-running server |
| `--launcher_args` | `""` | extra flags forwarded to `launch_mini_sglang.py` |

`launch_mini_sglang.py` (forward via `--launcher_args "..."`):

| flag | default | meaning |
| --- | --- | --- |
| `--dtype` | `bfloat16` | weight/activation dtype |
| `--tp-size` | `1` | tensor parallel size |
| `--memory-ratio` | `0.85` | fraction of GPU memory for KV cache |
| `--max-seq-len-override` | `4096` | max sequence length |
| `--max-running-requests` | `256` | concurrent in-flight requests |
| `--extra` | `""` | extra raw flags appended to `python -m minisgl` |

Example:

```bash
# Smoke test (10% of rows).
python run_benchmark.py --fold 10 --out_csv ../outputs/mini_sglang_smoke.csv

# Lower memory cap if 0.85 is too aggressive on a shared GPU.
python run_benchmark.py --launcher_args "--memory-ratio 0.7"
```

## 4. Files

- `launch_mini_sglang.py` — thin wrapper that runs `python -m minisgl`.
- `run_benchmark.py` — orchestrator: spawns the server, runs warmup,
  replays the timeline against `/generate` (no LoRA), writes the results
  CSV.
- `../outputs/mini_sglang_results.csv` — created on a successful run.

## 5. Notes & gotchas

- **No `/health` endpoint.** Readiness is probed via `/v1/models`. If you
  fork mini-sglang and add a health route, point `try_ready()` at it.
- **No JSON in the stream.** Unlike SGLang and vLLM, mini-sglang's
  `/generate` streams *raw* delta text per `data:` line, terminated by
  `data: [DONE]`. The client unwraps this and treats each non-empty,
  non-`[DONE]` line as one token-arrival event for TBT.
- **Default port 1919.** The other engines use 8000 (vLLM) and 30000
  (SGLang) — port stays 1919 here to match mini-sglang's CLI default and
  avoid collisions if you happen to run all three side-by-side.
