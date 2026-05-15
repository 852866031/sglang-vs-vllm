# vLLM multi-LoRA timeline benchmark

This directory drives the same multi-LoRA timeline workload used to compare
vLLM, SGLang, nano-vllm and mini-sglang.

The two LoRA adapters in `../llama3-toy-lora-0` and `../llama3-toy-lora-1`
are loaded into the same vLLM server and selected per-request in round-robin
fashion (even `row_id` → adapter 0, odd → adapter 1). The request schedule
comes from `../timeline.csv`. Per-request latency / TTFT / TBT stats are
written to `../outputs/vllm_results.csv` and have the same columns as
`../example_script/example_output.csv`.

## 1. Create the conda env

Tested on a single A100 40 GB.

```bash
conda create -n vllm-bench python=3.11 -y
conda activate vllm-bench

# vLLM. Use a recent release that supports Llama 3 + multi-LoRA serving.
pip install --upgrade pip
pip install "vllm>=0.20"

# Client deps for run_benchmark.py.
pip install aiohttp tqdm
```

The base model defaults to a local Llama-3.1-8B snapshot at

```
/home/jiaxuan_chen/scratch/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
```

so no HF login is needed. Pass `--model <path-or-hub-id>` to override.

## 2. Run the benchmark

From this directory:

```bash
conda activate vllm-bench
python run_benchmark.py
```

That single command will:

1. Spawn `launch_vllm.py`, which runs `vllm serve meta-llama/Meta-Llama-3-8B`
   with `--enable-lora` and both adapters registered as `llama3-toy-lora-0`
   and `llama3-toy-lora-1`.
2. Wait for `GET /health` to return 200 (model load + LoRA registration can
   take a minute or two the first time).
3. Replay the first ~200 timeline rows (capped at 20 s of schedule) as a
   warmup — these requests are sent but their metrics are NOT written.
4. Replay the full `../timeline.csv` at the timestamps it specifies,
   streaming `/v1/completions` so TTFT and per-token gaps can be measured.
5. Write `../outputs/vllm_results.csv` and tear the server down.

The output CSV columns match the reference:
`idx, t_rel_s, latency_s, status, ttft_s, avg_tbt_s, worst_tbt_s`.

## 3. Useful knobs

`run_benchmark.py` flags:

| flag | default | meaning |
| --- | --- | --- |
| `--port` | `8000` | server port |
| `--out_csv` | `../outputs/vllm_results.csv` | results path |
| `--timeline_csv` | `../timeline.csv` | request schedule |
| `--warmup_count` | `200` | max warmup rows |
| `--warmup_duration_s` | `20.0` | warmup time cap |
| `--warmup_rest_s` | `2.0` | pause between warmup and main timeline |
| `--fold N` | `1` | subsample: keep every N-th row (smoke tests) |
| `--no-spawn` | off | use an already-running server at `--host:--port` |
| `--launcher_args` | `""` | extra flags forwarded to `launch_vllm.py` |

`launch_vllm.py` flags (forward via `--launcher_args "..."`):

| flag | default | meaning |
| --- | --- | --- |
| `--max-model-len` | `4096` | context length |
| `--max-loras` | `2` | concurrent LoRAs vLLM batches |
| `--max-lora-rank` | `16` | must be ≥ adapter rank (these are r=16) |
| `--gpu-memory-utilization` | `0.9` | KV-cache fraction |
| `--dtype` | `bfloat16` | weight/activation dtype |
| `--extra` | `""` | extra raw flags appended to `vllm serve` |

Examples:

```bash
# Smoke test (10% of rows):
python run_benchmark.py --fold 10 --out_csv ../outputs/vllm_smoke.csv

# Talk to a server you already started by hand:
python launch_vllm.py --port 8000   # in one shell
python run_benchmark.py --no-spawn  # in another

# Pass through vLLM flags:
python run_benchmark.py --launcher_args "--max-model-len 8192 --gpu-memory-utilization 0.85"
```

## 4. Files

- `launch_vllm.py` — thin wrapper that runs `vllm serve` with both LoRA
  adapters preloaded.
- `run_benchmark.py` — orchestrator. Spawns the server, runs warmup, replays
  the timeline with streaming, and writes the results CSV.
- `../outputs/vllm_results.csv` — created on a successful run.
