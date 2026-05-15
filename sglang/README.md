# SGLang multi-LoRA timeline benchmark

This directory drives the same multi-LoRA timeline workload that
`../vllm/` runs, against SGLang instead of vLLM. The two LoRA adapters in
`../llama3-toy-lora-0` and `../llama3-toy-lora-1` are preloaded into a
single SGLang server and selected per-request in round-robin order (even
`row_id` → adapter 0, odd → adapter 1). Per-request `lora_path` selection
uses SGLang's native `/generate` endpoint.

Results land in `../outputs/sglang_<tag>.csv` (one file per `--run_tag`)
with the same columns as `../example_script/example_output.csv`:
`idx, t_rel_s, latency_s, status, ttft_s, avg_tbt_s, worst_tbt_s`.

## 1. Create the conda env

Tested on a single A100 40 GB.

```bash
conda create -n sglang-bench python=3.11 -y
conda activate sglang-bench

pip install --upgrade pip

# SGLang. Install the "all" extra so the server, FlashInfer attention, and
# OpenAI-compatible wrappers are pulled in.
pip install "sglang[all]>=0.4.0"

# Client deps for run_benchmark.py.
pip install aiohttp tqdm
```

The base model defaults to a local Llama-3.1-8B snapshot at

```
/home/jiaxuan_chen/scratch/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
```

so no HF login is needed. Pass `--model <path-or-hub-id>` to override.

## 2. Run the benchmark

```bash
conda activate sglang-bench
cd sglang
python run_benchmark.py
```

The single command:

1. Spawns `launch_sglang.py`, which runs
   `python -m sglang.launch_server --model-path <model> --enable-lora
   --lora-paths llama3-toy-lora-0=... llama3-toy-lora-1=...
   --max-loras-per-batch 2 --port 30000`.
2. Polls `/health` until it returns 200 (model load + LoRA registration can
   take a few minutes the first time, especially while CUDA graphs are
   captured).
3. Runs the warmup phase (first ≤200 rows, capped at 20 s of schedule).
4. Replays `../timeline.csv` at its native timestamps, streaming
   `/generate` to measure TTFT and per-token gaps.
5. Writes `../outputs/sglang_results.csv` and tears the server down.

## 3. Useful knobs

`run_benchmark.py`:

| flag | default | meaning |
| --- | --- | --- |
| `--port` | `30000` | server port |
| `--out_csv` | `../outputs/sglang_results.csv` | results path |
| `--timeline_csv` | `../timeline.csv` | request schedule |
| `--warmup_count` | `200` | max warmup rows |
| `--warmup_duration_s` | `20.0` | warmup time cap |
| `--warmup_rest_s` | `2.0` | pause after warmup |
| `--fold N` | `1` | subsample: keep every N-th row (smoke tests) |
| `--no-spawn` | off | talk to an already-running SGLang server |
| `--launcher_args` | `""` | extra flags forwarded to `launch_sglang.py` |

`launch_sglang.py` (forward via `--launcher_args "..."`):

| flag | default | meaning |
| --- | --- | --- |
| `--context-length` | `4096` | max sequence length |
| `--max-loras-per-batch` | `2` | concurrent LoRAs in one batched step |
| `--mem-fraction-static` | `0.85` | GPU memory budget for weights+KV |
| `--dtype` | `bfloat16` | weight/activation dtype |
| `--disable-radix-cache` | off | turn off prefix caching (matches vLLM's `--no-enable-prefix-caching`) |
| `--max-running-requests N` | unset (auto) | cap concurrent in-flight requests. The auto-tuner picks high values (e.g. 200+) on small models with lots of KV space, which inflates per-decode-step cost. 64–128 is usually a better fit on this workload. |
| `--lora-backend {triton,csgmv}` | unset (`triton`) | LoRA kernel backend. `csgmv` is Punica/SGMV-class and faster than the default `triton`. |
| `--extra` | `""` | extra raw flags appended to `sglang.launch_server` |

Examples:

```bash
# Smoke test (10% of rows):
python run_benchmark.py --fold 10 --run_tag smoke

# Talk to a server you already started by hand:
python launch_sglang.py --port 30000   # in one shell
python run_benchmark.py --no-spawn     # in another

# Disable radix-cache to match vLLM's no-prefix-cache config:
python run_benchmark.py --launcher_args "--disable-radix-cache" --run_tag noradix

# Pass through SGLang flags (e.g. attention backend):
python run_benchmark.py --launcher_args "--extra '--attention-backend triton'" \
                       --run_tag triton_attn
```

## 4. Sweep: incremental ladder of changes expected to close the gap with vLLM

Each command builds on the previous one — every step adds **one** more
flag — and the variants are ordered by my best guess at which closes the
most remaining distance to vLLM. We deliberately don't pass
`--max-running-requests`: capping admission tunes SGLang's scheduler
toward our specific workload using info vLLM doesn't get, so leaving
SGLang on its auto-tuned admission keeps the comparison fair. Instead we
fix the bottleneck (CUDA-graph coverage) so the auto-picked large running
batch isn't punished.

Each command writes to `../outputs/sglang_<tag>.csv`. After the sweep,
`python ../plot_results.py` picks up every variant automatically.

**Shortcut — `run_sweep.py`** runs the four-step incremental ladder
below in one shot, skipping any variant whose output CSV already exists
(so re-runs only fill in the gaps).

```bash
# Full sweep, skip variants whose output CSV already exists:
python run_sweep.py

# Inspect the plan without running anything:
python run_sweep.py --list

# Pick a subset:
python run_sweep.py --tags default,graph256

# Re-run a variant even if its CSV exists:
python run_sweep.py --force --tags graph256

# Forward extra flags to every run_benchmark.py invocation (everything
# after `--pass` goes through verbatim):
python run_sweep.py --pass --fold 10   # 10% smoke test of every variant
```

You can still run any individual variant by hand with the commands below.

```bash
# Step 0 — Baseline. No flags. SGLang's defaults on this build already
# include flashinfer attention and the csgmv LoRA backend, so this is
# already on the fast kernels — the gap to vLLM here is mostly CUDA-graph
# coverage (default cuda_graph_max_bs=32) vs vLLM's capture out to 512.
python run_benchmark.py --run_tag default

# Step 1 — Bump CUDA-graph capture to cover the full running batch.
# Single biggest expected win. SGLang's auto-picked max_running_requests
# is ~200 on this workload; with the default cuda_graph_max_bs=32, every
# step bigger than 32 runs eager. Capturing up to 256 brings essentially
# every decode step back inside a captured graph.
python run_benchmark.py --run_tag graph256 \
    --launcher_args "--extra '--cuda-graph-max-bs 256'"

# Step 2 — Add: disable chunked prefill. Our prompts are 1–30 tokens, so
# they always fit in a single chunk; chunked prefill is pure scheduler
# overhead in this regime. Small expected win, but cleans up a code path.
python run_benchmark.py --run_tag graph256_nochunk \
    --launcher_args "--extra '--cuda-graph-max-bs 256 --chunked-prefill-size -1'"

# Step 3 — Add: disable radix-cache. Matches vLLM's
# --no-enable-prefix-caching for an apples-to-apples comparison. With
# short, varied prompts the radix cache buys SGLang little; turning it
# off removes a small constant cost and equalizes configs.
python run_benchmark.py --run_tag graph256_nochunk_noradix \
    --launcher_args "--disable-radix-cache --extra '--cuda-graph-max-bs 256 --chunked-prefill-size -1'"
```

After the sweep, generate all plots in one go:

```bash
cd ..
python plot_results.py
# One PNG per variant in outputs/, all sglang variants in tab:orange,
# vllm in tab:blue.
```

If you want to also probe the *individual* contribution of each lever
(rather than the cumulative ladder), the natural extra runs are:

```bash
# Disable chunked prefill alone, no graph change.
python run_benchmark.py --run_tag nochunk \
    --launcher_args "--extra '--chunked-prefill-size -1'"

# Disable radix-cache alone.
python run_benchmark.py --run_tag noradix \
    --launcher_args "--disable-radix-cache"

# Force the slower LoRA backend (triton) as an A/B against the csgmv
# default. Useful only as a sanity check that csgmv is faster on this
# hardware.
python run_benchmark.py --run_tag triton_lora \
    --launcher_args "--lora-backend triton"
```

## 5. Files

- `launch_sglang.py` — thin wrapper that runs `sglang.launch_server` with
  both LoRA adapters preloaded.
- `run_benchmark.py` — orchestrator. Spawns the server, runs warmup,
  replays the timeline against the native `/generate` endpoint with
  per-request `lora_path`, and writes `../outputs/sglang_<tag>.csv`.
- `run_sweep.py` — runs every variant in section 4 sequentially, skipping
  any whose output CSV already exists. See `python run_sweep.py --help`.
- `../outputs/sglang_<tag>.csv` — created on a successful run.

## 6. Notes & gotchas

- **Adapter rank.** Both adapters have rank 16. SGLang infers max LoRA
  rank from the loaded adapters; no extra flag is needed.
- **LoRA + attention backend.** Older SGLang builds required
  `--attention-backend triton` for LoRA. Recent releases (≥0.4) support
  FlashInfer too — leave the default unless you hit a kernel error. If
  you do, retry with `--launcher_args "--extra '--attention-backend triton'"`.
- **Adapter base model.** The adapters' `adapter_config.json` records
  `base_model_name_or_path: meta-llama/Meta-Llama-3-8B` (3.0), but we
  serve them against 3.1-8B. The architectures match and the q/k/v/o
  target modules align, so loading succeeds; generations may differ
  qualitatively from a 3.0-trained baseline. This is fine for
  throughput/latency comparison but not for output quality.
