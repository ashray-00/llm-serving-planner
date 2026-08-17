# LLM Inference Benchmark Harness

Reproducible load-testing wrapper around [`vllm bench serve`](https://docs.vllm.ai/en/stable/cli/bench/serve/). Raw tool output is normalized into a single canonical schema (`schema.py`) so results stay comparable across runs, hardware, and vLLM versions.

Everything in this folder is testable offline except the actual `vllm bench serve` subprocess calls inside `bench.sh`.

## Methodology (non-negotiable)

1. **Fixed token lengths** — Every run uses explicit `--input-len` and `--output-len` with `--ignore-eos` so generation length is deterministic, not dataset-default or EOS-driven.
2. **Warmup discarded** — One low `--num-prompts` warmup per `(mode, load)` config runs before measured repeats; warmup output is never written to `results/`.
3. **Repeat + average** — Each load point runs `REPEATS` times (default 3). `run_baseline.sh` averages numeric metrics per `(mode, load_value)` into one `BenchResult` per load point in `results/baseline.json`.
4. **Dual sweep** — Concurrency sweep (`--max-concurrency`) and optional request-rate sweep (`--request-rate`) share the same harness.
5. **Full metadata tagging** — Sidecar `*.meta.json` files capture GPU, dtype, vLLM version, flags, mode, load value, and optional SLO thresholds that vLLM does not emit itself.

## vLLM raw output mapping

The normalizer targets vLLM's current `serve.py` `--save-result` JSON (see upstream `vllm/benchmarks/serve.py`). Key raw fields:

| Raw field | BenchResult field |
|-----------|-------------------|
| `model_id` | `model` (meta overrides) |
| `num_prompts` | `num_prompts` |
| `mean_ttft_ms`, `median_ttft_ms`, `p95_ttft_ms` | `ttft_ms.{mean,p50,p95}` |
| `mean_itl_ms`, `median_itl_ms`, `p95_itl_ms` | `itl_ms.{mean,p50,p95}` |
| `mean_e2el_ms`, `p95_e2el_ms` | `e2e_latency_ms.{mean,p95}` |
| `output_throughput` | `total_output_tok_per_s` |
| `total_output_tokens`, `completed`, `mean_e2el_ms` | `output_tok_per_s_per_request` (derived) |
| `request_goodput` | `goodput_req_per_s` (only when SLO set + `--goodput` passed) |

`bench.sh` passes `--metric-percentiles 50,95,99` and `--percentile-metrics ttft,itl,e2el` so p50/p95 fields exist in saved JSON.

## Dry-run (zero GPU, zero vLLM)

Verify loop structure and flag construction without executing vLLM:

```bash
cd bench
chmod +x bench.sh run_baseline.sh
./bench.sh --dry-run \
  --concurrencies "1 4" \
  --repeats 2 \
  --model Qwen/Qwen3-8B
```

## Run normalizer tests (offline)

```bash
cd bench
python3 -m pip install -e ".[dev]"
pytest tests/ -v
```

Tests use `tests/fixtures/mock_vllm_bench_output.json` shaped like real `--save-result` output. No GPU, no vLLM import.

## Normalize a single run manually

```bash
python normalize.py raw/max_concurrency_16_run1.json raw/max_concurrency_16_run1.meta.json \
  --out results/max_concurrency_16_run1.json
```

## Live baseline run (requires vLLM server)

1. Start a vLLM server (separate process/pod):

   ```bash
   vllm serve Qwen/Qwen3-8B --dtype bfloat16
   ```

2. Set GPU metadata and run the frozen baseline:

   ```bash
   cd bench
   export GPU_NAME="NVIDIA RTX 4090"
   # If the vLLM server requires auth (common on RunPod):
   # export VLLM_API_KEY="your-server-key"
   ./run_baseline.sh
   ```

   On RunPod, `VLLM_API_KEY` is usually already set for the vLLM server. Export it
   (or rely on the pod env) before running the baseline; `bench.sh` injects
   `--header Authorization=Bearer <key>` automatically. Do **not** pass the header
   via `VLLM_FLAGS` — spaces in `Bearer <key>` get split incorrectly.

   Output: `results/baseline.json` — a JSON array of averaged `BenchResult` objects, one entry per concurrency level.

Dry-run the baseline wrapper:

```bash
./run_baseline.sh --dry-run
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-8B` | Model name |
| `DTYPE` | `bf16` | Recorded in metadata |
| `CONCURRENCIES` | `1 4 16 64` | Space-separated `--max-concurrency` values |
| `REQUEST_RATES` | *(empty)* | Optional space-separated `--request-rate` values |
| `REPEATS` | `3` | Measured runs per load point |
| `INPUT_LEN` / `OUTPUT_LEN` | `1024` / `256` | Fixed token lengths |
| `SLO_TTFT_MS` | *(empty)* | If set, passes `--goodput ttft:<ms>` to vLLM |

Raw JSON lands in `raw/`; normalized output in `results/`.
