# kvcalc

Analytical GPU memory and bandwidth model for LLM serving. Quickly estimate KV cache footprint, VRAM-limited concurrency, and memory-bandwidth-bound decode throughput without running benchmarks.

## Install

```bash
pip install -e .
```

For loading configs from Hugging Face Hub:

```bash
pip install -e ".[hf]"
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from kvcalc import (
    GPU_SPECS,
    decode_ceiling,
    kv_bytes_per_token,
    kv_cache_size_bytes,
    load_model_config,
    max_concurrency,
)

config = load_model_config("tests/fixtures/qwen3-8b-config.json")
per_token = kv_bytes_per_token(
    config["n_layers"],
    config["n_kv_heads"],
    config["head_dim"],
    dtype_bytes=2,  # bf16
)

print(f"KV bytes/token: {per_token:,}")
print(f"32k context KV: {kv_cache_size_bytes(32_000, per_token) / 1e9:.2f} GB")

vram_gb, bandwidth_gbps = GPU_SPECS["RTX4090"]
conc = max_concurrency(vram_gb, model_params_b=8.0, dtype_bytes=2,
                       kv_bytes_per_token=per_token, context_len=4096)
print(f"Max concurrent 4k requests: {conc['max_concurrent_requests']}")

ceil = decode_ceiling(8.0, dtype_bytes=2, hbm_bandwidth_gbps=bandwidth_gbps)
print(f"Decode ceiling: {ceil['tokens_per_sec']:.0f} tok/s")
```

Run the full worked example:

```bash
python examples/qwen3_8b_4090.py
```

## What this predicts well vs what it doesn't

### Predicts well (near-deterministic)

- **KV bytes per token** — derived directly from architecture (layers, KV heads, head dim, dtype).
- **KV cache size at a given context length** — linear in token count.
- **Max concurrency** — given GPU VRAM, model weight footprint, and per-request context length.
- **Memory headroom** — how much VRAM remains for KV cache after weights and overhead.
- **Decode ceiling (tok/s)** — batch-1 throughput when limited by HBM bandwidth reading full weights each step.

### Does NOT predict

- **End-to-end throughput under load** — kernel efficiency, batching, scheduler behavior, CUDA graphs, and framework overhead dominate real serving throughput.
- **Prefill latency** — compute-bound prefill has different bottlenecks than decode.
- **Multi-GPU or disaggregated serving** — no network or pipeline parallelism modeling.

For production capacity planning beyond these analytical bounds, you need actual benchmarking on your target hardware and software stack.

## License

MIT
