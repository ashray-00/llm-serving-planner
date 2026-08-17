#!/usr/bin/env python3
"""Qwen3-8B on RTX 4090 worked example."""

from pathlib import Path

from kvcalc import (
    GPU_SPECS,
    decode_ceiling,
    kv_bytes_per_token,
    kv_cache_size_bytes,
    load_model_config,
    max_concurrency,
)

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "qwen3-8b-config.json"
MODEL_PARAMS_B = 8.0
DTYPE_BYTES = 2
CONTEXT_32K = 32_000
CONTEXT_4K = 4_096


def main() -> None:
    config = load_model_config(str(FIXTURE))
    per_token = kv_bytes_per_token(
        config["n_layers"],
        config["n_kv_heads"],
        config["head_dim"],
        DTYPE_BYTES,
    )

    kv_32k_bytes = kv_cache_size_bytes(CONTEXT_32K, per_token)
    kv_32k_gb = kv_32k_bytes / 1e9

    vram_gb, hbm_bandwidth_gbps, _ = GPU_SPECS["RTX4090"]
    concurrency = max_concurrency(
        gpu_vram_gb=vram_gb,
        model_params_b=MODEL_PARAMS_B,
        dtype_bytes=DTYPE_BYTES,
        kv_bytes_per_token=per_token,
        context_len=CONTEXT_4K,
    )

    ceiling = decode_ceiling(
        model_params_b=MODEL_PARAMS_B,
        dtype_bytes=DTYPE_BYTES,
        hbm_bandwidth_gbps=hbm_bandwidth_gbps,
    )

    print("Qwen3-8B on RTX 4090 (bf16)")
    print("=" * 40)
    print(f"KV bytes/token:              {per_token:,}")
    print(f"KV cache at 32k context:     {kv_32k_gb:.2f} GB")
    print(f"Max concurrency at 4k ctx:   {concurrency['max_concurrent_requests']} requests")
    print(f"  (weights: {concurrency['weights_gb']:.1f} GB, "
          f"KV headroom: {concurrency['kv_headroom_gb']:.1f} GB)")
    print(f"Decode ceiling:              {ceiling['ms_per_token']:.2f} ms/token")
    print(f"                             {ceiling['tokens_per_sec']:.1f} tok/s")


if __name__ == "__main__":
    main()
