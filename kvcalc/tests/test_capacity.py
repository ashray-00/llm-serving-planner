"""Tests for kvcalc.capacity."""

from pathlib import Path

import pytest

from kvcalc import (
    GPU_SPECS,
    kv_bytes_per_token,
    kv_cache_size_bytes,
    load_model_config,
    max_concurrency,
)

FIXTURE = Path(__file__).parent / "fixtures" / "qwen3-8b-config.json"

N_LAYERS = 36
N_KV_HEADS = 8
HEAD_DIM = 128
DTYPE_BYTES = 2
KV_BYTES_PER_TOKEN = 147456
CONTEXT_32K = 32_000
CONTEXT_4K = 4_096
MODEL_PARAMS_B = 8.0


@pytest.fixture
def qwen_config():
    return load_model_config(str(FIXTURE))


def test_load_model_config_from_fixture(qwen_config):
    assert qwen_config["n_layers"] == N_LAYERS
    assert qwen_config["n_kv_heads"] == N_KV_HEADS
    assert qwen_config["head_dim"] == HEAD_DIM
    assert qwen_config["hidden_size"] == 4096


def test_kv_cache_size_32k_context():
    size_bytes = kv_cache_size_bytes(CONTEXT_32K, KV_BYTES_PER_TOKEN)
    size_gb = size_bytes / 1e9
    assert size_gb == pytest.approx(4.6, rel=0.05)


def test_max_concurrency_rtx4090_4k_context():
    vram_gb, _, _ = GPU_SPECS["RTX4090"]
    result = max_concurrency(
        gpu_vram_gb=vram_gb,
        model_params_b=MODEL_PARAMS_B,
        dtype_bytes=DTYPE_BYTES,
        kv_bytes_per_token=KV_BYTES_PER_TOKEN,
        context_len=CONTEXT_4K,
    )
    assert result["weights_gb"] == pytest.approx(16.0)
    assert result["max_concurrent_requests"] == pytest.approx(10, abs=1)


def test_kv_bytes_from_loaded_config(qwen_config):
    per_token = kv_bytes_per_token(
        qwen_config["n_layers"],
        qwen_config["n_kv_heads"],
        qwen_config["head_dim"],
        DTYPE_BYTES,
    )
    assert per_token == KV_BYTES_PER_TOKEN
