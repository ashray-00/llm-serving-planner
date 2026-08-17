"""Tests for kvcalc.bandwidth."""

import pytest

from kvcalc import GPU_SPECS, decode_ceiling

MODEL_PARAMS_B = 8.0
DTYPE_BYTES = 2


def test_decode_ceiling_rtx4090_8b_bf16():
    _, hbm_bandwidth_gbps, _ = GPU_SPECS["RTX4090"]
    result = decode_ceiling(
        model_params_b=MODEL_PARAMS_B,
        dtype_bytes=DTYPE_BYTES,
        hbm_bandwidth_gbps=hbm_bandwidth_gbps,
    )
    assert result["tokens_per_sec"] == pytest.approx(63, rel=0.02)
    assert result["ms_per_token"] == pytest.approx(1000.0 / 63, rel=0.02)
