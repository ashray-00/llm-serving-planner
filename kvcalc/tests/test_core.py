"""Tests for kvcalc.core."""

from kvcalc import kv_bytes_per_token


def test_kv_bytes_per_token_qwen3_8b():
    assert kv_bytes_per_token(36, 8, 128, 2) == 147456
