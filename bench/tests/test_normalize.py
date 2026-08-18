"""Offline normalizer tests (no GPU, no vLLM install)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kvcalc import cost_per_million_output_tokens
from normalize import NormalizationError, normalize
from schema import BenchResult

FIXTURES = Path(__file__).parent / "fixtures"
MOCK_RAW = FIXTURES / "mock_vllm_bench_output.json"
MOCK_METRICS = FIXTURES / "mock_metrics_output.json"

BASE_META = {
    "model": "Qwen/Qwen3-8B",
    "dtype": "bf16",
    "gpu": "NVIDIA RTX 4090",
    "vllm_version": "0.9.0",
    "flags": ["--ignore-eos"],
    "input_len": 1024,
    "output_len": 256,
    "mode": "max_concurrency",
    "load_value": 16.0,
    "timestamp": "2025-08-13T10:30:45+00:00",
}


@pytest.fixture
def raw() -> dict:
    with MOCK_RAW.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_normalize_maps_all_fields(raw: dict) -> None:
    result = normalize(raw, BASE_META)

    assert isinstance(result, BenchResult)
    assert result.model == "Qwen/Qwen3-8B"
    assert result.dtype == "bf16"
    assert result.gpu == "NVIDIA RTX 4090"
    assert result.vllm_version == "0.9.0"
    assert result.flags == ["--ignore-eos"]
    assert result.input_len == 1024
    assert result.output_len == 256
    assert result.mode == "max_concurrency"
    assert result.load_value == 16.0
    assert result.num_prompts == 100

    assert result.ttft_ms.mean == pytest.approx(118.42)
    assert result.ttft_ms.p50 == pytest.approx(109.87)
    assert result.ttft_ms.p95 == pytest.approx(178.63)

    assert result.itl_ms.mean == pytest.approx(12.91)
    assert result.itl_ms.p50 == pytest.approx(12.28)
    assert result.itl_ms.p95 == pytest.approx(16.11)

    assert result.e2e_latency_ms.mean == pytest.approx(3412.55)
    assert result.e2e_latency_ms.p95 == pytest.approx(4021.77)

    assert result.total_output_tok_per_s == pytest.approx(597.15)
    assert result.gpu_dollars_per_hr == pytest.approx(0.34)
    assert result.cost_per_1m_output_tokens_raw == pytest.approx(
        cost_per_million_output_tokens(0.34, 597.15)
    )
    expected_per_request = (25600 / 100) / (3412.55 / 1000.0)
    assert result.output_tok_per_s_per_request == pytest.approx(expected_per_request)

    assert result.goodput_req_per_s is None
    assert result.cost_per_1m_output_tokens_goodput is None
    assert result.slo_ttft_ms is None
    assert result.queue_depth_mean is None
    assert result.gpu_cache_usage_perc is None
    assert result.num_preemptions_total is None
    assert result.timestamp == "2025-08-13T10:30:45+00:00"


def test_latency_units_are_milliseconds(raw: dict) -> None:
    result = normalize(raw, BASE_META)
    assert result.ttft_ms.mean > 1.0
    assert result.itl_ms.mean > 1.0
    assert result.e2e_latency_ms.mean > 100.0


def test_missing_required_field_raises(raw: dict) -> None:
    broken = dict(raw)
    del broken["mean_ttft_ms"]
    with pytest.raises(NormalizationError, match="mean_ttft_ms"):
        normalize(broken, BASE_META)


def test_goodput_none_without_slo(raw: dict) -> None:
    result = normalize(raw, BASE_META)
    assert result.goodput_req_per_s is None
    assert result.cost_per_1m_output_tokens_goodput is None
    assert result.slo_ttft_ms is None


def test_goodput_computed_with_slo(raw: dict) -> None:
    raw_with_goodput = dict(raw)
    raw_with_goodput["request_goodput"] = 2.01
    meta = dict(BASE_META)
    meta["slo_ttft_ms"] = 200.0

    result = normalize(raw_with_goodput, meta)
    assert result.slo_ttft_ms == pytest.approx(200.0)
    assert result.goodput_req_per_s == pytest.approx(2.01)
    assert result.cost_per_1m_output_tokens_goodput == pytest.approx(
        cost_per_million_output_tokens(0.34, 2.01 * BASE_META["output_len"])
    )


def test_goodput_raises_when_slo_but_missing_request_goodput(raw: dict) -> None:
    meta = dict(BASE_META)
    meta["slo_ttft_ms"] = 200.0
    with pytest.raises(NormalizationError, match="request_goodput"):
        normalize(raw, meta)


def test_roundtrip_json(raw: dict) -> None:
    result = normalize(raw, BASE_META)
    restored = BenchResult.from_json(result.to_json())
    assert restored == result


@pytest.mark.parametrize("output_tok_per_s", [0.0, -1.0])
def test_cost_per_million_output_tokens_raises_on_non_positive_rate(
    output_tok_per_s: float,
) -> None:
    with pytest.raises(ValueError, match="output_tok_per_s must be > 0"):
        cost_per_million_output_tokens(0.34, output_tok_per_s)


def test_normalize_uses_metrics_file_for_cache_and_preemption(raw: dict) -> None:
    with MOCK_METRICS.open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    result = normalize(raw, BASE_META, metrics)

    assert result.gpu_cache_usage_perc == pytest.approx(0.87)
    assert result.num_preemptions_total == 2


def test_normalize_metrics_file_optional_leaves_fields_none(raw: dict) -> None:
    result = normalize(raw, BASE_META, None)

    assert result.gpu_cache_usage_perc is None
    assert result.num_preemptions_total is None
