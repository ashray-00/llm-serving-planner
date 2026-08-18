"""Offline tests for metrics_sampler parsing and summarization."""

from __future__ import annotations

from metrics_sampler import _parse_prometheus_text, _pick_metric, summarize_samples

SAMPLE_PROMETHEUS = """
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.25
# HELP vllm:num_preemptions_total Cumulative preemptions.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total 100
"""


def test_parse_prometheus_text_extracts_cache_and_preemption_metrics() -> None:
    parsed = _parse_prometheus_text(SAMPLE_PROMETHEUS)

    assert _pick_metric(parsed, ("vllm:gpu_cache_usage_perc",)) == 0.25
    assert _pick_metric(parsed, ("vllm:num_preemptions_total",)) == 100.0


def test_summarize_samples_tracks_max_cache_and_preemption_delta() -> None:
    samples = [
        {
            "timestamp": "t0",
            "gpu_cache_usage_perc": 0.4,
            "num_preemptions_total": 10,
        },
        {
            "timestamp": "t1",
            "gpu_cache_usage_perc": 0.9,
            "num_preemptions_total": 15,
        },
    ]

    summary = summarize_samples(samples)

    assert summary["max_gpu_cache_usage_perc"] == 0.9
    assert summary["num_preemptions_total"] == 5
