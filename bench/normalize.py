"""Normalize raw vLLM bench serve JSON into the canonical BenchResult schema."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "kvcalc" / "src"))

from kvcalc import cost_per_million_output_tokens, get_gpu_specs
from schema import BenchResult, E2ELatencyStatsMs, LatencyStatsMs

# Required keys in vLLM bench serve --save-result output (vllm/benchmarks/serve.py,
# current main branch). Percentile keys use p{int}_* form from --metric-percentiles.
_REQUIRED_RAW_FIELDS = (
    "model_id",
    "num_prompts",
    "mean_ttft_ms",
    "median_ttft_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "mean_e2el_ms",
    "output_throughput",
    "request_throughput",
    "completed",
    "duration",
)

_PERCENTILE_SUFFIXES = ("50", "95")


class NormalizationError(ValueError):
    """Raised when raw vLLM output cannot be mapped to BenchResult."""


def _require(raw: dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise NormalizationError(
            f"Missing required raw field {key!r}. "
            f"Available keys: {sorted(raw.keys())}"
        )
    return raw[key]


def _percentile(raw: dict[str, Any], metric: str, percentile: int) -> float:
    key = f"p{percentile}_{metric}_ms"
    if key in raw:
        return float(raw[key])
    for alt in (float(percentile), f"{percentile}.0"):
        alt_key = f"p{alt}_{metric}_ms"
        if alt_key in raw:
            return float(raw[alt_key])
    raise NormalizationError(
        f"Missing percentile field {key!r} for metric {metric!r}. "
        f"Re-run bench.sh with --metric-percentiles 50,95,99."
    )


def _latency_stats(raw: dict[str, Any], metric: str) -> LatencyStatsMs:
    mean_key = f"mean_{metric}_ms"
    median_key = f"median_{metric}_ms"
    mean = float(_require(raw, mean_key))
    p50 = float(raw.get(median_key, _percentile(raw, metric, 50)))
    p95 = _percentile(raw, metric, 95)
    return LatencyStatsMs(mean=mean, p50=p50, p95=p95)


def _e2e_stats(raw: dict[str, Any]) -> E2ELatencyStatsMs:
    mean = float(_require(raw, "mean_e2el_ms"))
    p95 = _percentile(raw, "e2el", 95)
    return E2ELatencyStatsMs(mean=mean, p95=p95)


def _metrics_summary_fields(
    metrics: dict[str, Any] | None,
) -> tuple[float | None, int | None]:
    """Pull cache/preemption summary stats from a metrics sampler JSON file."""

    if not metrics:
        return None, None

    summary = metrics.get("summary")
    if not isinstance(summary, dict):
        return None, None

    cache = summary.get("max_gpu_cache_usage_perc")
    preempt = summary.get("num_preemptions_total")

    gpu_cache_usage_perc = float(cache) if cache is not None else None
    num_preemptions_total = int(preempt) if preempt is not None else None
    return gpu_cache_usage_perc, num_preemptions_total


def _optional_gpu_cache_usage_perc(
    raw: dict[str, Any],
    meta: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> float | None:
    """Map KV cache occupancy (0–1) if present in raw or sidecar metadata.

    vLLM ``bench serve --save-result`` does not emit this field; it is exposed
    on the running server's Prometheus ``/metrics`` endpoint as
    ``vllm:gpu_cache_usage_perc`` (v0) or ``vllm:kv_cache_usage_perc`` (v1).
    """
    metrics_cache, _ = _metrics_summary_fields(metrics)
    if metrics_cache is not None:
        return metrics_cache

    for source in (raw, meta):
        for key in (
            "gpu_cache_usage_perc",
            "kv_cache_usage_perc",
            "gpu_kv_cache_usage_perc",
        ):
            if key in source and source[key] is not None:
                return float(source[key])
    return None


def _optional_num_preemptions_total(
    raw: dict[str, Any],
    meta: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> int | None:
    """Map cumulative preemption count if present in raw or sidecar metadata.

    vLLM ``bench serve --save-result`` does not emit this field; it is exposed
    on the running server's Prometheus ``/metrics`` endpoint as
    ``vllm:num_preemptions_total``.
    """
    _, metrics_preempt = _metrics_summary_fields(metrics)
    if metrics_preempt is not None:
        return metrics_preempt

    for source in (raw, meta):
        for key in ("num_preemptions_total", "num_preemptions"):
            if key in source and source[key] is not None:
                return int(source[key])
    return None


def _output_tok_per_s_per_request(raw: dict[str, Any]) -> float:
    completed = int(_require(raw, "completed"))
    mean_e2el_ms = float(_require(raw, "mean_e2el_ms"))
    total_output_tokens = raw.get("total_output_tokens")
    if total_output_tokens is None:
        raise NormalizationError(
            "Missing required raw field 'total_output_tokens' needed to compute "
            "per-request output throughput."
        )
    if completed <= 0:
        raise NormalizationError("completed must be > 0 to compute throughput.")
    if mean_e2el_ms <= 0:
        raise NormalizationError("mean_e2el_ms must be > 0 to compute throughput.")
    avg_output_tokens = float(total_output_tokens) / completed
    return avg_output_tokens / (mean_e2el_ms / 1000.0)


def compute_cost_fields(
    gpu_name: str,
    output_len: int,
    total_output_tok_per_s: float,
    goodput_req_per_s: float | None,
) -> dict[str, float | None]:
    """Compute GPU hourly cost and token-normalized raw/goodput cost fields."""

    _, _, gpu_dollars_per_hr = get_gpu_specs(gpu_name)
    goodput_tok_per_s = (
        float(goodput_req_per_s) * int(output_len)
        if goodput_req_per_s is not None
        else None
    )
    return {
        "gpu_dollars_per_hr": gpu_dollars_per_hr,
        "cost_per_1m_output_tokens_raw": cost_per_million_output_tokens(
            gpu_dollars_per_hr=gpu_dollars_per_hr,
            output_tok_per_s=float(total_output_tok_per_s),
        ),
        "cost_per_1m_output_tokens_goodput": (
            cost_per_million_output_tokens(
                gpu_dollars_per_hr=gpu_dollars_per_hr,
                output_tok_per_s=goodput_tok_per_s,
            )
            if goodput_tok_per_s is not None
            else None
        ),
    }


def normalize(
    raw_json: dict[str, Any],
    meta: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> BenchResult:
    """Map vLLM bench serve output + sidecar metadata to BenchResult."""
    for key in _REQUIRED_RAW_FIELDS:
        _require(raw_json, key)

    slo_ttft_ms = meta.get("slo_ttft_ms")
    if slo_ttft_ms is not None:
        slo_ttft_ms = float(slo_ttft_ms)

    goodput: float | None = None
    if slo_ttft_ms is not None:
        if "request_goodput" not in raw_json:
            raise NormalizationError(
                "slo_ttft_ms provided in metadata but raw JSON lacks "
                "'request_goodput'. Re-run with vLLM --goodput ttft:<ms>."
            )
        raw_goodput = raw_json["request_goodput"]
        if raw_goodput is None:
            raise NormalizationError(
                "request_goodput is null despite SLO metadata; ensure bench.sh "
                "passes --goodput ttft:<slo_ms>."
            )
        goodput = float(raw_goodput)

    mode = meta.get("mode")
    if mode not in ("max_concurrency", "request_rate"):
        raise NormalizationError(
            f"meta.mode must be 'max_concurrency' or 'request_rate', got {mode!r}"
        )

    load_value = meta.get("load_value")
    if load_value is None:
        raise NormalizationError("meta.load_value is required.")

    timestamp = meta.get("timestamp")
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    total_output_tok_per_s = float(raw_json["output_throughput"])
    cost_fields = compute_cost_fields(
        gpu_name=str(meta["gpu"]),
        output_len=int(meta["output_len"]),
        total_output_tok_per_s=total_output_tok_per_s,
        goodput_req_per_s=goodput,
    )

    return BenchResult(
        model=str(meta.get("model") or raw_json.get("model_id")),
        dtype=str(meta["dtype"]),
        gpu=str(meta["gpu"]),
        vllm_version=str(meta["vllm_version"]),
        flags=list(meta.get("flags") or []),
        input_len=int(meta["input_len"]),
        output_len=int(meta["output_len"]),
        mode=mode,
        load_value=float(load_value),
        num_prompts=int(raw_json["num_prompts"]),
        ttft_ms=_latency_stats(raw_json, "ttft"),
        itl_ms=_latency_stats(raw_json, "itl"),
        e2e_latency_ms=_e2e_stats(raw_json),
        output_tok_per_s_per_request=_output_tok_per_s_per_request(raw_json),
        total_output_tok_per_s=total_output_tok_per_s,
        gpu_dollars_per_hr=float(cost_fields["gpu_dollars_per_hr"]),
        cost_per_1m_output_tokens_raw=float(
            cost_fields["cost_per_1m_output_tokens_raw"]
        ),
        cost_per_1m_output_tokens_goodput=cost_fields[
            "cost_per_1m_output_tokens_goodput"
        ],
        goodput_req_per_s=goodput,
        slo_ttft_ms=slo_ttft_ms,
        queue_depth_mean=(
            float(meta["queue_depth_mean"])
            if meta.get("queue_depth_mean") is not None
            else None
        ),
        gpu_cache_usage_perc=_optional_gpu_cache_usage_perc(raw_json, meta, metrics),
        num_preemptions_total=_optional_num_preemptions_total(raw_json, meta, metrics),
        timestamp=str(timestamp),
    )


def average_results(results: list[BenchResult]) -> BenchResult:
    """Average numeric fields across repeated runs with identical config."""
    if not results:
        raise ValueError("average_results requires at least one BenchResult.")
    if len(results) == 1:
        return results[0]

    def avg(values: list[float]) -> float:
        return statistics.mean(values)

    first = results[0]
    total_output_tok_per_s = avg([r.total_output_tok_per_s for r in results])
    goodput_req_per_s = (
        avg([r.goodput_req_per_s for r in results if r.goodput_req_per_s is not None])
        if all(r.goodput_req_per_s is not None for r in results)
        else None
    )
    cost_fields = compute_cost_fields(
        gpu_name=first.gpu,
        output_len=first.output_len,
        total_output_tok_per_s=total_output_tok_per_s,
        goodput_req_per_s=goodput_req_per_s,
    )
    return BenchResult(
        model=first.model,
        dtype=first.dtype,
        gpu=first.gpu,
        vllm_version=first.vllm_version,
        flags=first.flags,
        input_len=first.input_len,
        output_len=first.output_len,
        mode=first.mode,
        load_value=first.load_value,
        num_prompts=first.num_prompts,
        ttft_ms=LatencyStatsMs(
            mean=avg([r.ttft_ms.mean for r in results]),
            p50=avg([r.ttft_ms.p50 for r in results]),
            p95=avg([r.ttft_ms.p95 for r in results]),
        ),
        itl_ms=LatencyStatsMs(
            mean=avg([r.itl_ms.mean for r in results]),
            p50=avg([r.itl_ms.p50 for r in results]),
            p95=avg([r.itl_ms.p95 for r in results]),
        ),
        e2e_latency_ms=E2ELatencyStatsMs(
            mean=avg([r.e2e_latency_ms.mean for r in results]),
            p95=avg([r.e2e_latency_ms.p95 for r in results]),
        ),
        output_tok_per_s_per_request=avg(
            [r.output_tok_per_s_per_request for r in results]
        ),
        total_output_tok_per_s=total_output_tok_per_s,
        gpu_dollars_per_hr=float(cost_fields["gpu_dollars_per_hr"]),
        cost_per_1m_output_tokens_raw=float(
            cost_fields["cost_per_1m_output_tokens_raw"]
        ),
        cost_per_1m_output_tokens_goodput=cost_fields[
            "cost_per_1m_output_tokens_goodput"
        ],
        goodput_req_per_s=goodput_req_per_s,
        slo_ttft_ms=first.slo_ttft_ms,
        queue_depth_mean=(
            avg([r.queue_depth_mean for r in results if r.queue_depth_mean is not None])
            if all(r.queue_depth_mean is not None for r in results)
            else None
        ),
        gpu_cache_usage_perc=(
            max(r.gpu_cache_usage_perc for r in results if r.gpu_cache_usage_perc is not None)
            if any(r.gpu_cache_usage_perc is not None for r in results)
            else None
        ),
        num_preemptions_total=(
            max(r.num_preemptions_total for r in results if r.num_preemptions_total is not None)
            if any(r.num_preemptions_total is not None for r in results)
            else None
        ),
        timestamp=max(r.timestamp for r in results),
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize vLLM bench serve output.")
    parser.add_argument("raw_path", type=Path, help="Path to raw vLLM JSON output.")
    parser.add_argument("meta_path", type=Path, help="Path to sidecar metadata JSON.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Optional metrics sampler JSON with cache/preemption summary stats.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Write normalized BenchResult JSON here.",
    )
    args = parser.parse_args(argv)

    raw = _load_json(args.raw_path)
    meta = _load_json(args.meta_path)
    metrics = _load_json(args.metrics) if args.metrics is not None else None
    result = normalize(raw, meta, metrics)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result.to_json() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
