#!/usr/bin/env python3
"""Poll vLLM Prometheus /metrics during a benchmark run."""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CACHE_METRIC_NAMES = (
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_kv_cache_usage_perc",
)
PREEMPTION_METRIC_NAMES = (
    "vllm:num_preemptions_total",
    "vllm:num_preemptions",
)

_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+(?P<value>[-+0-9.eE]+)(?:\s+\d+)?$"
)


def _metric_base(name: str) -> str:
    return name.split("{", 1)[0]


def _parse_prometheus_text(payload: str) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {}
    for line in payload.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_RE.match(line)
        if not match:
            continue
        base = _metric_base(match.group("name"))
        metrics.setdefault(base, []).append(float(match.group("value")))
    return metrics


def _pick_metric(metrics: dict[str, list[float]], names: tuple[str, ...]) -> float | None:
    for name in names:
        values = metrics.get(name)
        if values:
            return max(values)
    return None


def _fetch_metrics(url: str, api_key: str | None, timeout: float) -> str:
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _print_matching_metric_lines(payload: str) -> None:
    print("[metrics_sampler] lines containing 'cache' or 'preempt':", file=sys.stderr)
    for line in payload.splitlines():
        lowered = line.lower()
        if "cache" in lowered or "preempt" in lowered:
            print(f"  {line}", file=sys.stderr)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    cache_values = [
        float(sample["gpu_cache_usage_perc"])
        for sample in samples
        if sample.get("gpu_cache_usage_perc") is not None
    ]
    preempt_values = [
        float(sample["num_preemptions_total"])
        for sample in samples
        if sample.get("num_preemptions_total") is not None
    ]

    summary: dict[str, Any] = {
        "sample_count": len(samples),
        "max_gpu_cache_usage_perc": max(cache_values) if cache_values else None,
        "num_preemptions_total": None,
    }
    if preempt_values:
        summary["num_preemptions_total"] = int(max(preempt_values) - min(preempt_values))
    return summary


def sample_once(
    url: str,
    api_key: str | None,
    timeout: float,
    *,
    print_matches: bool = False,
) -> tuple[dict[str, Any], str | None]:
    payload = _fetch_metrics(url, api_key, timeout)
    if print_matches:
        _print_matching_metric_lines(payload)

    parsed = _parse_prometheus_text(payload)
    sample = {
        "timestamp": _utc_now_iso(),
        "gpu_cache_usage_perc": _pick_metric(parsed, CACHE_METRIC_NAMES),
        "num_preemptions_total": _pick_metric(parsed, PREEMPTION_METRIC_NAMES),
    }
    if sample["num_preemptions_total"] is not None:
        sample["num_preemptions_total"] = int(sample["num_preemptions_total"])
    return sample, payload


def run_sampler(
    *,
    url: str,
    out: Path,
    duration: float,
    interval: float,
    api_key: str | None,
    timeout: float,
) -> int:
    stop = False

    def _handle_stop(signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    samples: list[dict[str, Any]] = []
    started = time.monotonic()
    printed_matches = False

    while not stop and (time.monotonic() - started) < duration:
        loop_start = time.monotonic()
        try:
            sample, _payload = sample_once(
                url,
                api_key,
                timeout,
                print_matches=not printed_matches,
            )
            if not printed_matches:
                printed_matches = True
            samples.append(sample)
        except urllib.error.HTTPError as exc:
            print(
                f"[metrics_sampler] HTTP error fetching {url}: {exc.code} {exc.reason}",
                file=sys.stderr,
            )
            return 1
        except urllib.error.URLError as exc:
            print(
                f"[metrics_sampler] failed to fetch {url}: {exc.reason}",
                file=sys.stderr,
            )
            return 1

        elapsed = time.monotonic() - loop_start
        sleep_for = interval - elapsed
        if sleep_for > 0 and not stop and (time.monotonic() - started) < duration:
            time.sleep(sleep_for)

    payload = {
        "url": url,
        "interval_s": interval,
        "duration_s": duration,
        "started_at": _utc_now_iso(),
        "samples": samples,
        "summary": summarize_samples(samples),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[metrics_sampler] wrote {out} ({len(samples)} samples)", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample vLLM Prometheus /metrics.")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/metrics",
        help="Prometheus metrics endpoint URL.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Write collected samples and summary stats to this JSON file.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Maximum sampling duration in seconds.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional Bearer token for authenticated /metrics endpoints.",
    )
    args = parser.parse_args(argv)

    if args.duration <= 0:
        parser.error("--duration must be > 0")
    if args.interval <= 0:
        parser.error("--interval must be > 0")

    api_key = args.api_key
    if api_key is None:
        import os

        api_key = os.environ.get("VLLM_API_KEY") or None

    return run_sampler(
        url=args.url,
        out=args.out,
        duration=args.duration,
        interval=args.interval,
        api_key=api_key,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
