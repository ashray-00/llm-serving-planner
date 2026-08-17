#!/usr/bin/env python3
"""Compare kvcalc predictions against bench baseline measurements (offline)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "kvcalc" / "src"))
sys.path.insert(0, str(REPO_ROOT / "bench"))

from kvcalc import (  # noqa: E402
    GPU_SPECS,
    decode_ceiling,
    kv_bytes_per_token,
    load_model_config,
    max_concurrency,
)
from schema import BenchResult  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_GPU = "RTX4090"
DEFAULT_DTYPE = "bf16"
DEFAULT_INPUT_LEN = 1024
DEFAULT_OUTPUT_LEN = 256
DEFAULT_MODEL_PARAMS_B = 8.0
DEFAULT_MODEL_CONFIG = (
    REPO_ROOT / "kvcalc" / "tests" / "fixtures" / "qwen3-8b-config.json"
)
DEFAULT_BASELINE = REPO_ROOT / "bench" / "results" / "baseline.json"
DEFAULT_OUTPUT = REPO_ROOT / "validation" / "validation_table.md"

_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


def _dtype_bytes(dtype: str) -> int:
    key = dtype.lower()
    if key not in _DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {sorted(_DTYPE_BYTES)}")
    return _DTYPE_BYTES[key]


def _load_baseline(path: Path) -> list[BenchResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array of BenchResult objects.")
    return [BenchResult.from_dict(item) for item in payload]


def _max_concurrency_results(results: list[BenchResult]) -> list[BenchResult]:
    return sorted(
        [r for r in results if r.mode == "max_concurrency"],
        key=lambda r: r.load_value,
    )


def _decode_at_concurrency_one(results: list[BenchResult]) -> float | None:
    for result in _max_concurrency_results(results):
        if int(result.load_value) == 1:
            return result.output_tok_per_s_per_request
    return None


def _highest_tested_concurrency(results: list[BenchResult]) -> int | None:
    conc_results = _max_concurrency_results(results)
    if not conc_results:
        return None
    return int(max(r.load_value for r in conc_results))


def _measured_max_concurrency(results: list[BenchResult]) -> int | None:
    """Highest concurrency without observed preemption or full KV cache."""
    conc_results = _max_concurrency_results(results)
    if not conc_results:
        return None

    stable: list[int] = []
    for result in conc_results:
        load = int(result.load_value)
        preempted = (
            result.num_preemptions_total is not None
            and result.num_preemptions_total > 0
        )
        cache_full = (
            result.gpu_cache_usage_perc is not None
            and result.gpu_cache_usage_perc >= 0.99
        )
        if not preempted and not cache_full:
            stable.append(load)

    if stable:
        return max(stable)
    return _highest_tested_concurrency(results)


def _preemption_onset_concurrency(results: list[BenchResult]) -> int | None:
    for result in _max_concurrency_results(results):
        if (
            result.num_preemptions_total is not None
            and result.num_preemptions_total > 0
        ):
            return int(result.load_value)
        if (
            result.gpu_cache_usage_perc is not None
            and result.gpu_cache_usage_perc >= 0.99
        ):
            return int(result.load_value)
    return None


def _metrics_missing_everywhere(results: list[BenchResult]) -> bool:
    conc_results = _max_concurrency_results(results)
    if not conc_results:
        return True
    for result in conc_results:
        if result.gpu_cache_usage_perc is not None:
            return False
        if result.num_preemptions_total is not None:
            return False
    return True


def _suggest_concurrencies(predicted_max: int) -> list[int]:
    if predicted_max <= 0:
        return [32, 48, 64, 96, 128]
    candidates = {
        max(1, predicted_max // 2),
        max(1, int(predicted_max * 0.75)),
        predicted_max,
        int(predicted_max * 1.25),
        int(predicted_max * 1.5),
        predicted_max * 2,
    }
    return sorted(candidates)


def _fmt_num(value: float | int | None, precision: int = 1) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{precision}f}"


def _fmt_delta(
    predicted: float | int | None,
    measured: float | int | None,
    precision: int = 1,
) -> str:
    if predicted is None or measured is None:
        return "N/A"
    delta = float(measured) - float(predicted)
    if precision == 0:
        return str(int(round(delta)))
    return f"{delta:+.{precision}f}"


def _build_table(
    *,
    context_len: int,
    predicted_decode_tps: float,
    predicted_max_concurrency: int,
    measured_decode_tps: float | None,
    measured_max_concurrency: int | None,
    measured_preemption_onset: int | None,
) -> str:
    rows = [
        (
            "Batch-1 decode tok/s",
            _fmt_num(predicted_decode_tps),
            _fmt_num(measured_decode_tps),
            _fmt_delta(predicted_decode_tps, measured_decode_tps),
            "TBD",
        ),
        (
            f"Max concurrency @ {context_len}",
            _fmt_num(predicted_max_concurrency, precision=0),
            _fmt_num(measured_max_concurrency, precision=0),
            _fmt_delta(predicted_max_concurrency, measured_max_concurrency, precision=0),
            "TBD",
        ),
        (
            "Preemption onset concurrency",
            _fmt_num(predicted_max_concurrency, precision=0),
            _fmt_num(measured_preemption_onset, precision=0),
            _fmt_delta(predicted_max_concurrency, measured_preemption_onset, precision=0),
            "TBD",
        ),
    ]

    lines = [
        "# kvcalc vs baseline validation",
        "",
        "| Metric | Predicted | Measured | Delta | Notes |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for metric, predicted, measured, delta, notes in rows:
        lines.append(
            f"| {metric} | {predicted} | {measured} | {delta} | {notes} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare kvcalc predictions to bench baseline results."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--input-len", type=int, default=DEFAULT_INPUT_LEN)
    parser.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN)
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help="Per-request KV context (default: input-len + output-len).",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Local config.json path (avoids Hugging Face network fetch).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Path to bench/results/baseline.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Markdown table output path.",
    )
    args = parser.parse_args(argv)

    if args.gpu not in GPU_SPECS:
        raise SystemExit(
            f"Unknown GPU {args.gpu!r}; known keys: {sorted(GPU_SPECS)}"
        )
    if not args.model_config.is_file():
        raise SystemExit(f"Model config not found: {args.model_config}")
    if not args.baseline.is_file():
        raise SystemExit(f"Baseline results not found: {args.baseline}")

    context_len = args.context_len
    if context_len is None:
        context_len = args.input_len + args.output_len

    dtype_bytes = _dtype_bytes(args.dtype)
    config = load_model_config(str(args.model_config))
    per_token = kv_bytes_per_token(
        config["n_layers"],
        config["n_kv_heads"],
        config["head_dim"],
        dtype_bytes,
    )

    vram_gb, hbm_bandwidth_gbps, _ = GPU_SPECS[args.gpu]
    conc = max_concurrency(
        gpu_vram_gb=vram_gb,
        model_params_b=DEFAULT_MODEL_PARAMS_B,
        dtype_bytes=dtype_bytes,
        kv_bytes_per_token=per_token,
        context_len=context_len,
    )
    ceiling = decode_ceiling(
        model_params_b=DEFAULT_MODEL_PARAMS_B,
        dtype_bytes=dtype_bytes,
        hbm_bandwidth_gbps=hbm_bandwidth_gbps,
    )

    baseline = _load_baseline(args.baseline)
    measured_decode = _decode_at_concurrency_one(baseline)
    measured_max_conc = _measured_max_concurrency(baseline)
    measured_preemption = _preemption_onset_concurrency(baseline)
    highest_tested = _highest_tested_concurrency(baseline)

    table = _build_table(
        context_len=context_len,
        predicted_decode_tps=ceiling["tokens_per_sec"],
        predicted_max_concurrency=conc["max_concurrent_requests"],
        measured_decode_tps=measured_decode,
        measured_max_concurrency=measured_max_conc,
        measured_preemption_onset=measured_preemption,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")

    print(table, end="")
    print(
        f"[compare] model={args.model} gpu={args.gpu} dtype={args.dtype} "
        f"context_len={context_len}",
        file=sys.stderr,
    )
    print(f"[compare] wrote {args.output}", file=sys.stderr)

    if _metrics_missing_everywhere(baseline):
        suggested = _suggest_concurrencies(conc["max_concurrent_requests"])
        tested = (
            ", ".join(str(int(r.load_value)) for r in _max_concurrency_results(baseline))
            or "none"
        )
        print(
            "\nWARNING: gpu_cache_usage_perc and num_preemptions_total are None "
            f"across all baseline entries (tested concurrencies: {tested}). "
            "No preemption cliff was observed in this sweep — run higher "
            f"concurrency levels to find it. Suggested values based on kvcalc "
            f"predicted max concurrency ({conc['max_concurrent_requests']}): "
            f"{', '.join(str(v) for v in suggested)}.",
            file=sys.stderr,
        )
        if highest_tested is not None:
            print(
                f"  Highest concurrency tested so far: {highest_tested}.",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
