#!/usr/bin/env bash
# Frozen baseline sweep: concurrency 1/4/16/64, 3 repeats, fixed token lengths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MODEL="${MODEL:-Qwen/Qwen3-8B}"
export DTYPE="${DTYPE:-bf16}"
export GPU_NAME="${GPU_NAME:-unknown-gpu}"
export VLLM_VERSION="${VLLM_VERSION:-$(vllm --version 2>/dev/null | head -n1 || echo unknown)}"
export INPUT_LEN="${INPUT_LEN:-1024}"
export OUTPUT_LEN="${OUTPUT_LEN:-256}"
export REPEATS="${REPEATS:-3}"
export CONCURRENCIES="${CONCURRENCIES:-1 4 16 64}"
export REQUEST_RATES="${REQUEST_RATES:-}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR}"
export VLLM_FLAGS="${VLLM_FLAGS:-}"

if [[ "${1:-}" == "--dry-run" ]]; then
  exec "$SCRIPT_DIR/bench.sh" --dry-run \
    --model "$MODEL" \
    --dtype "$DTYPE" \
    --gpu-name "$GPU_NAME" \
    --vllm-version "$VLLM_VERSION" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --repeats "$REPEATS" \
    --concurrencies "$CONCURRENCIES" \
    --output-dir "$OUTPUT_DIR"
fi

"$SCRIPT_DIR/bench.sh" \
  --model "$MODEL" \
  --dtype "$DTYPE" \
  --gpu-name "$GPU_NAME" \
  --vllm-version "$VLLM_VERSION" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --repeats "$REPEATS" \
  --concurrencies "$CONCURRENCIES" \
  --output-dir "$OUTPUT_DIR"

python3 - <<'PY'
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from normalize import average_results, normalize
from schema import BenchResult

bench_dir = Path.cwd()
raw_dir = bench_dir / "raw"
results_dir = bench_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

pattern = re.compile(
    r"^(?P<mode>max_concurrency|request_rate)_(?P<load>[^_]+)_run(?P<repeat>\d+)\.json$"
)

groups: dict[tuple[str, str], list[BenchResult]] = defaultdict(list)

for raw_path in sorted(raw_dir.glob("*.json")):
    match = pattern.match(raw_path.name)
    if not match:
        continue
    meta_path = raw_path.with_suffix(".meta.json")
    if not meta_path.exists():
        meta_path = raw_path.parent / f"{raw_path.stem}.meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Missing metadata for {raw_path.name}")

    result = normalize(json.loads(raw_path.read_text()), json.loads(meta_path.read_text()))
    key = (result.mode, str(result.load_value))
    groups[key].append(result)

if not groups:
    raise SystemExit("No measured raw files found to normalize.")

baseline: list[dict] = []
for key in sorted(groups, key=lambda item: (item[0], float(item[1]))):
    averaged = average_results(groups[key])
    baseline.append(averaged.to_dict())

out_path = results_dir / "baseline.json"
out_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
print(f"[run_baseline] wrote {out_path} ({len(baseline)} load points)")
PY
