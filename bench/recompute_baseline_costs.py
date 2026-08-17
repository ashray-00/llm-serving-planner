#!/usr/bin/env python3
"""Backfill cost fields onto already-normalized benchmark baseline results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from normalize import compute_cost_fields
from schema import BenchResult

BASELINE_PATH = Path(__file__).resolve().parent / "results" / "baseline.json"


def recompute_baseline_costs(path: Path = BASELINE_PATH) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Baseline results not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array of BenchResult objects.")

    updated: list[dict] = []
    for item in payload:
        cost_fields = compute_cost_fields(
            gpu_name=str(item["gpu"]),
            output_len=int(item["output_len"]),
            total_output_tok_per_s=float(item["total_output_tok_per_s"]),
            goodput_req_per_s=(
                float(item["goodput_req_per_s"])
                if item.get("goodput_req_per_s") is not None
                else None
            ),
        )
        entry = dict(item)
        entry.update(cost_fields)
        updated.append(BenchResult.from_dict(entry).to_dict())

    path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    return len(updated)


def main() -> int:
    updated_count = recompute_baseline_costs()
    print(f"Updated {updated_count} baseline result entries in {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
