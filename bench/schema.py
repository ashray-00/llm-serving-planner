"""Canonical benchmark result schema (the contract)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LatencyStatsMs(BaseModel):
    mean: float
    p50: float
    p95: float


class E2ELatencyStatsMs(BaseModel):
    mean: float
    p95: float


class BenchResult(BaseModel):
    model: str
    dtype: str
    gpu: str
    vllm_version: str
    flags: list[str]
    input_len: int
    output_len: int
    mode: Literal["max_concurrency", "request_rate"]
    load_value: float
    num_prompts: int
    ttft_ms: LatencyStatsMs
    itl_ms: LatencyStatsMs
    e2e_latency_ms: E2ELatencyStatsMs
    output_tok_per_s_per_request: float
    total_output_tok_per_s: float
    gpu_dollars_per_hr: float
    cost_per_1m_output_tokens_raw: float
    cost_per_1m_output_tokens_goodput: float | None = None
    goodput_req_per_s: float | None = None
    slo_ttft_ms: float | None = None
    queue_depth_mean: float | None = None
    gpu_cache_usage_perc: float | None = None
    num_preemptions_total: int | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @field_validator("timestamp")
    @classmethod
    def validate_iso8601(cls, value: str) -> str:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, payload: str) -> BenchResult:
        return cls.model_validate_json(payload)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, payload: dict) -> BenchResult:
        return cls.model_validate(payload)
