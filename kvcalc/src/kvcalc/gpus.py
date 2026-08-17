"""GPU hardware specifications for capacity, bandwidth, and cost modeling."""

from __future__ import annotations

GPU_SPECS: dict[str, tuple[float, float, float]] = {
    "RTX4090": (24, 1008, 0.34),
    # Placeholder community-cloud-style hourly prices only.
    # Verify against current real rental rates before trusting cost outputs.
    "A100_80GB": (80, 2039, 1.85),
    "A100_40GB": (40, 1555, 1.25),
    "H100_80GB": (80, 3350, 3.85),
    "L40S": (48, 864, 0.80),
}

_GPU_NAME_ALIASES = {
    "NVIDIA RTX 4090": "RTX4090",
    "NVIDIA A100 80GB": "A100_80GB",
    "NVIDIA A100 40GB": "A100_40GB",
    "NVIDIA H100 80GB": "H100_80GB",
    "NVIDIA L40S": "L40S",
}


def canonical_gpu_name(name: str) -> str:
    """Map human-readable GPU labels to ``GPU_SPECS`` keys."""

    return _GPU_NAME_ALIASES.get(name, name)


def get_gpu_specs(name: str) -> tuple[float, float, float]:
    """Return ``(vram_gb, hbm_bandwidth_gbps, dollars_per_hr)`` for a GPU."""

    key = canonical_gpu_name(name)
    try:
        return GPU_SPECS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown GPU {name!r}; known keys: {sorted(GPU_SPECS)}"
        ) from exc
