"""GPU VRAM capacity and KV cache concurrency calculations."""


def kv_cache_size_bytes(n_tokens: int, kv_bytes_per_token: int) -> int:
    """Return total KV cache size in bytes for ``n_tokens``."""
    return n_tokens * kv_bytes_per_token


def max_concurrency(
    gpu_vram_gb: float,
    model_params_b: float,
    dtype_bytes: int,
    kv_bytes_per_token: int,
    context_len: int,
    overhead_gb: float = 1.5,
) -> dict:
    """Estimate max concurrent requests given VRAM and per-request context.

    Returns a dict with:
    - ``weights_gb``: model weight footprint on GPU
    - ``kv_headroom_gb``: remaining VRAM available for KV cache
    - ``max_kv_tokens``: total KV tokens that fit in headroom
    - ``max_concurrent_requests``: requests at ``context_len`` per request
    """
    weights_gb = model_params_b * dtype_bytes
    kv_headroom_gb = gpu_vram_gb - weights_gb - overhead_gb

    if kv_headroom_gb <= 0:
        max_kv_tokens = 0
        max_concurrent_requests = 0
    else:
        max_kv_tokens = int((kv_headroom_gb * 1e9) / kv_bytes_per_token)
        max_concurrent_requests = max_kv_tokens // context_len

    return {
        "weights_gb": weights_gb,
        "kv_headroom_gb": kv_headroom_gb,
        "max_kv_tokens": max_kv_tokens,
        "max_concurrent_requests": max_concurrent_requests,
    }
