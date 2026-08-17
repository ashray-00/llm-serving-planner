"""Memory-bandwidth-bound decode throughput estimates."""


def decode_ceiling(
    model_params_b: float,
    dtype_bytes: int,
    hbm_bandwidth_gbps: float,
) -> dict:
    """Estimate batch-1 decode ceiling when limited by HBM bandwidth.

    Each generated token reads the full model weights once. Returns:
    - ``ms_per_token``: milliseconds per token at bandwidth ceiling
    - ``tokens_per_sec``: sustainable decode tokens/sec
    """
    bytes_per_token = model_params_b * 1e9 * dtype_bytes
    bandwidth_bytes_per_sec = hbm_bandwidth_gbps * 1e9

    tokens_per_sec = bandwidth_bytes_per_sec / bytes_per_token
    ms_per_token = 1000.0 / tokens_per_sec

    return {
        "ms_per_token": ms_per_token,
        "tokens_per_sec": tokens_per_sec,
    }
