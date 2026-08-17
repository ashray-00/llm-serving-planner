"""Core KV cache size calculations."""


def kv_bytes_per_token(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> int:
    """Return the KV cache bytes consumed per generated token.

    Formula: ``2 * n_layers * n_kv_heads * head_dim * dtype_bytes``

    The leading factor of 2 accounts for separate K and V tensors stored
    per layer, per KV head, per token.
    """
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
