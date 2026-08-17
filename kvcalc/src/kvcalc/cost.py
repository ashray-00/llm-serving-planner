"""Cost helpers for throughput-normalized serving estimates."""


def cost_per_million_output_tokens(
    gpu_dollars_per_hr: float, output_tok_per_s: float
) -> float:
    """Convert hourly GPU cost and output throughput into $/1M output tokens.

    Unit derivation:
    ``gpu_dollars_per_hr / output_tok_per_s`` gives dollars times seconds per hour
    per output token. Multiplying by ``1e6`` converts to dollars per million output
    tokens, and dividing by ``3600`` converts seconds per hour back to hours.
    """

    if output_tok_per_s <= 0:
        raise ValueError(
            f"output_tok_per_s must be > 0 to compute cost, got {output_tok_per_s}."
        )
    return (gpu_dollars_per_hr / output_tok_per_s) * 1e6 / 3600
