"""Load Hugging Face model configs from local files or remote repos."""

from __future__ import annotations

import json
import os
from typing import Any


def _extract_fields(raw: dict[str, Any]) -> dict[str, int]:
    """Extract normalized architecture fields from a raw HF config dict."""
    missing: list[str] = []

    if "num_hidden_layers" not in raw:
        missing.append("num_hidden_layers")
    if "num_key_value_heads" not in raw:
        missing.append("num_key_value_heads")
    if "hidden_size" not in raw:
        missing.append("hidden_size")

    if missing:
        raise ValueError(
            f"Config is missing required fields: {', '.join(missing)}"
        )

    n_layers = int(raw["num_hidden_layers"])
    n_kv_heads = int(raw["num_key_value_heads"])
    hidden_size = int(raw["hidden_size"])

    if "head_dim" in raw:
        head_dim = int(raw["head_dim"])
    elif "num_attention_heads" in raw:
        num_attention_heads = int(raw["num_attention_heads"])
        if num_attention_heads == 0:
            raise ValueError("num_attention_heads must be non-zero to derive head_dim")
        head_dim = hidden_size // num_attention_heads
    else:
        raise ValueError(
            "Config is missing head_dim and num_attention_heads; "
            "cannot derive head_dim"
        )

    return {
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
    }


def load_model_config(source: str) -> dict:
    """Load a model config from a local ``config.json`` path or HF repo id.

    Parameters
    ----------
    source:
        Local filesystem path to a ``config.json`` file, or a Hugging Face
        model repository id (e.g. ``"Qwen/Qwen3-8B"``).

    Returns
    -------
    dict
        Normalized config with keys ``n_layers``, ``n_kv_heads``,
        ``head_dim``, and ``hidden_size``.
    """
    if os.path.isfile(source):
        with open(source, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "Remote config fetch requires huggingface_hub. "
                "Install with: pip install 'kvcalc[hf]'"
            ) from exc

        config_path = hf_hub_download(repo_id=source, filename="config.json")
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)

    return _extract_fields(raw)
