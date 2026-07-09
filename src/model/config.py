"""Architecture hyperparameters for the baseline transformer.

Values are the "Default" column of docs/design.md section 2. This is a
plain dataclass (not argparse) so it can be constructed identically from
tests, training scripts, and future ablation sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    d_model: int = 512
    n_layers: int = 10
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_hidden: int = 1280
    context_len: int = 1024
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must divide n_kv_heads (GQA)"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
