import torch

from src.model.config import ModelConfig
from src.model.layers import (
    RMSNorm,
    SwiGLUMLP,
    TransformerBlock,
    apply_rotary,
    precompute_rope,
)


def test_rmsnorm_preserves_shape_and_normalizes():
    x = torch.randn(2, 5, 16) * 10
    norm = RMSNorm(16)
    out = norm(x)
    assert out.shape == x.shape
    # RMS of output (before learned weight, which starts at 1) should be ~1
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_rope_shapes():
    head_dim, max_seq_len = 8, 32
    cos, sin = precompute_rope(head_dim, max_seq_len, theta=10000.0)
    assert cos.shape == (max_seq_len, head_dim // 2)
    assert sin.shape == (max_seq_len, head_dim // 2)


def test_apply_rotary_preserves_shape_and_norm():
    B, n_heads, T, head_dim = 2, 4, 10, 8
    x = torch.randn(B, n_heads, T, head_dim)
    cos, sin = precompute_rope(head_dim, T, theta=10000.0)
    out = apply_rotary(x, cos, sin)
    assert out.shape == x.shape
    # rotation preserves per-position vector norm
    x1, x2 = x[..., 0::2], x[..., 1::2]
    in_norm = (x1.pow(2) + x2.pow(2)).sum(dim=-1).sqrt()
    o1, o2 = out[..., 0::2], out[..., 1::2]
    out_norm = (o1.pow(2) + o2.pow(2)).sum(dim=-1).sqrt()
    assert torch.allclose(in_norm, out_norm, atol=1e-4)


def test_swiglu_mlp_shape():
    mlp = SwiGLUMLP(d_model=16, hidden=40)
    x = torch.randn(2, 5, 16)
    assert mlp(x).shape == x.shape


def test_transformer_block_shape():
    config = ModelConfig(d_model=16, n_heads=4, n_kv_heads=2, ffn_hidden=40, context_len=32)
    block = TransformerBlock(config)
    x = torch.randn(2, 10, 16)
    cos, sin = precompute_rope(config.head_dim, config.context_len, config.rope_theta)
    out = block(x, cos[:10], sin[:10])
    assert out.shape == x.shape
