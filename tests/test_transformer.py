import torch

from src.model.config import ModelConfig
from src.model.transformer import Transformer


def make_baseline_model() -> Transformer:
    return Transformer(ModelConfig())  # defaults = docs/design.md baseline


def test_param_count_matches_design_doc():
    model = make_baseline_model()
    n = model.num_params()
    # docs/design.md section 2: ~34M (8.2M tied embeddings + ~2.6M/layer x 10)
    assert 33_000_000 < n < 36_000_000, f"param count {n:,} outside expected ~34M range"


def test_embeddings_are_tied():
    model = make_baseline_model()
    assert model.lm_head.weight is model.tok_emb.weight


def test_forward_shape_with_and_without_targets():
    config = ModelConfig(d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=64, context_len=64, vocab_size=100)
    model = Transformer(config)
    idx = torch.randint(0, config.vocab_size, (3, 16))

    logits, loss = model(idx)
    assert logits.shape == (3, 16, config.vocab_size)
    assert loss is None

    logits, loss = model(idx, targets=idx)
    assert logits.shape == (3, 16, config.vocab_size)
    assert loss is not None and loss.item() > 0


def test_loss_ignores_index_minus_100():
    config = ModelConfig(d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=64, context_len=64, vocab_size=50)
    model = Transformer(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = idx.clone()
    targets[:, -1] = -100  # last position masked out
    _, loss_masked = model(idx, targets=targets)
    _, loss_full = model(idx, targets=idx)
    assert loss_masked.item() != loss_full.item()


def test_causal_mask_future_tokens_do_not_affect_past_logits():
    config = ModelConfig(d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=64, context_len=64, vocab_size=100)
    model = Transformer(config)
    model.eval()

    torch.manual_seed(0)
    idx_a = torch.randint(0, config.vocab_size, (1, 10))
    idx_b = idx_a.clone()
    idx_b[0, -1] = (idx_b[0, -1] + 1) % config.vocab_size  # change only the last token

    with torch.no_grad():
        logits_a, _ = model(idx_a)
        logits_b, _ = model(idx_b)

    # logits at all positions except the last must be identical: nothing after
    # a position may influence it (causal masking).
    assert torch.allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-5)
    assert not torch.allclose(logits_a[:, -1, :], logits_b[:, -1, :], atol=1e-5)
