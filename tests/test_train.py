import torch

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.model.train import cosine_with_warmup, save_checkpoint, load_checkpoint


def test_cosine_with_warmup_start_is_small_positive_fraction():
    base_lr = 3e-4
    warmup_steps = 100
    total_steps = 1000
    lr = cosine_with_warmup(0, warmup_steps, total_steps, base_lr)
    assert lr == base_lr * (0 + 1) / warmup_steps
    assert 0 < lr < base_lr


def test_cosine_with_warmup_end_of_warmup_close_to_base_lr():
    base_lr = 3e-4
    warmup_steps = 100
    total_steps = 1000
    lr = cosine_with_warmup(warmup_steps - 1, warmup_steps, total_steps, base_lr)
    # at step=warmup_steps-1 the linear warmup ramp reaches exactly base_lr;
    # it must never exceed base_lr and should be close to it.
    assert lr <= base_lr
    assert lr > 0.9 * base_lr


def test_cosine_with_warmup_end_of_schedule_close_to_floor():
    base_lr = 3e-4
    warmup_steps = 100
    total_steps = 1000
    min_lr_ratio = 0.1
    lr = cosine_with_warmup(total_steps, warmup_steps, total_steps, base_lr, min_lr_ratio=min_lr_ratio)
    assert abs(lr - base_lr * min_lr_ratio) < 1e-9


def test_cosine_with_warmup_never_exceeds_base_lr():
    base_lr = 3e-4
    warmup_steps = 100
    total_steps = 1000
    for step in range(0, total_steps + 1, 17):
        lr = cosine_with_warmup(step, warmup_steps, total_steps, base_lr)
        assert lr <= base_lr + 1e-12


def test_checkpoint_round_trip(tmp_path):
    config = ModelConfig(d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=64, context_len=64, vocab_size=100)
    model = Transformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))

    idx = torch.randint(0, config.vocab_size, (2, 16))
    _, loss = model(idx, targets=idx)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model, optimizer, step=42)

    loaded_model, loaded_optimizer, loaded_step = load_checkpoint(ckpt_path, torch.device("cpu"))

    assert loaded_step == 42
    assert loaded_model.config.d_model == config.d_model
    assert loaded_model.config.n_layers == config.n_layers
    assert loaded_model.config.vocab_size == config.vocab_size

    orig_state = model.state_dict()
    loaded_state = loaded_model.state_dict()
    for key in orig_state:
        assert torch.equal(orig_state[key], loaded_state[key]), f"mismatch in {key}"

    assert len(loaded_optimizer.state) > 0
