# Baseline Transformer Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the ~34M-parameter decoder-only transformer specified in `docs/design.md` §2, plus the data-packing and training scripts needed to train the baseline on Kaggle T4.

**Architecture:** PyTorch from scratch. 10-layer, d=512 deep-thin decoder with GQA (8 query heads / 2 KV heads), RoPE, SwiGLU FFN (hidden 1280), RMSNorm pre-norm, tied 16K embeddings, context length 1024 — exactly the "Default" column of `docs/design.md` §2's table. Config is a dataclass so later ablation arms (NoPE/learned PE, MHA, GELU-FFN, wide-shallow) can be added as alternate config values without rewriting the model, but this plan only implements the defaults — ablation variants are future work.

**Tech Stack:** PyTorch >=2.2 (uses `F.scaled_dot_product_attention` for causal attention), SentencePiece (existing trained tokenizers in `tokenizers/`), NumPy (memmap token packing), pytest.

## Global Constraints

- Parameter count must land at ≈34M (8.2M tied embeddings + ~2.6M/layer × 10), per `docs/design.md` §2 — verified by a test, not eyeballed.
- Tokenizer: use the already-trained `tokenizers/bpe16k_digits.model` as the default (digit-split is the literature-predicted winner per `docs/design.md` §1); `tokenizers/bpe16k_baseline.model` must also work via a flag, since the tokenizer ablation needs both.
- Special token ids are fixed by `src/tokenizer/train_tokenizer.py`: pad=0, unk=1, bos=2, eos=3. Do not renumber.
- Context length 1024 (`docs/design.md` §2). Token ids fit in `uint16` (vocab 16000 < 65536) — use `uint16` for packed data files to keep them small.
- No dropout, no bias terms in Linear layers (modern small-LM convention MobileLLM/Llama-style, consistent with tied embeddings + RMSNorm already chosen in the design doc).
- Every new Python module goes under `src/model/`; every test goes under `tests/`, mirroring the module name (`src/model/transformer.py` → `tests/test_transformer.py`).

---

## File Structure

- `src/model/config.py` — `ModelConfig` dataclass (architecture hyperparameters).
- `src/model/layers.py` — `RMSNorm`, RoPE helpers (`precompute_rope`, `apply_rotary`), `CausalSelfAttention` (GQA+RoPE), `SwiGLUMLP`, `TransformerBlock`.
- `src/model/transformer.py` — `Transformer` (full model: embed → blocks → norm → tied head), `num_params()`.
- `src/model/dataset.py` — `pack_split()` (jsonl → tokenized `.bin` via SentencePiece) and `PackedDataset` (memmap-backed `torch.utils.data.Dataset`), plus a CLI (`python -m src.model.dataset ...`).
- `src/model/train.py` — training loop: AdamW, cosine LR w/ warmup, grad clipping, checkpoint save/resume, CLI (`python -m src.model.train ...`).
- `tests/test_layers.py`, `tests/test_transformer.py`, `tests/test_dataset.py` — pytest unit tests.
- `requirements.txt` — add `pytest`.

---

### Task 1: Model config + core layers (RMSNorm, RoPE, GQA attention, SwiGLU)

**Files:**
- Create: `src/model/__init__.py` (empty)
- Create: `src/model/config.py`
- Create: `src/model/layers.py`
- Test: `tests/test_layers.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `ModelConfig` dataclass with fields `vocab_size:int=16000, d_model:int=512, n_layers:int=10, n_heads:int=8, n_kv_heads:int=2, ffn_hidden:int=1280, context_len:int=1024, rope_theta:float=10000.0`.
- Produces: `RMSNorm(dim:int, eps:float=1e-6)` — `nn.Module`, `forward(x) -> Tensor` same shape as `x`.
- Produces: `precompute_rope(head_dim:int, max_seq_len:int, theta:float) -> tuple[Tensor, Tensor]` returning `(cos, sin)` each shaped `(max_seq_len, head_dim//2)`.
- Produces: `apply_rotary(x:Tensor, cos:Tensor, sin:Tensor) -> Tensor` where `x` is `(B, n_heads, T, head_dim)`.
- Produces: `CausalSelfAttention(config:ModelConfig)` — `nn.Module`, `forward(x, cos, sin) -> Tensor` same shape as `x`.
- Produces: `SwiGLUMLP(d_model:int, hidden:int)` — `nn.Module`, `forward(x) -> Tensor` same shape as `x`.
- Produces: `TransformerBlock(config:ModelConfig)` — `nn.Module`, `forward(x, cos, sin) -> Tensor` same shape as `x`.

- [ ] **Step 1: Add pytest to requirements**

Add a line `pytest>=8.0` to `requirements.txt` (after `tqdm`).

- [ ] **Step 2: Write `src/model/__init__.py`**

Empty file (makes `src/model` a package).

- [ ] **Step 3: Write `src/model/config.py`**

```python
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
```

- [ ] **Step 4: Write `src/model/layers.py`**

```python
"""Core transformer building blocks: RMSNorm, RoPE, GQA attention, SwiGLU.

Pre-norm, no bias, no dropout (small-model / big-data regime, matching
MobileLLM-style choices already locked in docs/design.md section 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin), each (max_seq_len, head_dim // 2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)  # (max_seq_len, head_dim // 2)
    return torch.cos(angles), torch.sin(angles)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_heads, T, head_dim). cos/sin: (T, head_dim // 2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.repeat = config.n_heads // config.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        k = k.repeat_interleave(self.repeat, dim=1)
        v = v.repeat_interleave(self.repeat, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLUMLP(config.d_model, config.ffn_hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x
```

- [ ] **Step 5: Write `tests/test_layers.py`**

```python
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
```

- [ ] **Step 6: Run tests, verify they pass**

Run: `python -m pytest tests/test_layers.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add src/model/__init__.py src/model/config.py src/model/layers.py tests/test_layers.py requirements.txt
git commit -m "feat: add transformer config and core layers (RMSNorm, RoPE, GQA, SwiGLU)"
```

---

### Task 2: Full Transformer model + parameter-count and causality tests

**Files:**
- Create: `src/model/transformer.py`
- Test: `tests/test_transformer.py`

**Interfaces:**
- Consumes: `ModelConfig` from `src/model/config.py`; `RMSNorm`, `TransformerBlock`, `precompute_rope` from `src/model/layers.py`.
- Produces: `Transformer(config: ModelConfig)` — `nn.Module`, `forward(idx: LongTensor[B,T], targets: LongTensor[B,T]|None = None) -> tuple[Tensor, Tensor|None]` returning `(logits[B,T,vocab_size], loss_or_None)`. `loss` uses `ignore_index=-100`. Also exposes `num_params(self) -> int`.

- [ ] **Step 1: Write `src/model/transformer.py`**

```python
"""Full decoder-only transformer: tied embeddings, GQA+RoPE, SwiGLU, RMSNorm.

Matches the "Default" baseline in docs/design.md section 2 (~34M params:
8.2M tied embeddings + ~2.6M/layer x 10).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.layers import RMSNorm, TransformerBlock, precompute_rope


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied embeddings

        cos, sin = precompute_rope(config.head_dim, config.context_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.config.context_len, f"sequence length {T} exceeds context_len {self.config.context_len}"

        x = self.tok_emb(idx)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 2: Write `tests/test_transformer.py`**

```python
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
```

- [ ] **Step 3: Run tests, verify they pass**

Run: `python -m pytest tests/test_transformer.py -v`
Expected: 5 passed. (`test_param_count_matches_design_doc` is the important one — if it fails, recheck `ModelConfig` defaults against `docs/design.md` §2 before touching anything else.)

- [ ] **Step 4: Commit**

```bash
git add src/model/transformer.py tests/test_transformer.py
git commit -m "feat: add full Transformer model matching docs/design.md baseline spec"
```

---

### Task 3: Tokenized data packing (jsonl -> memmap) + PackedDataset

**Files:**
- Create: `src/model/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `data/processed/{train,val,test}.jsonl` (fields `question`, `cot`, `answer`, per `src/tokenizer/train_tokenizer.py`); `tokenizers/bpe16k_{baseline,digits}.model` (SentencePiece, special ids pad=0/unk=1/bos=2/eos=3).
- Produces: `pack_split(jsonl_path: Path, sp_model_path: Path, out_path: Path) -> int` — writes a `uint16` binary token-id file, returns token count.
- Produces: `PackedDataset(bin_path: Path, block_size: int)` — `torch.utils.data.Dataset`, `__len__() -> int`, `__getitem__(i) -> tuple[LongTensor[block_size], LongTensor[block_size]]` (x, y = next-token targets).
- CLI: `python -m src.model.dataset --tokenizer {baseline,digits} --splits train val test` packs all requested splits to `data/packed/<tokenizer>_<split>.bin`.

- [ ] **Step 1: Write `src/model/dataset.py`**

```python
"""Pack tokenized train/val/test splits into flat uint16 binaries for
fast memmap-backed loading during training.

Each example is encoded as: bos_id, <question>\n<cot>\nAnswer: <answer>
tokens, eos_id. Examples are concatenated end-to-end (no padding); the
training script cuts fixed-size, non-overlapping windows out of the
concatenated stream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PACKED = ROOT / "data" / "packed"
TOKENIZERS = ROOT / "tokenizers"

TOKENIZER_PATHS = {
    "baseline": TOKENIZERS / "bpe16k_baseline.model",
    "digits": TOKENIZERS / "bpe16k_digits.model",
}


def pack_split(jsonl_path: Path, sp_model_path: Path, out_path: Path) -> int:
    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    bos_id, eos_id = sp.bos_id(), sp.eos_id()

    ids: list[int] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = f"{row['question']}\n{row['cot']}\nAnswer: {row['answer']}"
            ids.append(bos_id)
            ids.extend(sp.encode(text))
            ids.append(eos_id)

    arr = np.array(ids, dtype=np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(out_path)
    return len(arr)


class PackedDataset(Dataset):
    def __init__(self, bin_path: Path, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.n_chunks = (len(self.data) - 1) // block_size
        if self.n_chunks <= 0:
            raise ValueError(f"{bin_path} has too few tokens for block_size={block_size}")

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        chunk = self.data[start : start + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", choices=["baseline", "digits"], default="digits")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    sp_model_path = TOKENIZER_PATHS[args.tokenizer]
    for split in args.splits:
        jsonl_path = PROCESSED / f"{split}.jsonl"
        out_path = PACKED / f"{args.tokenizer}_{split}.bin"
        n_tokens = pack_split(jsonl_path, sp_model_path, out_path)
        print(f"{out_path}: {n_tokens:,} tokens")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `tests/test_dataset.py`**

Uses the tokenizer already committed at `tokenizers/bpe16k_digits.model` (no need to train a throwaway tokenizer in the test).

```python
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.model.dataset import PackedDataset, pack_split

TOKENIZER = Path(__file__).resolve().parents[1] / "tokenizers" / "bpe16k_digits.model"


@pytest.fixture
def tiny_jsonl(tmp_path: Path) -> Path:
    rows = [
        {"question": "১ + ১ = কত?", "cot": "১ + ১ = ২", "answer": "২"},
        {"question": "২ + ২ = কত?", "cot": "২ + ২ = ৪", "answer": "৪"},
        {"question": "৩ + ৩ = কত?", "cot": "৩ + ৩ = ৬", "answer": "৬"},
    ]
    path = tmp_path / "tiny.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def test_pack_split_writes_uint16_file(tmp_path: Path, tiny_jsonl: Path):
    if not TOKENIZER.exists():
        pytest.skip("tokenizer not trained yet")
    out_path = tmp_path / "packed.bin"
    n_tokens = pack_split(tiny_jsonl, TOKENIZER, out_path)

    assert out_path.exists()
    assert n_tokens > 0
    arr = np.fromfile(out_path, dtype=np.uint16)
    assert len(arr) == n_tokens
    # 3 examples each wrapped in bos/eos -> at least 6 special tokens present
    assert (arr == 2).sum() == 3  # bos_id
    assert (arr == 3).sum() == 3  # eos_id


def test_packed_dataset_shapes_and_shift(tmp_path: Path, tiny_jsonl: Path):
    if not TOKENIZER.exists():
        pytest.skip("tokenizer not trained yet")
    out_path = tmp_path / "packed.bin"
    n_tokens = pack_split(tiny_jsonl, TOKENIZER, out_path)

    block_size = 8
    if n_tokens <= block_size:
        pytest.skip("fixture too small for this block_size")

    ds = PackedDataset(out_path, block_size=block_size)
    assert len(ds) == (n_tokens - 1) // block_size

    x, y = ds[0]
    assert x.shape == (block_size,)
    assert y.shape == (block_size,)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    # y is x shifted by one token
    raw = np.fromfile(out_path, dtype=np.uint16)
    assert torch.equal(y[:-1], x[1:])
    assert y[-1].item() == int(raw[block_size])
```

- [ ] **Step 3: Run tests, verify they pass**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: 2 passed (or 2 skipped if `tokenizers/bpe16k_digits.model` is missing — it should already exist from the prior tokenizer-training commit, so passed is expected).

- [ ] **Step 4: Pack the real train/val/test splits for both tokenizers**

Run: `python -m src.model.dataset --tokenizer digits --splits train val test`
Run: `python -m src.model.dataset --tokenizer baseline --splits train val test`
Expected: prints token counts per split; creates `data/packed/{digits,baseline}_{train,val,test}.bin`.

- [ ] **Step 5: Commit**

`.gitignore` already has a blanket `*.bin` rule, so `data/packed/*.bin` is covered — no gitignore changes needed.

```bash
git add src/model/dataset.py tests/test_dataset.py
git commit -m "feat: add jsonl-to-token-binary packing and PackedDataset"
```

---

### Task 4: Training script (AdamW, cosine schedule, checkpoint/resume)

**Files:**
- Create: `src/model/train.py`

**Interfaces:**
- Consumes: `ModelConfig`, `Transformer` from Task 2; `PackedDataset` from Task 3; packed binaries at `data/packed/<tokenizer>_{train,val}.bin`.
- Produces: CLI `python -m src.model.train --tokenizer digits --steps 2000 --batch-size 32 --out-dir checkpoints/baseline_digits` that trains and periodically writes `checkpoints/<out-dir>/step_<N>.pt` containing `{"model": state_dict, "optimizer": state_dict, "step": int, "config": ModelConfig-as-dict}`, and supports `--resume path/to/checkpoint.pt`.

- [ ] **Step 1: Write `src/model/train.py`**

```python
"""Train the baseline Transformer on packed token binaries.

Checkpoint/resume pattern mirrors src/data/translate_indictrans2.py so the
same discipline (safe to interrupt on Kaggle's 12h session limit) applies
to training too.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model.config import ModelConfig
from src.model.dataset import PACKED, PackedDataset
from src.model.transformer import Transformer

ROOT = Path(__file__).resolve().parents[2]


def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * coeff)


def save_checkpoint(path: Path, model: Transformer, optimizer: torch.optim.Optimizer, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": dataclasses.asdict(model.config),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[Transformer, torch.optim.Optimizer, int]:
    ckpt = torch.load(path, map_location=device)
    config = ModelConfig(**ckpt["config"])
    model = Transformer(config).to(device)
    model.load_state_dict(ckpt["model"])
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(ckpt["optimizer"])
    return model, optimizer, ckpt["step"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", choices=["baseline", "digits"], default="digits")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--out-dir", type=str, default="checkpoints/baseline")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.out_dir

    train_bin = PACKED / f"{args.tokenizer}_train.bin"
    config = ModelConfig()
    train_ds = PackedDataset(train_bin, block_size=config.context_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    start_step = 0
    if args.resume:
        model, optimizer, start_step = load_checkpoint(Path(args.resume), device)
        print(f"resumed from {args.resume} at step {start_step}")
    else:
        model = Transformer(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))

    print(f"model params: {model.num_params():,} | device: {device} | train chunks: {len(train_ds):,}")

    model.train()
    step = start_step
    data_iter = iter(train_loader)
    t0 = time.time()

    while step < args.steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        lr = cosine_with_warmup(step, args.warmup_steps, args.steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        _, loss = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        step += 1
        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"step {step}/{args.steps} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s")

        if step % args.checkpoint_interval == 0 or step == args.steps:
            ckpt_path = out_dir / f"step_{step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step)
            print(f"saved {ckpt_path}")

    print("done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test training on real packed data for a few steps**

Run: `python -m src.model.train --tokenizer digits --steps 20 --batch-size 4 --checkpoint-interval 20 --out-dir checkpoints/smoke_test`
Expected: prints `model params: ~34,4XX,XXX`, loss printed at step 1 and step 20, loss at step 20 should be visibly lower than at step 1 (sanity: model is learning, not just running). Ends with `saved checkpoints/smoke_test/step_20.pt` and `done`.

- [ ] **Step 3: Smoke-test resume**

Run: `python -m src.model.train --tokenizer digits --steps 30 --batch-size 4 --checkpoint-interval 20 --out-dir checkpoints/smoke_test --resume checkpoints/smoke_test/step_20.pt`
Expected: prints `resumed from checkpoints/smoke_test/step_20.pt at step 20`, continues to step 30, saves `step_30.pt`.

- [ ] **Step 4: Clean up smoke-test checkpoints**

Run: `rm -rf checkpoints/smoke_test`
(`.gitignore` already has `checkpoints/` — nothing further needed.)

- [ ] **Step 5: Commit**

```bash
git add src/model/train.py
git commit -m "feat: add training script with cosine schedule and checkpoint/resume"
```

---

## Explicitly out of scope for this plan

- Ablation arms (NoPE/learned PE, MHA, GELU-FFN, wide-shallow 6x704, curriculum, Number Token Loss, cross-script digit-embedding tying) — `docs/design.md` §3 novel contributions. These reuse `ModelConfig`/`Transformer` but need their own design pass once the baseline trains successfully end-to-end.
- Evaluation harness (MGSM-bn, GSM-Plus-BN, Ganit dev accuracy scoring) — separate plan once a trained checkpoint exists.
- Generation/sampling code (top-k/top-p decoding) — only needed for evaluation, not training.
