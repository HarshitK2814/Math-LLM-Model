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
        {"question": "१ + १ = कत?", "cot": "१ + १ = २", "answer": "२"},
        {"question": "२ + २ = कत?", "cot": "२ + २ = ४", "answer": "४"},
        {"question": "३ + ३ = कत?", "cot": "३ + ३ = ६", "answer": "६"},
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
