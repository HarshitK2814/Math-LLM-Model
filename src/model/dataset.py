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
