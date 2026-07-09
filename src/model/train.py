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
        if step % 50 == 0 or step == 1 or step == args.steps:
            elapsed = time.time() - t0
            print(f"step {step}/{args.steps} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s")

        if step % args.checkpoint_interval == 0 or step == args.steps:
            ckpt_path = out_dir / f"step_{step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step)
            print(f"saved {ckpt_path}")

    print("done")


if __name__ == "__main__":
    main()
