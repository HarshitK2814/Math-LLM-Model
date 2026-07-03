"""Train SentencePiece BPE tokenizers on the unified corpus.

Two variants (ablation arm #1: number tokenization):
    baseline  - plain BPE; multi-digit numbers may merge into single tokens
    digits    - every digit (ASCII 0-9 and Bengali 0-9) is forced to be its
                own token; prior work shows this helps small models do
                arithmetic because place values stay aligned across tokens

Outputs:
    tokenizers/bpe{V}_baseline.{model,vocab}
    tokenizers/bpe{V}_digits.{model,vocab}
    docs/tokenizer_report.md   - fertility comparison on the val split
"""

from __future__ import annotations

import json
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TOK_DIR = ROOT / "tokenizers"
DOCS = ROOT / "docs"

VOCAB_SIZE = 16000
BN_DIGITS = list("০১২৩৪৫৬৭৮৯")


def iter_texts(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield f"{row['question']}\n{row['cot']}\nAnswer: {row['answer']}"


def build_corpus(corpus_path: Path) -> None:
    if corpus_path.exists():
        print(f"corpus exists: {corpus_path}")
        return
    with corpus_path.open("w", encoding="utf-8") as out:
        for text in iter_texts(PROCESSED / "train.jsonl"):
            out.write(text.replace("\n", " ") + "\n")
    print(f"wrote {corpus_path} ({corpus_path.stat().st_size / 1e6:.1f} MB)")


def train(corpus_path: Path, name: str, split_digits: bool) -> Path:
    prefix = TOK_DIR / f"bpe{VOCAB_SIZE // 1000}k_{name}"
    if prefix.with_suffix(".model").exists():
        print(f"skip {prefix.name} (exists)")
        return prefix.with_suffix(".model")
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(prefix),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,  # keep rare Bengali conjuncts
        split_digits=split_digits,
        # Bengali digits are not covered by split_digits; force them to be
        # standalone pieces so both scripts behave identically in this arm.
        user_defined_symbols=BN_DIGITS if split_digits else [],
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        num_threads=8,
    )
    print(f"trained {prefix.name}")
    return prefix.with_suffix(".model")


def fertility(model_path: Path, split: str) -> dict:
    """Average tokens per problem and per character on a split."""
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    n_tokens = n_chars = n_rows = 0
    for text in iter_texts(PROCESSED / f"{split}.jsonl"):
        n_tokens += len(sp.encode(text))
        n_chars += len(text)
        n_rows += 1
    return {
        "tokens_per_example": n_tokens / n_rows,
        "chars_per_token": n_chars / n_tokens,
        "total_tokens": n_tokens,
    }


def main() -> None:
    TOK_DIR.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    corpus = PROCESSED / "corpus.txt"
    build_corpus(corpus)

    results = {}
    for name, split_digits in [("baseline", False), ("digits", True)]:
        model = train(corpus, name, split_digits)
        results[name] = fertility(model, "val")
        print(name, results[name])

    b, d = results["baseline"], results["digits"]
    report = f"""# Tokenizer report

SentencePiece BPE, vocab {VOCAB_SIZE}, trained on `data/processed/corpus.txt`
(train split only — val/test never seen). Fertility measured on the val split.

| Variant | Tokens/example (val) | Chars/token | Val tokens total |
|---|---:|---:|---:|
| baseline (numbers merge) | {b["tokens_per_example"]:.1f} | {b["chars_per_token"]:.2f} | {b["total_tokens"]:,} |
| digit-split (1 digit = 1 token) | {d["tokens_per_example"]:.1f} | {d["chars_per_token"]:.2f} | {d["total_tokens"]:,} |

Digit-splitting costs {d["tokens_per_example"] / b["tokens_per_example"] - 1:+.1%} sequence length.
The ablation question is whether that cost buys better arithmetic accuracy at
20-30M parameters.

Special ids: pad=0, unk=1, bos=2, eos=3. Bengali digits are forced standalone
in the digit-split variant via user_defined_symbols (SentencePiece's
split_digits only covers ASCII).
"""
    (DOCS / "tokenizer_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
