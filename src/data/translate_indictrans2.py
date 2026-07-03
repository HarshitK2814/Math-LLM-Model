"""Translate English synthetic math problems to Bangla with IndicTrans2.

Designed to run on a Kaggle GPU notebook (T4/P100), not locally:

    !pip install -q IndicTransToolkit
    !git clone https://github.com/HarshitK2814/Math-LLM-Model.git
    %cd Math-LLM-Model
    !python src/data/translate_indictrans2.py --limit 50000

Reads GSM8K-style English problems from the Hugging Face parquet export of
whynlp/gsm8k-aug (385K synthetic problems), translates question + answer
rationale to Bangla with ai4bharat/indictrans2-en-indic-1B, and writes
data/processed/synthetic_bn.jsonl in the project's unified schema.

Checkpoints every --save-every rows so a 12h Kaggle session limit never
loses work; rerun with the same output file to resume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "synthetic_bn.jsonl"

GSM8K_AUG_URL = (
    "https://huggingface.co/datasets/whynlp/gsm8k-aug/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
MODEL = "ai4bharat/indictrans2-en-indic-1B"
SRC_LANG, TGT_LANG = "eng_Latn", "ben_Beng"


def load_pool(limit: int) -> pd.DataFrame:
    raw = ROOT / "data" / "raw" / "gsm8k_aug_train.parquet"
    if not raw.exists():
        print("downloading gsm8k-aug parquet ...")
        from urllib.request import urlretrieve

        raw.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(GSM8K_AUG_URL, raw)
    df = pd.read_parquet(raw)
    return df.head(limit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()

    from IndicTransToolkit.processor import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)
    ip = IndicProcessor(inference=True)

    df = load_pool(args.limit)
    done = 0
    if OUT.exists():  # resume support
        done = sum(1 for _ in OUT.open(encoding="utf-8"))
        print(f"resuming after {done} rows")

    def translate(batch: list[str]) -> list[str]:
        batch = ip.preprocess_batch(batch, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
        inputs = tokenizer(
            batch, truncation=True, padding="longest", max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=512, num_beams=1)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        return ip.postprocess_batch(decoded, lang=TGT_LANG)

    rows = df.iloc[done:]
    with OUT.open("a", encoding="utf-8") as f:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows.iloc[start : start + args.batch_size]
            questions = translate(chunk["question"].tolist())
            # gsm8k-aug stores reasoning as a list of step strings
            cots = translate([" ".join(s) for s in chunk["steps"]])
            for i, (q_bn, cot_bn) in enumerate(zip(questions, cots)):
                r = chunk.iloc[i]
                f.write(
                    json.dumps(
                        {
                            "id": f"gsm8k-aug-bn:{done + start + i}",
                            "source": "gsm8k-aug-translated",
                            "question": q_bn,
                            "cot": cot_bn,
                            "answer": str(r["answer"]).strip(),
                            "difficulty": "unknown",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if (start // args.batch_size) % (args.save_every // args.batch_size or 1) == 0:
                f.flush()
                print(f"{done + start + len(chunk)}/{done + len(rows)} translated")

    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
