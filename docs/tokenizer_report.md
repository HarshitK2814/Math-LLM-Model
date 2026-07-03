# Tokenizer report

SentencePiece BPE, vocab 16000, trained on `data/processed/corpus.txt`
(train split only — val/test never seen). Fertility measured on the val split.

| Variant | Tokens/example (val) | Chars/token | Val tokens total |
|---|---:|---:|---:|
| baseline (numbers merge) | 287.8 | 3.96 | 259,895 |
| digit-split (1 digit = 1 token) | 350.3 | 3.26 | 316,337 |

Digit-splitting costs +21.7% sequence length.
The ablation question is whether that cost buys better arithmetic accuracy at
20-30M parameters.

Special ids: pad=0, unk=1, bos=2, eos=3. Bengali digits are forced standalone
in the digit-split variant via user_defined_symbols (SentencePiece's
split_digits only covers ASCII).
