# Math-LLM-Model

**What architectural choices matter most for math reasoning at <50M parameters?**

A systematic ablation study of small transformer language models for mathematical
reasoning, designed to train end-to-end on free Kaggle GPUs (T4/P100).

## Research question

Small language models (<50M params) are cheap to train and study, but design
choices that are "settled" at large scale (positional encoding, activation
functions, tokenization granularity) may matter very differently at small scale,
especially for arithmetic and multi-step math reasoning. This project ablates:

- **Positional encoding:** RoPE vs. learned absolute embeddings
- **Activation / FFN:** SwiGLU vs. GELU
- **Tokenization:** digit-level vs. BPE for numbers
- **Supervision format:** with vs. without chain-of-thought solutions

## Datasets

Bangla math word-problem datasets (low-resource math reasoning):

- [kawchar85/Bangla-Math](https://huggingface.co/datasets/kawchar85/Bangla-Math)
- [dipta007/Ganit](https://huggingface.co/datasets/dipta007/Ganit)
- [GSM-Plus-BN](https://data.mendeley.com/datasets/74dscnmrhv/2)

## Repository layout

```
src/
  data/        # dataset download, cleaning, formatting
  tokenizer/   # tokenizer training (SentencePiece/BPE + digit-level variants)
  model/       # transformer implementation and config
  train.py     # training loop
  eval.py      # exact-match / accuracy evaluation
notebooks/     # Kaggle notebooks (thin wrappers around src/)
configs/       # experiment configs (one per ablation run)
```

## Status

- [x] Repo setup
- [ ] Data preprocessing & stats
- [ ] Tokenizer training
- [ ] Baseline model (~30M params) on Kaggle T4
- [ ] Ablation grid
- [ ] Evaluation & write-up

## Author

**Harshit Kumar** — CS undergrad, Manipal University. First author; all
experiments, data processing, tokenizer, architecture, and training in this
repository.
