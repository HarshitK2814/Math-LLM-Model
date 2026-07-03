# Model design: best practices from the literature + our novel contributions

This document distills the key findings from prior work on small math models,
states which design decisions we adopt from each, and defines what is novel in
our study. It doubles as the seed of the paper's related-work section.

## 1. Key insights from prior work

### Data (what to train on)

| Paper | Key finding | What we adopt |
|---|---|---|
| **TinyGSM** (arXiv 2312.09241) | Data quality+scale is THE lever: 12.3M synthetic problems let a 125M model beat 34B models on GSM8K. Python-style solutions outperform natural language. | Scale synthetic data via translation (gsm8k-aug → Bangla); consider PoT-style output format as an ablation arm. |
| **OpenMathInstruct-2** (2410.01560) | Solution format matters: overly verbose CoT *hurts* SFT; question diversity is the main scaling driver; SFT is robust to some noisy solutions (imprecise filtering is OK — relevant for MT noise). | Keep translated CoT concise; prioritize question diversity when choosing what to translate; don't over-filter MT output. |
| **Teaching Arithmetic to Small Transformers** (2307.03381) | Small transformers learn arithmetic dramatically faster with the right data *format*; CoT with intermediate steps improves accuracy, sample efficiency, and convergence simultaneously. | All training data keeps step-by-step solutions; formatting is a first-class experimental variable. |
| **Ganit** (2601.06767) | Difficulty stratification enables curriculum training for Bangla math. | Use Ganit's difficulty labels for an optional curriculum arm (easy→hard). |

### Tokenization of numbers

| Paper | Key finding | What we adopt |
|---|---|---|
| **Numeral-system scaling** (2409.17391) | Single-digit (base-10) tokenization is consistently more data-efficient than multi-digit for *from-scratch* training — exactly our regime. | Digit-split tokenizer is our expected winner; the ablation quantifies it for Bangla (two digit scripts!). |
| **Tokenization counts** (2402.14903) | Number token *direction* matters (R2L chunking beats L2R at small scale). | Noted as a formatting ablation if budget allows. |
| **NumeroLogic** (2404.00459) | Prefixing digit-count ("{2:42}") acts as micro-CoT for numbers and improves arithmetic. | Candidate formatting arm. |

### Positional encoding & length generalization

| Paper | Key finding | What we adopt |
|---|---|---|
| **NoPE** (2305.19466) | *No* positional encoding beats RoPE/ALiBi/APE for length generalization on math tasks in decoder-only models. | Upgrade ablation from {RoPE vs learned} to **{RoPE, learned, NoPE}** — 3-way. |
| **Length gen is fragile** (2402.09371) | Success depends on format × PE combo, with large variance across seeds. | Run every ablation cell with ≥3 seeds and report variance — reviewers know single-seed small-model results are noise. |
| **Abacus embeddings** (2405.17399) | Digit-position embeddings (position *within* the number) give 99% on 100-digit addition; enables input injection & recurrence gains. | Inspiration for our cross-script digit embedding idea (§3). |
| **Positional Description Matters** (2311.14737) | A 100M model + positional fixes does 15-digit multiplication; naive PE reliance is the bottleneck. | Supports treating PE as a primary ablation axis. |

### Architecture at small scale

| Paper | Key finding | What we adopt |
|---|---|---|
| **MobileLLM** (2402.14905) | At sub-billion scale, architecture matters more than believed: deep-and-thin > wide-and-shallow; embedding tying and grouped-query attention are free wins. | Deep-thin default (10-12 layers @ d=512, not 6 @ d=768); tied input/output embeddings; GQA. |
| **Tapered LMs** (2606.23670, Jun 2026) | Tapering MLP width across depth (wide early, narrow late) beats uniform width at equal params, for free. | Candidate "modern upgrades" arm — untested at 30M scale and on math. |
| **Number Token Loss** (2411.02083) | Adding a regression-like loss on number tokens (proximity-aware, not just CE) significantly improves numerical accuracy; drop-in for any LM. | Our auxiliary-loss arm (§3). |

### Evaluation

| Paper | What we adopt |
|---|---|
| **MGSM** (2210.03057) | MGSM-bn (250 problems) as the external, citable benchmark. |
| **GSM-Plus-BN** (Mendeley) | Perturbed variants → robustness analysis (memorization vs reasoning). |
| Ganit dev (776, difficulty-stratified) | Primary in-domain test set, accuracy broken down by difficulty. |

## 2. Resulting baseline design (~30M "modern small transformer")

Decoder-only, PyTorch from scratch. Every ablatable choice defaults to the
literature-supported best guess:

| Component | Default | Ablation alternatives |
|---|---|---|
| Layers × width | 10 × 512 (deep-thin) | 6 × 704 (wide-shallow, param-matched) |
| Attention | 8 heads, GQA (2 KV heads) | MHA |
| Positional | RoPE | NoPE, learned absolute |
| FFN | SwiGLU, hidden 1280 | GELU (param-matched) |
| Norm | RMSNorm, pre-norm | — |
| Embeddings | tied in/out, 16K BPE | — |
| Tokenizer | digit-split | baseline BPE (numbers merge) |
| Loss | CE | CE + Number Token Loss |
| Context | 1024 | — |

Parameter count ≈ 34M (8.2M tied embeddings + ~2.6M/layer × 10).
A ~20M variant (8 × 448) exists for T4-budget sweeps.

## 3. Novel contributions (beyond combining best practices)

1. **First systematic architecture/tokenization ablation for low-resource
   math reasoning.** All the ablation literature above is English/synthetic-
   arithmetic. Nobody has measured which of these choices matter when the
   language is low-resource and the training data is partly machine-translated.
   The interaction (e.g., does digit-splitting matter *more* when data is
   scarce?) is the headline result.
2. **Cross-script numeral alignment (new technique).** Bangla math text mixes
   Bengali digits (০-৯) and ASCII digits (0-9) — same semantics, disjoint
   tokens. We propose tying (or initializing shared) embeddings for digit
   pairs across scripts, so arithmetic competence transfers between scripts.
   Nothing in the literature studies multi-script numerals in one corpus; the
   digit-split tokenizer makes the intervention trivial to implement and
   cleanly measurable (train on mostly-ASCII synthetic data, test on
   Bengali-digit problems, and vice versa).
3. **Number Token Loss in a multilingual/low-resource setting.** NTL (2411.02083)
   was shown on English T5. Testing it from-scratch, at 30M params, with
   two digit scripts, is new — and it composes naturally with contribution 2.
4. **A 150M-token open Bangla math corpus built with open MT on free GPUs** —
   a reusable artifact (dataset release) that low-resource NLP reviewers value
   independently of the modeling results.

Priority order if compute runs short: tokenizer arm > PE arm > cross-script
alignment > NTL > activation arm > depth/width arm > curriculum.

## 4. Experimental hygiene (reviewer-proofing)

- ≥3 seeds per cell (length-gen results are seed-fragile, 2402.09371).
- Param-matched alternatives (never compare 34M vs 38M).
- Train tokenizers on the train split only; test sets held out end-to-end.
- Report tokens-seen, wall-clock, and GPU type per run (all free-tier) —
  reproducibility on free compute is part of the paper's pitch.
