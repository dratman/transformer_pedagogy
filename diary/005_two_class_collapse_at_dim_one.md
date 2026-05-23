# Diary 005 — Two-class collapse in the first dim-1 runs

Date: 2026-05-22

## Context

Diary 004 picked the rung-1 language (ABC periodic, vocab 3, deterministic
length-1 rule) and predicted that the standard head would hit the
2-token bottleneck from diary 002 while the distance head would not.
This entry records the first training runs and what actually happened.

## Setup

`train.py` was run twice with all settings identical except
`--distance_readout`:

    python3.11 py/train.py \
      --input txt_local/abc_periodic.txt \
      --mode continuous --tokenizer char \
      --n_layer 1 --n_head 1 --n_embd 1 --block_size 8 \
      --batch_size 32 --max_iters 3000 \
      --learning_rate 0.05 --warmup_iters 50 \
      --no_gelu --disable_layernorm \
      [--distance_readout]

`bias=True` is left on inside the model (attention and MLP biases learn),
so the model is dim-1 with all other parameters free.

## Results

Both runs converged to a **2-class collapse** — the three tokens did not
spread across three distinct positions on the embedding number line. The
standard head's collapse was the predicted bottleneck. The distance
head's collapse was different in *which* pair collapsed, and is the more
interesting finding.

### Standard head — best val loss 0.4476

Learned embeddings: `A = +3.50, B = -0.04, C = -0.04` — A separated, B and
C collapsed. The no-bias `lm_head` computes `logit_i = ŷ · e_i`; since B
and C are nearly identical, the model cannot distinguish them. It can
only output A (when ŷ > 0) or "one of {B, C}" (when ŷ < 0).

Per-position trace on `ABCABCAB`:

| pos | input | P(A) | P(B) | P(C) | pred | want |
|---:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | A | 0.00 | 0.93 | 0.07 | B | B ✓ |
| 1 | B | 0.00 | 0.57 | 0.43 | B | C ✗ |
| 2 | C | 1.00 | 0.00 | 0.00 | A | A ✓ |
| 3 | A | 0.00 | 0.91 | 0.09 | B | B ✓ |
| 4 | B | 0.00 | 0.52 | 0.48 | B | C ✗ |
| 5 | C | 0.98 | 0.01 | 0.01 | A | A ✓ |
| 6 | A | 0.00 | 0.90 | 0.10 | B | B ✓ |
| 7 | B | 0.00 | 0.50 | 0.50 | B | C ✗ |

5 of 8 positions correct. All three failures are "after B," where the
model must choose between B and C and they are indistinguishable.

### Distance head — best val loss 0.4552

Learned embeddings: `A = -0.09, B = -0.08, C = +2.38` — C separated, A and
B collapsed. **Not the predicted outcome.** The distance readout is
mathematically *capable* of distinguishing all three tokens (diary 002);
it failed to here because the optimization fell into a local minimum
where A and B share an embedding.

Per-position trace on `ABCABCAB`:

| pos | input | P(A) | P(B) | P(C) | pred | want |
|---:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | A | 0.17 | 0.17 | 0.66 | C | B ✗ |
| 1 | B | 0.00 | 0.00 | 1.00 | C | C ✓ |
| 2 | C | 0.69 | 0.31 | 0.00 | A | A ✓ |
| 3 | A | 0.49 | 0.50 | 0.01 | B | B ✓ |
| 4 | B | 0.00 | 0.00 | 1.00 | C | C ✓ |
| 5 | C | 0.70 | 0.30 | 0.00 | A | A ✓ |
| 6 | A | 0.48 | 0.48 | 0.04 | B | B ✓ |
| 7 | B | 0.00 | 0.00 | 1.00 | C | C ✓ |

7 of 8 positions correct — meaningfully better than the standard head's
5 of 8. But the embeddings have *not* spread out, and at position 0 the
model still fails on "after A." The accuracy at later positions comes
from *position-dependent* behaviour (the position embedding and the
attention mix of past values produce a different effective ŷ at different
positions), not from a clean three-way embedding split.

## What this says

**1. The standard-head bottleneck shows up cleanly** — exactly as predicted
in diary 002. With three tokens on one number line, the no-bias tied head
cannot distinguish three of them. The fact that the collapsed pair is
{B, C} rather than {A, B} is a random outcome of optimization; the
*structure* of the collapse is the architectural prediction.

**2. The distance head also collapses, for a different reason.** The
distance head is mathematically *capable* of distinguishing all three
tokens. It failed to here because the optimization fell into a 2-class
local minimum. The collapse is not architectural; it is a property of
training with these particular settings. Cross-entropy does penalise
collapse (diary 002), but evidently not strongly enough at this scale
and learning rate to drive a third embedding away from the cluster.

The distance head still does meaningfully better in raw accuracy (7/8 vs
5/8), but the win comes from *exploiting position-dependent contributions
from attention* to break ties — not from doing what we want it to do.
What we want is three clean embedding positions.

## Next moves to try

- **Initialisation.** Push the three embeddings apart at init (instead of
  the standard small Gaussian) so optimisation starts outside the collapse
  basin.
- **Longer training with constant (non-decaying) LR.** The cosine schedule
  reaches very small LR by iter 2500 and stops driving the embeddings.
- **Smaller `block_size` (1 or 2).** Reduces position-embedding contributions
  and the attention-mix diversity that is currently letting the distance
  head "cheat" via position dependence.
- **Freeze attention and position embeddings.** Force the readout and token
  embedding to do all the work; check that the model can reach near-zero
  loss in that controlled setting, isolating the optimisation issue.

The aim of the next experiment is either to show the distance head *can*
reach near-zero loss on this language (vindicating diary 002), or to
identify exactly what is preventing it.

## Status

Two training runs complete. Apparatus works. The standard head's predicted
bottleneck is visible. The distance head shows a different 2-class collapse
driven by optimisation, not architecture. Next: break the collapse.
