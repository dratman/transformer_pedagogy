# Diary 009 — Vocab 3 context-2: dim 2 starts to fail, dim 3 solves it (in argmax)

Date: 2026-05-22

## Context

Diary 008 had dim 2 cleanly solve the vocab-2 context-2 language (AABB)
up to a known cold-start floor. The conjecture left open was: does dim 2
still suffice when vocab grows to 3? This entry tests it on a de Bruijn
sequence over `{A, B, C}` that visits every length-2 pair exactly once,
which gives a genuinely length-2-dependent rule with no symmetry to lean
on.

## The language

Vocab `{A, B, C}`, periodic pattern **`AABCCACBB`** (period 9) — a de
Bruijn sequence of order 2. The cyclic stream contains each of the nine
length-2 pairs exactly once, so the context-2 rule is

    AA → B    AB → C    AC → B
    BA → A    BB → A    BC → C
    CA → C    CB → B    CC → A

Length-1 statistics: after each of A, B, C, the next token is uniform
over {A, B, C}; context-1 view has loss `ln(3) ≈ 1.098`. So the
cold-start floor for a block of size `bs` is `ln(3)/bs` (e.g. `0.275`
at `bs=4`, `0.137` at `bs=8`).

Corpus: 6,666 copies of the period, ~60,000 chars total.

## Results

Same training settings as the earlier diaries: `--no_gelu`,
`--disable_layernorm`, `--distance_readout`, single layer, single head,
distance-readout, cross-entropy, cosine LR 0.05 with warmup 50–100,
batch_size 32.

| n_embd | bs | iters | best val | floor | argmax acc on 9 pairs |
|--:|--:|--:|--:|--:|:-:|
| 1 | 1 | 3000 | 1.0969 | `ln(3) = 1.10` | — (no context) |
| 1 | 4 | 3000 | 0.9158 | `0.275` | — |
| 2 | 4 | 3000 | 0.7047 | `0.275` | 7 / 9 |
| 2 | 4 | 10000 | 0.4520 | `0.275` | 8 / 9 |
| 2 | 8 | 3000 | 0.4058 | `0.137` | — |
| 2 | 8 | 10000 | 0.7024 | `0.137` | — (worse: different basin) |
| 3 | 4 | 3000 | 0.3714 | `0.275` | — |
| 3 | 4 | 10000 | 0.3775 | `0.275` | **9 / 9** |

## What the dim 3 model actually does

Per-pair argmax is correct on all nine. But confidence varies a lot —
several pairs hover around `P(correct) ≈ 0.65–0.90` rather than `≈ 1.00`:

    AA → P(B) = 1.00          (sharp)
    AB → P(C) = 0.65          (split with A)
    AC → P(B) = 1.00          (sharp)
    BA → P(A) = 0.87          (slight split with C)
    BB → P(A) = 0.90          (slight split with B and C)
    BC → P(C) = 1.00          (sharp)
    CA → P(C) = 1.00          (sharp)
    CB → P(B) = 1.00          (sharp)
    CC → P(A) = 0.96          (sharp-ish)

If those nine probabilities are turned into the cross-entropy loss on
the context-having positions (1, 2, 3 of every bs=4 block), and
combined with the unavoidable `ln(3)` at position 0, the average
predicted loss is about `0.33` — close to the observed `0.378`.

So **dim 3 has learned the rule** (argmax everywhere correct) but its
predictions are *not saturated* — distances between the correct target
embedding and the others are not pushed apart enough for `P → 1` on a
few of the harder pairs. The remaining gap to the floor (`0.275`) is
this confidence slack, not wrong answers.

## What the dim 2 model actually does

8 / 9 argmax correct after 10K iters. The one failure: `AC → C` (wants
B), with `P(B) = 0.47`, `P(C) = 0.47` — a tie the model resolves the
wrong way. So dim 2 is **partially representing** the rule — most
pairs are correct, but at least one stays unresolvable. Loss `0.452`
sits well above the `0.275` floor accordingly.

## The conjecture, revised

Two findings:

1. **The dim floor for argmax-correct context-2 over vocab 3 (this
   language) is 3, not 2.** Dim 2 with one head can place most pairs
   right but cannot fully resolve all nine; one pair ends in a tie.
2. **Even at dim 3, training does not fully saturate confidence in
   3,000–10,000 iters under the cosine schedule.** Argmax is right but
   distances are not pushed apart enough; the loss sits above the
   structural floor.

The first is an architectural statement about this specific language at
single-layer one-head. The second is an optimisation observation, not
an architectural one — longer training with a flatter LR schedule
would likely push the dim 3 confidence to 1 and the loss to the floor.

## Surprises

- The dim 2 bs=8 run at 10K iters did **worse** than the same setup at
  3K iters (`0.70` vs `0.41`). Different random init, different basin.
  The de Bruijn task has a tangled loss landscape; dim-2 results are
  sensitive to init in a way the earlier (vocab 2 / cyclic) tasks were
  not. Worth noting before drawing strong dim-floor conclusions from a
  single run at the edge.

## Open threads

- Does dim 3 reach the floor with longer or constant-LR training?
- Does dim 2 ever cleanly solve this with extra heads or layers (more
  expressive power without changing the embedding dimension)?
- What is the exact structural / topological reason dim 2 cannot resolve
  all 9 pairs into 3 output classes here? An analytical answer would
  generalise to other context-2 rules.

## Status

Nine diary entries. The dim floor question now has texture beyond "1
vs 2": for context-2 over vocab 3, the floor moves to 3 — and even
there, training saturation matters. Next obvious experiments:
constant-LR longer runs to settle the optimisation question; or, more
heads / more layers to see if dim 2 can do this with more capacity.
