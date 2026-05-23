# Diary 008 — AABB context-2 language: dim 2 solves it at the cold-start floor

Date: 2026-05-22

## Context

Diary 007 conjectured that dim 2 suffices for any *length-1 cyclic*
language and proposed the next step: a context-dependent language to
test where dim 2 finally fails. This entry runs the simplest such test
— vocab 2, periodic `AABB AABB AABB...` — and finds dim 2 still
sufficient, but uncovers a structural floor that future runs will need
to account for.

## The language

Vocab `{A, B}`, periodic pattern `AABB`. Period 4. Context-2 rule (with
state = last two tokens):

| (tok_{t-1}, tok_t) | tok_{t+1} |
|---|---|
| AA | B |
| AB | B |
| BB | A |
| BA | A |

Equivalently: `next = NOT(tok_{t-1})` — the next token is the *negation
of the older* token in the pair. The length-1 view is uninformative —
after either A or B, the next token is 50/50 — so the model **must**
use context length 2.

## Results

`bs` = block_size. `--distance_readout`, `--no_gelu`, `--disable_layernorm`,
single layer, all-parameters-jointly-learned, 3000 iters, lr 0.05.

| n_embd | bs | best val loss | comment |
|--:|--:|--:|---|
| 1 | 1 | 0.6920 | ≈ ln(2) — no context, must guess |
| 1 | 4 | 0.4212 | partial — dim 1 cannot fully exploit attention here |
| 2 | 4 | 0.1761 | matches `ln(2)/4 = 0.173` — see below |
| 2 | 8 | 0.0893 | matches `ln(2)/8 = 0.0866` |

## The cold-start floor

At every block of length `bs`, **position 0 has no prior context**. The
training target at position 0 is the token at position 1, but the model
sees only the token at position 0 — and the length-1 view of AABB is
genuinely 50/50. So position 0 *unavoidably* contributes `ln(2)` to the
per-position loss, regardless of what the model learns.

Positions 1 through `bs-1` each see ≥ 2 tokens of context, which is
enough for the context-2 rule. If the model masters the rule there, the
average per-position loss is

    (ln(2) + 0 + 0 + ... + 0) / bs  =  ln(2) / bs.

That gives the predicted floor in the table:

    bs = 4:  ln(2)/4 = 0.173   (observed 0.176)
    bs = 8:  ln(2)/8 = 0.087   (observed 0.089)

Per-position trace of the dim 2, bs 4 model on the four cyclic slices
(`AABB`, `ABBA`, `BBAA`, `BAAB`):

    slice AABB  (targets ABBA)
      pos 0  input A  P(A)=0.49 P(B)=0.51   ← cold start
      pos 1  input A  P(B)=1.00              ← context (A,A) → B
      pos 2  input B  P(B)=1.00              ← context (A,B) → B
      pos 3  input B  P(A)=1.00              ← context (B,B) → A

The same pattern holds for every slice: position 0 is at the 50/50
floor, every later position is at probability ≈ 1 on the correct token.
**Dim 2 is sufficient.**

## Why dim 1 + attention falls short

The dim-1 model at bs 4 reached 0.4212 — well above the 0.173 floor.
It does see context, so it does some of the work, but it cannot fully
implement the context-2 rule. Two pieces of structure conspire:

1. Dim-1 attention is constrained (diary 001) — the attention weights
   are softmax of `q·k = W_Q W_K x_t x_j`, so the position the model
   can attend to is determined by which `x_j` is the most extreme,
   modulated by a query-dependent temperature. It is hard to attend
   *specifically* to position `t−1` for every `t` when everything has
   to fit on one number line.
2. With dim 1, the whole residual stream is a scalar, and the model
   must encode both "which token is current" and "what attention
   brought in" in a single number. There is no room to keep them apart.

Dim 2 has enough room: separate axes can carry "current token" and
"attention output" without overwriting each other, and 2-D attention
can produce sharper "attend to position t−1" patterns.

So the AABB experiment is also the first concrete case of dim-1
**still failing** when the language demands real context — even though
the *output cycle* in this language (the 2-cycle `A ↔ B`) is something
1-D affine can do in principle. Dim 1's failure here is in the *route*,
not the destination.

## Pattern after eight diary entries

For a length-1 cyclic language of vocab N:
- dim 1 fails for N ≥ 3 (architectural; diary 006).
- dim 2 succeeds for N = 3 and N = 4 (diary 006 / 007).

For a context-2 language (this entry):
- dim 1 plus attention is insufficient even on vocab 2.
- dim 2 plus attention is sufficient on vocab 2 (reaches the structural
  cold-start floor `ln(2)/bs`).

The expected next failure for dim 2 is vocab ≥ 3 with context-2, where
the *output structure* per context has more states than the 2-cycle
that 1-D affine can do. Whether 2-D affine can also represent that
output structure under attention's mediation is the next test.

## Status

Six diary-008-relevant runs in the table. Dim 2 sufficient for vocab 2
context-2. Cold-start floor `ln(2)/bs` is now a known structural feature
of any `bs`-block training on this kind of language and should be
factored in when reading future losses. Next: vocab 3 context-2.
