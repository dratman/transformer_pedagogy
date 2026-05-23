# Diary 006 — Dimension 1 cannot deterministically cycle through three tokens

Date: 2026-05-22

## Context

Diary 005 left the dim-1 runs in a 2-class collapse: both heads stuck near
loss 0.45, with the distance head doing slightly better only by using
attention's position-dependent contributions to break ties. The next move
in the list was to *reduce* `block_size` to remove that position-dependent
freedom and see whether the distance head could reach the predicted
near-zero loss on its own.

Trying that surfaced a fundamental result.

## What we saw

`block_size = 1, n_embd = 1, vocab = 3` (ABC periodic):

| head | best val loss |
|---|--:|
| standard | 0.8995 |
| distance | 0.9322 |

Both stuck near the uniform-prior loss `ln(3) ≈ 1.098`. Worse than the
block-size-8 runs (which reached 0.45). Reducing the block size made
things *worse*, not better. That looked wrong — until the linear algebra
made it inevitable.

## Why: an affine map of ℝ cannot cycle through three points

At `block_size = 1` the model receives a single token; there is one
position embedding; attention attends only to itself; the MLP is two
linear layers (no GELU); LayerNorm is off. Every component is affine
in the input. The whole network therefore implements a single 1-D
affine map from token embedding to final activation:

    ŷ = α · e_token + γ

The ABC language requires `A → B → C → A`. Translated to the affine map:

    α · e_A + γ = e_B
    α · e_B + γ = e_C
    α · e_C + γ = e_A

Subtracting and combining the three equations gives

    a² + b² + c² − ab − bc − ca = 0

(with `a, b, c = e_A, e_B, e_C`). That expression equals
`½[(a−b)² + (b−c)² + (c−a)²]`, so it vanishes only when `a = b = c`.

**A 1-D affine map has no length-3 cycle.** Whatever optimizer we use,
whatever readout we pick, dim 1 with block_size 1 cannot represent the
ABC language deterministically. The empirical val-loss plateau at ~0.9
is the model finding the best constrained approximation it can.

The same impossibility argument generalises: a 1-D affine map has only
fixed points or shifts. No finite cycle of length ≥ 2 (other than the
trivial 2-cycle that requires α = −1 plus a specific shift). Length-3
cycles or longer simply do not exist in 1-D affine dynamics.

## The fix: dim = 2 is sufficient

Same setup, only `n_embd` raised from 1 to 2:

| head | best val loss |
|---|--:|
| distance | 0.0000 |
| standard | 0.0000 |

Both heads solved it to numerical zero. The dim-2 distance model placed
the three tokens at three points of a triangle:

    A = ( +0.128, −0.888 )
    B = ( +0.899, +0.260 )
    C = ( −2.233, +0.147 )

with `P(B | A) = P(C | B) = P(A | C) = 1.0000`. The dim-2 standard model
arrived at a different but qualitatively similar triangle and also reaches
`P = 1.0000` on every transition.

In two dimensions, an affine map *can* permute three points — three points
are an affine basis of the plane, and one can pick `α` (a 2×2 matrix) and
`γ` so the map cycles them as desired. The model finds such a placement
under cross-entropy with no special initialization.

## What this resolves about diary 005

The 2-class collapse at `dim 1, block_size 8` was the model doing the
best it could under the same fundamental constraint, while attention's
position-dependent contributions gave it a small partial escape (a
different effective affine map at each position). That explained:

- why both heads ended up near 0.45 — they cannot reach 0;
- why the distance head's accuracy was better at later positions —
  attention has more past to mix in;
- why the embeddings collapsed into two classes — three distinct
  embeddings under one affine map are over-constrained.

Diary 005's "next moves to try" included things like better
initialization and longer training. None of them would have helped.
The constraint was algebraic, not optimization-related.

## A clean pedagogical statement

> **The minimum embedding dimension to represent a deterministic
> length-1 N-cycle, with a single-layer linear/affine model and
> block_size 1, is 2 for any N ≥ 3.**

This is exactly the kind of "every knob has a known meaning" finding the
project is for. Dimension is not just a knob on capacity; it has a
discrete lower bound set by the structure of the language. For the ABC
3-cycle, that bound is 2.

## Status

Five training runs total, summarized:

| run | dim | block_size | head | best val |
|---|--:|--:|:--|--:|
| 1 | 1 | 8 | standard | 0.4476 |
| 2 | 1 | 8 | distance | 0.4552 |
| 3 | 1 | 1 | standard | 0.8995 |
| 4 | 1 | 1 | distance | 0.9322 |
| 5 | 2 | 1 | distance | 0.0000 |
| 6 | 2 | 1 | standard | 0.0000 |

The distance-vs-standard distinction (diary 002) is real at dim 1 but
*irrelevant* once dim ≥ 2 — both heads have enough room to spread the
embeddings. The interesting axis for the next experiments is therefore
the dimension floor of the language: how does the minimum-dim move as
the language grows in vocab, in cycle structure, or in context length?
