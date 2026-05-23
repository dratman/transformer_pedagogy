# Diary 007 — Dim 2 handles the 4-cycle; dim 1 fails at exactly ln(2)

Date: 2026-05-22

## Context

Diary 006 established that the minimum embedding dimension for a length-1
3-cycle (ABC periodic) is 2 — by linear-algebra impossibility of an
affine map of ℝ to have a length-3 cycle. Two conjectures were left
open: does dim 2 remain sufficient as the vocabulary grows, and does the
1-D constrained optimum follow a clean pattern?

This entry tests both on the next case: vocab 4, the periodic language
A → B → C → D → A.

A small parameterised corpus generator was added,
`py/make_corpus_periodic.py`, so subsequent experiments can dial the
vocabulary without writing a new script each time.

## Setup and results

Same as diary 006 — `block_size=1`, single layer, `no_gelu`,
`disable_layernorm`, `distance_readout`, all parameters learned jointly,
3000 iters, lr 0.05, cosine schedule with 50-step warmup.

| n_embd | best val loss | reading |
|--:|--:|---|
| 1 | 0.6939 | `ln(2) ≈ 0.6931` |
| 2 | 0.0000 | solved |

## What dim 1 does, in detail

Learned embeddings:

    A = +0.4164    B = -0.4458
    C = +0.4373    D = -0.4234

Two clusters: `{A, C}` near `+0.43`, `{B, D}` near `-0.43`. Predictions:

    after A → P(B) = 0.53, P(D) = 0.47, others ≈ 0
    after B → P(A) = 0.46, P(C) = 0.54, others ≈ 0
    after C → P(B) = 0.53, P(D) = 0.47   (same as after A)
    after D → P(A) = 0.46, P(C) = 0.54   (same as after B)

The model has implemented the **2-cycle approximation** of the 4-cycle.
The affine map `y = α x + γ` with `α = −1` and `γ = e_A + e_B` exactly
swaps the two clusters. Each prediction puts approximately ½ on each
member of the target cluster, costing `−log(½) = ln(2)` per context. The
average loss is therefore `ln(2)`, and `0.6939` matches `0.6931` to
three decimals. **The model hit its constrained optimum exactly.**

This is the cleanest demonstration so far of diary 006's statement that
a 1-D affine map has only 2-cycles, never longer.

## What dim 2 does, in detail

The four embedding points in the plane:

    A = (−0.660, +1.421)
    B = (+1.785, +2.918)
    C = (+1.766, −1.063)
    D = (−0.665, −2.256)

A non-regular quadrilateral. (A regular square is one valid solution,
but the joint optimisation has rotational, translational, and
scaling/distortion freedom and need not pick that one.) Pairwise
distances range from 2.7 to 5.7.

Predictions:

    after A → P(B) = 1.0000
    after B → P(C) = 1.0000
    after C → P(D) = 1.0000
    after D → P(A) = 1.0000

A 2-D affine matrix can be a rotation by 2π/4 = 90°, so four points
placed appropriately are permuted exactly. The model found a
configuration in which the full affine pipeline (token embedding,
attention, MLP, readout) effectively cycles the four points.

## Pattern emerging

A clean statement is taking shape:

> For a length-1 cyclic language of vocabulary N at `block_size = 1`
> with a single-layer affine model:
>
> - **dim 1** can implement only 2-cycles. For `N ≥ 3` the best
>   approximation is a 2-class collapse; for symmetric even N (like
>   N = 4) its average loss is exactly `ln(2)`.
> - **dim 2** suffices, since a 2-D affine matrix can be a rotation
>   by `2π/N` for any N, exactly permuting N points in an N-gon-like
>   configuration.

Conjecture, worth testing next: **dim 2 remains sufficient for
arbitrary N** at block_size 1, length-1 cyclic. Higher dimension is
needed only when the language has structure beyond a single cycle —
branching, or context dependence beyond length 1.

## Open thread

The ABC (vocab 3) dim-1 runs in diaries 005 / 006 reached ~0.9 loss,
not the `ln(2)` that the 4-token case hit cleanly. The 2-class
theoretical optimum for an odd-N cycle is asymmetric (one class must
hold either two source tokens of equal class or one alone, with no
symmetric pairing), and the optimisation did not reach a tight
analytical optimum. Worth a note, not pursued today.

## Status

Six rung-1 runs in the table now across vocab 3 and vocab 4 at
block_sizes 1 and 8. The dimension floor for the cyclic case has been
confirmed at N = 4. Next obvious moves: vocab 5 (or higher) at dim 2
for further confirmation; or shift to a context-dependent language to
see where dim 2 finally fails.
