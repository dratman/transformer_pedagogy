# Diary 002 — Scalar-to-token readout at dimension one

Date: 2026-05-22

## Context

Following diary 001, two model decisions for the dim-1 model are now
settled, and one new design — the readout — has been worked out.

**Settled.** No nonlinearity except in attention: `disable_layernorm=True`
(already wired) and `no_gelu=True` (already in `model.py`). The model is
then linear embedding → attention → (affine) MLP → linear readout, with
the attention softmax the lone nonlinearity in the whole model. With
`no_gelu` the MLP is two linear layers in series — at dimension 1 it
composes to one affine map of a scalar, effectively `a·x + b`, two
scalars of real content.

**Today's design — the readout.** Ralph proposed: the model produces a
scalar prediction ŷ, and that scalar leads to a token via the tied
embedding. At dimension 1 this is natural — the residual stream is one
number wide, so the final activation already *is* a scalar. The design
question is purely how ŷ becomes a token.

## The readout, made precise

Each token sits at a point e_i on the (1-D, tied) embedding number line.
Decode by nearest token — argmin_i (ŷ − e_i)². Expanding the squared
distance,

    −(ŷ − e_i)² = 2 e_i · ŷ − e_i² − ŷ²

and the −ŷ² term is identical across tokens, so it drops out of any
comparison. What remains is a linear logit:

    effective logit_i = (2 e_i) · ŷ + (−e_i²)

So the distance readout *is* a linear output head — one whose per-token
**weight is 2 e_i** and per-token **bias is −e_i²**, both derived from
the embedding. With tied weights the e_i are the input embeddings, so
this readout adds **zero new parameters**.

## What it fixes

`model.py`'s `lm_head` has `bias=False`. At dimension 1 that means
logit_i = ŷ · e_i, and the argmax of that is always at an extreme
embedding — the current model can predict only **two distinct tokens
ever**, the ones at the ends of the embedding line, regardless of
context. It is a pure readout bottleneck, independent of attention or
anything upstream.

The distance readout's −e_i² term is exactly the missing bias, derived
from the embedding itself — and it is what lets a dim-1 model predict
*any* token, parameter-free.

(Honest precision: this is not more powerful than a linear head with a
*free* output bias. It is the special case in which bias_i is tied to
−e_i² and the weight to 2 e_i. Its virtue is not extra expressive
power; it is that everything stays pinned to a single embedding number
line — the parameter-free, fully interpretable form.)

## What "won't work as stated" — and the fix

Ralph anticipated that the idea would not work as initially stated. The
failure is in the *training*, not the readout. If ŷ is trained to
regress the next token's embedding scalar — loss (ŷ − e_true)² — with
the embedding learned, there is a degenerate global optimum: collapse
every embedding to one value, predict that value, loss zero, nothing
learned. That is representation collapse, the same failure that forces
self-distillation methods to use stop-gradients.

The change that fixes it while keeping the rest: don't regress the
scalar — train the *distribution*. With

    P(token i) ∝ exp( −(ŷ − e_i)² )

and cross-entropy loss, the model decodes by nearest-neighbour as
intended, the embedding stays tied and learned, and the loss **cannot be
cheated by collapse**: collapsed embeddings give equal distances, a
uniform distribution, and therefore *maximal* cross-entropy — so the loss
actively drives the embeddings apart.

In code it is a one-line change to the readout:

    logits = 2 · ŷ · e − e²        # in place of   logits = ŷ · e

This is also exactly the "turn the deterministic scalar into a probability
distribution" step that was separated out earlier. It is a readout/loss
operation, not an internal layer, so the "no nonlinearity except
attention" decision is untouched.

## The picture

The whole model becomes one number line. Tokens sit at fixed points on
it; the activation ŷ is a moving point on it; prediction is "which
token-point is ŷ nearest." Maximally visualizable — which is the point
of the project.

## Status

Concept recorded. Still no code. The corpus / language choice remains
open, but this readout design does not depend on the language and can be
settled in advance of it.
