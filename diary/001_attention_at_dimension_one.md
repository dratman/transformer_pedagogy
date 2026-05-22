# Diary 001 — Attention at dimension one

Date: 2026-05-22

## Context

First diary entry of `transformer_pedagogy`. The project builds minimal
transformer language models on tiny, explicitly defined languages and climbs
upward from the simplest possible case — the aim being to make plain, the way
a compiler makes a programming language plain, what a transformer and a
language are really doing. (See the README for the broader framing.)

The immediate setting is **rung 1**: a single-layer model at embedding
dimension 1, with LayerNorm disabled (it is degenerate at dimension 1 — see
the `disable_layernorm` option in `model.py`), and all parameters —
embeddings, weights, attention — learned jointly. The rung-1 language is a
**2-token alternation language**: strings over {A, B} in which the two
symbols strictly take turns. This entry records one design decision settled
before any rung-1 code is written.

## The concept: at dimension 1, Q, K, V are scalars

In standard attention, Q, K, V are vectors only because the embedding
dimension is normally greater than 1. Set the embedding dimension to 1 and
each is necessarily a single number. This is not a simplification we impose;
it is what dimension-1 attention *is*.

Precisely:

- The **projection weights** w_Q, w_K, w_V are scalars. In `model.py`,
  `c_attn` is `Linear(1, 3)` — three weight scalars (plus three biases if
  bias is kept on). The output projection W_O (`c_proj`) is one more scalar.
- The **q, k, v values at each position** are therefore scalars too. With
  only two tokens in the alternation language, there are ultimately just two
  distinct q-values across the whole vocabulary, two k-values, two v-values.
- What does **not** collapse to a scalar is the attention **pattern** — the
  weight position i places on position j. That remains a T×T object by
  necessity: one number per ordered pair of positions. The scalars are what
  is *learned*; the T×T pattern is what is *computed* from them at run time.

So the entire learned content of the attention sublayer, at dimension 1, is
a handful of scalars.

## Two ways to have "Q, K, V each a scalar"

A genuine choice sits inside this:

- **Factored.** Each token has one embedding scalar `e`; then q = w_Q·e,
  k = w_K·e, v = w_V·e. A token's q, k, v are forced *proportional* to its
  embedding. This is what `model.py` does.
- **Direct.** Give each token its own free q, k, v scalars, learned
  directly, untied from the embedding.

At dimension 1 the two are nearly the same — both deliver "a scalar per
token per role" — and they differ only in whether q, k, v must be
proportional to the embedding.

## Decision: the factored form

Rung 1 uses the **factored** form. Three reasons:

1. It is the actual transformer. We want to study the real object, not a
   reparameterized cousin.
2. At dimension 1 it is already trivially small, so nothing is gained by
   reparameterizing for size.
3. **It is the form that survives the climb to depth.** The direct form
   works only for a single layer at the input. In layer 2 and beyond, q, k,
   v must be projected from a *computed* residual stream — by then no longer
   just the embedding — not looked up per token. The factored form
   generalizes; the direct form does not.

## A result worth keeping

The observation is itself a piece of the pedagogy: at dimension 1 the whole
Q/K/V projection apparatus collapses to a few per-token scalars, and the only
residue of the projection machinery is a single proportionality constraint
(q, k, v ∝ embedding). The elaborate attention parameterization has a small,
fully stateable meaning at the floor.

## Status

Concept recorded. No rung-1 code written yet.
