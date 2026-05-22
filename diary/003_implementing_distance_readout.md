# Diary 003 — Implementing the distance readout

Date: 2026-05-22

## Context

Diary 002 worked out the math for a squared-distance, tied-embedding
readout: `logits_i = -||x - e_i||^2`, equivalent to a linear head with
per-token weight `2 e_i` and bias `-||e_i||^2`. This entry records the
implementation in `model.py` and the smoke-test case that exercises it.

## What was added

A single new `GPTConfig` flag, off by default so all existing behaviour is
unchanged:

    distance_readout: bool = False  # -||x - e_i||^2 readout instead of x . e_i

A `compute_logits` helper on `GPT` that branches on the flag:

    def compute_logits(self, x):
        if getattr(self.config, 'distance_readout', False):
            e = self.lm_head.weight                 # tied to wte
            x_sq = (x * x).sum(-1, keepdim=True)    # (B, T, 1)
            e_sq = (e * e).sum(-1)                  # (V,)
            xe = x @ e.T                            # (B, T, V)
            return -(x_sq - 2.0 * xe + e_sq)
        return self.lm_head(x)

The expansion `||x - e||^2 = ||x||^2 - 2 x.e + ||e||^2` is used directly so
the implementation works in any embedding dimension and broadcasts cleanly
across the batch and time axes.

`GPT.forward` was changed in two places to call `self.compute_logits(...)`
in place of `self.lm_head(...)` — one for the training path (full sequence)
and one for the single-position inference path. Nothing else changed;
parameter count and the rest of the model are untouched.

## Smoke test

`py/smoke_test.py` gained a third case:

    distance = GPTConfig(vocab_size=8, block_size=16, n_layer=1, n_head=1,
                         n_embd=1, dropout=0.0, bias=False, no_gelu=True,
                         disable_layernorm=True, distance_readout=True)

Result of `python3.11 py/smoke_test.py`:

    [ok] minimal 1-D:    loss 2.0791
    [ok] standard:       loss 2.7093
    [ok] distance head:  loss 2.0792
    smoke test passed   (exit 0)

The untrained distance head sits at the uniform-prior loss `ln(8) ≈ 2.079`,
essentially identical to the untrained standard head — both random-init
heads give near-uniform predictions. The distance head will only diverge
from the standard head once trained on a corpus that exposes the bottleneck
that the standard no-bias head suffers at dimension 1 (cf. diary 002).

## Status

Distance-readout code in place and exercised by the smoke test. No
training run yet; the language and corpus for the first experiment is the
next step.
