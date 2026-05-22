"""
Smoke test for the transformer_pedagogy model code.

Two quick checks, no training involved:

  1. The minimal 1-D, single-layer model builds with disable_layernorm=True,
     every LayerNorm is switched off, and a forward pass gives a finite loss.
  2. A small model with LayerNorm left on (the default path) still builds
     and runs -- i.e. the disable_layernorm wiring did not disturb it.

An untrained model should sit near the uniform-prior loss ln(vocab_size);
the test allows a wide margin around that.

Run with the interpreter that has torch installed:

    python3.11 py/smoke_test.py

Exits 0 on success, 1 on any failure.
"""
import math

import torch

from model import GPT, GPTConfig, LayerNorm


def check(cfg, *, expect_ln_disabled):
    """Build a model, run one forward pass; return (list of failures, loss)."""
    failures = []
    model = GPT(cfg)

    lns = [m for m in model.modules() if isinstance(m, LayerNorm)]
    disabled = [ln for ln in lns if ln.disabled]
    if expect_ln_disabled and len(disabled) != len(lns):
        failures.append(f"expected all {len(lns)} LayerNorms disabled, "
                        f"got {len(disabled)}")
    if not expect_ln_disabled and disabled:
        failures.append(f"expected LayerNorms active, {len(disabled)} were off")

    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(x, x)

    want_shape = (2, cfg.block_size, cfg.vocab_size)
    if tuple(logits.shape) != want_shape:
        failures.append(f"logits shape {tuple(logits.shape)}, expected {want_shape}")
    if not torch.isfinite(loss):
        failures.append(f"loss not finite: {loss.item()}")
    elif abs(loss.item() - math.log(cfg.vocab_size)) > 1.5:
        failures.append(f"loss {loss.item():.3f} far from ln(vocab) "
                        f"{math.log(cfg.vocab_size):.3f}")
    return failures, loss


def main():
    # 1. Minimal 1-D, single-layer model -- affine except for attention.
    minimal = GPTConfig(vocab_size=8, block_size=16, n_layer=1, n_head=1,
                        n_embd=1, dropout=0.0, bias=False, no_gelu=True,
                        disable_layernorm=True)
    # 2. Small model on the default path, LayerNorm left on.
    standard = GPTConfig(vocab_size=16, block_size=16, n_layer=2, n_head=2,
                         n_embd=4, dropout=0.0)
    # 3. Minimal 1-D with the squared-distance readout (diary 002).
    distance = GPTConfig(vocab_size=8, block_size=16, n_layer=1, n_head=1,
                         n_embd=1, dropout=0.0, bias=False, no_gelu=True,
                         disable_layernorm=True, distance_readout=True)

    all_failures = []
    for label, cfg, expect in [("minimal 1-D",   minimal,  True),
                               ("standard",      standard, False),
                               ("distance head", distance, True)]:
        failures, loss = check(cfg, expect_ln_disabled=expect)
        print(f"[{'ok' if not failures else 'FAIL'}] {label}: loss {loss.item():.4f}")
        all_failures += [f"{label}: {f}" for f in failures]

    if all_failures:
        print("\nSMOKE TEST FAILED:")
        for failure in all_failures:
            print("  -", failure)
        return 1
    print("\nsmoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
