# Diary 004 — First language: ABC periodic

Date: 2026-05-22

## Context

Rung 1 needs a language. The 2-token alternation language was set aside
(diary 001 / conversation). The distance readout is now wired (diary 003).
This entry records the choice of the first language and its rationale.

## The language

Vocab `{A, B, C}` (size 3). Legal strings are prefixes of the single
infinite periodic stream `ABCABCABC...`. Equivalently, the legal next
token is *next-in-cycle*, depending only on the last token:

| context (last token) | legal next token |
|---|---|
| A | B |
| B | C |
| C | A |

It is deterministic, length-1, three rows.

## Why this choice

Three things, each load-bearing.

1. **It exposes the standard head's bottleneck and lets the distance head
   demonstrate the fix.** As shown in diary 002, `model.py`'s no-bias tied
   `lm_head` at dimension 1 computes `logit_i = ŷ · e_i`, whose argmax is
   always at an extreme embedding — so a standard dim-1 model can predict
   only *two* distinct tokens ever. A vocab of three with a rule that
   visits all three is the smallest setting where this limitation actually
   bites. Expectation: standard head fails (cannot model the language);
   distance head succeeds.

2. **It keeps attention idle.** The rule is length-1 — only the last
   token matters — so attention has nothing to do. The model can fit the
   language using embeddings and the readout alone. Good for a first
   experiment: it isolates the embedding + readout, leaving attention as
   a control to be examined separately (does it learn to do nothing, or
   does it do something irrelevant?).

3. **The answer key is trivial.** Deterministic, three rules, hand-
   checkable. The optimal loss is 0 in the limit. We know what the trained
   model *should* do, so we can grade it.

## Why not skip ahead to a context-dependent language

Eventually, yes; not yet. The first experiment is also a calibration of
the apparatus: does the training pipeline (train.py extended with the new
flags) actually train a dim-1 model cleanly? Settling that on the simplest
case — length-1, attention idle — before introducing context dependence
keeps the next failure attributable when it comes.

## The corpus

A long strictly periodic stream — `ABCABCABC...` repeated 20,000 times,
60,000 characters total. Generator at `py/make_corpus_abc.py`; output at
`txt_local/abc_periodic.txt` (gitignored). The generator is part of the
repo so a fresh clone can reproduce the corpus exactly.

## Status

Language defined; corpus generator written. Training is next.
