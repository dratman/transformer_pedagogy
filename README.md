# transformer_pedagogy

A pedagogical series of **minimal transformer examples**, built up from the
simplest possible cases, to make plain what a transformer — and a language —
are really doing.

## Goal

Most transformer tutorials begin with models already large enough that their
behavior is opaque. This project goes the other way: it starts at the floor
and climbs slowly.

- **Single-layer** transformers — linear or affine, before any nonlinearity.
- The **smallest possible dimension**, down to and including **one dimension**.
- **Artificial languages** (not natural language) and small-vocabulary text,
  so the data itself is fully understood before the model ever touches it.

The aim is for each example to be small enough to reason about completely —
by hand where possible — so every weight and every activation has a known,
stateable meaning.

## Layout

Each example lives in its own numbered directory, all visible side by side:

    01_.../
    02_.../
    ...

Shared training and inference code lives in `py/`, adapted from the author's
larger small-transformer work.

## Status

Started 2026-05-22. Scaffold only so far.
