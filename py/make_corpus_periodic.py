#!/usr/bin/env python3
"""
Generate a periodic corpus for the rung-1 experiments.

Given a *pattern* (e.g. "ABCD", "AABB"), writes the strictly periodic
stream pattern*pattern*pattern... to txt_local/<lowercase>_periodic.txt.
The pattern may contain repeated characters; the vocab is the set of
distinct characters appearing in it. Total length is fixed at ~60,000
chars across patterns so corpus exposure is comparable.

Usage:
    python3.11 py/make_corpus_periodic.py            # default ABCD
    python3.11 py/make_corpus_periodic.py ABCDE      # vocab 5 cycle
    python3.11 py/make_corpus_periodic.py AABB       # context-2 pattern
"""

import sys
from pathlib import Path

TARGET_CHARS = 60_000

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "ABCD"
    if len(pattern) < 2 or len(set(pattern)) < 2:
        sys.exit(f"pattern must be length >= 2 with >= 2 distinct chars: got {pattern!r}")
    cycles = TARGET_CHARS // len(pattern)
    text = pattern * cycles
    out = PROJECT_ROOT / "txt_local" / f"{pattern.lower()}_periodic.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}  --  {len(text):,} chars "
          f"({cycles:,} cycles of {pattern!r})")


if __name__ == "__main__":
    main()
