#!/usr/bin/env python3
"""
Generate a periodic-cycle corpus for the rung-1 dimension-floor experiments.

Given an alphabet (e.g. "ABCD"), writes the strictly periodic stream
ABCDABCDABCD... (vocab = the distinct letters, deterministic length-1
next-token rule) to txt_local/<lowercase>_periodic.txt. Total length is
fixed at ~60,000 chars across alphabets so corpus exposure is comparable.

Usage:
    python3.11 py/make_corpus_periodic.py            # default ABCD
    python3.11 py/make_corpus_periodic.py ABCDE      # vocab 5 cycle
"""

import sys
from pathlib import Path

TARGET_CHARS = 60_000

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "ABCD"
    if len(pattern) < 2 or len(set(pattern)) != len(pattern):
        sys.exit(f"alphabet must be at least 2 distinct chars: got {pattern!r}")
    cycles = TARGET_CHARS // len(pattern)
    text = pattern * cycles
    out = PROJECT_ROOT / "txt_local" / f"{pattern.lower()}_periodic.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}  --  {len(text):,} chars "
          f"({cycles:,} cycles of {pattern!r})")


if __name__ == "__main__":
    main()
