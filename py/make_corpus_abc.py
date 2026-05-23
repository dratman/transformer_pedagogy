#!/usr/bin/env python3
"""
Generate the rung-1 ABC periodic corpus.

Language: vocab {A, B, C}, the strictly periodic stream ABCABCABC...
Deterministic, length-1 next-token rule. See diary 004.

Writes the corpus to txt_local/abc_periodic.txt. That file is gitignored
so the generator is the durable definition; rerun this script after a
fresh clone to reproduce the corpus.

Usage:
    python3.11 py/make_corpus_abc.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "txt_local" / "abc_periodic.txt"
N_CYCLES = 20_000


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = "ABC" * N_CYCLES
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}  --  {len(text):,} characters "
          f"({N_CYCLES:,} cycles of ABC)")


if __name__ == "__main__":
    main()
