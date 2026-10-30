"""Rebuild chapters/data/shakespeare.txt from the tiny Shakespeare text.

The language models chapter reads a local copy of the first 100,000
characters of the text Karpathy published with char-rnn. This keeps those
characters and changes nothing else.

Usage: python3 tools/get-shakespeare.py
"""

import urllib.request
from pathlib import Path

SOURCE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
N_CHARS = 100_000
OUT = Path(__file__).resolve().parent.parent / "chapters" / "data" / "shakespeare.txt"


def main():
    with urllib.request.urlopen(SOURCE) as response:
        text = response.read().decode("utf-8")
    OUT.write_text(text[:N_CHARS], encoding="utf-8")
    print(f"wrote the first {N_CHARS:,} of {len(text):,} characters to {OUT}")


if __name__ == "__main__":
    main()
