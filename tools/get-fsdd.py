"""Rebuild chapters/data/7_jackson_0.wav from the Free Spoken Digit Dataset.

The applications chapter reads one recording, a speaker saying "seven",
from release v1.0.10 of the dataset, unchanged.
The dataset is licensed CC BY-SA 4.0.

Usage: python3 tools/get-fsdd.py
"""

import urllib.request
from pathlib import Path

NAME = "7_jackson_0.wav"
SOURCE = (
    "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/"
    f"v1.0.10/recordings/{NAME}"
)
OUT = Path(__file__).resolve().parent.parent / "chapters" / "data" / NAME


def main():
    with urllib.request.urlopen(SOURCE) as response:
        data = response.read()
    OUT.write_bytes(data)
    print(f"wrote {len(data):,} bytes to {OUT}")


if __name__ == "__main__":
    main()
