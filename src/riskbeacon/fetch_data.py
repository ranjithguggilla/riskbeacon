"""Fetch the training data from its public mirror.

The raw csv is not committed to keep the repository light. This script pulls it
from the GitHub mirror of the Kaggle "Give Me Some Credit" competition into the
data/ folder. It is safe to run repeatedly.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from .config import Paths

SOURCE_REPO = "https://github.com/vivekkalyan/give-me-some-credit.git"
SOURCE_FILE = "cs-training.csv"


def main() -> int:
    paths = Paths()
    paths.data.mkdir(parents=True, exist_ok=True)
    target = paths.raw_csv
    if target.exists():
        print(f"already present: {target}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["git", "clone", "--depth", "1", SOURCE_REPO, tmp],
            check=True,
        )
        src = Path(tmp) / SOURCE_FILE
        target.write_bytes(src.read_bytes())
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
