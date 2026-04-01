"""Run nateemma import batches 007 and 008 using the standard heavy screen."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_nateemma_next50 as base_runner


def main():
    base_runner.BATCHES = {
        "007": base_runner.BATCHES["007"],
        "008": base_runner.BATCHES["008"],
    }
    base_runner.MAX_WORKERS = 10
    base_runner.main()


if __name__ == "__main__":
    main()
