#!/usr/bin/env python3
"""Run full experiment from project root: python run_experiment.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.analyze import plot_complexity_tradeoff, write_analysis_report
from src.datasets import iter_datasets
from src.train import run_bundle, save_all


def main() -> None:
    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    all_blocks = []
    for bundle in iter_datasets():
        print(f"=== {bundle.key} ({bundle.task}) ===", flush=True)
        block = run_bundle(bundle)
        all_blocks.append(block)

    out_json = Path(config.RESULTS_DIR) / "all_results.json"
    save_all(all_blocks, str(out_json))
    write_analysis_report(all_blocks, str(Path(config.RESULTS_DIR) / "ANALYSIS.md"))
    plot_complexity_tradeoff(str(out_json), str(Path(config.RESULTS_DIR) / "complexity_tradeoff.png"))
    print(f"Wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
