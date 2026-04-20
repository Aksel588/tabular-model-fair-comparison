#!/usr/bin/env python3
"""Run full experiment from project root: python run_experiment.py"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.analyze import plot_complexity_tradeoff, write_analysis_report
from src.datasets import get_dataset, list_dataset_keys
from src.train import run_bundle, save_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fair tabular model comparison (matched RandomizedSearchCV budget per model).",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        metavar="KEYS",
        help=(
            "Comma-separated dataset keys (default: all). Valid keys: "
            + ", ".join(list_dataset_keys())
        ),
    )
    args = parser.parse_args()

    wanted: set[str] | None = None
    if args.datasets:
        wanted = {k.strip() for k in args.datasets.split(",") if k.strip()}
        valid = set(list_dataset_keys())
        unknown = wanted - valid
        if unknown:
            raise SystemExit(
                f"Unknown dataset keys: {sorted(unknown)}. Valid: {sorted(valid)}"
            )

    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    if wanted is not None:
        keys_to_run = [k for k in list_dataset_keys() if k in wanted]
    else:
        keys_to_run = list_dataset_keys()

    all_blocks = []
    for key in keys_to_run:
        bundle = get_dataset(key)
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
