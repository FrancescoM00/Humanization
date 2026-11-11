#!/usr/bin/env python3
"""Analyze saved compact-feature predictions with paired group bootstrap."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stylometry.shortlist_statistics import analyze_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results_shortlist_validation")
    parser.add_argument("--protocol", type=Path, default=None)
    args = parser.parse_args()
    summary = analyze_results(args.results, args.protocol)
    print(json.dumps({"primary_metrics": summary["primary_metrics"],
                      "elapsed_seconds": summary["elapsed_seconds"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
