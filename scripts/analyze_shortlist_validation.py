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
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

from stylometry.shortlist_statistics import analyze_results
from stylometry.shortlist_report import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results_shortlist_validation")
    parser.add_argument("--protocol", type=Path, default=None)
    args = parser.parse_args()
    summary = analyze_results(args.results, args.protocol)
    report = write_report(args.results, args.protocol)
    print(json.dumps({"primary_metrics": summary["primary_metrics"],
                      "report": str(report),
                      "elapsed_seconds": summary["elapsed_seconds"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
