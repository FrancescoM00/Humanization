#!/usr/bin/env python3
"""Analyze the candidate-pool robustness check without fitting any model."""
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

from stylometry.shortlist_length_analysis import analyze_results, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results_shortlist_length_robustness")
    parser.add_argument("--protocol", type=Path, default=None)
    args = parser.parse_args()
    summary = analyze_results(args.results, args.protocol, ROOT)
    report = write_report(args.results, args.protocol)
    print(json.dumps({"primary_metrics": summary["primary_metrics"],
                      "macro_differences": summary["macro_differences"],
                      "parent_C3_reproduced": summary["parent_C3_reproduced"],
                      "report": str(report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
