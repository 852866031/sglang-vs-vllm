#!/usr/bin/env python3
"""
shift_timeline.py — shift every timestamp_s in a timeline CSV by a constant
delta. Rows whose post-shift timestamp is < 0 are dropped.

Examples:
  # Shift everything 5s later, write to a new file.
  python shift_timeline.py --shift 5.0 -i timeline.csv -o timeline_shifted.csv

  # Shift earlier by 2.5s (drops rows with timestamp_s < 2.5).
  python shift_timeline.py --shift -2.5 -i timeline.csv -o timeline_early.csv

  # In-place shift.
  python shift_timeline.py --shift 1.0 -i timeline.csv --in-place
"""
import argparse
import csv
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input CSV path.")
    ap.add_argument("-o", "--output", help="Output CSV path.")
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite the input file. Mutually exclusive with -o.")
    ap.add_argument("--shift", type=float, required=True,
                    help="Seconds to add to every timestamp_s. Negative = earlier.")
    args = ap.parse_args()

    if args.in_place and args.output:
        ap.error("--in-place and --output are mutually exclusive")
    if not args.in_place and not args.output:
        ap.error("provide --output PATH or --in-place")

    in_path = Path(args.input)
    out_path = in_path if args.in_place else Path(args.output)

    with open(in_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "timestamp_s" not in fieldnames:
            sys.exit(f"{in_path}: missing required column 'timestamp_s'")
        rows = list(reader)

    total = len(rows)
    kept = []
    for r in rows:
        new_ts = float(r["timestamp_s"]) + args.shift
        if new_ts < 0:
            continue
        r["timestamp_s"] = f"{new_ts:.6f}"
        kept.append(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    dropped = total - len(kept)
    print(f"shifted by {args.shift:+.6f}s: kept {len(kept)}/{total} rows "
          f"(dropped {dropped}) -> {out_path}")


if __name__ == "__main__":
    main()
