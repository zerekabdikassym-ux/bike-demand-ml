"""Inspect the original UCI hour.csv without third-party dependencies."""

import argparse
import csv
from datetime import datetime
from pathlib import Path


def inspect(path: Path) -> int:
    required = {"dteday", "hr", "cnt", "casual", "registered"}
    seen = set()
    timestamps = []
    missing = duplicates = invalid = mismatches = rows = 0
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        absent = required - set(reader.fieldnames or [])
        if absent:
            raise ValueError(f"Missing columns: {', '.join(sorted(absent))}")
        for row in reader:
            rows += 1
            missing += sum(value is None or not value.strip() for value in row.values() if not isinstance(value, list))
            try:
                hour = int(row["hr"])
                if not 0 <= hour <= 23:
                    raise ValueError("Hour outside 0..23")
                stamp = datetime.strptime(row["dteday"], "%Y-%m-%d").replace(hour=hour)
                counts = [int(row[key]) for key in ("cnt", "casual", "registered")]
                if min(counts) < 0:
                    raise ValueError("Negative rental count")
            except (ValueError, TypeError):
                invalid += 1
                continue
            mismatches += counts[0] != counts[1] + counts[2]
            duplicates += stamp in seen
            seen.add(stamp)
            timestamps.append(stamp)
    ordered = timestamps == sorted(timestamps)
    print(f"Rows: {rows}")
    print(f"Missing cells: {missing}")
    print(f"Invalid dates/hours/counts: {invalid}")
    print(f"Duplicate timestamps: {duplicates}")
    print(f"Rental-total mismatches: {mismatches}")
    print(f"Chronological input: {ordered}")
    if timestamps:
        print(f"Date range: {min(timestamps).isoformat()} to {max(timestamps).isoformat()}")
        expected = int((max(timestamps) - min(timestamps)).total_seconds() // 3600) + 1
        print(f"Absent hours within range: {expected - len(seen)} (not assumed zero demand)")
    return int(not rows or bool(missing or invalid or duplicates or mismatches or not ordered))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Path to the original UCI hour.csv")
    args = parser.parse_args()
    try:
        return inspect(args.csv)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"Inspection failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
