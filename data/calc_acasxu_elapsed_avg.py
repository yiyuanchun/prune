from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    src_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Calculate the average elapsed_seconds value for each ACASXu CSV file."
    )
    parser.add_argument(
        "--acasxu-dir",
        type=Path,
        default=src_dir / "ACASXu",
        help="Directory containing ACASXu CSV files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search CSV files recursively under --acasxu-dir.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of decimal places to print for averages.",
    )
    return parser.parse_args()


def parse_elapsed_seconds(row: dict[str, str]) -> float | None:
    raw_value = (row.get("elapsed_seconds") or "").strip()
    if not raw_value:
        return None
    try:
        value = float(raw_value)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def iter_csv_paths(directory: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(path for path in directory.glob(pattern) if path.is_file())


def calculate_average(csv_path: Path) -> tuple[float | None, int, int]:
    total = 0.0
    valid_count = 0
    invalid_count = 0

    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or "elapsed_seconds" not in reader.fieldnames:
            return None, 0, 0

        for row in reader:
            value = parse_elapsed_seconds(row)
            if value is None:
                invalid_count += 1
                continue
            total += value
            valid_count += 1

    if valid_count == 0:
        return None, valid_count, invalid_count
    return total / valid_count, valid_count, invalid_count


def main() -> None:
    args = parse_args()
    csv_paths = iter_csv_paths(args.acasxu_dir, args.recursive)
    if not csv_paths:
        print(f"warning: no CSV files found in {args.acasxu_dir.resolve().as_posix()}")
        return

    print("file,average_elapsed_seconds,valid_rows,invalid_rows")
    for csv_path in csv_paths:
        average, valid_count, invalid_count = calculate_average(csv_path)
        average_text = "NA" if average is None else f"{average:.{args.precision}f}"
        print(f"{csv_path.name},{average_text},{valid_count},{invalid_count}")


if __name__ == "__main__":
    main()
