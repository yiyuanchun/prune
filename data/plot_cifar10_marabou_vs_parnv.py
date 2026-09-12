from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

TIMEOUT_SECONDS = 3600.0
AXIS_MIN = 1e-1
PARNV_TIME_FIELDS = ("elapsed_seconds",)
MARABOU_TIME_FIELDS = ("elapsed_seconds",)


def parse_args() -> argparse.Namespace:
    src_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Plot Marabou elapsed_seconds against split-CIFAR10 elapsed_seconds "
            "using sample_index to pair rows."
        )
    )
    parser.add_argument(
        "--marabou-csv",
        type=Path,
        default=src_dir / "CIFAR10" / "verify_cifar10_robustness_marabou_10.csv",
        help="CSV containing sample_index and elapsed_seconds for Marabou.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=src_dir / "CIFAR10" / "verify_split_cifar10_10_marabou.csv",
        help="CSV containing sample_index and elapsed_seconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=src_dir / "CIFAR10" / "cifar10_marabou_vs_parnv_10_scatter.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def parse_positive_float(row: dict[str, str], field: str) -> float | None:
    raw_value = (row.get(field) or "").strip()
    if not raw_value:
        return None
    try:
        value = float(raw_value)
    except ValueError:
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def parse_positive_float_by_fields(row: dict[str, str], fields: tuple[str, ...]) -> float | None:
    for field in fields:
        value = parse_positive_float(row, field)
        if value is not None:
            return value
    return None


def is_skipped_row(row: dict[str, str]) -> bool:
    return any(
        (row.get(field) or "").strip().lower() == "skipped"
        for field in ("status", "overall_status")
    )


def build_paired_points(
    marabou_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], int]:
    split_by_index: dict[int, dict[str, str]] = {}
    for row in split_rows:
        if is_skipped_row(row):
            continue
        sample_index_text = (row.get("sample_index") or "").strip()
        if not sample_index_text:
            continue
        try:
            split_by_index[int(sample_index_text)] = row
        except ValueError:
            continue

    finished_points: list[tuple[float, float]] = []
    timeout_points: list[tuple[float, float]] = []
    skipped_both_timeout = 0
    for marabou_row in marabou_rows:
        if is_skipped_row(marabou_row):
            continue
        sample_index_text = (marabou_row.get("sample_index") or "").strip()
        if not sample_index_text:
            continue
        try:
            sample_index = int(sample_index_text)
        except ValueError:
            continue
        split_row = split_by_index.get(sample_index)
        if split_row is None:
            continue

        x_value = parse_positive_float_by_fields(split_row, PARNV_TIME_FIELDS)
        y_value = parse_positive_float_by_fields(marabou_row, MARABOU_TIME_FIELDS)
        if x_value is None or y_value is None:
            continue

        parnv_timeout = x_value > TIMEOUT_SECONDS
        marabou_timeout = y_value > TIMEOUT_SECONDS

        if parnv_timeout and marabou_timeout:
            skipped_both_timeout += 1
            continue
        if parnv_timeout:
            timeout_points.append((TIMEOUT_SECONDS, y_value))
            continue
        if marabou_timeout:
            timeout_points.append((x_value, TIMEOUT_SECONDS))
            continue

        finished_points.append((x_value, y_value))

    return finished_points, timeout_points, skipped_both_timeout


def plot_points(
    finished_points: list[tuple[float, float]],
    timeout_points: list[tuple[float, float]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    if finished_points:
        x_values, y_values = zip(*finished_points, strict=True)
        ax.scatter(x_values, y_values, color="blue", s=48, label="finished")
    else:
        ax.scatter([], [], color="blue", s=48, label="finished")

    if timeout_points:
        x_values, y_values = zip(*timeout_points, strict=True)
        ax.scatter(
            x_values,
            y_values,
            color="#e63c2d",
            marker="x",
            s=72,
            linewidths=2,
            label="timeout",
            clip_on=False,
            zorder=4,
        )
    else:
        ax.scatter(
            [],
            [],
            color="#e63c2d",
            marker="x",
            s=72,
            linewidths=2,
            label="timeout",
            clip_on=False,
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(AXIS_MIN, TIMEOUT_SECONDS)
    ax.set_ylim(AXIS_MIN, TIMEOUT_SECONDS)
    ax.plot(
        [AXIS_MIN, TIMEOUT_SECONDS],
        [AXIS_MIN, TIMEOUT_SECONDS],
        color="limegreen",
        linestyle="--",
        linewidth=2,
        label="y=x",
    )

    ax.set_xlabel("PARNV(sec)", fontname="Times New Roman", fontsize=20)
    ax.set_ylabel("Marabou(sec)", fontname="Times New Roman", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=17)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black", fontsize=15)
    ax.grid(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    marabou_rows = read_csv_rows(args.marabou_csv)
    split_rows = read_csv_rows(args.split_csv)
    finished_points, timeout_points, skipped_both_timeout = build_paired_points(marabou_rows, split_rows)
    if not finished_points and not timeout_points:
        print(
            "warning: no plottable points (all pairs are invalid or both methods exceed 3600s); "
            "an empty figure will still be saved."
        )
    plot_points(finished_points, timeout_points, args.output)
    print(
        f"saved plot to {args.output.resolve().as_posix()} "
        f"(finished={len(finished_points)}, timeout={len(timeout_points)}, "
        f"skipped_both_timeout={skipped_both_timeout})"
    )


if __name__ == "__main__":
    main()
