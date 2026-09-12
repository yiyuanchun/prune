from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired B3 REDNet + NARv experiments.")
    parser.add_argument("--narv-csv", type=Path, required=True)
    parser.add_argument("--rednet-narv-csv", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--paired-output-csv", type=Path, default=Path("b3_paired_results.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("b3_summary.json"))
    return parser.parse_args()


def _read(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["net_name"], row["property_id"]): row for row in rows}


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float(default)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "")))
    except (TypeError, ValueError):
        return int(default)


def _solved(row: dict[str, str]) -> bool:
    return str(row.get("verification_result", "")).upper() in {"VERIFIED", "UNSAFE"}


def _par2(row: dict[str, str], timeout: float) -> float:
    return _float(row, "total_time_seconds", timeout) if _solved(row) else 2.0 * timeout


def _geomean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> None:
    args = parse_args()
    narv = _read(args.narv_csv.resolve())
    combo = _read(args.rednet_narv_csv.resolve())
    common_keys = sorted(set(narv) & set(combo))
    if not common_keys:
        raise ValueError("No paired network/property rows were found.")

    paired_rows: list[dict[str, Any]] = []
    common_solved_speedups: list[float] = []
    result_mismatches = 0
    for key in common_keys:
        base = narv[key]
        combined = combo[key]
        base_time = _float(base, "total_time_seconds", args.timeout_seconds)
        combo_time = _float(combined, "total_time_seconds", args.timeout_seconds)
        base_solved = _solved(base)
        combo_solved = _solved(combined)
        speedup = base_time / combo_time if base_solved and combo_solved and combo_time > 0.0 else None
        if speedup is not None:
            common_solved_speedups.append(speedup)
        base_answer = str(base.get("verification_result", "")).upper()
        combo_answer = str(combined.get("verification_result", "")).upper()
        mismatch = base_solved and combo_solved and base_answer != combo_answer
        result_mismatches += int(mismatch)
        paired_rows.append({
            "net_name": key[0],
            "property_id": key[1],
            "narv_result": base_answer,
            "rednet_narv_result": combo_answer,
            "result_mismatch": mismatch,
            "narv_total_time_seconds": base_time,
            "rednet_narv_total_time_seconds": combo_time,
            "paired_speedup": "" if speedup is None else speedup,
            "rednet_preprocessing_time_seconds": _float(combined, "rednet_preprocessing_time_seconds"),
            "relu_reduction_ratio": _float(combined, "relu_reduction_ratio", 1.0),
            "original_relu_count": _int(combined, "original_relu_count"),
            "reduced_relu_count": _int(combined, "reduced_relu_count"),
            "narv_refinement_steps": _int(base, "refinement_steps"),
            "rednet_narv_refinement_steps": _int(combined, "refinement_steps"),
            "narv_iterations": _int(base, "iterations"),
            "rednet_narv_iterations": _int(combined, "iterations"),
            "equivalence_max_abs_error": _float(combined, "equivalence_max_abs_error"),
        })

    args.paired_output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.paired_output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    narv_rows = [narv[key] for key in common_keys]
    combo_rows = [combo[key] for key in common_keys]
    narv_solved_times = [_float(row, "total_time_seconds") for row in narv_rows if _solved(row)]
    combo_solved_times = [_float(row, "total_time_seconds") for row in combo_rows if _solved(row)]
    reduction_ratios = [_float(row, "relu_reduction_ratio", 1.0) for row in combo_rows]
    summary = {
        "paired_instances": len(common_keys),
        "timeout_seconds": float(args.timeout_seconds),
        "result_mismatches": result_mismatches,
        "narv": {
            "solved": sum(_solved(row) for row in narv_rows),
            "mean_solved_time_seconds": _mean(narv_solved_times),
            "median_solved_time_seconds": _median(narv_solved_times),
            "par2_mean_seconds": _mean([_par2(row, args.timeout_seconds) for row in narv_rows]),
            "mean_refinement_steps": _mean([float(_int(row, "refinement_steps")) for row in narv_rows]),
        },
        "rednet_narv": {
            "solved": sum(_solved(row) for row in combo_rows),
            "mean_solved_time_seconds": _mean(combo_solved_times),
            "median_solved_time_seconds": _median(combo_solved_times),
            "par2_mean_seconds": _mean([_par2(row, args.timeout_seconds) for row in combo_rows]),
            "mean_refinement_steps": _mean([float(_int(row, "refinement_steps")) for row in combo_rows]),
            "mean_reduction_ratio": _mean(reduction_ratios),
            "median_reduction_ratio": _median(reduction_ratios),
            "mean_preprocessing_time_seconds": _mean([
                _float(row, "rednet_preprocessing_time_seconds") for row in combo_rows
            ]),
        },
        "paired_common_solved": {
            "count": len(common_solved_speedups),
            "geometric_mean_speedup": _geomean(common_solved_speedups),
            "median_speedup": _median(common_solved_speedups),
            "combo_faster_fraction": (
                sum(value > 1.0 for value in common_solved_speedups) / len(common_solved_speedups)
                if common_solved_speedups else None
            ),
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
