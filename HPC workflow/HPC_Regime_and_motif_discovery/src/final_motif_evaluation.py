from __future__ import annotations

import ast
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


Interval = tuple[int, int]


@dataclass(frozen=True)
class EvaluationCase:
    slice_id: str
    asset: str
    frequency: str
    regime_label: str
    slice_path: Path
    primary_mp_window: int
    locomotif_raw_path: Path | None = None


def validate_interval(start: int, end: int, series_length: int | None = None) -> Interval:
    """Validate a half-open [start, end) interval."""
    start_i = int(start)
    end_i = int(end)
    if start_i < 0:
        raise ValueError(f"Interval start must be non-negative, got {start_i}.")
    if end_i <= start_i:
        raise ValueError(f"Interval end must be greater than start, got ({start_i}, {end_i}).")
    if series_length is not None and end_i > int(series_length):
        raise ValueError(f"Interval ({start_i}, {end_i}) exceeds series length {series_length}.")
    return start_i, end_i


def interval_union(intervals: Iterable[Interval]) -> list[Interval]:
    """Return sorted non-overlapping half-open intervals covering the same bars."""
    sorted_intervals = sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s))
    if not sorted_intervals:
        return []
    merged: list[Interval] = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_mass(intervals: Iterable[Interval]) -> int:
    return int(sum(max(0, int(end) - int(start)) for start, end in intervals))


def coverage(intervals: Iterable[Interval], eligible_observations: int) -> float:
    if eligible_observations <= 0:
        return math.nan
    return interval_mass(interval_union(intervals)) / float(eligible_observations)


def redundancy_fraction(intervals: Iterable[Interval]) -> float:
    intervals_list = list(intervals)
    total = interval_mass(intervals_list)
    if total == 0:
        return math.nan
    union = interval_mass(interval_union(intervals_list))
    return 1.0 - union / float(total)


def interval_iou(a: Interval, b: Interval) -> float:
    a_start, a_end = validate_interval(a[0], a[1])
    b_start, b_end = validate_interval(b[0], b[1])
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return float(intersection / union) if union else 0.0


def covered_points(intervals: Iterable[Interval]) -> set[int]:
    points: set[int] = set()
    for start, end in interval_union(intervals):
        points.update(range(start, end))
    return points


def union_jaccard(left: Iterable[Interval], right: Iterable[Interval]) -> float:
    left_points = covered_points(left)
    right_points = covered_points(right)
    union = left_points | right_points
    if not union:
        return math.nan
    return len(left_points & right_points) / float(len(union))


def _numeric_timestamp_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        median = numeric.dropna().median()
        if median > 1e17:
            return pd.to_datetime(numeric, unit="ns", utc=True, errors="coerce")
        if median > 1e14:
            return pd.to_datetime(numeric, unit="us", utc=True, errors="coerce")
        if median > 1e11:
            return pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
        if median > 1e9:
            return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_price_slice(path: Path) -> pd.DataFrame:
    df = read_table(path).copy()
    required = {"timestamp", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["timestamp"] = _numeric_timestamp_series(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(f"Timestamps are not monotonic after sorting: {path}")
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def validate_controlled_slice(
    slice_id: str,
    df: pd.DataFrame,
    expected: dict[str, Any],
    required_columns: Sequence[str] = ("timestamp", "open", "high", "low", "close", "close_z"),
) -> None:
    """Fail fast when a controlled input slice does not match the audited slice contract."""
    errors: list[str] = []
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        errors.append(f"missing required columns {missing_columns}")

    if "rows" in expected and int(expected["rows"]) != len(df):
        errors.append(f"expected {int(expected['rows'])} rows, observed {len(df)}")

    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if timestamps.isna().any():
            errors.append(f"observed {int(timestamps.isna().sum())} unparsable timestamps")
        if not timestamps.is_monotonic_increasing:
            errors.append("timestamps are not monotonic increasing")
        duplicates = int(timestamps.duplicated().sum())
        if duplicates:
            errors.append(f"observed {duplicates} duplicate timestamps")
        if len(timestamps) and timestamps.notna().any():
            observed_start = timestamps.iloc[0]
            observed_end = timestamps.iloc[-1]
            if "start" in expected:
                expected_start = pd.Timestamp(expected["start"])
                if expected_start.tzinfo is None:
                    expected_start = expected_start.tz_localize("UTC")
                else:
                    expected_start = expected_start.tz_convert("UTC")
                if observed_start != expected_start:
                    errors.append(f"expected start {expected_start.isoformat()}, observed {observed_start.isoformat()}")
            if "end" in expected:
                expected_end = pd.Timestamp(expected["end"])
                if expected_end.tzinfo is None:
                    expected_end = expected_end.tz_localize("UTC")
                else:
                    expected_end = expected_end.tz_convert("UTC")
                if observed_end != expected_end:
                    errors.append(f"expected end {expected_end.isoformat()}, observed {observed_end.isoformat()}")

    if errors:
        detail = ".\n".join(errors)
        raise ValueError(
            f"Controlled input mismatch for {slice_id}:\n"
            f"{detail}.\n"
            "Do not use this file for final thesis evaluation."
        )


def file_profile(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        df = read_table(path)
        columns = list(map(str, df.columns))
        rows = int(len(df))
    except Exception as exc:
        columns = []
        rows = None
        return {"path": str(path), "exists": True, "file_size": path.stat().st_size, "read_error": str(exc)}
    return {
        "path": str(path),
        "exists": True,
        "file_size": int(path.stat().st_size),
        "rows": rows,
        "columns": columns,
    }


def dataframe_schema(df: pd.DataFrame) -> dict[str, Any]:
    timestamps: dict[str, Any] = {}
    for column in df.columns:
        if "timestamp" in str(column).lower() or str(column).lower() in {"time", "date", "datetime"}:
            parsed = _numeric_timestamp_series(df[column])
            timestamps[str(column)] = {
                "non_null": int(parsed.notna().sum()),
                "min": parsed.min().isoformat() if parsed.notna().any() else None,
                "max": parsed.max().isoformat() if parsed.notna().any() else None,
            }
    return {"rows": int(len(df)), "columns": list(map(str, df.columns)), "timestamps": timestamps}


def parse_locomotif_raw_output(path: Path) -> list[dict[str, Any]]:
    """Parse stored LoCoMotif repr output without executing LoCoMotif."""
    if path is None or not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = "# repr(motif_sets)"
    if marker not in text:
        return []
    payload = text.split(marker, 1)[1].strip()
    payload = re.sub(r"np\.int\d+\(([-+]?\d+)\)", r"\1", payload)
    try:
        parsed = ast.literal_eval(payload)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Could not parse LoCoMotif raw output {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for motif_set_id, item in enumerate(parsed, start=1):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        representative, occurrences = item
        rep_start, rep_end = validate_interval(int(representative[0]), int(representative[1]))
        seen: set[Interval] = set()
        for occurrence_id, interval in enumerate(occurrences, start=1):
            start, end = validate_interval(int(interval[0]), int(interval[1]))
            if (start, end) in seen:
                continue
            seen.add((start, end))
            rows.append(
                {
                    "motif_set_id": motif_set_id,
                    "occurrence_id": occurrence_id,
                    "representative_start": rep_start,
                    "representative_end": rep_end,
                    "start_idx": start,
                    "end_idx": end,
                    "motif_length_bars": end - start,
                    "source_format": "raw_locomotif_repr",
                }
            )
    return rows


def resolve_cases(workflow_root: Path, config: dict[str, Any]) -> list[EvaluationCase]:
    table_dir = resolve_path(workflow_root, config["inputs"]["controlled_table_dir"])
    cases: list[EvaluationCase] = []
    for slice_id, spec in config["inputs"]["slices"].items():
        raw_name = spec.get("locomotif_raw_output")
        cases.append(
            EvaluationCase(
                slice_id=slice_id,
                asset=str(spec["asset"]),
                frequency=str(spec["frequency"]),
                regime_label=str(spec["regime_label"]),
                slice_path=table_dir / str(spec["slice_input"]),
                primary_mp_window=int(spec["primary_mp_window"]),
                locomotif_raw_path=(table_dir / str(raw_name)) if raw_name else None,
            )
        )
    return cases


def resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _timestamp_at(df: pd.DataFrame, index: int) -> str:
    if index < 0 or index >= len(df):
        raise IndexError(f"Timestamp index {index} is outside slice length {len(df)}.")
    value = pd.Timestamp(df.loc[index, "timestamp"])
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    return value.isoformat()


def _select_mp_rows_for_case(mp_rows: pd.DataFrame, case: EvaluationCase, slice_len: int) -> pd.DataFrame:
    subset = mp_rows[
        (mp_rows["slice_id"].astype(str) == case.slice_id)
        & (mp_rows["asset"].astype(str) == case.asset)
        & (mp_rows["frequency"].astype(str) == case.frequency)
        & (mp_rows["regime_label"].astype(str) == case.regime_label)
        & (mp_rows["feature"].astype(str) == "close_z")
    ].copy()
    if subset.empty:
        return subset
    exact_suffix = f"_n{slice_len}"
    exact = subset[subset["run_key"].astype(str).str.endswith(exact_suffix)].copy()
    if not exact.empty:
        return exact
    subset["_valid_count"] = 0
    for idx, row in subset.iterrows():
        m = int(row["window_length"])
        valid = 0
        for col in ["motif_start_1", "motif_start_2"]:
            start = int(row[col])
            valid += int(0 <= start < start + m <= slice_len)
        subset.loc[idx, "_valid_count"] = valid
    return subset[subset["_valid_count"] > 0].drop(columns=["_valid_count"])


def build_mp_occurrences(mp_rows: pd.DataFrame, case: EvaluationCase, price_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    selected = _select_mp_rows_for_case(mp_rows, case, len(price_df))
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        window = int(row["window_length"])
        for occurrence_number, start_column in [(1, "motif_start_1"), (2, "motif_start_2")]:
            start = int(row[start_column])
            end = start + window
            try:
                validate_interval(start, end, len(price_df))
            except ValueError as exc:
                notes.append(f"Skipped invalid MP occurrence {row['run_key']} occurrence {occurrence_number}: {exc}")
                continue
            rows.append(
                {
                    "method": "Matrix Profile",
                    "slice_id": case.slice_id,
                    "asset": case.asset,
                    "frequency": case.frequency,
                    "regime_label": case.regime_label,
                    "feature": "close_z",
                    "configuration_id": f"m={window}",
                    "window_length": window,
                    "motif_set_id": str(row["run_key"]),
                    "occurrence_id": f"{row['run_key']}_{occurrence_number}",
                    "start_idx": start,
                    "end_idx": end,
                    "motif_length_bars": window,
                    "start_timestamp": _timestamp_at(price_df, start),
                    "end_timestamp": _timestamp_at(price_df, end - 1),
                    "event_anchor_idx": end - 1,
                    "event_time": _timestamp_at(price_df, end - 1),
                    "native_distance": float(row["best_motif_distance"]),
                    "mean_candidate_distance": float(row["mean_motif_distance"]),
                    "median_candidate_distance": float(row["median_motif_distance"]),
                    "native_similarity": math.nan,
                    "native_fitness": math.nan,
                    "runtime_seconds": float(row.get("runtime_seconds", math.nan)),
                    "runtime_context": "stored_controlled_matrix_profile_runtime",
                    "possible_jit_warmup": bool(float(row.get("runtime_seconds", 0.0)) > 1.0),
                    "source": "mp_controlled_slice_motifs.csv",
                }
            )
    return pd.DataFrame(rows), notes


def build_locomotif_occurrences(
    case: EvaluationCase,
    price_df: pd.DataFrame,
    invalid_interval_policy: str = "error",
) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    raw_rows = parse_locomotif_raw_output(case.locomotif_raw_path) if case.locomotif_raw_path else []
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        start = int(row["start_idx"])
        end = int(row["end_idx"])
        try:
            validate_interval(start, end, len(price_df))
        except ValueError as exc:
            message = (
                f"LoCoMotif interval mismatch for {case.slice_id} motif_set_id={row['motif_set_id']} "
                f"occurrence_id={row['occurrence_id']}: {exc}. "
                "The supplied controlled slice and LoCoMotif output do not correspond; "
                "do not use this file for final thesis evaluation."
            )
            if invalid_interval_policy == "error":
                raise ValueError(message) from exc
            if invalid_interval_policy == "skip":
                notes.append(f"Skipped invalid LoCoMotif interval: {message}")
                continue
            raise ValueError(f"Unknown invalid_interval_policy={invalid_interval_policy!r}; expected 'error' or 'skip'.")
        rows.append(
            {
                "method": "LoCoMotif",
                "slice_id": case.slice_id,
                "asset": case.asset,
                "frequency": case.frequency,
                "regime_label": case.regime_label,
                "feature": "close_z",
                "configuration_id": "l_min=12,l_max=48,rho=0.65,nb=3,overlap=0.2,warping=True",
                "window_length": math.nan,
                "motif_set_id": int(row["motif_set_id"]),
                "occurrence_id": f"{case.slice_id}_set{row['motif_set_id']}_occ{row['occurrence_id']}",
                "start_idx": start,
                "end_idx": end,
                "motif_length_bars": end - start,
                "start_timestamp": _timestamp_at(price_df, start),
                "end_timestamp": _timestamp_at(price_df, end - 1),
                "event_anchor_idx": end - 1,
                "event_time": _timestamp_at(price_df, end - 1),
                "native_distance": math.nan,
                "mean_candidate_distance": math.nan,
                "median_candidate_distance": math.nan,
                "native_similarity": math.nan,
                "native_fitness": math.nan,
                "runtime_seconds": math.nan,
                "runtime_context": "raw_output_only_structured_runtime_unavailable",
                "possible_jit_warmup": False,
                "source": str(case.locomotif_raw_path.name if case.locomotif_raw_path else ""),
            }
        )
    return pd.DataFrame(rows), notes


def intrinsic_metrics(occurrences: pd.DataFrame, slice_lengths: dict[str, int]) -> pd.DataFrame:
    if occurrences.empty:
        return pd.DataFrame()
    group_cols = ["method", "slice_id", "asset", "frequency", "regime_label", "feature", "configuration_id"]
    rows: list[dict[str, Any]] = []
    for keys, group in occurrences.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        intervals = list(zip(group["start_idx"].astype(int), group["end_idx"].astype(int)))
        lengths = group["motif_length_bars"].astype(float)
        distance_values = pd.to_numeric(group.get("native_distance"), errors="coerce")
        row.update(
            {
                "motif_set_count": int(group["motif_set_id"].nunique()),
                "occurrence_count": int(group["occurrence_id"].nunique()),
                "motif_length_mean": float(lengths.mean()) if len(lengths) else math.nan,
                "motif_length_median": float(lengths.median()) if len(lengths) else math.nan,
                "motif_length_std": float(lengths.std(ddof=1)) if len(lengths) > 1 else 0.0,
                "motif_length_min": float(lengths.min()) if len(lengths) else math.nan,
                "motif_length_max": float(lengths.max()) if len(lengths) else math.nan,
                "temporal_coverage": coverage(intervals, slice_lengths[str(row["slice_id"])]),
                "total_interval_mass": interval_mass(intervals),
                "union_interval_mass": interval_mass(interval_union(intervals)),
                "redundancy_fraction": redundancy_fraction(intervals),
                "redundancy_label": "thesis-derived redundancy statistic",
                "best_distance": float(distance_values.min()) if distance_values.notna().any() else math.nan,
                "mean_candidate_distance": float(pd.to_numeric(group.get("mean_candidate_distance"), errors="coerce").mean())
                if pd.to_numeric(group.get("mean_candidate_distance"), errors="coerce").notna().any()
                else math.nan,
                "median_candidate_distance": float(pd.to_numeric(group.get("median_candidate_distance"), errors="coerce").median())
                if pd.to_numeric(group.get("median_candidate_distance"), errors="coerce").notna().any()
                else math.nan,
                "native_similarity": math.nan,
                "native_fitness": math.nan,
                "shape_dispersion_proxy": math.nan,
                "metric_family": "literature-derived plus thesis-specific descriptive metrics",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def cross_method_agreement(occurrences: pd.DataFrame, primary_windows: dict[str, int], thresholds: Sequence[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slice_id, group in occurrences.groupby("slice_id", dropna=False):
        mp = group[(group["method"] == "Matrix Profile") & (group["window_length"].astype(float) == float(primary_windows[str(slice_id)]))]
        lm = group[group["method"] == "LoCoMotif"]
        mp_intervals = list(zip(mp["start_idx"].astype(int), mp["end_idx"].astype(int)))
        lm_intervals = list(zip(lm["start_idx"].astype(int), lm["end_idx"].astype(int)))
        max_ious: list[float] = []
        for interval in mp_intervals:
            max_ious.append(max((interval_iou(interval, other) for other in lm_intervals), default=0.0))
        row = {
            "slice_id": slice_id,
            "asset": group["asset"].iloc[0],
            "frequency": group["frequency"].iloc[0],
            "regime_label": group["regime_label"].iloc[0],
            "mp_primary_window": primary_windows[str(slice_id)],
            "mp_occurrences": int(len(mp_intervals)),
            "locomotif_occurrences": int(len(lm_intervals)),
            "union_coverage_jaccard": union_jaccard(mp_intervals, lm_intervals),
            "mean_mp_max_iou": float(np.mean(max_ious)) if max_ious else math.nan,
            "median_mp_max_iou": float(np.median(max_ious)) if max_ious else math.nan,
            "proportion_mp_any_overlap": float(np.mean([value > 0 for value in max_ious])) if max_ious else math.nan,
            "metric_label": "cross-method descriptive agreement metric, not ground-truth accuracy",
        }
        for threshold in thresholds:
            row[f"proportion_mp_iou_ge_{str(threshold).replace('.', 'p')}"] = (
                float(np.mean([value >= float(threshold) for value in max_ious])) if max_ious else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_future_outcome(price_df: pd.DataFrame, anchor_idx: int, horizon_bars: int) -> dict[str, float] | None:
    """Compute future outcomes from the bar after the motif ends."""
    anchor = int(anchor_idx)
    horizon = int(horizon_bars)
    future_end = anchor + horizon
    if anchor < 0 or future_end >= len(price_df):
        return None
    p0 = float(price_df.loc[anchor, "close"])
    p1 = float(price_df.loc[future_end, "close"])
    future = price_df.iloc[anchor + 1 : future_end + 1]
    simple_return = p1 / p0 - 1.0
    log_return = math.log(p1 / p0)
    future_log_returns = pd.to_numeric(future["log_return"], errors="coerce").fillna(0.0).to_numpy()
    rv = float(np.sqrt(np.sum(np.square(future_log_returns))))
    high_col = "high" if "high" in price_df.columns else "close"
    low_col = "low" if "low" in price_df.columns else "close"
    max_up = float(pd.to_numeric(future[high_col], errors="coerce").max() / p0 - 1.0)
    max_down = float(pd.to_numeric(future[low_col], errors="coerce").min() / p0 - 1.0)
    return {
        "simple_forward_return": float(simple_return),
        "log_forward_return": float(log_return),
        "future_realized_volatility": rv,
        "max_upward_excursion": max_up,
        "max_downward_excursion": max_down,
    }


def financial_event_outcomes(occurrences: pd.DataFrame, price_frames: dict[str, pd.DataFrame], horizons: dict[str, dict[str, int]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, occurrence in occurrences.iterrows():
        df = price_frames[str(occurrence["slice_id"])]
        for horizon_label, horizon_bars in horizons[str(occurrence["frequency"])].items():
            outcome = compute_future_outcome(df, int(occurrence["event_anchor_idx"]), int(horizon_bars))
            base = occurrence.to_dict()
            row = {
                key: base[key]
                for key in [
                    "method",
                    "slice_id",
                    "asset",
                    "frequency",
                    "regime_label",
                    "feature",
                    "configuration_id",
                    "motif_set_id",
                    "occurrence_id",
                    "start_idx",
                    "end_idx",
                    "event_anchor_idx",
                    "event_time",
                ]
            }
            row.update({"horizon_label": horizon_label, "horizon_bars": int(horizon_bars)})
            if outcome is None:
                row.update({"eligible": False, "exclusion_reason": "insufficient_future_data"})
                for name in OUTCOME_COLUMNS:
                    row[name] = math.nan
            else:
                row.update({"eligible": True, "exclusion_reason": ""})
                row.update(outcome)
            rows.append(row)
    return pd.DataFrame(rows)


OUTCOME_COLUMNS = [
    "simple_forward_return",
    "log_forward_return",
    "future_realized_volatility",
    "max_upward_excursion",
    "max_downward_excursion",
]


def summarize_outcomes(outcomes: pd.DataFrame, minimum_inference_n: int = 10) -> pd.DataFrame:
    group_cols = ["method", "slice_id", "asset", "frequency", "regime_label", "configuration_id", "horizon_label", "horizon_bars"]
    rows: list[dict[str, Any]] = []
    for keys, group in outcomes.groupby(group_cols, dropna=False):
        eligible = group[group["eligible"] == True].copy()
        returns = pd.to_numeric(eligible["simple_forward_return"], errors="coerce").dropna()
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_raw_occurrences": int(group["occurrence_id"].nunique()),
                "n_eligible_events": int(len(eligible)),
                "n_excluded_end_of_sample": int((group["eligible"] == False).sum()),
                "minimum_inference_n": int(minimum_inference_n),
                "inference_status": "eligible" if len(eligible) >= minimum_inference_n else "insufficient_n",
            }
        )
        for metric in OUTCOME_COLUMNS:
            values = pd.to_numeric(eligible[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else math.nan
            row[f"{metric}_median"] = float(values.median()) if len(values) else math.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
            row[f"{metric}_iqr"] = float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else math.nan
        row["positive_return_fraction"] = float((returns > 0).mean()) if len(returns) else math.nan
        row["negative_return_fraction"] = float((returns < 0).mean()) if len(returns) else math.nan
        row["directional_consistency"] = (
            float(max(row["positive_return_fraction"], row["negative_return_fraction"])) if len(returns) else math.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def anchors_inside_intervals(anchor_indices: np.ndarray, intervals: Iterable[Interval]) -> np.ndarray:
    mask = np.zeros(len(anchor_indices), dtype=bool)
    for start, end in interval_union(intervals):
        mask |= (anchor_indices >= int(start)) & (anchor_indices < int(end))
    return mask


def eligible_baseline_anchors(
    price_df: pd.DataFrame,
    horizon_bars: int,
    motif_intervals: Iterable[Interval],
    regime_label: str | None = None,
) -> np.ndarray:
    max_anchor = len(price_df) - int(horizon_bars) - 1
    if max_anchor < 0:
        return np.array([], dtype=int)
    anchors = np.arange(0, max_anchor + 1, dtype=int)
    if motif_intervals:
        anchors = anchors[~anchors_inside_intervals(anchors, motif_intervals)]
    if regime_label and regime_label != "agnostic":
        if "regime_label" not in price_df.columns:
            return np.array([], dtype=int)
        regimes = price_df.loc[anchors, "regime_label"].astype(str).to_numpy()
        anchors = anchors[regimes == str(regime_label)]
    return anchors


def sample_random_baseline(
    price_df: pd.DataFrame,
    horizon_bars: int,
    n_events: int,
    repetitions: int,
    seed: int,
    motif_intervals: Iterable[Interval] = (),
    regime_label: str | None = None,
) -> pd.DataFrame:
    anchors = eligible_baseline_anchors(price_df, horizon_bars, motif_intervals, regime_label=regime_label)
    if n_events <= 0 or len(anchors) == 0:
        return pd.DataFrame()
    valid_anchors: list[int] = []
    metric_values: dict[str, list[float]] = {metric: [] for metric in OUTCOME_COLUMNS}
    for anchor in anchors:
        outcome = compute_future_outcome(price_df, int(anchor), int(horizon_bars))
        if outcome is None:
            continue
        valid_anchors.append(int(anchor))
        for metric in OUTCOME_COLUMNS:
            metric_values[metric].append(float(outcome[metric]))
    if not valid_anchors:
        return pd.DataFrame()
    arrays = {metric: np.asarray(values, dtype=float) for metric, values in metric_values.items()}
    rng = np.random.default_rng(seed)
    replace = len(valid_anchors) < n_events
    rows: list[dict[str, Any]] = []
    for draw in range(int(repetitions)):
        sampled_positions = rng.choice(np.arange(len(valid_anchors)), size=int(n_events), replace=replace)
        row: dict[str, Any] = {
            "draw": draw,
            "sampled_events": int(len(sampled_positions)),
            "eligible_anchor_count": int(len(valid_anchors)),
            "sampled_with_replacement": bool(replace),
        }
        for metric in OUTCOME_COLUMNS:
            sampled_values = arrays[metric][sampled_positions]
            row[f"{metric}_mean"] = float(np.nanmean(sampled_values)) if len(sampled_values) else math.nan
            row[f"{metric}_median"] = float(np.nanmedian(sampled_values)) if len(sampled_values) else math.nan
        returns = arrays["simple_forward_return"][sampled_positions]
        row["positive_return_fraction"] = float(np.mean(returns > 0)) if len(returns) else math.nan
        row["directional_consistency"] = float(max(np.mean(returns > 0), np.mean(returns < 0))) if len(returns) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    y_arr = np.asarray(pd.Series(y).dropna(), dtype=float)
    if len(x_arr) == 0 or len(y_arr) == 0:
        return math.nan
    greater = 0
    less = 0
    for value in x_arr:
        greater += int(np.sum(value > y_arr))
        less += int(np.sum(value < y_arr))
    return float((greater - less) / (len(x_arr) * len(y_arr)))


def permutation_test_mean_difference(x: Sequence[float], y: Sequence[float], repetitions: int, seed: int) -> float:
    x_arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    y_arr = np.asarray(pd.Series(y).dropna(), dtype=float)
    if len(x_arr) == 0 or len(y_arr) == 0:
        return math.nan
    observed = abs(float(np.mean(x_arr) - np.mean(y_arr)))
    combined = np.concatenate([x_arr, y_arr])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(int(repetitions)):
        shuffled = rng.permutation(combined)
        diff = abs(float(np.mean(shuffled[: len(x_arr)]) - np.mean(shuffled[len(x_arr) :])))
        count += int(diff >= observed)
    return float((count + 1) / (int(repetitions) + 1))


def _block_sample(values: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if len(values) == 0:
        return values
    block = max(1, min(int(block_length), len(values)))
    starts = rng.integers(0, len(values), size=int(math.ceil(len(values) / block)))
    sampled: list[float] = []
    for start in starts:
        for offset in range(block):
            sampled.append(float(values[(start + offset) % len(values)]))
            if len(sampled) >= len(values):
                break
    return np.asarray(sampled, dtype=float)


def moving_block_bootstrap_mean_diff_ci(
    x: Sequence[float],
    y: Sequence[float],
    repetitions: int,
    seed: int,
    block_length: int | str = "auto",
    alpha: float = 0.05,
) -> tuple[float, float]:
    x_arr = np.asarray(pd.Series(x).dropna(), dtype=float)
    y_arr = np.asarray(pd.Series(y).dropna(), dtype=float)
    if len(x_arr) < 2 or len(y_arr) < 2:
        return math.nan, math.nan
    if block_length == "auto":
        block = max(1, int(round(math.sqrt(min(len(x_arr), len(y_arr))))))
    else:
        block = int(block_length)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(int(repetitions)):
        bx = _block_sample(x_arr, block, rng)
        by = _block_sample(y_arr, block, rng)
        diffs.append(float(np.mean(bx) - np.mean(by)))
    return float(np.quantile(diffs, alpha / 2.0)), float(np.quantile(diffs, 1.0 - alpha / 2.0))


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    p = np.asarray([np.nan if value is None else float(value) for value in p_values], dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.where(~np.isnan(p))[0]
    if len(valid) == 0:
        return q.tolist()
    order = valid[np.argsort(p[valid])]
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.minimum(adjusted, 1.0)
    return q.tolist()


def compare_to_baseline(
    outcomes: pd.DataFrame,
    occurrences: pd.DataFrame,
    price_frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
    baseline_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repetitions = int(config["baseline_repetitions"])
    min_n = int(config["minimum_inference_n"])
    seed = int(config["random_seed"])
    bootstrap_reps = int(config.get("bootstrap", {}).get("repetitions", 2000))
    block_length = config.get("bootstrap", {}).get("block_length", "auto")
    group_cols = ["method", "slice_id", "asset", "frequency", "regime_label", "configuration_id", "horizon_label", "horizon_bars"]
    comparison_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    for group_index, (keys, group) in enumerate(outcomes.groupby(group_cols, dropna=False)):
        key = dict(zip(group_cols, keys))
        eligible = group[group["eligible"] == True].copy()
        n = int(len(eligible))
        if n == 0:
            continue
        slice_id = str(key["slice_id"])
        price_df = price_frames[slice_id]
        intervals = list(
            zip(
                occurrences[occurrences["slice_id"].astype(str) == slice_id]["start_idx"].astype(int),
                occurrences[occurrences["slice_id"].astype(str) == slice_id]["end_idx"].astype(int),
            )
        )
        regime = str(key["regime_label"]) if baseline_kind == "regime_matched" else None
        if baseline_kind == "regime_matched" and regime == "agnostic":
            status = "not_applicable_for_agnostic_case"
            baseline = pd.DataFrame()
        else:
            baseline = sample_random_baseline(
                price_df=price_df,
                horizon_bars=int(key["horizon_bars"]),
                n_events=n,
                repetitions=repetitions,
                seed=seed + group_index,
                motif_intervals=intervals,
                regime_label=regime,
            )
            status = "available" if not baseline.empty else "not_available"
        base_row = dict(key)
        base_row.update({"baseline_kind": baseline_kind, "n_motif_events": n, "baseline_status": status})
        if not baseline.empty:
            base_row.update(
                {
                    "baseline_repetitions": int(len(baseline)),
                    "eligible_anchor_count": int(baseline["eligible_anchor_count"].max()),
                    "sampled_with_replacement": bool(baseline["sampled_with_replacement"].any()),
                }
            )
        comparison_rows.append(base_row)
        if baseline.empty:
            continue
        for metric in OUTCOME_COLUMNS:
            observed = pd.to_numeric(eligible[metric], errors="coerce").dropna().to_numpy(dtype=float)
            baseline_draw_means = pd.to_numeric(baseline[f"{metric}_mean"], errors="coerce").dropna().to_numpy(dtype=float)
            observed_mean = float(np.mean(observed)) if len(observed) else math.nan
            baseline_mean = float(np.mean(baseline_draw_means)) if len(baseline_draw_means) else math.nan
            diff = observed_mean - baseline_mean if math.isfinite(observed_mean) and math.isfinite(baseline_mean) else math.nan
            status_i = "eligible" if n >= min_n and len(baseline_draw_means) else "insufficient_n"
            if status_i == "eligible":
                centered = baseline_draw_means - np.mean(baseline_draw_means)
                p_mc = float((np.sum(np.abs(centered) >= abs(diff)) + 1) / (len(centered) + 1))
                baseline_one_draw = []
                anchors = eligible_baseline_anchors(
                    price_df,
                    int(key["horizon_bars"]),
                    intervals,
                    regime_label=regime,
                )
                if len(anchors):
                    rng = np.random.default_rng(seed + group_index + 999)
                    sampled = rng.choice(anchors, size=n, replace=len(anchors) < n)
                    for anchor in sampled:
                        out = compute_future_outcome(price_df, int(anchor), int(key["horizon_bars"]))
                        if out is not None:
                            baseline_one_draw.append(out[metric])
                delta = cliffs_delta(observed, baseline_one_draw)
                perm_p = permutation_test_mean_difference(observed, baseline_one_draw, min(2000, repetitions), seed + group_index + 123)
                ci_low, ci_high = moving_block_bootstrap_mean_diff_ci(
                    observed,
                    baseline_one_draw,
                    repetitions=bootstrap_reps,
                    seed=seed + group_index + 456,
                    block_length=block_length,
                )
            else:
                p_mc = math.nan
                delta = math.nan
                perm_p = math.nan
                ci_low = math.nan
                ci_high = math.nan
            row = dict(key)
            row.update(
                {
                    "baseline_kind": baseline_kind,
                    "metric": metric,
                    "n": n,
                    "observed_mean": observed_mean,
                    "baseline_mean_of_draw_means": baseline_mean,
                    "difference_in_means": diff,
                    "observed_median": float(np.median(observed)) if len(observed) else math.nan,
                    "baseline_median_of_draw_medians": float(np.nanmean(baseline[f"{metric}_median"])) if not baseline.empty else math.nan,
                    "monte_carlo_two_sided_p_value": p_mc,
                    "permutation_two_sided_p_value": perm_p,
                    "cliffs_delta": delta,
                    "bootstrap_mean_diff_ci_low": ci_low,
                    "bootstrap_mean_diff_ci_high": ci_high,
                    "inference_status": status_i,
                    "resampling_note": "moving/circular block bootstrap over ordered event outcomes" if status_i == "eligible" else "not run; n below threshold",
                }
            )
            inference_rows.append(row)
    inference = pd.DataFrame(inference_rows)
    if not inference.empty and "permutation_two_sided_p_value" in inference.columns:
        inference["fdr_q_value"] = np.nan
        fam_cols = ["baseline_kind", "metric"]
        for _, fam in inference.groupby(fam_cols, dropna=False):
            q_values = benjamini_hochberg(fam["permutation_two_sided_p_value"].tolist())
            inference.loc[fam.index, "fdr_q_value"] = q_values
    return pd.DataFrame(comparison_rows), inference


def parameter_sensitivity(intrinsic: pd.DataFrame) -> pd.DataFrame:
    if intrinsic.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    mp = intrinsic[intrinsic["method"] == "Matrix Profile"].copy()
    if not mp.empty:
        rows.append(
            {
                "method": "Matrix Profile",
                "sensitivity_type": "window_length",
                "status": "available",
                "configurations": ", ".join(sorted(mp["configuration_id"].astype(str).unique())),
                "notes": "Controlled MP windows are summarized across all available m values.",
            }
        )
    lm = intrinsic[intrinsic["method"] == "LoCoMotif"].copy()
    rows.append(
        {
            "method": "LoCoMotif",
            "sensitivity_type": "rho_or_parameter_grid",
            "status": "not_available" if lm["configuration_id"].nunique() <= 1 else "available",
            "configurations": ", ".join(sorted(lm["configuration_id"].astype(str).unique())) if not lm.empty else "",
            "notes": "Do not infer LoCoMotif rho sensitivity from a single rho=0.65 setting.",
        }
    )
    return pd.DataFrame(rows)


def runtime_summary(occurrences: pd.DataFrame, runtime_tables: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not occurrences.empty:
        for keys, group in occurrences.groupby(["method", "slice_id", "configuration_id"], dropna=False):
            method, slice_id, configuration_id = keys
            values = pd.to_numeric(group["runtime_seconds"], errors="coerce").dropna()
            rows.append(
                {
                    "method": method,
                    "slice_id": slice_id,
                    "configuration_id": configuration_id,
                    "runtime_seconds": float(values.iloc[0]) if len(values) else math.nan,
                    "runtime_context": str(group["runtime_context"].iloc[0]),
                    "possible_jit_warmup": bool(group["possible_jit_warmup"].any()),
                }
            )
    for table in runtime_tables:
        if table.empty:
            continue
        for _, row in table.iterrows():
            rows.append(
                {
                    "method": "LoCoMotif" if "l_min" in table.columns else "Matrix Profile",
                    "slice_id": row.get("slice_id", ""),
                    "configuration_id": row.get("run_key", ""),
                    "runtime_seconds": row.get("runtime_seconds", math.nan),
                    "runtime_context": "stored_runtime_table",
                    "possible_jit_warmup": False,
                    "success": row.get("success", math.nan),
                    "error_message": row.get("error_message", ""),
                }
            )
    return pd.DataFrame(rows).drop_duplicates()


def git_commit_hash(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True)
    except Exception:
        return None
    return result.stdout.strip()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def provenance_payload(
    repo_root: Path,
    workflow_root: Path,
    config: dict[str, Any],
    input_paths: Sequence[Path],
    schemas: dict[str, Any],
    notes: Sequence[str],
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": git_commit_hash(repo_root),
        "repo_root": str(repo_root),
        "workflow_root": str(workflow_root),
        "input_files": [file_profile(path) for path in input_paths],
        "schemas": schemas,
        "configuration": config,
        "random_seed": config.get("random_seed"),
        "notes": list(notes),
    }
