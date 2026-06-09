from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def interval_overlap_ratio(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    left = max(int(start_a), int(start_b))
    right = min(int(end_a), int(end_b))
    overlap = max(0, right - left)
    union = max(int(end_a), int(end_b)) - min(int(start_a), int(start_b))
    return float(overlap / union) if union > 0 else 0.0


def _time_split_stability(group: pd.DataFrame, timestamp_col: str) -> float:
    if group.empty or timestamp_col not in group.columns:
        return np.nan
    timestamps = pd.to_datetime(group[timestamp_col], utc=True, errors="coerce")
    midpoint = timestamps.min() + (timestamps.max() - timestamps.min()) / 2
    first = int((timestamps <= midpoint).sum())
    second = int((timestamps > midpoint).sum())
    denominator = max(first, second, 1)
    return float(min(first, second) / denominator)


def evaluate_matrix_profile_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    success = results[results.get("status", "success") == "success"].copy()
    if success.empty:
        return pd.DataFrame()
    group_cols = [
        "asset",
        "frequency",
        "method",
        "mode",
        "regime_method",
        "regime_label",
        "window_length",
        "feature_set",
    ]
    group_cols = [column for column in group_cols if column in success.columns]
    rows: list[dict[str, Any]] = []
    for keys, group in success.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values))
        row.update(
            {
                "number_of_motifs": int(len(group)),
                "mean_motif_distance_or_score": float(group["motif_distance"].mean()),
                "median_motif_distance": float(group["motif_distance"].median()),
                "recurrence_count": int(len(group) * 2),
                "runtime_seconds": float(group["runtime_seconds"].sum()),
                "time_split_stability": _time_split_stability(group, "motif_timestamp_1"),
                "cross_regime_overlap": np.nan,
                "notes": "Matrix Profile motif pairs; recurrence count treats both motif windows in each pair as occurrences.",
            }
        )
        if "cross_regime_pair" in group.columns:
            row["cross_regime_overlap"] = float(group["cross_regime_pair"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_locomotif_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    success = results[results.get("status", "success") == "success"].copy()
    if success.empty:
        return pd.DataFrame()
    group_cols = [
        "asset",
        "frequency",
        "method",
        "mode",
        "regime_method",
        "regime_label",
        "l_min",
        "l_max",
        "rho",
    ]
    group_cols = [column for column in group_cols if column in success.columns]
    rows: list[dict[str, Any]] = []
    for keys, group in success.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values))
        motif_sets = int(group["motif_set_rank"].nunique())
        row.update(
            {
                "number_of_motifs": motif_sets,
                "mean_motif_distance_or_score": float(group["motif_score"].mean()) if group["motif_score"].notna().any() else np.nan,
                "recurrence_count": int(len(group)),
                "mean_motif_length": float(group["motif_length"].mean()),
                "median_motif_length": float(group["motif_length"].median()),
                "runtime_seconds": float(group["runtime_seconds"].sum()),
                "time_split_stability": _time_split_stability(group, "motif_start_timestamp"),
                "cross_regime_overlap": np.nan,
                "notes": "Real LoCoMotif motif intervals parsed from apply_locomotif output.",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_mp_locomotif(mp_results: pd.DataFrame, loco_results: pd.DataFrame) -> pd.DataFrame:
    if mp_results.empty or loco_results.empty:
        return pd.DataFrame()
    mp = mp_results[mp_results.get("status", "success") == "success"].copy()
    loco = loco_results[loco_results.get("status", "success") == "success"].copy()
    if mp.empty or loco.empty:
        return pd.DataFrame()
    rows = []
    for keys, mp_group in mp.groupby(["asset", "frequency"], dropna=False):
        asset, frequency = keys
        loco_group = loco[(loco["asset"].astype(str) == str(asset)) & (loco["frequency"].astype(str) == str(frequency))]
        if loco_group.empty:
            continue
        overlaps = []
        for _, mp_row in mp_group.iterrows():
            mp_start = int(mp_row["motif_start_1"])
            mp_end = mp_start + int(mp_row["window_length"])
            for _, loco_row in loco_group.iterrows():
                overlaps.append(interval_overlap_ratio(mp_start, mp_end, int(loco_row["motif_start"]), int(loco_row["motif_end"])))
        rows.append(
            {
                "asset": asset,
                "frequency": frequency,
                "mp_motifs": int(len(mp_group)),
                "locomotif_instances": int(len(loco_group)),
                "mean_interval_overlap": float(np.mean(overlaps)) if overlaps else np.nan,
                "max_interval_overlap": float(np.max(overlaps)) if overlaps else np.nan,
            }
        )
    return pd.DataFrame(rows)


def thesis_key_results_table(
    mp_eval: pd.DataFrame,
    loco_eval: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    if not mp_eval.empty:
        mp = mp_eval.copy()
        mp["window_or_length_setting"] = mp.get("window_length", pd.Series(index=mp.index, dtype=object)).astype(str)
        frames.append(mp)
    if not loco_eval.empty:
        loco = loco_eval.copy()
        loco["window_or_length_setting"] = "l_min=" + loco.get("l_min", pd.Series(index=loco.index)).astype(str) + ",l_max=" + loco.get("l_max", pd.Series(index=loco.index)).astype(str)
        frames.append(loco)
    if not frames:
        return pd.DataFrame(
            columns=[
                "asset",
                "frequency",
                "method",
                "regime_method",
                "mode",
                "window_or_length_setting",
                "number_of_motifs",
                "mean_motif_distance_or_score",
                "recurrence_count",
                "cross_regime_overlap",
                "time_split_stability",
                "runtime_seconds",
                "notes",
            ]
        )
    combined = pd.concat(frames, ignore_index=True, sort=False)
    columns = [
        "asset",
        "frequency",
        "method",
        "regime_method",
        "mode",
        "window_or_length_setting",
        "number_of_motifs",
        "mean_motif_distance_or_score",
        "recurrence_count",
        "cross_regime_overlap",
        "time_split_stability",
        "runtime_seconds",
        "notes",
    ]
    for column in columns:
        if column not in combined.columns:
            combined[column] = np.nan
    return combined[columns]

