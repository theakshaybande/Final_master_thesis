from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd


def import_stumpy():
    try:
        import stumpy

        return stumpy
    except Exception as exc:
        raise RuntimeError(f"stumpy is not available. Install it with: pip install stumpy. Import error: {exc}") from exc


def gpu_available() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


def _timestamp_at(timestamps: pd.Series, position: int) -> Any:
    position = int(np.clip(position, 0, len(timestamps) - 1))
    value = timestamps.iloc[position]
    return pd.Timestamp(value).isoformat()


def _extract_top_pairs(
    profile_values: np.ndarray,
    profile_indices: np.ndarray,
    timestamps: pd.Series,
    window_length: int,
    top_k: int,
    exclusion_zone: int,
    context: dict[str, Any],
) -> pd.DataFrame:
    profile_values = np.asarray(profile_values, dtype=float)
    profile_indices = np.asarray(profile_indices, dtype=int)
    unavailable = np.zeros(len(profile_values), dtype=bool)
    rows: list[dict[str, Any]] = []
    for candidate in np.argsort(profile_values):
        candidate = int(candidate)
        if len(rows) >= top_k:
            break
        if candidate < 0 or candidate >= len(profile_values):
            continue
        if unavailable[candidate] or not np.isfinite(profile_values[candidate]):
            continue
        neighbor = int(profile_indices[candidate])
        if neighbor < 0 or neighbor >= len(profile_values):
            continue
        if abs(candidate - neighbor) < exclusion_zone or unavailable[neighbor]:
            continue
        row = {
            **context,
            "motif_rank": len(rows) + 1,
            "motif_start_1": candidate,
            "motif_start_2": neighbor,
            "motif_distance": float(profile_values[candidate]),
            "motif_timestamp_1": _timestamp_at(timestamps, candidate),
            "motif_timestamp_2": _timestamp_at(timestamps, neighbor),
            "motif_end_timestamp_1": _timestamp_at(timestamps, candidate + window_length - 1),
            "motif_end_timestamp_2": _timestamp_at(timestamps, neighbor + window_length - 1),
            "exclusion_zone": int(exclusion_zone),
            "status": "success",
        }
        rows.append(row)
        for center in [candidate, neighbor]:
            low = max(0, center - exclusion_zone)
            high = min(len(profile_values), center + exclusion_zone + 1)
            unavailable[low:high] = True
    return pd.DataFrame(rows)


def run_univariate_matrix_profile(
    series: np.ndarray,
    timestamps: pd.Series,
    window_length: int,
    top_k: int,
    context: dict[str, Any],
    use_gpu: bool = False,
    exclusion_zone_factor: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stumpy = import_stumpy()
    arr = np.asarray(series, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if len(arr) < 2 * int(window_length) + 2:
        raise ValueError(f"Series length {len(arr)} is too short for Matrix Profile window {window_length}.")
    if np.nanstd(arr) == 0:
        raise ValueError("Series has zero variance after cleaning.")

    t0 = time.perf_counter()
    used_gpu = False
    if use_gpu and hasattr(stumpy, "gpu_stump") and gpu_available():
        try:
            mp = stumpy.gpu_stump(arr, int(window_length))
            used_gpu = True
        except Exception:
            mp = stumpy.stump(arr, int(window_length))
    else:
        mp = stumpy.stump(arr, int(window_length))
    runtime = time.perf_counter() - t0
    profile_values = np.asarray(mp[:, 0], dtype=float)
    profile_indices = np.asarray(mp[:, 1], dtype=int)
    exclusion_zone = max(1, int(window_length * exclusion_zone_factor))
    row_context = {
        **context,
        "method": "matrix_profile",
        "window_length": int(window_length),
        "runtime_seconds": float(runtime),
        "n_observations": int(len(arr)),
        "used_gpu": bool(used_gpu),
    }
    motifs = _extract_top_pairs(profile_values, profile_indices, timestamps, window_length, top_k, exclusion_zone, row_context)
    profile = pd.DataFrame({"profile_index": np.arange(len(profile_values)), "matrix_profile": profile_values, "neighbor_index": profile_indices})
    return motifs, profile


def run_multivariate_matrix_profile(
    X: np.ndarray,
    timestamps: pd.Series,
    feature_columns: list[str],
    window_length: int,
    top_k: int,
    context: dict[str, Any],
    exclusion_zone_factor: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stumpy = import_stumpy()
    matrix = np.asarray(X, dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a time x feature matrix; got shape {matrix.shape}.")
    if len(matrix) < 2 * int(window_length) + 2:
        raise ValueError(f"Matrix length {len(matrix)} is too short for Matrix Profile window {window_length}.")
    if matrix.shape[1] < 2:
        raise ValueError("Multivariate Matrix Profile requires at least two channels.")

    t0 = time.perf_counter()
    profile_matrix, index_matrix = stumpy.mstump(matrix.T, int(window_length))
    runtime = time.perf_counter() - t0
    dimension_row = min(matrix.shape[1] - 1, profile_matrix.shape[0] - 1)
    profile_values = np.asarray(profile_matrix[dimension_row], dtype=float)
    profile_indices = np.asarray(index_matrix[dimension_row], dtype=int)
    exclusion_zone = max(1, int(window_length * exclusion_zone_factor))
    row_context = {
        **context,
        "method": "matrix_profile",
        "window_length": int(window_length),
        "runtime_seconds": float(runtime),
        "n_observations": int(len(matrix)),
        "used_gpu": False,
        "mstump_dimension_row": int(dimension_row),
        "feature_set": ",".join(feature_columns),
    }
    motifs = _extract_top_pairs(profile_values, profile_indices, timestamps, window_length, top_k, exclusion_zone, row_context)
    profile = pd.DataFrame({"profile_index": np.arange(len(profile_values)), "matrix_profile": profile_values, "neighbor_index": profile_indices})
    return motifs, profile

