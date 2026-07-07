from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "locomotif_controlled_slice_comparison"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"
CONFIG_DIR = OUT_DIR / "configs"
LATEX_DIR = OUT_DIR / "latex"
LOCOMOTIF_TIMEOUT_SECONDS = 120

BTC_15M = ROOT / "final_dataset" / "features" / "crypto" / "BTCUSDT_15m_features_2020_2025.parquet"
BTC_1H = ROOT / "final_dataset" / "features" / "crypto" / "BTCUSDT_1h_features_2020_2025.parquet"
KNOWN_REGIME_LABELS = [
    ROOT / "reports" / "results" / "regime_studies" / "01_volatility_quantile_regimes" / "regime_labels" / "BTCUSDT_quantile_regime_labels.parquet",
    ROOT / "HPC workflow" / "HPC_Regime_and_motif_discovery" / "results" / "regimes" / "quantile" / "quantile_regime_labels_LOCAL_SMOKE_TEST.parquet",
]

LOCAL_REFUSAL_MESSAGE = (
    "Refusing to run LoCoMotif locally. Submit this job on HPC with SLURM, "
    "or pass --allow-local for a deliberate tiny local smoke test."
)


def configure_output_root(output_root: str | None) -> None:
    global OUT_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR, CONFIG_DIR, LATEX_DIR
    if not output_root:
        return
    candidate = Path(output_root).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    OUT_DIR = candidate
    TABLE_DIR = OUT_DIR / "tables"
    FIGURE_DIR = OUT_DIR / "figures"
    LOG_DIR = OUT_DIR / "logs"
    CONFIG_DIR = OUT_DIR / "configs"
    LATEX_DIR = OUT_DIR / "latex"


def is_local_windows_or_user_path() -> bool:
    root_text = str(ROOT).replace("\\", "/").lower()
    return platform.system().lower() == "windows" or "c:/users/learn" in root_text or "/users/learn" in root_text


def is_hpc_environment(hpc_flag: bool) -> bool:
    root_text = str(ROOT).replace("\\", "/")
    return (
        hpc_flag
        or bool(os.environ.get("SLURM_JOB_ID"))
        or (platform.system().lower() == "linux" and "Final_master_thesis" in root_text)
    )


def enforce_execution_safety(args: argparse.Namespace) -> int | None:
    hpc_env = is_hpc_environment(args.hpc)
    local_env = is_local_windows_or_user_path()
    if args.run_locomotif and local_env and not hpc_env and not args.allow_local:
        print(LOCAL_REFUSAL_MESSAGE)
        return 2
    if args.run_mp and local_env and not hpc_env:
        tiny_smoke = args.mode == "smoke" and args.max_points <= 300
        if not tiny_smoke and not args.allow_local:
            print("Refusing to run non-tiny Matrix Profile locally. Use --mode smoke --max-points 300 or submit on HPC.")
            return 2
    return None


def ensure_dirs() -> None:
    for directory in [TABLE_DIR, FIGURE_DIR, LOG_DIR, CONFIG_DIR, LATEX_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def yn(value: str) -> bool:
    value = value.lower().strip()
    if value in {"yes", "y", "true", "1"}:
        return True
    if value in {"no", "n", "false", "0"}:
        return False
    raise argparse.ArgumentTypeError("Use yes or no.")


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def ensure_base_csvs() -> None:
    base_tables = {
        "locomotif_controlled_motif_sets.csv": [
            "run_key",
            "slice_id",
            "asset",
            "frequency",
            "regime_label",
            "feature",
            "motif_set_id",
            "motif_set_rank",
            "score_or_quality",
            "occurrence_count",
            "representative_start",
            "representative_end",
            "representative_start_timestamp",
            "representative_end_timestamp",
            "mean_interval_length",
            "median_interval_length",
            "l_min",
            "l_max",
            "rho",
            "nb",
            "overlap",
            "warping",
        ],
        "locomotif_controlled_occurrences.csv": [
            "run_key",
            "slice_id",
            "asset",
            "frequency",
            "regime_label",
            "feature",
            "motif_set_id",
            "occurrence_id",
            "role",
            "occurrence_start",
            "occurrence_end",
            "occurrence_length",
            "occurrence_start_timestamp",
            "occurrence_end_timestamp",
            "l_min",
            "l_max",
            "rho",
            "nb",
            "overlap",
            "warping",
        ],
    }
    for filename, columns in base_tables.items():
        path = TABLE_DIR / filename
        if not path.exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False)


def append_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        if not path.exists():
            df.to_csv(path, index=False)
        return
    if path.exists():
        previous = pd.read_csv(path)
        if "run_key" in df.columns:
            incoming_keys = set(df["run_key"].dropna().astype(str))
            previous = previous[~previous["run_key"].astype(str).isin(incoming_keys)]
        df = pd.concat([previous, df], ignore_index=True)
    df.to_csv(path, index=False)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_feature_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input data file not found: {path}")
    df = pd.read_parquet(path).copy()
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} does not contain timestamp.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    if "rolling_volatility_240" not in df.columns:
        df["rolling_volatility_240"] = df["log_return"].rolling(240).std()
    if "rolling_volatility_60" not in df.columns:
        df["rolling_volatility_60"] = df["log_return"].rolling(60).std()
    return df


def find_exact_15m_labels(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    for path in KNOWN_REGIME_LABELS:
        if not path.exists():
            continue
        try:
            labels = read_table(path)
        except Exception:
            continue
        if "timestamp" not in labels.columns:
            continue
        label_col = None
        for candidate in ["regime_label", "regime_quantile_label"]:
            if candidate in labels.columns:
                label_col = candidate
                break
        if label_col is None:
            continue
        labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
        merged = feature_df[["timestamp"]].merge(labels[["timestamp", label_col]], on="timestamp", how="left")
        coverage = merged[label_col].notna().mean()
        interval_seconds = feature_df["timestamp"].diff().dt.total_seconds().dropna().median()
        if coverage > 0.90 and 800 <= interval_seconds <= 1000:
            out = labels[["timestamp", label_col]].rename(columns={label_col: "regime_label"})
            return out, str(path), "found_exact_15m_labels"
    return pd.DataFrame(), "", "not_found"


def regenerate_15m_labels(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    labels = feature_df[["timestamp", "close", "log_return", "rolling_volatility_240"]].copy()
    labels = labels.dropna(subset=["rolling_volatility_240"]).reset_index(drop=True)
    threshold = float(labels["rolling_volatility_240"].median())
    labels["regime_method"] = "quantile_2_rolling_240"
    labels["regime_label"] = np.where(labels["rolling_volatility_240"] <= threshold, "low_vol", "high_vol")
    labels["volatility_threshold_q50"] = threshold
    out = TABLE_DIR / "btcusdt_15m_quantile2rolling240_regenerated_labels.csv"
    labels.to_csv(out, index=False)
    return labels[["timestamp", "regime_method", "regime_label", "rolling_volatility_240", "volatility_threshold_q50"]], str(out), "regenerated_for_controlled_experiment"


def add_regime_labels(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    found, source, status = find_exact_15m_labels(feature_df)
    if found.empty:
        labels, source, status = regenerate_15m_labels(feature_df)
    else:
        labels = found
    merged = feature_df.merge(labels, on="timestamp", how="left", suffixes=("", "_label"))
    if "regime_method" not in merged.columns:
        merged["regime_method"] = "quantile_2_rolling_240"
    return merged, source, status


def contiguous_segments(df: pd.DataFrame, label: str) -> list[tuple[int, int]]:
    mask = (df["regime_label"] == label).fillna(False).to_numpy()
    segments: list[tuple[int, int]] = []
    start = None
    for idx, ok in enumerate(mask):
        if ok and start is None:
            start = idx
        if (not ok or idx == len(mask) - 1) and start is not None:
            end = idx + 1 if ok and idx == len(mask) - 1 else idx
            segments.append((start, end))
            start = None
    return segments


def choose_regime_slice(df: pd.DataFrame, label: str, max_points: int) -> tuple[pd.DataFrame, str]:
    valid = df.dropna(subset=["regime_label", "close", "rolling_volatility_240"]).reset_index(drop=True)
    segments = contiguous_segments(valid, label)
    if not segments:
        raise ValueError(f"No contiguous {label} segment was found.")
    longest = max(segments, key=lambda pair: pair[1] - pair[0])
    length = longest[1] - longest[0]
    if length >= 1000:
        start, end = longest
        rule = f"longest_contiguous_{label}_segment"
    else:
        cluster_mid = (longest[0] + longest[1]) // 2
        window = min(max_points, len(valid), 3000)
        start = max(0, cluster_mid - window // 2)
        end = min(len(valid), start + window)
        start = max(0, end - window)
        rule = f"calendar_window_around_largest_{label}_cluster"
    selected = valid.iloc[start:end].head(max_points).copy().reset_index(drop=True)
    return selected, rule


def choose_agnostic_1h_slice(df: pd.DataFrame, max_points: int) -> tuple[pd.DataFrame, str]:
    valid = df.dropna(subset=["close"]).reset_index(drop=True)
    start = min(240, max(0, len(valid) - max_points))
    selected = valid.iloc[start : start + max_points].copy().reset_index(drop=True)
    selected["regime_method"] = "agnostic"
    selected["regime_label"] = "agnostic"
    return selected, "first_valid_contiguous_1h_window_after_warmup"


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = float(values.std(ddof=0))
    if not math.isfinite(std) or std == 0.0:
        raise ValueError(f"Cannot z-normalize {series.name}; standard deviation is zero or invalid.")
    return (values - float(values.mean())) / std


def prepare_slice(slice_name: str, feature: str, max_points: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if slice_name in {"high_vol", "low_vol"}:
        source = BTC_15M
        raw = load_feature_data(source)
        labeled, label_source, label_status = add_regime_labels(raw)
        selected, rule = choose_regime_slice(labeled, slice_name, max_points)
        frequency = "15m"
        mode = "conditioned"
        csv_name = f"btcusdt_15m_{slice_name}_slice_input.csv"
    elif slice_name == "agnostic_1h":
        source = BTC_1H
        raw = load_feature_data(source)
        selected, rule = choose_agnostic_1h_slice(raw, max_points)
        label_source = "not_applicable"
        label_status = "agnostic"
        frequency = "1h"
        mode = "agnostic"
        csv_name = "btcusdt_1h_agnostic_slice_input.csv"
    else:
        raise ValueError(f"Unsupported slice: {slice_name}")

    selected = selected.copy()
    selected["close_z"] = zscore(selected["close"])
    selected["log_return_z"] = zscore(selected["log_return"].fillna(0.0))
    selected["rolling_volatility_60_z"] = zscore(selected["rolling_volatility_60"].fillna(selected["rolling_volatility_60"].median()))
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, "timestamp"]).reset_index(drop=True)
    selected.to_csv(TABLE_DIR / csv_name, index=False)
    metadata = {
        "slice_id": slice_name,
        "asset": "BTCUSDT",
        "frequency": frequency,
        "mode": mode,
        "regime_method": "quantile_2_rolling_240" if mode == "conditioned" else "agnostic",
        "regime_label": slice_name if mode == "conditioned" else "agnostic",
        "start_timestamp": selected["timestamp"].iloc[0].isoformat(),
        "end_timestamp": selected["timestamp"].iloc[-1].isoformat(),
        "n_observations": int(len(selected)),
        "selection_rule": rule,
        "max_points": int(max_points),
        "feature_columns": ["close_z", "log_return_z", "rolling_volatility_60_z"],
        "source_data_file": rel(source),
        "regime_label_source": label_source,
        "regime_label_status": label_status,
        "notes": "Contiguous chronological slice; no non-contiguous regime concatenation.",
    }
    update_slice_metadata(metadata)
    return selected, metadata


def update_slice_metadata(metadata: dict[str, Any]) -> None:
    path = CONFIG_DIR / "selected_slices_metadata.json"
    current: list[dict[str, Any]] = []
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current = [item for item in current if item.get("slice_id") != metadata["slice_id"]]
    current.append(metadata)
    save_json(path, sorted(current, key=lambda item: item["slice_id"]))


def import_locomotif_module() -> tuple[str, Any]:
    errors: dict[str, str] = {}
    for module_name in ["locomotif.locomotif", "locomotif"]:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "apply_locomotif"):
                return module_name, module
            errors[module_name] = "apply_locomotif missing"
        except Exception as exc:
            errors[module_name] = repr(exc)
    raise RuntimeError(f"Real dtai-locomotif apply_locomotif is unavailable: {errors}")


def locomotif_worker(series: np.ndarray, params: dict[str, Any], queue: mp.Queue) -> None:
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    try:
        module_name, module = import_locomotif_module()
        signature = str(inspect.signature(module.apply_locomotif))
        t0 = time.perf_counter()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            motif_sets = module.apply_locomotif(
                series,
                l_min=params["l_min"],
                l_max=params["l_max"],
                rho=params["rho"],
                nb=params["nb"],
                overlap=params["overlap"],
                warping=params["warping"],
            )
        queue.put(
            {
                "success": True,
                "module": module_name,
                "signature": signature,
                "runtime_seconds": time.perf_counter() - t0,
                "motif_sets": motif_sets,
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
            }
        )
    except Exception as exc:
        queue.put(
            {
                "success": False,
                "error_message": repr(exc),
                "traceback": traceback.format_exc(),
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
            }
        )


def locomotif_params(frequency: str, rho: float, lmin: int | None = None, lmax: int | None = None, nb: int | None = None) -> dict[str, Any]:
    if frequency == "1h":
        defaults = {"l_min": 24, "l_max": 72, "rho": float(rho), "nb": 10, "overlap": 0.20, "warping": True}
    else:
        defaults = {"l_min": 32, "l_max": 128, "rho": float(rho), "nb": 10, "overlap": 0.20, "warping": True}
    if lmin is not None:
        defaults["l_min"] = int(lmin)
    if lmax is not None:
        defaults["l_max"] = int(lmax)
    if nb is not None:
        defaults["nb"] = int(nb)
    return defaults


def raw_repr(obj: Any) -> str:
    try:
        return repr(obj)
    except Exception as exc:
        return f"<repr failed: {exc}>"


def validate_interval(interval: Any) -> tuple[int, int]:
    if not isinstance(interval, (tuple, list)) or len(interval) != 2:
        raise ValueError(f"Expected interval pair, got {interval!r}")
    start, end = int(interval[0]), int(interval[1])
    if end <= start:
        raise ValueError(f"Invalid interval {interval!r}")
    return start, end


def parse_locomotif_result(motif_sets: Any, timestamps: pd.Series, context: dict[str, Any], params: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if motif_sets is None:
        return pd.DataFrame(), pd.DataFrame()
    if not isinstance(motif_sets, list):
        raise ValueError(f"LoCoMotif returned {type(motif_sets)}, expected list.")
    set_rows: list[dict[str, Any]] = []
    occurrence_rows: list[dict[str, Any]] = []
    for set_idx, item in enumerate(motif_sets, start=1):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"Motif set {set_idx} has unsupported structure: {item!r}")
        representative, occurrences = item
        rep_start, rep_end = validate_interval(representative)
        intervals = [("representative", rep_start, rep_end)]
        for occurrence in list(occurrences):
            start, end = validate_interval(occurrence)
            intervals.append(("occurrence", start, end))
        lengths = [end - start for _, start, end in intervals]
        set_rows.append(
            {
                **context,
                "motif_set_id": set_idx,
                "motif_set_rank": set_idx,
                "score_or_quality": np.nan,
                "occurrence_count": len(intervals),
                "representative_start": rep_start,
                "representative_end": rep_end,
                "representative_start_timestamp": timestamps.iloc[min(rep_start, len(timestamps) - 1)].isoformat(),
                "representative_end_timestamp": timestamps.iloc[min(rep_end - 1, len(timestamps) - 1)].isoformat(),
                "mean_interval_length": float(np.mean(lengths)),
                "median_interval_length": float(np.median(lengths)),
                **params,
            }
        )
        for occurrence_id, (role, start, end) in enumerate(intervals, start=1):
            occurrence_rows.append(
                {
                    **context,
                    "motif_set_id": set_idx,
                    "occurrence_id": occurrence_id,
                    "role": role,
                    "occurrence_start": start,
                    "occurrence_end": end,
                    "occurrence_length": end - start,
                    "occurrence_start_timestamp": timestamps.iloc[min(start, len(timestamps) - 1)].isoformat(),
                    "occurrence_end_timestamp": timestamps.iloc[min(end - 1, len(timestamps) - 1)].isoformat(),
                    **params,
                }
            )
    return pd.DataFrame(set_rows), pd.DataFrame(occurrence_rows)


def run_locomotif(
    slice_df: pd.DataFrame,
    metadata: dict[str, Any],
    feature: str,
    rho: float,
    lmin: int | None = None,
    lmax: int | None = None,
    nb: int | None = None,
    timeout_seconds: int | None = None,
    output_prefix: str = "",
) -> dict[str, Any]:
    params = locomotif_params(metadata["frequency"], rho, lmin=lmin, lmax=lmax, nb=nb)
    context = {
        "run_key": f"{metadata['slice_id']}_{feature}_rho{rho}_n{len(slice_df)}",
        "slice_id": metadata["slice_id"],
        "asset": "BTCUSDT",
        "frequency": metadata["frequency"],
        "regime_label": metadata["regime_label"],
        "feature": feature,
    }
    raw_path = TABLE_DIR / f"locomotif_raw_output_{metadata['slice_id']}_{feature}_rho{str(rho).replace('.', 'p')}.txt"
    runtime_row = {**context, **params, "start_time": pd.Timestamp.now("UTC").isoformat(), "success": False}
    try:
        series = slice_df[[feature]].to_numpy(dtype=np.float32)
        queue: mp.Queue = mp.Queue()
        process = mp.Process(target=locomotif_worker, args=(series, params, queue))
        process.start()
        timeout_limit = int(timeout_seconds or LOCOMOTIF_TIMEOUT_SECONDS)
        process.join(timeout_limit)
        if process.is_alive():
            process.terminate()
            process.join(10)
            raise TimeoutError(f"Real LoCoMotif call exceeded {timeout_limit} seconds.")
        if queue.empty():
            raise RuntimeError(f"Real LoCoMotif worker exited with code {process.exitcode} without returning output.")
        worker_result = queue.get()
        raw_path.write_text(
            "# stdout\n"
            + worker_result.get("stdout", "")
            + "\n# stderr\n"
            + worker_result.get("stderr", "")
            + "\n# repr(motif_sets)\n"
            + raw_repr(worker_result.get("motif_sets")),
            encoding="utf-8",
        )
        if not worker_result.get("success"):
            raise RuntimeError(worker_result.get("traceback") or worker_result.get("error_message", "LoCoMotif worker failed."))
        motif_sets = worker_result["motif_sets"]
        runtime = float(worker_result["runtime_seconds"])
        set_df, occ_df = parse_locomotif_result(motif_sets, slice_df["timestamp"], context, params)
        append_csv(TABLE_DIR / "locomotif_controlled_motif_sets.csv", set_df)
        append_csv(TABLE_DIR / "locomotif_controlled_occurrences.csv", occ_df)
        if output_prefix == "micro":
            append_csv(TABLE_DIR / "micro_locomotif_motif_sets.csv", set_df)
        runtime_row.update(
            {
                "end_time": pd.Timestamp.now("UTC").isoformat(),
                "runtime_seconds": runtime,
                "success": True,
                "error_message": "",
                "module": worker_result.get("module", ""),
                "apply_locomotif_signature": worker_result.get("signature", ""),
                "raw_motif_sets_count": len(motif_sets),
                "filtered_motif_sets_count": len(set_df),
                "occurrence_count": len(occ_df),
                "raw_output_file": rel(raw_path),
            }
        )
    except Exception as exc:
        raw_path.write_text(traceback.format_exc(), encoding="utf-8")
        runtime_row.update(
            {
                "end_time": pd.Timestamp.now("UTC").isoformat(),
                "runtime_seconds": np.nan,
                "success": False,
                "error_message": repr(exc),
                "module": "",
                "raw_motif_sets_count": 0,
                "filtered_motif_sets_count": 0,
                "occurrence_count": 0,
                "raw_output_file": rel(raw_path),
            }
        )
    append_csv(TABLE_DIR / "locomotif_controlled_runtime.csv", [runtime_row])
    if output_prefix == "micro":
        append_csv(TABLE_DIR / "micro_locomotif_runtime.csv", [runtime_row])
        if not bool(runtime_row["success"]):
            append_csv(
                TABLE_DIR / "micro_locomotif_failure_summary.csv",
                [
                    {
                        "asset": "BTCUSDT",
                        "frequency": metadata["frequency"],
                        "max_points": metadata["n_observations"],
                        "lmin": params["l_min"],
                        "lmax": params["l_max"],
                        "rho": params["rho"],
                        "nb": params["nb"],
                        "timeout_seconds": int(timeout_seconds or LOCOMOTIF_TIMEOUT_SECONDS),
                        "status": "failed",
                        "error_message": runtime_row["error_message"],
                    }
                ],
            )
    save_locomotif_parameter_mapping()
    return runtime_row


def save_locomotif_parameter_mapping() -> None:
    save_json(
        CONFIG_DIR / "locomotif_parameter_mapping.json",
        {
            "implementation": "locomotif.locomotif.apply_locomotif",
            "package": "dtai-locomotif",
            "mapping": {
                "lmin": "l_min",
                "lmax": "l_max",
                "rho": "rho",
                "max motif sets or nb": "nb",
                "overlap": "overlap",
                "warping enabled": "warping=True",
            },
        },
    )


def matrix_profile_windows(frequency: str) -> list[int]:
    return [24, 48, 72] if frequency == "1h" else [32, 64, 128]


def run_matrix_profile(slice_df: pd.DataFrame, metadata: dict[str, Any], feature: str) -> None:
    try:
        import stumpy
    except Exception as exc:
        rows = []
        for window in matrix_profile_windows(metadata["frequency"]):
            rows.append(
                {
                    "run_key": f"{metadata['slice_id']}_{feature}_m{window}_n{len(slice_df)}",
                    "slice_id": metadata["slice_id"],
                    "asset": "BTCUSDT",
                    "frequency": metadata["frequency"],
                    "regime_label": metadata["regime_label"],
                    "feature": feature,
                    "window_length": window,
                    "success": False,
                    "runtime_seconds": np.nan,
                    "error_message": repr(exc),
                }
            )
        append_csv(TABLE_DIR / "mp_controlled_slice_runtime.csv", rows)
        return

    series = slice_df[feature].to_numpy(dtype=float)
    runtime_rows: list[dict[str, Any]] = []
    motif_rows: list[dict[str, Any]] = []
    for window in matrix_profile_windows(metadata["frequency"]):
        context = {
            "run_key": f"{metadata['slice_id']}_{feature}_m{window}_n{len(slice_df)}",
            "slice_id": metadata["slice_id"],
            "asset": "BTCUSDT",
            "frequency": metadata["frequency"],
            "regime_label": metadata["regime_label"],
            "feature": feature,
            "window_length": window,
        }
        try:
            t0 = time.perf_counter()
            profile = stumpy.stump(series, m=window)
            runtime = time.perf_counter() - t0
            distances = pd.to_numeric(pd.Series(profile[:, 0]), errors="coerce").replace([np.inf, -np.inf], np.nan)
            best_idx = int(distances.idxmin())
            neighbor_idx = int(profile[best_idx, 1])
            best_distance = float(distances.iloc[best_idx])
            runtime_rows.append({**context, "success": True, "runtime_seconds": runtime, "error_message": ""})
            motif_rows.append(
                {
                    **context,
                    "runtime_seconds": runtime,
                    "best_motif_distance": best_distance,
                    "mean_motif_distance": float(distances.mean()),
                    "median_motif_distance": float(distances.median()),
                    "motif_start_1": best_idx,
                    "motif_start_2": neighbor_idx,
                    "motif_timestamp_1": slice_df["timestamp"].iloc[best_idx].isoformat(),
                    "motif_timestamp_2": slice_df["timestamp"].iloc[neighbor_idx].isoformat(),
                }
            )
        except Exception as exc:
            runtime_rows.append({**context, "success": False, "runtime_seconds": np.nan, "error_message": repr(exc)})
    append_csv(TABLE_DIR / "mp_controlled_slice_runtime.csv", runtime_rows)
    append_csv(TABLE_DIR / "mp_controlled_slice_motifs.csv", motif_rows)
    summary = pd.DataFrame(motif_rows)
    if not summary.empty:
        summary.to_csv(TABLE_DIR / "mp_controlled_slice_summary.csv", index=False)


def plot_slice_overview(slice_df: pd.DataFrame, metadata: dict[str, Any]) -> None:
    if metadata["frequency"] != "15m":
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(slice_df["timestamp"], slice_df["close"], color="#234F73", linewidth=1.0)
    axes[0].set_ylabel("Close")
    axes[1].plot(slice_df["timestamp"], slice_df["rolling_volatility_240"], color="#B55D2A", linewidth=1.0)
    axes[1].set_ylabel("Rolling vol 240")
    axes[1].set_xlabel("Timestamp")
    fig.suptitle(
        f"BTCUSDT {metadata['frequency']} {metadata['regime_label']} slice: "
        f"{metadata['start_timestamp']} to {metadata['end_timestamp']} ({metadata['n_observations']} obs.)"
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"btcusdt_15m_{metadata['regime_label']}_slice_overview.png", dpi=300)
    plt.close(fig)


def get_locomotif_occurrences(slice_id: str, feature: str) -> pd.DataFrame:
    occ_path = TABLE_DIR / "locomotif_controlled_occurrences.csv"
    if not occ_path.exists():
        return pd.DataFrame()
    occ = pd.read_csv(occ_path)
    occ = occ[(occ["slice_id"] == slice_id) & (occ["feature"] == feature)]
    if occ.empty:
        return occ
    rho_values = sorted(occ["rho"].dropna().unique())
    if rho_values:
        occ = occ[occ["rho"] == rho_values[0]]
    return occ


def plot_locomotif_overlay(slice_df: pd.DataFrame, metadata: dict[str, Any], feature: str) -> None:
    if metadata["frequency"] != "15m":
        return
    occ = get_locomotif_occurrences(metadata["slice_id"], feature)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(slice_df["timestamp"], slice_df[feature], color="#234F73", linewidth=0.8)
    if not occ.empty:
        top_set = int(occ["motif_set_id"].min())
        top = occ[occ["motif_set_id"] == top_set]
        for _, row in top.iterrows():
            start = int(row["occurrence_start"])
            end = int(row["occurrence_end"])
            ax.axvspan(slice_df["timestamp"].iloc[start], slice_df["timestamp"].iloc[min(end - 1, len(slice_df) - 1)], color="#D95F02", alpha=0.24)
        rho = float(top["rho"].iloc[0])
        l_min = int(top["l_min"].iloc[0])
        l_max = int(top["l_max"].iloc[0])
        title_tail = f"lmin={l_min}, lmax={l_max}, rho={rho}, motif sets={occ['motif_set_id'].nunique()}, occurrences={len(occ)}"
    else:
        title_tail = "no successful LoCoMotif intervals available"
    ax.set_title(f"LoCoMotif BTCUSDT 15m {metadata['regime_label']} top motif set ({title_tail})")
    ax.set_ylabel(feature)
    ax.set_xlabel("Timestamp")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"locomotif_btcusdt_15m_{metadata['regime_label']}_top_motif_set.png", dpi=300)
    plt.close(fig)


def get_best_mp(slice_id: str, feature: str) -> pd.Series | None:
    path = TABLE_DIR / "mp_controlled_slice_motifs.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[(df["slice_id"] == slice_id) & (df["feature"] == feature)]
    if df.empty:
        return None
    return df.sort_values("best_motif_distance").iloc[0]


def plot_mp_overlay(slice_df: pd.DataFrame, metadata: dict[str, Any], feature: str) -> None:
    if metadata["frequency"] != "15m":
        return
    best = get_best_mp(metadata["slice_id"], feature)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(slice_df["timestamp"], slice_df[feature], color="#234F73", linewidth=0.8)
    if best is not None:
        window = int(best["window_length"])
        for start in [int(best["motif_start_1"]), int(best["motif_start_2"])]:
            end = min(start + window - 1, len(slice_df) - 1)
            ax.axvspan(slice_df["timestamp"].iloc[start], slice_df["timestamp"].iloc[end], color="#1B9E77", alpha=0.25)
        title_tail = f"window={window}, best distance={float(best['best_motif_distance']):.4f}"
    else:
        title_tail = "no successful Matrix Profile motif available"
    ax.set_title(f"Matrix Profile BTCUSDT 15m {metadata['regime_label']} top motif overlay ({title_tail})")
    ax.set_ylabel(feature)
    ax.set_xlabel("Timestamp")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"mp_btcusdt_15m_{metadata['regime_label']}_top_motif_overlay.png", dpi=300)
    plt.close(fig)


def plot_side_by_side(slice_df: pd.DataFrame, metadata: dict[str, Any], feature: str) -> None:
    if metadata["frequency"] != "15m":
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax in axes:
        ax.plot(slice_df["timestamp"], slice_df[feature], color="#234F73", linewidth=0.7)
        ax.tick_params(axis="x", rotation=30)
    best = get_best_mp(metadata["slice_id"], feature)
    if best is not None:
        window = int(best["window_length"])
        for start in [int(best["motif_start_1"]), int(best["motif_start_2"])]:
            axes[0].axvspan(slice_df["timestamp"].iloc[start], slice_df["timestamp"].iloc[min(start + window - 1, len(slice_df) - 1)], color="#1B9E77", alpha=0.25)
        axes[0].set_title(f"Matrix Profile fixed-length pair, m={window}")
    else:
        axes[0].set_title("Matrix Profile fixed-length pair unavailable")
    occ = get_locomotif_occurrences(metadata["slice_id"], feature)
    if not occ.empty:
        top = occ[occ["motif_set_id"] == int(occ["motif_set_id"].min())]
        for _, row in top.iterrows():
            axes[1].axvspan(
                slice_df["timestamp"].iloc[int(row["occurrence_start"])],
                slice_df["timestamp"].iloc[min(int(row["occurrence_end"]) - 1, len(slice_df) - 1)],
                color="#D95F02",
                alpha=0.24,
            )
        axes[1].set_title("LoCoMotif time-warped motif set")
    else:
        axes[1].set_title("LoCoMotif motif set unavailable")
    fig.suptitle(f"MP fixed-length motif pair versus LoCoMotif time-warped motif set: {metadata['regime_label']}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"mp_vs_locomotif_{metadata['regime_label']}_side_by_side.png", dpi=300)
    plt.close(fig)


def plot_summary_charts() -> None:
    loco_rt = read_table(TABLE_DIR / "locomotif_controlled_runtime.csv")
    mp_rt = read_table(TABLE_DIR / "mp_controlled_slice_runtime.csv")
    loco_sets = read_table(TABLE_DIR / "locomotif_controlled_motif_sets.csv")
    occ = read_table(TABLE_DIR / "locomotif_controlled_occurrences.csv")
    mp_motifs = read_table(TABLE_DIR / "mp_controlled_slice_motifs.csv")

    if not loco_rt.empty or not mp_rt.empty:
        rows = []
        if not loco_rt.empty:
            rows.extend(loco_rt[loco_rt["success"] == True].assign(method="LoCoMotif")[["slice_id", "method", "runtime_seconds"]].to_dict("records"))
        if not mp_rt.empty:
            rows.extend(mp_rt[mp_rt["success"] == True].assign(method="Matrix Profile")[["slice_id", "method", "runtime_seconds"]].to_dict("records"))
        runtime_df = pd.DataFrame(rows)
        if not runtime_df.empty:
            pivot = runtime_df.groupby(["slice_id", "method"])["runtime_seconds"].mean().unstack(fill_value=0)
            ax = pivot.plot(kind="bar", figsize=(9, 5), color=["#1B9E77", "#D95F02"])
            ax.set_ylabel("Runtime seconds")
            ax.set_title("Runtime on matched controlled slices")
            ax.figure.tight_layout()
            ax.figure.savefig(FIGURE_DIR / "mp_vs_locomotif_runtime.png", dpi=300)
            plt.close(ax.figure)

    count_rows = []
    if not loco_sets.empty:
        for slice_id, group in loco_sets.groupby("slice_id"):
            count_rows.append({"slice_id": slice_id, "method": "LoCoMotif", "count": group["motif_set_id"].nunique()})
    if not mp_motifs.empty:
        for slice_id, group in mp_motifs.groupby("slice_id"):
            count_rows.append({"slice_id": slice_id, "method": "Matrix Profile", "count": len(group)})
    count_df = pd.DataFrame(count_rows)
    if not count_df.empty:
        pivot = count_df.pivot(index="slice_id", columns="method", values="count").fillna(0)
        ax = pivot.plot(kind="bar", figsize=(9, 5), color=["#1B9E77", "#D95F02"])
        ax.set_ylabel("Count")
        ax.set_title("Counts are different object types: MP motif candidates vs LoCoMotif motif sets")
        ax.figure.tight_layout()
        ax.figure.savefig(FIGURE_DIR / "mp_vs_locomotif_counts.png", dpi=300)
        plt.close(ax.figure)

    if not occ.empty and "occurrence_length" in occ.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(pd.to_numeric(occ["occurrence_length"], errors="coerce").dropna(), bins=20, color="#7570B3", edgecolor="white")
        ax.set_title("LoCoMotif occurrence interval length distribution")
        ax.set_xlabel("Interval length")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "locomotif_interval_length_distribution.png", dpi=300)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No LoCoMotif occurrence intervals were returned.\nAll real LoCoMotif calls failed or timed out.", ha="center", va="center")
        ax.set_title("LoCoMotif occurrence interval length distribution")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "locomotif_interval_length_distribution.png", dpi=300)
        plt.close(fig)

    if not loco_rt.empty and "rho" in loco_rt.columns:
        df = loco_rt[loco_rt["success"] == True].copy()
        if not df.empty:
            pivot = df.groupby(["rho", "slice_id"])["runtime_seconds"].mean().unstack(fill_value=0)
            ax = pivot.plot(kind="bar", figsize=(9, 5))
            ax.set_ylabel("Runtime seconds")
            ax.set_title("LoCoMotif runtime by rho")
            ax.figure.tight_layout()
            ax.figure.savefig(FIGURE_DIR / "locomotif_runtime_by_rho.png", dpi=300)
            plt.close(ax.figure)
        else:
            failures = loco_rt.groupby(["rho", "slice_id"]).size().unstack(fill_value=0)
            ax = failures.plot(kind="bar", figsize=(9, 5))
            ax.set_ylabel("Timed-out or failed runs")
            ax.set_title("LoCoMotif runtime by rho unavailable: real calls did not complete")
            ax.figure.tight_layout()
            ax.figure.savefig(FIGURE_DIR / "locomotif_runtime_by_rho.png", dpi=300)
            plt.close(ax.figure)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No LoCoMotif runtime records available.", ha="center", va="center")
        ax.set_title("LoCoMotif runtime by rho")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "locomotif_runtime_by_rho.png", dpi=300)
        plt.close(fig)

    create_comparison_table()
    comparison = read_table(TABLE_DIR / "mp_vs_locomotif_controlled_comparison.csv")
    if not comparison.empty:
        fig, ax = plt.subplots(figsize=(12, max(3, 0.35 * len(comparison) + 1.5)))
        ax.axis("off")
        display_cols = ["slice_id", "method", "count", "total_occurrences", "runtime_seconds", "best_distance_or_score"]
        table = ax.table(cellText=comparison[display_cols].round(4).astype(str).values, colLabels=display_cols, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)
        ax.set_title("Controlled LoCoMotif experiment summary")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "controlled_locomotif_experiment_summary.png", dpi=300)
        plt.close(fig)


def create_comparison_table() -> None:
    loco_rt = read_table(TABLE_DIR / "locomotif_controlled_runtime.csv")
    loco_sets = read_table(TABLE_DIR / "locomotif_controlled_motif_sets.csv")
    occ = read_table(TABLE_DIR / "locomotif_controlled_occurrences.csv")
    mp_rt = read_table(TABLE_DIR / "mp_controlled_slice_runtime.csv")
    mp_motifs = read_table(TABLE_DIR / "mp_controlled_slice_motifs.csv")
    rows: list[dict[str, Any]] = []
    if not loco_rt.empty:
        for _, rt in loco_rt.iterrows():
            run_sets = loco_sets[loco_sets["run_key"] == rt["run_key"]] if not loco_sets.empty else pd.DataFrame()
            run_occ = occ[occ["run_key"] == rt["run_key"]] if not occ.empty else pd.DataFrame()
            succeeded = bool(rt["success"]) if not isinstance(rt["success"], str) else rt["success"].lower() == "true"
            rows.append(
                {
                    "asset": rt["asset"],
                    "frequency": rt["frequency"],
                    "slice_id": rt["slice_id"],
                    "regime_label": rt["regime_label"],
                    "feature": rt["feature"],
                    "method": "LoCoMotif",
                    "object_type": "time-warped motif set",
                    "parameter_summary": f"lmin={int(rt['l_min'])}, lmax={int(rt['l_max'])}, rho={rt['rho']}, nb={int(rt['nb'])}, overlap={rt['overlap']}, warping={rt['warping']}",
                    "count": int(run_sets["motif_set_id"].nunique()) if succeeded and not run_sets.empty else 0,
                    "total_occurrences": int(len(run_occ)) if succeeded else 0,
                    "best_distance_or_score": np.nan,
                    "mean_distance_or_score": np.nan,
                    "median_distance_or_score": np.nan,
                    "runtime_seconds": float(rt["runtime_seconds"]) if pd.notna(rt["runtime_seconds"]) else np.nan,
                    "visual_figure": f"figures/locomotif_btcusdt_15m_{rt['regime_label']}_top_motif_set.png" if rt["frequency"] == "15m" else "",
                    "interpretation_note": "LoCoMotif groups repeated occurrences that may be locally time-warped." if succeeded else f"Real LoCoMotif run failed; no motif sets were fabricated. Error: {rt.get('error_message', '')}",
                }
            )
    if not mp_motifs.empty:
        for _, mp in mp_motifs.iterrows():
            rows.append(
                {
                    "asset": mp["asset"],
                    "frequency": mp["frequency"],
                    "slice_id": mp["slice_id"],
                    "regime_label": mp["regime_label"],
                    "feature": mp["feature"],
                    "method": "Matrix Profile",
                    "object_type": "fixed-length nearest-neighbour motif pair/candidate",
                    "parameter_summary": f"window_length={int(mp['window_length'])}",
                    "count": 1,
                    "total_occurrences": 2,
                    "best_distance_or_score": float(mp["best_motif_distance"]),
                    "mean_distance_or_score": float(mp["mean_motif_distance"]),
                    "median_distance_or_score": float(mp["median_motif_distance"]),
                    "runtime_seconds": float(mp["runtime_seconds"]),
                    "visual_figure": f"figures/mp_btcusdt_15m_{mp['regime_label']}_top_motif_overlay.png" if mp["frequency"] == "15m" else "",
                    "interpretation_note": "MP gives exact fixed-length nearest-neighbour distance; runtime depends on slice length and window.",
                }
            )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "mp_vs_locomotif_controlled_comparison.csv", index=False)


def write_reuse_notes() -> None:
    notes = """# Reused LoCoMotif Code Notes

## Files inspected

- `notebooks/diagnostics/00_check_locomotif_installation_and_api.ipynb`
- `HPC workflow/HPC_Regime_and_motif_discovery/src/locomotif_utils.py`
- `HPC workflow/HPC_Regime_and_motif_discovery/notebooks/04_locomotif_motif_discovery.ipynb`
- `HPC workflow/HPC_Regime_and_motif_discovery/notebooks/study/04_locomotif_visual_study.ipynb`
- `notebooks/Locomotif Initial Study/03_year_2025_matrix_profile_vs_locomotif_motif_comparison.ipynb`

## Implementation pattern reused

The controlled experiment uses the same real API pattern documented in the diagnostic notebook and implemented in `locomotif_utils.py`:

`locomotif.locomotif.apply_locomotif(ts, l_min, l_max, rho=None, nb=None, start_mask=None, end_mask=None, overlap=0.0, warping=True)`

The input is passed as a time-by-channel NumPy array with shape `(n_observations, 1)` for the controlled univariate runs. Results are parsed as a list of `(representative_interval, occurrence_intervals)` motif sets.

## Real package status

The script imports the real `dtai-locomotif` package via `locomotif.locomotif` or `locomotif` and fails loudly if `apply_locomotif` is unavailable.

## Proxy logic avoided

No proxy motif discovery, fake motif generation, or simulated LoCoMotif output is used. If the real LoCoMotif call fails, the script writes the traceback to the raw output file, records failure in `locomotif_controlled_runtime.csv`, and continues only with honest failure reporting.
"""
    (LOG_DIR / "reused_locomotif_code_notes.md").write_text(notes, encoding="utf-8")


def tex_escape(value: Any) -> str:
    return str(value).replace("_", "\\_")


def write_latex_and_report() -> None:
    create_comparison_table()
    metadata = json.loads((CONFIG_DIR / "selected_slices_metadata.json").read_text(encoding="utf-8")) if (CONFIG_DIR / "selected_slices_metadata.json").exists() else []
    comparison = read_table(TABLE_DIR / "mp_vs_locomotif_controlled_comparison.csv")
    loco_rt = read_table(TABLE_DIR / "locomotif_controlled_runtime.csv")
    mp_motifs = read_table(TABLE_DIR / "mp_controlled_slice_motifs.csv")

    lines = [
        "\\section{Controlled LoCoMotif Comparison}",
        "\\subsection{Purpose of the Controlled Experiment}",
        "This controlled experiment compares Matrix Profile and LoCoMotif on matched BTCUSDT time slices rather than attempting a full multi-asset benchmark. The purpose is to provide direct evidence for Research Question 3 while keeping the LoCoMotif run reproducible and computationally bounded.",
        "\\subsection{Selected BTCUSDT Regime Slices}",
    ]
    for item in metadata:
        lines.append(
            f"The {tex_escape(item['slice_id'])} slice used {item['n_observations']} observations from {tex_escape(item['start_timestamp'])} to {tex_escape(item['end_timestamp'])}. "
            f"The selection rule was {tex_escape(item['selection_rule'])}, and the label source was {tex_escape(item['regime_label_status'])}."
        )
    lines.extend(
        [
            "\\subsection{LoCoMotif Motif-Set Evidence}",
            "LoCoMotif returns time-warped motif sets, so its counts are motif-set counts and occurrence counts rather than nearest-neighbour pair counts.",
        ]
    )
    if not loco_rt.empty:
        for _, row in loco_rt.iterrows():
            status = "succeeded" if bool(row.get("success")) else "failed"
            lines.append(
                f"For {tex_escape(row['slice_id'])} with feature {tex_escape(row['feature'])} and rho {row['rho']}, LoCoMotif {status}; "
                f"runtime was {row.get('runtime_seconds', np.nan)} seconds and {row.get('filtered_motif_sets_count', 0)} motif sets were recorded."
            )
    lines.extend(
        [
            "\\subsection{Matrix Profile versus LoCoMotif on Matched Slices}",
            "Matrix Profile gives fixed-length nearest-neighbour motif pairs. LoCoMotif groups repeated occurrences that may be locally time-warped. The comparison therefore uses matched slices and matched normalized features, but does not treat raw counts as equivalent objects.",
        ]
    )
    if not mp_motifs.empty:
        for _, row in mp_motifs.iterrows():
            lines.append(
                f"For {tex_escape(row['slice_id'])}, the Matrix Profile run with window {int(row['window_length'])} had best distance {float(row['best_motif_distance']):.6f} and runtime {float(row['runtime_seconds']):.3f} seconds."
            )
    lines.extend(
        [
            "\\subsection{Interpretation for Research Question 3}",
            "The controlled BTCUSDT evidence supports an honest method comparison: Matrix Profile is interpretable as a fixed-length distance search, while LoCoMotif provides variable-length motif-set evidence under time warping. These results should be interpreted as controlled evidence for RQ3; a full multi-asset LoCoMotif benchmark remains future work.",
        ]
    )
    (LATEX_DIR / "controlled_locomotif_results_section.tex").write_text("\n\n".join(lines), encoding="utf-8")

    discussion = [
        "\\subsection{Controlled LoCoMotif Evidence}",
        "The controlled LoCoMotif experiment was intentionally restricted to matched BTCUSDT slices. This avoids presenting LoCoMotif as if it had been benchmarked across the full thesis universe. The results show how LoCoMotif's time-warped motif sets complement Matrix Profile's fixed-length nearest-neighbour motif pairs, while preserving the limitation that runtime and motif-set recovery depend on slice length, rho, and length bounds.",
    ]
    (LATEX_DIR / "controlled_locomotif_discussion_snippet.tex").write_text("\n\n".join(discussion), encoding="utf-8")

    appendix = [
        "\\section{Controlled LoCoMotif Motif Examples}",
        "Table~\\ref{tab:controlled-locomotif-slices} reports the selected slice metadata. Table~\\ref{tab:controlled-locomotif-parameters} reports the LoCoMotif parameters. Figures include the high-volatility and low-volatility LoCoMotif overlays and the Matrix Profile versus LoCoMotif side-by-side comparisons. The machine-readable comparison table is `reports/locomotif_controlled_slice_comparison/tables/mp_vs_locomotif_controlled_comparison.csv`.",
        "\\begin{figure}[htbp]\\centering\\includegraphics[width=0.95\\textwidth]{figures/locomotif_btcusdt_15m_high_vol_top_motif_set.png}\\caption{Controlled high-volatility LoCoMotif motif-set overlay.}\\end{figure}",
        "\\begin{figure}[htbp]\\centering\\includegraphics[width=0.95\\textwidth]{figures/locomotif_btcusdt_15m_low_vol_top_motif_set.png}\\caption{Controlled low-volatility LoCoMotif motif-set overlay.}\\end{figure}",
        "\\begin{figure}[htbp]\\centering\\includegraphics[width=0.95\\textwidth]{figures/mp_vs_locomotif_high_vol_side_by_side.png}\\caption{Matrix Profile versus LoCoMotif on the high-volatility slice.}\\end{figure}",
        "\\begin{figure}[htbp]\\centering\\includegraphics[width=0.95\\textwidth]{figures/mp_vs_locomotif_low_vol_side_by_side.png}\\caption{Matrix Profile versus LoCoMotif on the low-volatility slice.}\\end{figure}",
    ]
    (LATEX_DIR / "controlled_locomotif_appendix_snippet.tex").write_text("\n\n".join(appendix), encoding="utf-8")

    report = [
        "# Controlled LoCoMotif Run Report",
        "",
        "## Data files used",
        f"- `{rel(BTC_15M)}`",
        f"- `{rel(BTC_1H)}`",
        "",
        "## Regime labels",
    ]
    if metadata:
        for item in metadata:
            report.append(f"- {item['slice_id']}: {item['regime_label_status']} from `{item['regime_label_source']}`")
    report.extend(["", "## Selected slices"])
    for item in metadata:
        report.append(f"- {item['slice_id']}: {item['start_timestamp']} to {item['end_timestamp']}, n={item['n_observations']}, rule={item['selection_rule']}")
    report.extend(
        [
            "",
            "## LoCoMotif package/import used",
            "- Real `dtai-locomotif` through `locomotif.locomotif.apply_locomotif` when available.",
            "",
            "## LoCoMotif parameter settings",
            "- 15m: lmin=32, lmax=128, rho from CLI, nb=10, overlap=0.20, warping=True.",
            "- 1h: lmin=24, lmax=72, rho from CLI, nb=10, overlap=0.20, warping=True.",
            "",
            "## Matrix Profile parameter settings",
            "- 15m windows: 32, 64, 128.",
            "- 1h windows: 24, 48, 72.",
            "",
            "## Success/failure status",
        ]
    )
    if not loco_rt.empty:
        for _, row in loco_rt.iterrows():
            report.append(f"- LoCoMotif {row['run_key']}: success={row['success']}, runtime={row.get('runtime_seconds')}, error={row.get('error_message', '')}")
    if not mp_motifs.empty:
        for _, row in mp_motifs.iterrows():
            report.append(f"- Matrix Profile {row['run_key']}: best_distance={row['best_motif_distance']}, runtime={row['runtime_seconds']}")
    report.extend(
        [
            "",
            "## Generated tables",
            "- `tables/locomotif_controlled_runtime.csv`",
            "- `tables/locomotif_controlled_motif_sets.csv`",
            "- `tables/locomotif_controlled_occurrences.csv`",
            "- `tables/mp_controlled_slice_runtime.csv`",
            "- `tables/mp_controlled_slice_motifs.csv`",
            "- `tables/mp_vs_locomotif_controlled_comparison.csv`",
            "",
            "## Generated figures",
            "- See `figures/*.png`.",
            "",
            "## Key numbers for thesis",
        ]
    )
    if not comparison.empty:
        for _, row in comparison.iterrows():
            report.append(f"- {row['slice_id']} {row['method']} {row['parameter_summary']}: count={row['count']}, runtime={row['runtime_seconds']}")
    report.extend(
        [
            "",
            "## Limitations",
            "- Controlled BTCUSDT slices only; not a full benchmark.",
            "- MP and LoCoMotif output different object types, so raw counts are not equivalent.",
            "- LoCoMotif runtime depends on slice length, rho, and motif length bounds.",
            "",
            "## Exact commands to reproduce",
            "```powershell",
            "python scripts/run_locomotif_controlled_slice_comparison.py --mode smoke --slice agnostic_1h --feature close_z --max-points 1000 --rho 0.65 --run-mp yes --run-locomotif yes",
            "python scripts/run_locomotif_controlled_slice_comparison.py --mode full --slice high_vol --feature close_z --max-points 2000 --rho 0.65 --run-mp yes --run-locomotif yes",
            "python scripts/run_locomotif_controlled_slice_comparison.py --mode full --slice low_vol --feature close_z --max-points 2000 --rho 0.65 --run-mp yes --run-locomotif yes",
            "```",
        ]
    )
    (OUT_DIR / "CONTROLLED_LOCOMOTIF_RUN_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def create_notebook() -> None:
    notebook_path = ROOT / "notebooks" / "05_locomotif_controlled_slice_comparison.ipynb"
    if notebook_path.exists():
        return
    payload = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Controlled LoCoMotif Slice Comparison\n", "\n", "This notebook is a lightweight companion to the reproducible script run.\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "OUT = Path('reports/locomotif_controlled_slice_comparison')\n",
                    "pd.read_csv(OUT / 'tables' / 'mp_vs_locomotif_controlled_comparison.csv')\n",
                ],
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_micro_comparison() -> None:
    comparison = read_table(TABLE_DIR / "mp_vs_locomotif_controlled_comparison.csv")
    if comparison.empty:
        pd.DataFrame().to_csv(TABLE_DIR / "micro_mp_vs_locomotif_comparison.csv", index=False)
        return
    micro = comparison[comparison["slice_id"] == "agnostic_1h"].copy()
    micro.to_csv(TABLE_DIR / "micro_mp_vs_locomotif_comparison.csv", index=False)


def save_micro_figures(slice_df: pd.DataFrame, metadata: dict[str, Any], feature: str) -> None:
    occ = get_locomotif_occurrences(metadata["slice_id"], feature)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(slice_df["timestamp"], slice_df[feature], color="#234F73", linewidth=0.8)
    if not occ.empty:
        top = occ[occ["motif_set_id"] == int(occ["motif_set_id"].min())]
        for _, row in top.iterrows():
            ax.axvspan(
                slice_df["timestamp"].iloc[int(row["occurrence_start"])],
                slice_df["timestamp"].iloc[min(int(row["occurrence_end"]) - 1, len(slice_df) - 1)],
                color="#D95F02",
                alpha=0.24,
            )
        title = "Micro LoCoMotif BTCUSDT 1h top motif set"
    else:
        title = "Micro LoCoMotif BTCUSDT 1h: no motif set returned"
    ax.set_title(title)
    ax.set_ylabel(feature)
    ax.set_xlabel("Timestamp")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "micro_locomotif_btcusdt_1h_top_motif_set.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax in axes:
        ax.plot(slice_df["timestamp"], slice_df[feature], color="#234F73", linewidth=0.8)
        ax.tick_params(axis="x", rotation=30)
    best = get_best_mp(metadata["slice_id"], feature)
    if best is not None:
        window = int(best["window_length"])
        for start in [int(best["motif_start_1"]), int(best["motif_start_2"])]:
            if 0 <= start < len(slice_df):
                axes[0].axvspan(slice_df["timestamp"].iloc[start], slice_df["timestamp"].iloc[min(start + window - 1, len(slice_df) - 1)], color="#1B9E77", alpha=0.25)
        axes[0].set_title(f"Matrix Profile, m={window}")
    else:
        axes[0].set_title("Matrix Profile unavailable")
    if not occ.empty:
        top = occ[occ["motif_set_id"] == int(occ["motif_set_id"].min())]
        for _, row in top.iterrows():
            axes[1].axvspan(
                slice_df["timestamp"].iloc[int(row["occurrence_start"])],
                slice_df["timestamp"].iloc[min(int(row["occurrence_end"]) - 1, len(slice_df) - 1)],
                color="#D95F02",
                alpha=0.24,
            )
        axes[1].set_title("LoCoMotif micro motif set")
    else:
        axes[1].set_title("LoCoMotif micro failed/timed out")
    fig.suptitle("Micro MP versus real LoCoMotif attempt on BTCUSDT 1h")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "micro_mp_vs_locomotif_btcusdt_1h_side_by_side.png", dpi=300)
    plt.close(fig)

    micro_rt = read_table(TABLE_DIR / "micro_locomotif_runtime.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not micro_rt.empty:
        labels = [str(v) for v in micro_rt["run_key"]]
        values = [float(v) if pd.notna(v) else 0.0 for v in micro_rt["runtime_seconds"]]
        ax.bar(labels, values, color="#D95F02")
        ax.set_ylabel("Runtime seconds")
        ax.tick_params(axis="x", rotation=20)
        ax.set_title("Micro LoCoMotif runtime; zero means failed before measured completion")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No micro LoCoMotif runtime rows available.", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "micro_locomotif_runtime.png", dpi=300)
    plt.close(fig)


def run_one(
    slice_name: str,
    feature: str,
    max_points: int,
    rho: float,
    run_mp: bool,
    run_loco: bool,
    mode: str = "full",
    lmin: int | None = None,
    lmax: int | None = None,
    nb: int | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    slice_df, metadata = prepare_slice(slice_name, feature, max_points)
    if run_mp:
        run_matrix_profile(slice_df, metadata, feature)
    loco_status = {"success": None}
    if run_loco:
        loco_status = run_locomotif(
            slice_df,
            metadata,
            feature,
            rho,
            lmin=lmin,
            lmax=lmax,
            nb=nb,
            timeout_seconds=timeout_seconds,
            output_prefix="micro" if mode == "micro" else "",
        )
    plot_slice_overview(slice_df, metadata)
    plot_mp_overlay(slice_df, metadata, feature)
    plot_locomotif_overlay(slice_df, metadata, feature)
    plot_side_by_side(slice_df, metadata, feature)
    plot_summary_charts()
    write_latex_and_report()
    if mode == "micro":
        write_micro_comparison()
        save_micro_figures(slice_df, metadata, feature)
    return loco_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled BTCUSDT LoCoMotif vs Matrix Profile slice comparison.")
    parser.add_argument("--mode", choices=["smoke", "full", "micro"], default="smoke")
    parser.add_argument("--slice", choices=["high_vol", "low_vol", "agnostic_1h", "all"], default="high_vol")
    parser.add_argument("--feature", choices=["close_z", "log_return_z"], default="close_z")
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--rho", type=float, default=0.65)
    parser.add_argument("--lmin", type=int, default=None)
    parser.add_argument("--lmax", type=int, default=None)
    parser.add_argument("--nb", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--run-mp", type=yn, default=True)
    parser.add_argument("--run-locomotif", type=yn, default=True)
    parser.add_argument("--hpc", type=yn, default=False)
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_output_root(args.output_root)
    safety_exit = enforce_execution_safety(args)
    if safety_exit is not None:
        return safety_exit
    ensure_dirs()
    ensure_base_csvs()
    write_reuse_notes()
    create_notebook()
    save_locomotif_parameter_mapping()
    try:
        targets = ["high_vol", "low_vol", "agnostic_1h"] if args.slice == "all" else [args.slice]
        statuses = [
            run_one(
                target,
                args.feature,
                args.max_points,
                args.rho,
                args.run_mp,
                args.run_locomotif,
                mode=args.mode,
                lmin=args.lmin,
                lmax=args.lmax,
                nb=args.nb,
                timeout_seconds=args.timeout_seconds,
            )
            for target in targets
        ]
        all_loco_ok = all(status.get("success") in {True, None} for status in statuses)
        if all_loco_ok:
            print("CONTROLLED LOCOMOTIF EXPERIMENT COMPLETE")
            return 0
        print("CONTROLLED LOCOMOTIF EXPERIMENT FAILED")
        for status in statuses:
            if status.get("success") is False:
                print(f"{status.get('run_key')}: {status.get('error_message')}")
        return 1
    except Exception:
        (LOG_DIR / "controlled_locomotif_uncaught_failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        print("CONTROLLED LOCOMOTIF EXPERIMENT FAILED")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
