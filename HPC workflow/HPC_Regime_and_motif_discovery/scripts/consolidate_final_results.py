#!/usr/bin/env python3
"""Consolidate final regime and motif outputs into compact thesis reports.

Run from the repository root:

    python "HPC workflow/HPC_Regime_and_motif_discovery/scripts/consolidate_final_results.py"

The script is intentionally read-only with respect to ``results``.  It writes
only to ``reports/final_results_consolidation`` and continues when inputs are
missing or unreadable.
"""

from __future__ import annotations

import math
import numbers
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
WORKFLOW_ROOT = SCRIPT_PATH.parents[1]
RESULTS_ROOT = WORKFLOW_ROOT / "results"
OUTPUT_ROOT = WORKFLOW_ROOT / "reports" / "final_results_consolidation"
SELECTED_FIGURES_ROOT = OUTPUT_ROOT / "selected_final_figures"

QUANTILE_CAVEAT = (
    "The quantile regime outputs store `rolling_volatility_60` as the actual "
    "volatility column across all quantile method identifiers. Therefore, "
    "quantile regimes should be interpreted as 60-period rolling-volatility "
    "regimes with different regime-count granularities rather than as separate "
    "30/60/240 volatility-horizon experiments."
)
LOCOMOTIF_SUBSET_CAVEAT = (
    "LoCoMotif is reported as a controlled subset experiment rather than a "
    "full-scale benchmark."
)

READ_CACHE: dict[Path, pd.DataFrame | None] = {}
READ_ERRORS: dict[Path, str] = {}
METADATA_CACHE: dict[Path, tuple[int | None, int | None, list[str]]] = {}


EXPECTED_FILES: list[dict[str, str]] = [
    {
        "method": "Quantile",
        "file_role": "labels",
        "path": "regimes/quantile/quantile_regime_labels.parquet",
        "thesis_use": "Primary quantile regime counts and coverage",
        "notes": "",
    },
    {
        "method": "Quantile",
        "file_role": "summary",
        "path": "regimes/quantile/quantile_regime_summary.parquet",
        "thesis_use": "Regime descriptive statistics",
        "notes": "",
    },
    {
        "method": "Quantile",
        "file_role": "transition matrix",
        "path": "regimes/quantile/quantile_transition_matrix.parquet",
        "thesis_use": "Regime persistence and transition evidence",
        "notes": "",
    },
    {
        "method": "Quantile",
        "file_role": "scope file",
        "path": "regimes/quantile/BTCUSDT_15m_quantile_regimes.parquet",
        "thesis_use": "BTCUSDT 15m scope confirmation",
        "notes": "",
    },
    {
        "method": "Quantile",
        "file_role": "scope file",
        "path": "regimes/quantile/BTCUSDT_1h_quantile_regimes.parquet",
        "thesis_use": "BTCUSDT 1h scope confirmation",
        "notes": "",
    },
    {
        "method": "Quantile",
        "file_role": "scope file",
        "path": "regimes/quantile/ETHUSDT_15m_quantile_regimes.parquet",
        "thesis_use": "ETHUSDT 15m scope confirmation",
        "notes": "",
    },
    {
        "method": "Quantile",
        "file_role": "scope file",
        "path": "regimes/quantile/ETHUSDT_1h_quantile_regimes.parquet",
        "thesis_use": "ETHUSDT 1h scope confirmation",
        "notes": "",
    },
    {
        "method": "Quantile",
        "file_role": "failure log",
        "path": "logs/01_quantile_failures.parquet",
        "thesis_use": "Completeness and failure audit",
        "notes": "",
    },
    {
        "method": "HMM",
        "file_role": "labels",
        "path": "regimes/hmm/hmm_regime_labels.parquet",
        "thesis_use": "Primary HMM regime counts and coverage",
        "notes": "",
    },
    {
        "method": "HMM",
        "file_role": "summary",
        "path": "regimes/hmm/hmm_regime_summary.parquet",
        "thesis_use": "HMM regime descriptive statistics",
        "notes": "",
    },
    {
        "method": "HMM",
        "file_role": "transition matrix",
        "path": "regimes/hmm/hmm_transition_matrix.parquet",
        "thesis_use": "HMM persistence and transition evidence",
        "notes": "",
    },
    {
        "method": "HMM",
        "file_role": "model selection",
        "path": "regimes/hmm/hmm_model_selection.parquet",
        "thesis_use": "Model-state selection evidence",
        "notes": "",
    },
    {
        "method": "HMM",
        "file_role": "persistence metrics",
        "path": "regimes/hmm/hmm_persistence_metrics.parquet",
        "thesis_use": "Expected state duration and persistence",
        "notes": "",
    },
    {
        "method": "HMM",
        "file_role": "feature diagnostics",
        "path": "regimes/hmm/hmm_feature_diagnostics.parquet",
        "thesis_use": "Input quality audit",
        "notes": "",
    },
    {
        "method": "HMM",
        "file_role": "quantile comparison",
        "path": "regimes/hmm/hmm_quantile_comparison.parquet",
        "thesis_use": "Cross-method regime validation",
        "notes": "",
    },
    {
        "method": "HMM",
        "file_role": "confusion table",
        "path": "regimes/hmm/hmm_quantile_confusion_table.parquet",
        "thesis_use": "Cross-method label agreement",
        "notes": "",
    },
    {
        "method": "HMM",
        "file_role": "scope file",
        "path": "regimes/hmm/BTCUSDT_15m_hmm_regimes.parquet",
        "thesis_use": "BTCUSDT 15m scope confirmation",
        "notes": "",
    },
    {
        "method": "HMM",
        "file_role": "scope file",
        "path": "regimes/hmm/BTCUSDT_1h_hmm_regimes.parquet",
        "thesis_use": "BTCUSDT 1h scope confirmation",
        "notes": "",
    },
    {
        "method": "HMM",
        "file_role": "scope file",
        "path": "regimes/hmm/ETHUSDT_15m_hmm_regimes.parquet",
        "thesis_use": "ETHUSDT 15m scope confirmation",
        "notes": "",
    },
    {
        "method": "HMM",
        "file_role": "scope file",
        "path": "regimes/hmm/ETHUSDT_1h_hmm_regimes.parquet",
        "thesis_use": "ETHUSDT 1h scope confirmation",
        "notes": "",
    },
    {
        "method": "HMM",
        "file_role": "failure log",
        "path": "logs/02_hmm_failures.parquet",
        "thesis_use": "Completeness and failure audit",
        "notes": "",
    },
    {
        "method": "Matrix Profile",
        "file_role": "motif results",
        "path": "motifs/matrix_profile/matrix_profile_motif_results.parquet",
        "thesis_use": "Primary Matrix Profile motif evidence",
        "notes": "",
    },
    {
        "method": "Matrix Profile",
        "file_role": "evaluation",
        "path": "motifs/matrix_profile/matrix_profile_evaluation.parquet",
        "thesis_use": "Motif recurrence, stability, and quality metrics",
        "notes": "",
    },
    {
        "method": "Matrix Profile",
        "file_role": "runtime",
        "path": "motifs/matrix_profile/matrix_profile_runtime.parquet",
        "thesis_use": "Computational performance",
        "notes": "",
    },
    {
        "method": "Matrix Profile",
        "file_role": "profiles",
        "path": "motifs/matrix_profile/matrix_profile_profiles.parquet",
        "thesis_use": "Profile availability audit only",
        "notes": "Metadata inspected without loading the full profile table",
    },
    {
        "method": "LoCoMotif",
        "file_role": "motif results",
        "path": "motifs/locomotif/locomotif_motif_results.parquet",
        "thesis_use": "Primary LoCoMotif interval evidence",
        "notes": "",
    },
    {
        "method": "LoCoMotif",
        "file_role": "evaluation",
        "path": "motifs/locomotif/locomotif_evaluation.parquet",
        "thesis_use": "Motif recurrence, length, and stability metrics",
        "notes": "",
    },
    {
        "method": "LoCoMotif",
        "file_role": "runtime",
        "path": "motifs/locomotif/locomotif_runtime.parquet",
        "thesis_use": "Computational performance and status",
        "notes": "",
    },
    {
        "method": "LoCoMotif",
        "file_role": "failure log",
        "path": "motifs/locomotif/04_locomotif_failures.parquet",
        "thesis_use": "Completeness and failure audit",
        "notes": "",
    },
]


def result_path(relative_path: str) -> Path:
    """Resolve a path relative to the results directory."""
    return RESULTS_ROOT / Path(relative_path)


def safe_read_parquet(path: Path) -> pd.DataFrame | None:
    """Read a parquet file, returning None and recording errors on failure."""
    path = Path(path)
    if path in READ_CACHE:
        return READ_CACHE[path]
    if not path.exists():
        READ_ERRORS[path] = "missing"
        READ_CACHE[path] = None
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # Continue consolidating all other outputs.
        READ_ERRORS[path] = f"{type(exc).__name__}: {exc}"
        frame = None
    READ_CACHE[path] = frame
    return frame


def parquet_metadata(path: Path) -> tuple[int | None, int | None, list[str]]:
    """Get parquet footer metadata without loading table values when possible."""
    path = Path(path)
    if path in METADATA_CACHE:
        return METADATA_CACHE[path]
    if not path.exists():
        result = (None, None, [])
        METADATA_CACHE[path] = result
        return result

    rows: int | None = None
    columns: list[str] = []
    try:
        # pandas already requires a parquet engine.  Accessing its footer keeps
        # the large matrix-profile profile table out of memory.
        from pandas.io.parquet import get_engine

        engine = get_engine("auto")
        if hasattr(engine, "api") and hasattr(engine.api, "parquet"):
            parquet_file = engine.api.parquet.ParquetFile(path)
            rows = int(parquet_file.metadata.num_rows)
            columns = list(parquet_file.schema_arrow.names)
        elif hasattr(engine, "api") and hasattr(engine.api, "ParquetFile"):
            parquet_file = engine.api.ParquetFile(str(path))
            rows = int(parquet_file.count())
            columns = list(parquet_file.columns)
    except Exception:
        pass

    if rows is None:
        try:
            empty_columns = pd.read_parquet(path, columns=[])
            rows = len(empty_columns)
        except Exception as exc:
            READ_ERRORS.setdefault(path, f"{type(exc).__name__}: {exc}")

    result = (rows, len(columns) if columns else None, columns)
    METADATA_CACHE[path] = result
    return result


def shape_of(path: Path) -> tuple[int | None, int | None]:
    """Return a parquet table shape without requiring a successful full read."""
    rows, cols, _ = parquet_metadata(path)
    if rows is not None:
        return rows, cols
    frame = safe_read_parquet(path)
    return frame.shape if frame is not None else (None, None)


def safe_value_counts(df: pd.DataFrame | None, col: str) -> pd.DataFrame:
    """Return value counts as a two-column DataFrame."""
    if df is None or col not in df.columns:
        return pd.DataFrame(columns=[col, "rows"])
    values = (
        df[col]
        .fillna("(missing)")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis(col)
        .reset_index(name="rows")
    )
    return values


def grouped_counts(
    df: pd.DataFrame | None, columns: Sequence[str]
) -> pd.DataFrame:
    """Count rows by the subset of requested columns that exists."""
    if df is None:
        return pd.DataFrame(columns=[*columns, "rows"])
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.DataFrame(columns=[*columns, "rows"])
    clean = df[available].copy()
    for column in available:
        clean[column] = clean[column].fillna("(missing)").astype(str)
    return (
        clean.groupby(available, dropna=False, sort=True)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def unique_values(df: pd.DataFrame | None, column: str) -> list[str]:
    """Return sorted, non-null unique values as strings."""
    if df is None or column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())


def first_existing_column(
    df: pd.DataFrame | None, candidates: Sequence[str]
) -> str | None:
    """Return the first candidate present in a DataFrame."""
    if df is None:
        return None
    return next((column for column in candidates if column in df.columns), None)


def timestamp_range(df: pd.DataFrame | None) -> tuple[str, str]:
    """Find a timestamp range using common timestamp column names."""
    column = first_existing_column(
        df, ("timestamp", "datetime", "date", "time", "motif_start_timestamp")
    )
    if column is None or df is None or df.empty:
        return "not available", "not available"
    converted = pd.to_datetime(df[column], errors="coerce", utc=True).dropna()
    if converted.empty:
        return "not available", "not available"
    return converted.min().isoformat(), converted.max().isoformat()


def format_number(value: Any) -> str:
    """Format scalar values compactly for reports."""
    if value is None or value is pd.NA:
        return "not available"
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        numeric_value = float(value)
        if math.isnan(numeric_value):
            return "not available"
        if numeric_value.is_integer():
            return f"{int(numeric_value):,}"
        return f"{numeric_value:,.4f}"
    return str(value)


def markdown_escape(value: Any) -> str:
    """Escape a scalar for a simple Markdown table."""
    text = format_number(value)
    return text.replace("|", r"\|").replace("\n", " ")


def write_markdown_table(df: pd.DataFrame | None, max_rows: int = 20) -> str:
    """Render a DataFrame without requiring the optional tabulate package."""
    if df is None:
        return "_Table unavailable._"
    if len(df.columns) == 0:
        return "_Table has no columns._"
    if df.empty:
        return "_No rows._"
    shown = df.head(max_rows)
    header = "| " + " | ".join(markdown_escape(c) for c in shown.columns) + " |"
    separator = "| " + " | ".join("---" for _ in shown.columns) + " |"
    rows = [
        "| " + " | ".join(markdown_escape(value) for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    if len(df) > max_rows:
        rows.append(f"\n_Showing {max_rows} of {len(df):,} rows._")
    return "\n".join([header, separator, *rows])


def shape_text(df: pd.DataFrame | None, path: Path) -> str:
    """Format an available DataFrame or metadata shape."""
    if df is not None:
        return f"{df.shape[0]:,} rows x {df.shape[1]:,} columns"
    rows, cols = shape_of(path)
    if rows is None:
        return "missing or unreadable"
    col_text = f"{cols:,}" if cols is not None else "unknown"
    return f"{rows:,} rows x {col_text} columns"


def numeric_summary(
    df: pd.DataFrame | None,
    group_columns: Sequence[str] = (),
    preferred_metrics: Sequence[str] = (),
    max_metrics: int = 8,
) -> tuple[list[str], pd.DataFrame]:
    """Summarize numeric metrics, optionally by available grouping columns."""
    if df is None or df.empty:
        return [], pd.DataFrame()
    numeric = df.select_dtypes(include="number").columns.tolist()
    preferred = [metric for metric in preferred_metrics if metric in numeric]
    metrics = preferred + [metric for metric in numeric if metric not in preferred]
    metrics = metrics[:max_metrics]
    groups = [column for column in group_columns if column in df.columns]
    if not metrics:
        return [], pd.DataFrame()
    if groups:
        summary = (
            df.groupby(groups, dropna=False)[metrics]
            .agg(["count", "mean", "median", "min", "max"])
            .reset_index()
        )
        summary.columns = [
            "_".join(str(part) for part in column if str(part))
            if isinstance(column, tuple)
            else str(column)
            for column in summary.columns
        ]
        return metrics, summary
    summary = df[metrics].agg(["count", "mean", "median", "min", "max"]).T
    summary.index.name = "metric"
    return metrics, summary.reset_index()


def runtime_summary(
    df: pd.DataFrame | None, group_columns: Sequence[str]
) -> tuple[str | None, pd.DataFrame, str | None]:
    """Summarize a runtime column and flag suspicious magnitudes."""
    runtime_col = first_existing_column(
        df,
        (
            "runtime_seconds",
            "elapsed_seconds",
            "duration_seconds",
            "runtime",
            "elapsed_time",
        ),
    )
    if df is None or runtime_col is None or df.empty:
        return runtime_col, pd.DataFrame(), None
    values = pd.to_numeric(df[runtime_col], errors="coerce")
    usable = df.loc[values.notna()].copy()
    usable[runtime_col] = values.dropna()
    groups = [column for column in group_columns if column in usable.columns]
    if usable.empty:
        return runtime_col, pd.DataFrame(), None
    if groups:
        summary = (
            usable.groupby(groups, dropna=False)[runtime_col]
            .agg(["count", "mean", "median", "min", "max"])
            .reset_index()
        )
    else:
        summary = usable[runtime_col].agg(["count", "mean", "median", "min", "max"])
        summary = summary.to_frame().T
    caution = None
    median = float(usable[runtime_col].median())
    maximum = float(usable[runtime_col].max())
    total = float(usable[runtime_col].sum())
    if maximum > 86_400 or (median > 0 and maximum > median * 100):
        caution = (
            "Some runtime values are unusually large relative to the median. "
            "Confirm whether these values are cumulative job time rather than "
            "per-experiment runtime before quoting them."
        )
    elif total > 0 and maximum > total * 0.9 and len(usable) > 2:
        caution = (
            "One runtime observation dominates the recorded total. Check "
            "whether runtime accounting is cumulative or includes setup time."
        )
    return runtime_col, summary, caution


def add_key_number(
    rows: list[dict[str, Any]],
    metric: str,
    value: Any,
    method: str,
    source_file: str,
    note: str = "",
) -> None:
    """Append one key-number record."""
    rows.append(
        {
            "metric": metric,
            "value": value if value is not None else "not available",
            "method": method,
            "source_file": source_file,
            "note": note,
        }
    )


def file_inventory() -> pd.DataFrame:
    """Build the required expected-file inventory."""
    rows: list[dict[str, Any]] = []
    for item in EXPECTED_FILES:
        path = result_path(item["path"])
        exists = path.is_file()
        row_count: int | None = None
        col_count: int | None = None
        columns: list[str] = []
        notes = item["notes"]
        if exists:
            row_count, col_count, columns = parquet_metadata(path)
            if row_count is None:
                notes = "; ".join(
                    part for part in (notes, READ_ERRORS.get(path, "unreadable")) if part
                )
        else:
            notes = "; ".join(part for part in (notes, "expected file missing") if part)
        rows.append(
            {
                "method": item["method"],
                "file_role": item["file_role"],
                "path": str(path.relative_to(WORKFLOW_ROOT)).replace("\\", "/"),
                "exists": exists,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3)
                if exists
                else None,
                "rows": row_count,
                "cols": col_count,
                "columns": ", ".join(columns),
                "thesis_use": item["thesis_use"],
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def status_table(method: str) -> pd.DataFrame:
    """Return thesis-scope file status for an algorithm."""
    suffix = "quantile_regimes.parquet" if method == "Quantile" else "hmm_regimes.parquet"
    folder = "quantile" if method == "Quantile" else "hmm"
    rows = []
    for asset in ("BTCUSDT", "ETHUSDT"):
        for frequency in ("15m", "1h"):
            path = RESULTS_ROOT / "regimes" / folder / f"{asset}_{frequency}_{suffix}"
            shape = shape_of(path)
            rows.append(
                {
                    "asset": asset,
                    "frequency": frequency,
                    "exists": path.exists(),
                    "rows": shape[0],
                    "path": str(path.relative_to(WORKFLOW_ROOT)).replace("\\", "/"),
                }
            )
    return pd.DataFrame(rows)


def summarize_regimes(
    method: str,
    labels_path: Path,
    related_paths: dict[str, Path],
) -> dict[str, Any]:
    """Create a schema-tolerant regime summary."""
    labels = safe_read_parquet(labels_path)
    start, end = timestamp_range(labels)
    method_col = first_existing_column(labels, ("regime_method", "method"))
    label_col = first_existing_column(labels, ("regime_label", "label", "state"))
    summary: dict[str, Any] = {
        "name": method,
        "labels": labels,
        "labels_path": labels_path,
        "total_rows": len(labels) if labels is not None else None,
        "assets": unique_values(labels, "asset"),
        "frequencies": unique_values(labels, "frequency"),
        "timestamp_start": start,
        "timestamp_end": end,
        "methods": unique_values(labels, method_col) if method_col else [],
        "regime_labels": unique_values(labels, label_col) if label_col else [],
        "rows_by_asset": safe_value_counts(labels, "asset"),
        "rows_by_frequency": safe_value_counts(labels, "frequency"),
        "rows_by_method": safe_value_counts(labels, method_col)
        if method_col
        else pd.DataFrame(),
        "rows_by_label": safe_value_counts(labels, label_col)
        if label_col
        else pd.DataFrame(),
        "scope_status": status_table(method),
        "related": {},
    }
    for role, path in related_paths.items():
        frame = safe_read_parquet(path)
        summary["related"][role] = {
            "path": path,
            "frame": frame,
            "shape": shape_text(frame, path),
        }
    failures = summary["related"].get("failures", {}).get("frame")
    failure_path = summary["related"].get("failures", {}).get("path")
    if failures is not None:
        summary["failure_rows"] = len(failures)
    elif failure_path is not None:
        summary["failure_rows"] = shape_of(failure_path)[0]
    else:
        summary["failure_rows"] = None
    return summary


def quantile_volatility_audit(summary: dict[str, Any]) -> dict[str, Any]:
    """Audit the volatility column recorded in quantile labels."""
    labels = summary["labels"]
    if labels is None:
        return {
            "column": None,
            "values": [],
            "all_rolling_60": False,
            "text": "Quantile volatility-column audit unavailable.",
        }
    column = first_existing_column(
        labels, ("volatility_column", "volatility_feature", "volatility_col")
    )
    if column is None:
        return {
            "column": None,
            "values": [],
            "all_rolling_60": False,
            "text": "No volatility-column identifier was found in quantile labels.",
        }
    values = unique_values(labels, column)
    all_rolling_60 = (
        bool(values)
        and labels[column].notna().all()
        and set(values) == {"rolling_volatility_60"}
    )
    return {
        "column": column,
        "values": values,
        "all_rolling_60": all_rolling_60,
        "text": (
            QUANTILE_CAVEAT
            if all_rolling_60
            else f"Recorded volatility columns: {', '.join(values) or 'none'}."
        ),
    }


def summarize_matrix_profile(paths: dict[str, Path]) -> dict[str, Any]:
    """Summarize Matrix Profile results, evaluation, runtime, and profiles."""
    motif = safe_read_parquet(paths["motif"])
    evaluation = safe_read_parquet(paths["evaluation"])
    runtime = safe_read_parquet(paths["runtime"])
    profile_rows, profile_cols = shape_of(paths["profiles"])
    runtime_col, runtime_table, runtime_caution = runtime_summary(
        runtime, ("asset", "frequency", "mode", "profile_type")
    )
    eval_metrics, eval_summary = numeric_summary(
        evaluation,
        ("asset", "frequency", "mode"),
        (
            "number_of_motifs",
            "recurrence_count",
            "mean_motif_distance_or_score",
            "median_motif_distance",
            "time_split_stability",
            "cross_regime_overlap",
            "runtime_seconds",
        ),
    )
    mode_counts = safe_value_counts(motif, "mode")
    gpu_counts = safe_value_counts(motif, "used_gpu")
    return {
        "motif": motif,
        "evaluation": evaluation,
        "runtime": runtime,
        "total_rows": len(motif) if motif is not None else None,
        "rows_by_asset_frequency_mode": grouped_counts(
            motif, ("asset", "frequency", "mode")
        ),
        "rows_by_regime_method": safe_value_counts(motif, "regime_method"),
        "rows_by_regime_label": safe_value_counts(motif, "regime_label"),
        "rows_by_frequency_window": grouped_counts(
            motif, ("frequency", "window_length")
        ),
        "rows_by_feature_set": safe_value_counts(motif, "feature_set"),
        "rows_by_profile_type": safe_value_counts(motif, "profile_type"),
        "rows_by_asset_frequency_label": grouped_counts(
            motif, ("asset", "frequency", "regime_label")
        ),
        "rows_by_asset_frequency_window": grouped_counts(
            motif, ("asset", "frequency", "window_length")
        ),
        "rows_by_asset_frequency_feature": grouped_counts(
            motif, ("asset", "frequency", "feature_set")
        ),
        "mode_counts": mode_counts,
        "gpu_counts": gpu_counts,
        "evaluation_shape": shape_text(evaluation, paths["evaluation"]),
        "evaluation_columns": list(evaluation.columns)
        if evaluation is not None
        else parquet_metadata(paths["evaluation"])[2],
        "evaluation_metrics": eval_metrics,
        "evaluation_summary": eval_summary,
        "runtime_shape": shape_text(runtime, paths["runtime"]),
        "runtime_column": runtime_col,
        "runtime_summary": runtime_table,
        "runtime_caution": runtime_caution,
        "profiles_shape": (profile_rows, profile_cols),
        "profiles_empty": profile_rows == 0 if profile_rows is not None else None,
        "assets": unique_values(motif, "asset"),
        "frequencies": unique_values(motif, "frequency"),
        "regime_labels": unique_values(motif, "regime_label"),
    }


def summarize_locomotif(paths: dict[str, Path]) -> dict[str, Any]:
    """Summarize LoCoMotif results, evaluation, runtime, and failures."""
    motif = safe_read_parquet(paths["motif"])
    evaluation = safe_read_parquet(paths["evaluation"])
    runtime = safe_read_parquet(paths["runtime"])
    failures = safe_read_parquet(paths["failures"])
    eval_metrics, eval_summary = numeric_summary(
        evaluation,
        ("asset", "frequency", "mode"),
        (
            "number_of_motifs",
            "recurrence_count",
            "mean_motif_length",
            "median_motif_length",
            "time_split_stability",
            "cross_regime_overlap",
            "runtime_seconds",
        ),
    )
    runtime_col, runtime_table, runtime_caution = runtime_summary(
        runtime, ("asset", "frequency", "mode", "status")
    )
    assets = sorted(
        {
            value
            for frame in (motif, evaluation, runtime)
            for value in unique_values(frame, "asset")
        }
    )
    frequencies = sorted(
        {
            value
            for frame in (motif, evaluation, runtime)
            for value in unique_values(frame, "frequency")
        }
    )
    appears_subset = bool(assets or frequencies) and set(assets) <= {"BTCUSDT"} and set(
        frequencies
    ) <= {"15m"}
    return {
        "motif": motif,
        "evaluation": evaluation,
        "runtime": runtime,
        "failures": failures,
        "total_rows": len(motif) if motif is not None else None,
        "assets": assets,
        "frequencies": frequencies,
        "rows_by_asset_frequency_mode": grouped_counts(
            motif, ("asset", "frequency", "mode")
        ),
        "rows_by_regime_method": safe_value_counts(motif, "regime_method"),
        "rows_by_regime_label": safe_value_counts(motif, "regime_label"),
        "rows_by_feature_set": safe_value_counts(motif, "feature_set"),
        "rows_by_status": safe_value_counts(motif, "status"),
        "evaluation_shape": shape_text(evaluation, paths["evaluation"]),
        "evaluation_rows": len(evaluation) if evaluation is not None else None,
        "evaluation_metrics": eval_metrics,
        "evaluation_summary": eval_summary,
        "runtime_rows": len(runtime) if runtime is not None else None,
        "runtime_status": safe_value_counts(runtime, "status"),
        "runtime_column": runtime_col,
        "runtime_summary": runtime_table,
        "runtime_caution": runtime_caution,
        "failure_shape": shape_text(failures, paths["failures"]),
        "failure_rows": len(failures) if failures is not None else shape_of(paths["failures"])[0],
        "regime_labels": unique_values(motif, "regime_label"),
        "appears_subset": appears_subset,
    }


def figure_category(path: Path) -> str | None:
    """Classify figures from the required filename prefixes."""
    if path.name.startswith(("01_", "02_")):
        return "regime"
    if path.name.startswith("03_"):
        return "matrix_profile"
    if path.name.startswith("04_"):
        return "locomotif"
    return None


def figure_score(path: Path, category: str) -> tuple[int, str]:
    """Rank figures for compact presentation evidence."""
    name = path.name.lower()
    score = 0
    reasons: list[str] = []
    for keyword, points, reason in (
        ("overlay", 10, "motif overlay"),
        ("profile", 9, "matrix profile"),
        ("transition", 8, "transition evidence"),
        ("close_by_regime", 8, "regime timeline"),
        ("vol_by_regime", 7, "volatility separation"),
        ("agnostic", 6, "agnostic example"),
        ("conditioned", 6, "conditioned example"),
        ("low_vol", 4, "low-volatility example"),
        ("high_vol", 4, "high-volatility example"),
        ("extreme_vol", 5, "extreme-volatility example"),
        ("distribution", 3, "distribution evidence"),
    ):
        if keyword in name:
            score += points
            reasons.append(reason)
    if category == "matrix_profile" and ("overlay" in name or "profile" in name):
        score += 3
    return score, ", ".join(reasons) or "available result figure"


def select_figures() -> tuple[pd.DataFrame, list[Path]]:
    """Inventory figures, select 15-25 where possible, and copy them."""
    figures_root = RESULTS_ROOT / "figures"
    candidates: list[tuple[Path, str, int, str]] = []
    if figures_root.exists():
        for path in sorted(figures_root.rglob("*")):
            if not path.is_file():
                continue
            category = figure_category(path)
            if category is None:
                continue
            score, reason = figure_score(path, category)
            candidates.append((path, category, score, reason))

    selected: list[Path] = []
    selected_set: set[Path] = set()

    def add_ranked(category: str, limit: int) -> None:
        ranked = sorted(
            (item for item in candidates if item[1] == category),
            key=lambda item: (-item[2], item[0].name),
        )
        # Greedy diversity: avoid selecting only one mode/label/plot type.
        tokens_seen: Counter[str] = Counter()
        while ranked and sum(1 for p in selected if figure_category(p) == category) < limit:
            best_index = 0
            best_value = -10**9
            for index, item in enumerate(ranked):
                name = item[0].name.lower()
                diversity_tokens = [
                    token
                    for token in (
                        "agnostic",
                        "conditioned",
                        "low_vol",
                        "high_vol",
                        "extreme_vol",
                        "overlay",
                        "profile",
                        "transition",
                    )
                    if token in name
                ]
                diversity_bonus = sum(5 if tokens_seen[token] == 0 else 0 for token in diversity_tokens)
                value = item[2] + diversity_bonus
                if value > best_value:
                    best_index, best_value = index, value
            item = ranked.pop(best_index)
            selected.append(item[0])
            selected_set.add(item[0])
            for token in (
                "agnostic",
                "conditioned",
                "low_vol",
                "high_vol",
                "extreme_vol",
                "overlay",
                "profile",
                "transition",
            ):
                if token in item[0].name.lower():
                    tokens_seen[token] += 1

    add_ranked("regime", 2)
    add_ranked("matrix_profile", 10)
    add_ranked("locomotif", 6)

    # If a category has fewer figures, fill toward 15 from all remaining useful figures.
    for item in sorted(candidates, key=lambda value: (-value[2], value[0].name)):
        if len(selected) >= 15:
            break
        if item[0] not in selected_set:
            selected.append(item[0])
            selected_set.add(item[0])
    selected = selected[:25]
    selected_set = set(selected)

    SELECTED_FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    # The folder is script-owned. Remove stale selections so its contents
    # always agree with the current recommendation CSV.
    for old_selection in SELECTED_FIGURES_ROOT.iterdir():
        if old_selection.is_file() or old_selection.is_symlink():
            try:
                old_selection.unlink()
            except OSError as exc:
                READ_ERRORS[old_selection] = (
                    f"stale selected figure could not be removed: {exc}"
                )
    copied_names: dict[Path, str] = {}
    for path in selected:
        destination = SELECTED_FIGURES_ROOT / path.name
        if destination.exists() and destination.resolve() != path.resolve():
            stem, suffix = destination.stem, destination.suffix
            counter = 2
            while destination.exists():
                destination = SELECTED_FIGURES_ROOT / f"{stem}_{counter}{suffix}"
                counter += 1
        try:
            shutil.copy2(path, destination)
            copied_names[path] = destination.name
        except OSError as exc:
            READ_ERRORS[path] = f"figure copy failed: {exc}"

    inventory_rows = []
    for path, category, score, reason in candidates:
        selected_ok = path in copied_names
        inventory_rows.append(
            {
                "category": category,
                "filename": path.name,
                "source_path": str(path.relative_to(WORKFLOW_ROOT)).replace("\\", "/"),
                "selected": selected_ok,
                "selected_filename": copied_names.get(path, ""),
                "priority_score": score,
                "recommendation_reason": reason,
                "recommended_use": (
                    "Presentation and thesis evidence" if selected_ok else "Supplementary"
                ),
            }
        )
    inventory = pd.DataFrame(
        inventory_rows,
        columns=[
            "category",
            "filename",
            "source_path",
            "selected",
            "selected_filename",
            "priority_score",
            "recommendation_reason",
            "recommended_use",
        ],
    )
    if not inventory.empty:
        inventory = inventory.sort_values(
            ["selected", "category", "priority_score", "filename"],
            ascending=[False, True, False, True],
            kind="stable",
        ).reset_index(drop=True)
    return inventory, [path for path in selected if path in copied_names]


def mode_count(summary: dict[str, Any], mode: str) -> int | None:
    """Extract a case-insensitive mode count."""
    table = summary.get("mode_counts")
    if table is None or table.empty or "mode" not in table.columns:
        return None
    match = table[table["mode"].astype(str).str.lower() == mode.lower()]
    return int(match["rows"].sum()) if not match.empty else 0


def bool_count(table: pd.DataFrame, value: bool) -> int | None:
    """Extract true/false counts from a value-count table."""
    if table.empty or "used_gpu" not in table.columns:
        return None
    normalized = table["used_gpu"].astype(str).str.lower()
    target = "true" if value else "false"
    match = table[normalized == target]
    return int(match["rows"].sum()) if not match.empty else 0


def figure_count(inventory: pd.DataFrame, category: str) -> int:
    """Count available figures by category."""
    if inventory.empty:
        return 0
    return int((inventory["category"] == category).sum())


def join_values(values: Iterable[str]) -> str:
    """Join values or state that they are unavailable."""
    values = list(values)
    return ", ".join(values) if values else "not available"


def comparison_table(
    matrix_profile: dict[str, Any], locomotif: dict[str, Any]
) -> pd.DataFrame:
    """Build the required algorithm comparison table."""
    mp_modes = unique_values(matrix_profile["motif"], "mode")
    loco_modes = unique_values(locomotif["motif"], "mode")
    mp_runtime = (
        "Recorded; grouped runtime statistics available"
        if not matrix_profile["runtime_summary"].empty
        else "Not available"
    )
    loco_runtime = (
        "Recorded; grouped runtime statistics available"
        if not locomotif["runtime_summary"].empty
        else "Not available"
    )
    loco_failure = locomotif["failure_rows"]
    rows = [
        (
            "scope completed",
            f"{format_number(matrix_profile['total_rows'])} motif rows",
            (
                f"{format_number(locomotif['total_rows'])} motif interval rows"
                + ("; controlled subset" if locomotif["appears_subset"] else "")
            ),
        ),
        (
            "assets/frequencies",
            f"{join_values(matrix_profile['assets'])}; {join_values(matrix_profile['frequencies'])}",
            f"{join_values(locomotif['assets'])}; {join_values(locomotif['frequencies'])}",
        ),
        (
            "result scale",
            "Pairwise recurring subsequence candidates across configured searches",
            "Motif interval instances grouped into motif sets",
        ),
        (
            "agnostic support",
            "Yes" if "agnostic" in [v.lower() for v in mp_modes] else "Not evidenced",
            "Yes" if "agnostic" in [v.lower() for v in loco_modes] else "Not evidenced",
        ),
        (
            "conditioned support",
            "Yes" if "conditioned" in [v.lower() for v in mp_modes] else "Not evidenced",
            "Yes" if "conditioned" in [v.lower() for v in loco_modes] else "Not evidenced",
        ),
        (
            "regime labels used",
            join_values(matrix_profile["regime_labels"]),
            join_values(locomotif["regime_labels"]),
        ),
        ("runtime characteristics", mp_runtime, loco_runtime),
        (
            "failure status",
            "No dedicated failure file requested; inspect runtime/status fields",
            (
                "No internal failures recorded"
                if loco_failure == 0
                else f"{format_number(loco_failure)} recorded failure rows"
            ),
        ),
        (
            "thesis interpretation",
            "Strong baseline for global and regime-conditioned shape recurrence",
            "Flexible-length motif evidence; interpret within completed scope",
        ),
        (
            "limitations",
            "Raw motif counts are search-dependent and do not imply significance or profitability",
            "Coverage may be capped/subset; counts are not directly comparable with Matrix Profile",
        ),
    ]
    return pd.DataFrame(rows, columns=["aspect", "matrix_profile", "locomotif"])


def missing_input_lines(inventory: pd.DataFrame) -> list[str]:
    """List missing or unreadable expected inputs."""
    missing = inventory[~inventory["exists"]]
    lines = [f"- Missing: `{path}`" for path in missing["path"].tolist()]
    for path, error in sorted(READ_ERRORS.items(), key=lambda item: str(item[0])):
        if error == "missing":
            continue
        try:
            display = path.relative_to(WORKFLOW_ROOT)
        except ValueError:
            display = path
        lines.append(f"- Unreadable: `{str(display).replace(chr(92), '/')}` — {error}")
    return lines or ["- No missing or unreadable expected inputs were detected."]


def merge_read_errors_into_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Attach read-time errors discovered after the metadata inventory pass."""
    updated = inventory.copy()
    for index, row in updated.iterrows():
        path = WORKFLOW_ROOT / Path(str(row["path"]))
        error = READ_ERRORS.get(path)
        if not error or error == "missing":
            continue
        existing = str(row["notes"]) if pd.notna(row["notes"]) else ""
        if error not in existing:
            updated.at[index, "notes"] = "; ".join(
                part for part in (existing, f"read error: {error}") if part
            )
    return updated


def selected_figure_table(figure_inventory: pd.DataFrame) -> pd.DataFrame:
    """Return compact selected-figure recommendations."""
    if figure_inventory.empty:
        return figure_inventory
    columns = [
        "category",
        "selected_filename",
        "recommendation_reason",
        "recommended_use",
    ]
    return figure_inventory.loc[figure_inventory["selected"], columns].reset_index(drop=True)


def recommended_figure(
    figure_inventory: pd.DataFrame, category: str, keywords: Sequence[str] = ()
) -> str:
    """Choose one selected figure for a presentation slide."""
    if figure_inventory.empty:
        return "No matching figure was available."
    subset = figure_inventory[
        (figure_inventory["selected"]) & (figure_inventory["category"] == category)
    ].copy()
    if subset.empty:
        return "No matching figure was available."
    if keywords:
        lower = subset["filename"].str.lower()
        score = pd.Series(0, index=subset.index)
        for keyword in keywords:
            score += lower.str.contains(keyword.lower(), regex=False).astype(int)
        subset = subset.assign(keyword_score=score).sort_values(
            ["keyword_score", "priority_score"], ascending=[False, False]
        )
    return f"`selected_final_figures/{subset.iloc[0]['selected_filename']}`"


def regime_section(summary: dict[str, Any], volatility_text: str | None = None) -> str:
    """Render a detailed Markdown section for one regime method."""
    related = summary["related"]
    lines = [
        f"## {summary['name']} regime results",
        "",
        f"- Total label rows: **{format_number(summary['total_rows'])}**",
        f"- Assets: {join_values(summary['assets'])}",
        f"- Frequencies: {join_values(summary['frequencies'])}",
        f"- Timestamp range: {summary['timestamp_start']} to {summary['timestamp_end']}",
        f"- Regime methods: {join_values(summary['methods'])}",
        f"- Regime labels: {join_values(summary['regime_labels'])}",
        f"- Failure rows: **{format_number(summary['failure_rows'])}**",
        "",
        "### Rows by asset",
        "",
        write_markdown_table(summary["rows_by_asset"]),
        "",
        "### Rows by frequency",
        "",
        write_markdown_table(summary["rows_by_frequency"]),
        "",
        "### Rows by regime method",
        "",
        write_markdown_table(summary["rows_by_method"]),
        "",
        "### Rows by regime label",
        "",
        write_markdown_table(summary["rows_by_label"]),
        "",
        "### Thesis-scope file status",
        "",
        write_markdown_table(summary["scope_status"]),
        "",
        "### Supporting table shapes",
        "",
    ]
    shape_rows = [
        {"table": role, "shape": details["shape"]}
        for role, details in related.items()
    ]
    lines.append(write_markdown_table(pd.DataFrame(shape_rows)))
    if volatility_text:
        lines.extend(["", "### Volatility-column audit", "", volatility_text])
    return "\n".join(lines)


def matrix_profile_section(summary: dict[str, Any]) -> str:
    """Render the Matrix Profile report section."""
    profile_rows, profile_cols = summary["profiles_shape"]
    profile_status = (
        "empty"
        if summary["profiles_empty"] is True
        else "not empty"
        if summary["profiles_empty"] is False
        else "missing or unreadable"
    )
    lines = [
        "## Matrix Profile motif discovery",
        "",
        f"- Total motif rows: **{format_number(summary['total_rows'])}**",
        f"- Assets: {join_values(summary['assets'])}",
        f"- Frequencies: {join_values(summary['frequencies'])}",
        f"- Evaluation table: {summary['evaluation_shape']}",
        f"- Evaluation numeric metrics: {join_values(summary['evaluation_metrics'])}",
        f"- Runtime table: {summary['runtime_shape']}",
        (
            f"- Profile table: {format_number(profile_rows)} rows x "
            f"{format_number(profile_cols)} columns; **{profile_status}**"
        ),
        "",
        "Matrix Profile successfully discovered motifs when motif result rows are "
        "present. Agnostic mode identifies global recurring subsequences, whereas "
        "conditioned mode repeats motif search within regimes, methods, and "
        "eligible segments. Consequently, a larger conditioned row count partly "
        "reflects repeated search scope. Raw motif count is not statistical "
        "significance, and shape similarity is not evidence of trading profitability.",
        "",
        "### Rows by asset, frequency, and mode",
        "",
        write_markdown_table(summary["rows_by_asset_frequency_mode"]),
        "",
        "### Rows by regime method",
        "",
        write_markdown_table(summary["rows_by_regime_method"]),
        "",
        "### Rows by regime label",
        "",
        write_markdown_table(summary["rows_by_regime_label"]),
        "",
        "### Rows by frequency and window length",
        "",
        write_markdown_table(summary["rows_by_frequency_window"]),
        "",
        "### Rows by feature set",
        "",
        write_markdown_table(summary["rows_by_feature_set"]),
        "",
        "### Rows by profile type",
        "",
        write_markdown_table(summary["rows_by_profile_type"]),
        "",
        "### Rows by asset, frequency, and regime label",
        "",
        write_markdown_table(summary["rows_by_asset_frequency_label"]),
        "",
        "### Rows by asset, frequency, and window length",
        "",
        write_markdown_table(summary["rows_by_asset_frequency_window"]),
        "",
        "### Rows by asset, frequency, and feature set",
        "",
        write_markdown_table(summary["rows_by_asset_frequency_feature"]),
        "",
        "### GPU audit",
        "",
        write_markdown_table(summary["gpu_counts"]),
        "",
        "### Evaluation grouped summaries",
        "",
        write_markdown_table(summary["evaluation_summary"]),
        "",
        "### Runtime summaries",
        "",
        write_markdown_table(summary["runtime_summary"]),
    ]
    if summary["runtime_caution"]:
        lines.extend(["", f"**Runtime caution:** {summary['runtime_caution']}"])
    return "\n".join(lines)


def locomotif_section(summary: dict[str, Any]) -> str:
    """Render the LoCoMotif report section."""
    failure_text = (
        "No internal failures were recorded."
        if summary["failure_rows"] == 0
        else f"Recorded failure rows: **{format_number(summary['failure_rows'])}**."
    )
    lines = [
        "## LoCoMotif motif discovery",
        "",
        f"- Total motif interval rows: **{format_number(summary['total_rows'])}**",
        f"- Assets: {join_values(summary['assets'])}",
        f"- Frequencies: {join_values(summary['frequencies'])}",
        f"- Evaluation table: {summary['evaluation_shape']}",
        f"- Evaluation rows: **{format_number(summary['evaluation_rows'])}**",
        f"- Evaluation metrics: {join_values(summary['evaluation_metrics'])}",
        f"- Runtime rows: **{format_number(summary['runtime_rows'])}**",
        f"- Failure table: {summary['failure_shape']}",
        f"- {failure_text}",
        "",
    ]
    if summary["appears_subset"]:
        lines.extend([LOCOMOTIF_SUBSET_CAVEAT, ""])
    lines.extend(
        [
            "### Rows by asset, frequency, and mode",
            "",
            write_markdown_table(summary["rows_by_asset_frequency_mode"]),
            "",
            "### Rows by regime method",
            "",
            write_markdown_table(summary["rows_by_regime_method"]),
            "",
            "### Rows by regime label",
            "",
            write_markdown_table(summary["rows_by_regime_label"]),
            "",
            "### Rows by feature set",
            "",
            write_markdown_table(summary["rows_by_feature_set"]),
            "",
            "### Motif-result status",
            "",
            write_markdown_table(summary["rows_by_status"]),
            "",
            "### Evaluation grouped summaries",
            "",
            write_markdown_table(summary["evaluation_summary"]),
            "",
            "### Runtime status",
            "",
            write_markdown_table(summary["runtime_status"]),
            "",
            "### Runtime summaries",
            "",
            write_markdown_table(summary["runtime_summary"]),
        ]
    )
    if summary["runtime_caution"]:
        lines.extend(["", f"**Runtime caution:** {summary['runtime_caution']}"])
    return "\n".join(lines)


def research_answer(
    quantile: dict[str, Any],
    hmm: dict[str, Any],
    matrix_profile: dict[str, Any],
    locomotif: dict[str, Any],
) -> str:
    """Construct a conservative answer to the empirical research question."""
    mp_total = matrix_profile["total_rows"]
    loco_total = locomotif["total_rows"]
    if not mp_total and not loco_total:
        return (
            "The available final files do not contain enough motif-result rows to "
            "answer the research question empirically. Regime and motif conclusions "
            "must remain pending until the missing or empty final outputs are supplied."
        )
    parts = [
        "The results support the presence of recurring subsequence shapes in the "
        "studied cryptocurrency series"
    ]
    if mode_count(matrix_profile, "conditioned"):
        parts.append(
            "and show that regime-conditioned discovery produces identifiable "
            "motifs within volatility states"
        )
    if mode_count(matrix_profile, "agnostic"):
        parts.append(
            "while agnostic discovery provides a global recurrence baseline"
        )
    answer = ", ".join(parts) + ". "
    answer += (
        "Differences in row counts across modes and algorithms describe the configured "
        "search process, not statistical significance, causality, or trading profitability. "
    )
    if locomotif["appears_subset"]:
        answer += LOCOMOTIF_SUBSET_CAVEAT
    elif loco_total:
        answer += (
            "LoCoMotif provides complementary flexible-length interval evidence, "
            "but its counts are not directly comparable with Matrix Profile pairs."
        )
    if quantile["total_rows"] is None and hmm["total_rows"] is None:
        answer += " Final regime-validation files were unavailable, limiting cross-method validation."
    return answer


def build_chatgpt_report(
    generated_at: str,
    inventory: pd.DataFrame,
    key_numbers: pd.DataFrame,
    quantile: dict[str, Any],
    volatility_audit: dict[str, Any],
    hmm: dict[str, Any],
    matrix_profile: dict[str, Any],
    locomotif: dict[str, Any],
    comparison: pd.DataFrame,
    figure_inventory: pd.DataFrame,
) -> str:
    """Build the most comprehensive Markdown output."""
    selected = selected_figure_table(figure_inventory)
    sections = [
        "# Final Results Consolidation for ChatGPT",
        "",
        f"Generated: {generated_at}",
        "",
        "## Direct answer to the research question",
        "",
        research_answer(quantile, hmm, matrix_profile, locomotif),
        "",
        "## Key numbers",
        "",
        write_markdown_table(key_numbers, max_rows=100),
        "",
        regime_section(quantile, volatility_audit["text"]),
        "",
        regime_section(hmm),
        "",
        matrix_profile_section(matrix_profile),
        "",
        locomotif_section(locomotif),
        "",
        "## Matrix Profile vs LoCoMotif",
        "",
        write_markdown_table(comparison, max_rows=50),
        "",
        "## Recommended figures",
        "",
        write_markdown_table(selected, max_rows=30),
        "",
        "## File inventory",
        "",
        write_markdown_table(inventory, max_rows=100),
        "",
        "## Caveats and missing inputs",
        "",
        *missing_input_lines(inventory),
        "",
        "- Motif discovery demonstrates recurring shape similarity; it does not "
        "establish profitability, causality, or predictive power.",
        "- Raw row counts depend on configured windows, feature sets, regimes, "
        "segments, and algorithm output structure.",
        "- Algorithm row counts are not directly comparable unless the search "
        "spaces and output units are harmonized.",
    ]
    return "\n".join(sections).rstrip() + "\n"


def slide(
    number: int,
    title: str,
    bullets: Sequence[str],
    figure: str,
    speaker_note: str,
) -> str:
    """Render one slide-ready Markdown block."""
    bullet_text = "\n".join(f"- {bullet}" for bullet in bullets[:5])
    return (
        f"## Slide {number}: {title}\n\n"
        f"{bullet_text}\n\n"
        f"**Recommended figure:** {figure}\n\n"
        f"**Speaker note:** {speaker_note}"
    )


def build_presentation_report(
    generated_at: str,
    quantile: dict[str, Any],
    hmm: dict[str, Any],
    volatility_audit: dict[str, Any],
    matrix_profile: dict[str, Any],
    locomotif: dict[str, Any],
    figure_inventory: pd.DataFrame,
) -> str:
    """Build the ten-slide presentation outline."""
    mp_agnostic = mode_count(matrix_profile, "agnostic")
    mp_conditioned = mode_count(matrix_profile, "conditioned")
    slides = [
        slide(
            1,
            "Research question",
            [
                "Do recurring cryptocurrency price/volatility patterns differ across market regimes?",
                "Compare regime-agnostic discovery with volatility-regime-conditioned discovery.",
                "Use Quantile and HMM labels for regime context.",
                "Compare fixed-length Matrix Profile motifs with flexible-length LoCoMotif intervals.",
            ],
            "Use a concise study-design diagram if one is available.",
            "The analysis asks whether conditioning motif discovery on market state changes the recurring structures that are observed. The empirical claims concern recurrence and regime context, not trading profitability.",
        ),
        slide(
            2,
            "Pipeline",
            [
                "Load precomputed feature and regime result files.",
                "Validate Quantile and HMM coverage, labels, transitions, and failures.",
                "Run agnostic and conditioned motif discovery.",
                "Evaluate recurrence, stability, overlap, runtime, and failure status.",
                "Consolidate tables and representative figures without rerunning notebooks.",
            ],
            "Use a pipeline diagram from the thesis, if available.",
            "All numbers in this presentation come from saved result files. The consolidation step is read-only for original results and preserves missing-file evidence.",
        ),
        slide(
            3,
            "Regime detection validation",
            [
                f"Quantile label rows: {format_number(quantile['total_rows'])}.",
                f"HMM label rows: {format_number(hmm['total_rows'])}.",
                f"Quantile failures: {format_number(quantile['failure_rows'])}; HMM failures: {format_number(hmm['failure_rows'])}.",
                f"Quantile labels: {join_values(quantile['regime_labels'])}.",
                volatility_audit["text"],
            ],
            recommended_figure(figure_inventory, "regime", ("close_by_regime", "transition")),
            "Regime validation establishes the conditioning variable used by the motif experiments. Missing files or failures should be reported as scope limitations rather than inferred away.",
        ),
        slide(
            4,
            "Matrix Profile setup",
            [
                f"Assets/frequencies: {join_values(matrix_profile['assets'])}; {join_values(matrix_profile['frequencies'])}.",
                "Agnostic mode searches the full eligible series.",
                "Conditioned mode repeats search inside regime-specific segments.",
                "Configured window lengths, feature sets, and profile types define the search space.",
                (
                    "GPU audit rows: "
                    f"true={format_number(bool_count(matrix_profile['gpu_counts'], True))}, "
                    f"false={format_number(bool_count(matrix_profile['gpu_counts'], False))}."
                ),
            ],
            recommended_figure(figure_inventory, "matrix_profile", ("profile", "agnostic")),
            "Matrix Profile measures subsequence shape similarity at configured fixed windows. Conditioned searches create more search units, so raw counts require normalization before comparative interpretation.",
        ),
        slide(
            5,
            "Matrix Profile key results",
            [
                f"Total motif rows: {format_number(matrix_profile['total_rows'])}.",
                f"Agnostic rows: {format_number(mp_agnostic)}.",
                f"Conditioned rows: {format_number(mp_conditioned)}.",
                f"Evaluation table: {matrix_profile['evaluation_shape']}.",
                "Recurring shapes were identified, but motif count is not a significance test.",
            ],
            recommended_figure(figure_inventory, "matrix_profile", ("overlay", "conditioned")),
            "The central result is successful recurrence discovery in global and, where present, regime-conditioned searches. Count differences partly reflect repeated searches over regimes, methods, and segments.",
        ),
        slide(
            6,
            "Matrix Profile figure evidence",
            [
                "Overlay plots show paired subsequences in the original context.",
                "Profile plots show low-distance candidate locations.",
                "Use both agnostic and conditioned examples.",
                "Include low-, high-, or extreme-volatility examples when available.",
                "Interpret visual similarity jointly with evaluation metrics.",
            ],
            recommended_figure(figure_inventory, "matrix_profile", ("overlay", "high_vol")),
            "Representative figures should demonstrate what the algorithm calls similar without implying economic value. The selected folder contains a compact set chosen for mode, regime, and plot-type diversity.",
        ),
        slide(
            7,
            "LoCoMotif setup",
            [
                f"Assets/frequencies: {join_values(locomotif['assets'])}; {join_values(locomotif['frequencies'])}.",
                "LoCoMotif searches flexible-length motif intervals.",
                "Agnostic and conditioned support is reported from completed outputs.",
                f"Runtime rows: {format_number(locomotif['runtime_rows'])}.",
                LOCOMOTIF_SUBSET_CAVEAT if locomotif["appears_subset"] else "Scope follows the completed result files.",
            ],
            recommended_figure(figure_inventory, "locomotif", ("agnostic", "overlay")),
            "LoCoMotif provides a complementary representation because interval lengths can vary. Its completed scope must be stated explicitly, especially if computation was capped.",
        ),
        slide(
            8,
            "LoCoMotif key results",
            [
                f"Motif interval rows: {format_number(locomotif['total_rows'])}.",
                f"Evaluation rows: {format_number(locomotif['evaluation_rows'])}.",
                f"Failure rows: {format_number(locomotif['failure_rows'])}.",
                f"Regime labels represented: {join_values(locomotif['regime_labels'])}.",
                "Use recurrence, motif length, and stability metrics where populated.",
            ],
            recommended_figure(figure_inventory, "locomotif", ("conditioned", "high_vol")),
            "LoCoMotif results should be interpreted within their completed computational scope. Empty outputs and failure logs are substantive audit evidence, not values to replace with assumptions.",
        ),
        slide(
            9,
            "Matrix Profile vs LoCoMotif",
            [
                "Matrix Profile: fixed-window shape recurrence and direct profile diagnostics.",
                "LoCoMotif: flexible-length interval motif sets.",
                "Output units differ, so raw motif counts are not directly comparable.",
                "Runtime comparisons require matched datasets, modes, and search settings.",
                "The methods provide complementary rather than interchangeable evidence.",
            ],
            recommended_figure(figure_inventory, "locomotif", ("overlay",)),
            "The comparison should focus on methodological behavior, completed scope, and qualitative evidence. A numerical winner cannot be inferred from unmatched raw counts.",
        ),
        slide(
            10,
            "Main findings and limitations",
            [
                research_answer(quantile, hmm, matrix_profile, locomotif),
                "Conditioned count inflation partly reflects repeated regime-specific searches.",
                "Shape recurrence does not establish statistical significance or profitability.",
                "Quantile interpretation must follow the actual stored volatility column.",
                "Missing files, failures, and capped experiments limit generalization.",
            ],
            recommended_figure(figure_inventory, "regime", ("transition",)),
            "The defensible conclusion is that recurring shapes can be characterized globally and within regimes when completed outputs support them. Predictive or economic claims require separate out-of-sample testing.",
        ),
    ]
    return (
        "# Final Results for Presentation\n\n"
        f"Generated: {generated_at}\n\n"
        + "\n\n".join(slides)
        + "\n"
    )


def latex_escape(value: Any) -> str:
    """Escape plain report text for safe inclusion in LaTeX prose."""
    text = format_number(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def build_latex_report(
    quantile: dict[str, Any],
    hmm: dict[str, Any],
    volatility_audit: dict[str, Any],
    matrix_profile: dict[str, Any],
    locomotif: dict[str, Any],
) -> str:
    """Build LaTeX-safe thesis prose in a Markdown container."""
    mp_agnostic = mode_count(matrix_profile, "agnostic")
    mp_conditioned = mode_count(matrix_profile, "conditioned")
    subset = (
        latex_escape(LOCOMOTIF_SUBSET_CAVEAT)
        if locomotif["appears_subset"]
        else "The reported LoCoMotif scope follows the completed output files."
    )
    caveat = (
        latex_escape(QUANTILE_CAVEAT)
        if volatility_audit["all_rolling_60"]
        else latex_escape(volatility_audit["text"])
    )
    return f"""# LaTeX-ready thesis text

\\section{{Regime Detection Validation}}

The quantile procedure produced {latex_escape(quantile['total_rows'])} labelled observations, while the hidden Markov model procedure produced {latex_escape(hmm['total_rows'])} labelled observations. The corresponding failure logs contained {latex_escape(quantile['failure_rows'])} and {latex_escape(hmm['failure_rows'])} rows, respectively. The observed quantile labels were {latex_escape(join_values(quantile['regime_labels']))}, and the observed HMM labels were {latex_escape(join_values(hmm['regime_labels']))}. Missing or empty outputs are treated as limitations of the completed computational scope rather than imputed results.

{caveat}

\\section{{Matrix Profile Motif Discovery Results}}

The Matrix Profile output contained {latex_escape(matrix_profile['total_rows'])} motif-result rows. Of these, {latex_escape(mp_agnostic)} rows were associated with agnostic searches and {latex_escape(mp_conditioned)} rows were associated with regime-conditioned searches. Agnostic discovery identifies recurring subsequences over the full eligible series, whereas conditioned discovery repeats the search within eligible regime-specific subsets. The resulting motifs constitute evidence of shape recurrence under the configured windows and features; they do not by themselves establish statistical significance, predictive ability, or economic profitability.

\\section{{LoCoMotif Motif Discovery Results}}

The LoCoMotif output contained {latex_escape(locomotif['total_rows'])} motif interval rows and {latex_escape(locomotif['evaluation_rows'])} evaluation rows. The failure log contained {latex_escape(locomotif['failure_rows'])} rows. {subset} LoCoMotif provides complementary evidence because it represents recurring structures as flexible-length intervals rather than fixed-window motif pairs.

\\section{{Comparison of Agnostic and Regime-Conditioned Discovery}}

The agnostic configuration provides a global recurrence baseline, while regime-conditioned configurations evaluate recurring shapes within market states. A larger conditioned output count can arise mechanically because motif search is repeated over multiple regimes, regime methods, and contiguous segments. Therefore, raw row counts should not be interpreted as direct evidence that one mode is statistically superior. Comparisons should instead consider matched search settings, recurrence metrics, stability, cross-regime overlap, and representative motif plots.

\\section{{Computational Performance}}

Runtime measurements were summarized by the available asset, frequency, mode, profile type, and status fields. Runtime comparisons are valid only when the underlying data volume and search configuration are comparable. Unusually large values should be checked for cumulative job-time accounting, initialization overhead, or retries before being quoted as per-experiment execution time.

\\section{{Summary of Empirical Findings}}

{latex_escape(research_answer(quantile, hmm, matrix_profile, locomotif))}

\\section{{Limitations}}

The study is limited by the completed computational scope, any missing result files, recorded failures, and potentially capped LoCoMotif experiments. Motif counts depend on the number of assets, frequencies, windows, features, regimes, and segments searched. Matrix Profile pair rows and LoCoMotif interval rows are different output units and are not directly comparable. Finally, recurring shape similarity does not establish causality, statistical significance, out-of-sample predictability, or trading profitability.
"""


def build_key_numbers(
    quantile: dict[str, Any],
    hmm: dict[str, Any],
    matrix_profile: dict[str, Any],
    locomotif: dict[str, Any],
    figure_inventory: pd.DataFrame,
    selected_figures: list[Path],
) -> pd.DataFrame:
    """Create the required compact key-number CSV."""
    rows: list[dict[str, Any]] = []
    add_key_number(rows, "quantile total rows", quantile["total_rows"], "Quantile", "results/regimes/quantile/quantile_regime_labels.parquet")
    add_key_number(rows, "HMM total rows", hmm["total_rows"], "HMM", "results/regimes/hmm/hmm_regime_labels.parquet")
    add_key_number(rows, "quantile failure rows", quantile["failure_rows"], "Quantile", "results/logs/01_quantile_failures.parquet")
    add_key_number(rows, "HMM failure rows", hmm["failure_rows"], "HMM", "results/logs/02_hmm_failures.parquet")
    add_key_number(rows, "Matrix Profile motif rows", matrix_profile["total_rows"], "Matrix Profile", "results/motifs/matrix_profile/matrix_profile_motif_results.parquet")
    add_key_number(rows, "Matrix Profile evaluation rows", len(matrix_profile["evaluation"]) if matrix_profile["evaluation"] is not None else None, "Matrix Profile", "results/motifs/matrix_profile/matrix_profile_evaluation.parquet")
    add_key_number(rows, "Matrix Profile runtime rows", len(matrix_profile["runtime"]) if matrix_profile["runtime"] is not None else None, "Matrix Profile", "results/motifs/matrix_profile/matrix_profile_runtime.parquet")
    add_key_number(rows, "Matrix Profile figure count", figure_count(figure_inventory, "matrix_profile"), "Matrix Profile", "results/figures")
    add_key_number(rows, "Matrix Profile agnostic rows", mode_count(matrix_profile, "agnostic"), "Matrix Profile", "results/motifs/matrix_profile/matrix_profile_motif_results.parquet")
    add_key_number(rows, "Matrix Profile conditioned rows", mode_count(matrix_profile, "conditioned"), "Matrix Profile", "results/motifs/matrix_profile/matrix_profile_motif_results.parquet")
    add_key_number(rows, "Matrix Profile GPU true rows", bool_count(matrix_profile["gpu_counts"], True), "Matrix Profile", "results/motifs/matrix_profile/matrix_profile_motif_results.parquet")
    add_key_number(rows, "Matrix Profile GPU false rows", bool_count(matrix_profile["gpu_counts"], False), "Matrix Profile", "results/motifs/matrix_profile/matrix_profile_motif_results.parquet")
    add_key_number(rows, "LoCoMotif motif rows", locomotif["total_rows"], "LoCoMotif", "results/motifs/locomotif/locomotif_motif_results.parquet")
    add_key_number(rows, "LoCoMotif evaluation rows", locomotif["evaluation_rows"], "LoCoMotif", "results/motifs/locomotif/locomotif_evaluation.parquet")
    add_key_number(rows, "LoCoMotif runtime rows", locomotif["runtime_rows"], "LoCoMotif", "results/motifs/locomotif/locomotif_runtime.parquet")
    add_key_number(rows, "LoCoMotif failure rows", locomotif["failure_rows"], "LoCoMotif", "results/motifs/locomotif/04_locomotif_failures.parquet")
    add_key_number(rows, "LoCoMotif figure count", figure_count(figure_inventory, "locomotif"), "LoCoMotif", "results/figures")
    add_key_number(rows, "selected final figure count", len(selected_figures), "All", "results/figures")
    return pd.DataFrame(rows, columns=["metric", "value", "method", "source_file", "note"])


def terminal_summary(
    key_numbers: pd.DataFrame,
    selected_figures: list[Path],
    inventory: pd.DataFrame,
    volatility_audit: dict[str, Any],
    locomotif: dict[str, Any],
) -> str:
    """Build the terminal and text-file summary."""
    key_lines = [
        f"- {row.metric}: {format_number(row.value)}"
        for row in key_numbers.itertuples(index=False)
    ]
    selected_lines = [
        f"- {path.name}" for path in selected_figures
    ] or ["- No eligible figures were available in results/figures."]
    caveats = [
        "- Motif counts do not establish significance, causality, or profitability.",
        "- Matrix Profile and LoCoMotif output rows are not directly comparable.",
    ]
    if volatility_audit["all_rolling_60"]:
        caveats.append(f"- {QUANTILE_CAVEAT}")
    if locomotif["appears_subset"]:
        caveats.append(f"- {LOCOMOTIF_SUBSET_CAVEAT}")
    missing_count = int((~inventory["exists"]).sum())
    files_created = [
        "FINAL_RESULTS_FOR_CHATGPT.md",
        "FINAL_RESULTS_FOR_PRESENTATION.md",
        "FINAL_RESULTS_FOR_THESIS_LATEX.md",
        "final_results_key_numbers.csv",
        "final_results_table_inventory.csv",
        "final_results_figure_recommendations.csv",
        "final_results_terminal_summary.txt",
        "selected_final_figures/",
    ]
    return "\n".join(
        [
            "FINAL RESULTS CONSOLIDATION",
            f"Output folder: {OUTPUT_ROOT}",
            "",
            "Files created:",
            *[f"- {name}" for name in files_created],
            "",
            "Key numbers:",
            *key_lines,
            "",
            "Top findings:",
            "- Saved regime and motif results were summarized without rerunning notebooks.",
            "- Agnostic and conditioned motif outputs are reported separately where available.",
            f"- Missing expected input files: {missing_count}.",
            "",
            "Selected figures:",
            *selected_lines,
            "",
            "Caveats:",
            *caveats,
        ]
    ) + "\n"


def main() -> int:
    """Run consolidation and write all requested outputs."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SELECTED_FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    inventory = file_inventory()

    quantile_paths = {
        "summary": result_path("regimes/quantile/quantile_regime_summary.parquet"),
        "transition_matrix": result_path("regimes/quantile/quantile_transition_matrix.parquet"),
        "failures": result_path("logs/01_quantile_failures.parquet"),
    }
    quantile = summarize_regimes(
        "Quantile",
        result_path("regimes/quantile/quantile_regime_labels.parquet"),
        quantile_paths,
    )
    volatility_audit = quantile_volatility_audit(quantile)

    hmm_paths = {
        "summary": result_path("regimes/hmm/hmm_regime_summary.parquet"),
        "transition_matrix": result_path("regimes/hmm/hmm_transition_matrix.parquet"),
        "model_selection": result_path("regimes/hmm/hmm_model_selection.parquet"),
        "persistence_metrics": result_path("regimes/hmm/hmm_persistence_metrics.parquet"),
        "feature_diagnostics": result_path("regimes/hmm/hmm_feature_diagnostics.parquet"),
        "quantile_comparison": result_path("regimes/hmm/hmm_quantile_comparison.parquet"),
        "confusion_table": result_path("regimes/hmm/hmm_quantile_confusion_table.parquet"),
        "failures": result_path("logs/02_hmm_failures.parquet"),
    }
    hmm = summarize_regimes(
        "HMM", result_path("regimes/hmm/hmm_regime_labels.parquet"), hmm_paths
    )

    matrix_profile_paths = {
        "motif": result_path("motifs/matrix_profile/matrix_profile_motif_results.parquet"),
        "evaluation": result_path("motifs/matrix_profile/matrix_profile_evaluation.parquet"),
        "runtime": result_path("motifs/matrix_profile/matrix_profile_runtime.parquet"),
        "profiles": result_path("motifs/matrix_profile/matrix_profile_profiles.parquet"),
    }
    matrix_profile = summarize_matrix_profile(matrix_profile_paths)

    locomotif_paths = {
        "motif": result_path("motifs/locomotif/locomotif_motif_results.parquet"),
        "evaluation": result_path("motifs/locomotif/locomotif_evaluation.parquet"),
        "runtime": result_path("motifs/locomotif/locomotif_runtime.parquet"),
        "failures": result_path("motifs/locomotif/04_locomotif_failures.parquet"),
    }
    locomotif = summarize_locomotif(locomotif_paths)

    figure_inventory, selected_figures = select_figures()
    inventory = merge_read_errors_into_inventory(inventory)
    key_numbers = build_key_numbers(
        quantile,
        hmm,
        matrix_profile,
        locomotif,
        figure_inventory,
        selected_figures,
    )
    comparison = comparison_table(matrix_profile, locomotif)

    inventory.to_csv(OUTPUT_ROOT / "final_results_table_inventory.csv", index=False)
    key_numbers.to_csv(OUTPUT_ROOT / "final_results_key_numbers.csv", index=False)
    figure_inventory.to_csv(
        OUTPUT_ROOT / "final_results_figure_recommendations.csv", index=False
    )

    chatgpt_report = build_chatgpt_report(
        generated_at,
        inventory,
        key_numbers,
        quantile,
        volatility_audit,
        hmm,
        matrix_profile,
        locomotif,
        comparison,
        figure_inventory,
    )
    presentation_report = build_presentation_report(
        generated_at,
        quantile,
        hmm,
        volatility_audit,
        matrix_profile,
        locomotif,
        figure_inventory,
    )
    latex_report = build_latex_report(
        quantile, hmm, volatility_audit, matrix_profile, locomotif
    )
    summary_text = terminal_summary(
        key_numbers,
        selected_figures,
        inventory,
        volatility_audit,
        locomotif,
    )

    (OUTPUT_ROOT / "FINAL_RESULTS_FOR_CHATGPT.md").write_text(
        chatgpt_report, encoding="utf-8"
    )
    (OUTPUT_ROOT / "FINAL_RESULTS_FOR_PRESENTATION.md").write_text(
        presentation_report, encoding="utf-8"
    )
    (OUTPUT_ROOT / "FINAL_RESULTS_FOR_THESIS_LATEX.md").write_text(
        latex_report, encoding="utf-8"
    )
    (OUTPUT_ROOT / "final_results_terminal_summary.txt").write_text(
        summary_text, encoding="utf-8"
    )

    print(summary_text, end="")
    print(str((OUTPUT_ROOT / "FINAL_RESULTS_FOR_CHATGPT.md").resolve()))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Consolidation interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
