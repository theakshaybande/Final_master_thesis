from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


THESIS_SCOPE_ASSETS = ["BTCUSDT", "ETHUSDT"]
THESIS_SCOPE_FREQUENCIES = ["15m", "1h"]


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / "HPC workflow" / "HPC_Regime_and_motif_discovery").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root containing HPC workflow/HPC_Regime_and_motif_discovery")


PROJECT_ROOT = find_project_root()
WORKFLOW_ROOT = PROJECT_ROOT / "HPC workflow" / "HPC_Regime_and_motif_discovery"
RESULTS_ROOT = WORKFLOW_ROOT / "results"
REPORT_ROOT = WORKFLOW_ROOT / "reports" / "study_notebooks"
TABLE_DIR = REPORT_ROOT / "tables"
HTML_EXPORT_DIR = REPORT_ROOT / "html_exports"
FIGURE_DIRS = {
    "regime": REPORT_ROOT / "figures" / "regime",
    "regime_comparison": REPORT_ROOT / "figures" / "regime_comparison",
    "matrix_profile": REPORT_ROOT / "figures" / "matrix_profile",
    "locomotif": REPORT_ROOT / "figures" / "locomotif",
}


def ensure_study_output_dirs() -> None:
    for path in [REPORT_ROOT, TABLE_DIR, HTML_EXPORT_DIR, *FIGURE_DIRS.values()]:
        path.mkdir(parents=True, exist_ok=True)


def result_path(*parts: str) -> Path:
    return RESULTS_ROOT.joinpath(*parts)


def resolve_existing_file(folder: Path, expected_name: str) -> Path:
    """Return exact path when present, otherwise a matching suffixed file if available."""
    folder = Path(folder)
    exact = folder / expected_name
    if exact.exists():
        return exact
    stem = Path(expected_name).stem
    suffix = Path(expected_name).suffix
    matches = sorted(folder.glob(f"{stem}*{suffix}"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    if matches:
        return matches[0]
    return exact


def safe_read_parquet(path: Path | str, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        print(f"Missing file: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path, columns=columns)
        print(f"Loaded {path.name}: {len(df):,} rows, {len(df.columns):,} columns")
        return df
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return pd.DataFrame()


def display_table(df: pd.DataFrame, n: int = 20):
    if df is None or df.empty:
        print("No rows available.")
        return df
    display(df.head(n))
    return df


def save_table(df: pd.DataFrame, name: str) -> Path | None:
    ensure_study_output_dirs()
    if df is None or df.empty:
        print(f"Table not saved because it is empty: {name}")
        return None
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved table: {path}")
    return path


def clean_filename(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name.strip("_") or "figure"


def save_fig(fig, name: str, folder: Path | str, dpi: int = 180, pdf: bool = True) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base = clean_filename(name)
    outputs = []
    png = folder / f"{base}.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    outputs.append(png)
    if pdf:
        try:
            pdf_path = folder / f"{base}.pdf"
            fig.savefig(pdf_path, bbox_inches="tight")
            outputs.append(pdf_path)
        except Exception as exc:
            print(f"PDF save failed for {base}: {exc}")
    print("Saved figure:", ", ".join(str(p) for p in outputs))
    return outputs


def z_normalize(x):
    arr = pd.Series(x).astype(float).to_numpy()
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std == 0:
        return arr - mean
    return (arr - mean) / std


def find_distance_column(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    preferred = ["motif_distance", "distance", "matrix_profile_value"]
    for col in preferred:
        if col in df.columns and col in numeric_cols:
            return col
    for col in numeric_cols:
        if "distance" in col.lower():
            return col
    for col in numeric_cols:
        if "profile" in col.lower():
            return col
    return None


def infer_feature_file(asset: str, frequency: str) -> Path:
    feature_root = PROJECT_ROOT / "final_dataset" / "features"
    exact = feature_root / "crypto" / f"{asset}_{frequency}_features_2020_2025.parquet"
    if exact.exists():
        return exact
    matches = sorted(feature_root.rglob(f"{asset}_{frequency}_features*.parquet"))
    if matches:
        return matches[0]
    return exact


def timestamp_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "timestamp", "datetime", "date", "time",
        "motif_timestamp_1", "motif_start_timestamp",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if "timestamp" in col.lower() or col.lower().endswith("time"):
            return col
    return None


def regime_column(df: pd.DataFrame) -> str | None:
    for col in ["regime_label", "regime", "state", "raw_state", "regime_code"]:
        if col in df.columns:
            return col
    return None


def coerce_timestamp(df: pd.DataFrame, col: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    col = col or timestamp_column(df)
    if col and col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def filter_scope(df: pd.DataFrame, asset: str | None = None, frequency: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if asset is not None and "asset" in out.columns:
        out = out[out["asset"].astype(str) == str(asset)]
    if frequency is not None and "frequency" in out.columns:
        out = out[out["frequency"].astype(str) == str(frequency)]
    return out


def choose_quantile_method(df: pd.DataFrame, preferred: str = "quantile_2_rolling_240") -> str | None:
    if df is None or df.empty or "regime_method" not in df.columns:
        return None
    methods = sorted(df["regime_method"].dropna().astype(str).unique())
    if preferred in methods:
        return preferred
    for method in methods:
        if "quantile_2" in method:
            return method
    return methods[0] if methods else None


def useful_columns(df: pd.DataFrame, max_cols: int = 12) -> str:
    if df is None or df.empty:
        return ""
    preferred = [
        "asset", "frequency", "mode", "regime_method", "regime_label", "feature_set",
        "profile_type", "window_length", "motif_distance", "distance", "matrix_profile",
        "runtime_seconds", "used_gpu", "status",
    ]
    cols = [c for c in preferred if c in df.columns]
    cols.extend([c for c in df.columns if c not in cols][: max_cols - len(cols)])
    return ", ".join(cols[:max_cols])


def load_feature_data(asset: str, frequency: str) -> pd.DataFrame:
    path = infer_feature_file(asset, frequency)
    df = safe_read_parquet(path)
    if df.empty:
        return df
    ts = timestamp_column(df)
    if ts:
        df = coerce_timestamp(df, ts).sort_values(ts)
    return df


def find_feature_value_column(df: pd.DataFrame, requested: str | None = None) -> str | None:
    if df is None or df.empty:
        return None
    if requested and requested in df.columns and pd.api.types.is_numeric_dtype(df[requested]):
        return requested
    if requested and requested == "multivariate":
        requested = None
    candidates = [requested, "close", "log_return", "rolling_volatility_60", "rolling_vol", "volume"]
    for col in candidates:
        if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return col
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numeric[0] if numeric else None


def group_count_table(df: pd.DataFrame, cols: list[str], name: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        print(f"No data for grouping {cols}")
        return pd.DataFrame(columns=cols + ["rows"])
    existing = [c for c in cols if c in df.columns]
    if not existing:
        print(f"None of these columns are available: {cols}")
        return pd.DataFrame()
    out = df.groupby(existing, dropna=False).size().reset_index(name="rows").sort_values("rows", ascending=False)
    if name:
        save_table(out, name)
    return out


def plot_heatmap_from_table(df: pd.DataFrame, title: str, fig_name: str, folder: Path, filters: dict | None = None):
    if df is None or df.empty:
        print(f"No transition/confusion data available for {title}")
        return None
    work = df.copy()
    filters = filters or {}
    for col, value in filters.items():
        if col in work.columns:
            work = work[work[col].astype(str) == str(value)]
    if work.empty:
        print(f"No rows after filtering for {title}: {filters}")
        return None
    from_candidates = ["from_regime", "source_regime", "regime_from", "hmm_regime", "hmm_label", "row_label", "from_label"]
    to_candidates = ["to_regime", "target_regime", "regime_to", "quantile_regime", "quantile_label", "column_label", "to_label"]
    value_candidates = ["probability", "transition_probability", "value", "count", "n", "rows"]
    from_col = next((c for c in from_candidates if c in work.columns), None)
    to_col = next((c for c in to_candidates if c in work.columns), None)
    value_col = next((c for c in value_candidates if c in work.columns and pd.api.types.is_numeric_dtype(work[c])), None)
    if from_col and to_col and value_col:
        matrix = work.pivot_table(index=from_col, columns=to_col, values=value_col, aggfunc="sum", fill_value=0)
    else:
        numeric = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
        label_cols = [c for c in work.columns if c not in numeric]
        if numeric and label_cols:
            matrix = work.set_index(label_cols[0])[numeric]
        elif numeric:
            matrix = work[numeric]
        else:
            print(f"Could not infer heatmap structure for {title}")
            return None
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel(matrix.columns.name or "To")
    ax.set_ylabel(matrix.index.name or "From")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([str(c) for c in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([str(i) for i in matrix.index])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_fig(fig, fig_name, folder)
    plt.show()
    return matrix
