from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = PROJECT_ROOT / "HPC workflow" / "HPC_Regime_and_motif_discovery"
STUDY_DIR = WORKFLOW_ROOT / "notebooks" / "study"
REPORT_ROOT = WORKFLOW_ROOT / "reports" / "study_notebooks"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


HELPERS = r'''
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
'''


SETUP_CODE = r'''
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NOTEBOOK_DIR = Path.cwd() / "HPC workflow" / "HPC_Regime_and_motif_discovery" / "notebooks" / "study"
if NOTEBOOK_DIR.exists() and str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from study_helpers import *

ensure_study_output_dirs()
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

print("Project root:", PROJECT_ROOT)
print("Workflow root:", WORKFLOW_ROOT)
print("Study outputs:", REPORT_ROOT)
'''


def notebook_1():
    cells = [
        md("""
# 01 Regime Detection Visual Study

## Objective
Study, visualize, and document the already-computed quantile and HMM regime labels for the thesis scope: BTCUSDT and ETHUSDT, 15-minute and 1-hour frequencies, 2020-2025.

## Input files
- Quantile regime outputs under `results/regimes/quantile`
- HMM regime outputs under `results/regimes/hmm`
- Feature files under `final_dataset/features`

## Output folder
`reports/study_notebooks`, especially `figures/regime` and `tables`.

## Thesis relevance
These figures explain what market regimes were detected and how those labels condition motif discovery. This notebook does not rerun regime detection or modify original result files.

## Analysis-only safety
This notebook never imports or calls STUMPY, STUMP/MSTUMP, HMM fitting, LoCoMotif search, or any other expensive experiment algorithm. It only reads saved result files and thesis-scope feature parquet files, then produces derived tables and figures.

**Quantile caveat.** The quantile regime outputs store `rolling_volatility_60` as the actual volatility column across all quantile method identifiers. Therefore, quantile regimes are interpreted as 60-period rolling-volatility regimes with different regime-count granularities rather than as separate 30/60/240 volatility-horizon experiments.
"""),
        code(SETUP_CODE),
        md("## Load Quantile and HMM Thesis-Scope Files"),
        code(r'''
quantile_dir = result_path("regimes", "quantile")
hmm_dir = result_path("regimes", "hmm")

quantile_files = {
    "labels": resolve_existing_file(quantile_dir, "quantile_regime_labels.parquet"),
    "summary": resolve_existing_file(quantile_dir, "quantile_regime_summary.parquet"),
    "transitions": resolve_existing_file(quantile_dir, "quantile_transition_matrix.parquet"),
}
hmm_files = {
    "labels": resolve_existing_file(hmm_dir, "hmm_regime_labels.parquet"),
    "summary": resolve_existing_file(hmm_dir, "hmm_regime_summary.parquet"),
    "transitions": resolve_existing_file(hmm_dir, "hmm_transition_matrix.parquet"),
    "persistence": resolve_existing_file(hmm_dir, "hmm_persistence_metrics.parquet"),
}

quantile_labels = coerce_timestamp(safe_read_parquet(quantile_files["labels"]))
quantile_summary = safe_read_parquet(quantile_files["summary"])
quantile_transitions = safe_read_parquet(quantile_files["transitions"])
hmm_labels = coerce_timestamp(safe_read_parquet(hmm_files["labels"]))
hmm_summary = safe_read_parquet(hmm_files["summary"])
hmm_transitions = safe_read_parquet(hmm_files["transitions"])
hmm_persistence = safe_read_parquet(hmm_files["persistence"])

file_inventory = pd.DataFrame([
    {"dataset": f"quantile_{k}", "path": str(v), "exists": v.exists()} for k, v in quantile_files.items()
] + [
    {"dataset": f"hmm_{k}", "path": str(v), "exists": v.exists()} for k, v in hmm_files.items()
])
display_table(file_inventory)
save_table(file_inventory, "study_regime_file_inventory")
'''),
        md("## Validate Files, Shapes, Columns, and Date Ranges"),
        code(r'''
def frame_schema(name, df):
    ts = timestamp_column(df)
    return {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns) if df is not None else 0,
        "timestamp_column": ts,
        "min_timestamp": df[ts].min() if ts and not df.empty else pd.NaT,
        "max_timestamp": df[ts].max() if ts and not df.empty else pd.NaT,
        "useful_columns": useful_columns(df),
    }

schema = pd.DataFrame([
    frame_schema("quantile_labels", quantile_labels),
    frame_schema("quantile_summary", quantile_summary),
    frame_schema("quantile_transitions", quantile_transitions),
    frame_schema("hmm_labels", hmm_labels),
    frame_schema("hmm_summary", hmm_summary),
    frame_schema("hmm_transitions", hmm_transitions),
    frame_schema("hmm_persistence", hmm_persistence),
])
display_table(schema, 20)
save_table(schema, "study_regime_schema_validation")
'''),
        md("## Regime Coverage Table"),
        code(r'''
def coverage_row(df, asset, frequency, method_name):
    scoped = filter_scope(df, asset, frequency)
    ts = timestamp_column(scoped)
    reg = regime_column(scoped)
    method_col = "regime_method" if "regime_method" in scoped.columns else None
    return {
        "asset": asset,
        "frequency": frequency,
        "method": method_name,
        "rows": len(scoped),
        "min_timestamp": scoped[ts].min() if ts and not scoped.empty else pd.NaT,
        "max_timestamp": scoped[ts].max() if ts and not scoped.empty else pd.NaT,
        "regime_methods": ", ".join(sorted(scoped[method_col].dropna().astype(str).unique())) if method_col and not scoped.empty else "",
        "regime_labels": ", ".join(sorted(scoped[reg].dropna().astype(str).unique())) if reg and not scoped.empty else "",
    }

coverage = pd.DataFrame(
    [coverage_row(quantile_labels, asset, frequency, "quantile") for asset in THESIS_SCOPE_ASSETS for frequency in THESIS_SCOPE_FREQUENCIES]
    + [coverage_row(hmm_labels, asset, frequency, "HMM") for asset in THESIS_SCOPE_ASSETS for frequency in THESIS_SCOPE_FREQUENCIES]
)
display_table(coverage, 20)
save_table(coverage, "study_regime_coverage_table")
'''),
        md("## Regime Label Counts"),
        code(r'''
def plot_regime_counts(df, method_name, asset, frequency, filename):
    scoped = filter_scope(df, asset, frequency)
    if method_name.lower() == "quantile":
        selected = choose_quantile_method(scoped)
        if selected and "regime_method" in scoped.columns:
            scoped = scoped[scoped["regime_method"].astype(str) == selected]
            method_title = f"{method_name}: {selected}"
        else:
            method_title = method_name
    else:
        method_title = method_name
    reg = regime_column(scoped)
    if scoped.empty or not reg:
        print(f"No label counts available for {asset} {frequency} {method_name}")
        return
    counts = scoped[reg].astype(str).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_title(f"Regime label counts - {asset} {frequency} {method_title}")
    ax.set_xlabel("Regime label")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    save_fig(fig, filename, FIGURE_DIRS["regime"])
    plt.show()

for asset in ["BTCUSDT", "ETHUSDT"]:
    plot_regime_counts(quantile_labels, "quantile", asset, "15m", f"study_regime_counts_{asset}_15m_quantile")
    plot_regime_counts(hmm_labels, "HMM", asset, "15m", f"study_regime_counts_{asset}_15m_hmm")
'''),
        md("## Price and Regime Overlays"),
        code(r'''
def plot_price_regime_overlay(labels, method_name, asset, frequency, filename, max_points=6000):
    scoped = filter_scope(labels, asset, frequency)
    if method_name.lower() == "quantile":
        selected = choose_quantile_method(scoped)
        if selected and "regime_method" in scoped.columns:
            scoped = scoped[scoped["regime_method"].astype(str) == selected]
    ts = timestamp_column(scoped)
    reg = regime_column(scoped)
    feature = load_feature_data(asset, frequency)
    fts = timestamp_column(feature)
    if scoped.empty or feature.empty or not ts or not fts or not reg or "close" not in feature.columns:
        print(f"Cannot draw overlay for {asset} {frequency} {method_name}: missing labels, timestamps, or close.")
        return
    merged = pd.merge(
        feature[[fts, "close"]].rename(columns={fts: "timestamp"}),
        scoped[[ts, reg]].rename(columns={ts: "timestamp", reg: "regime_label"}),
        on="timestamp",
        how="inner",
    ).dropna(subset=["timestamp", "close", "regime_label"]).sort_values("timestamp")
    if merged.empty:
        print(f"No timestamp overlap for {asset} {frequency} {method_name}")
        return
    if len(merged) > max_points:
        step = int(np.ceil(len(merged) / max_points))
        merged = merged.iloc[::step].copy()
    labels_sorted = sorted(merged["regime_label"].astype(str).unique())
    color_map = {label: plt.cm.tab10(i % 10) for i, label in enumerate(labels_sorted)}
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(merged["timestamp"], merged["close"], color="0.75", linewidth=0.8, label="close")
    for label in labels_sorted:
        part = merged[merged["regime_label"].astype(str) == label]
        ax.scatter(part["timestamp"], part["close"], s=7, color=color_map[label], label=str(label), alpha=0.75)
    ax.set_title(f"{asset} {frequency} close price by {method_name} regime")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Close price")
    ax.legend(title="Regime", ncol=min(4, len(labels_sorted)))
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, filename, FIGURE_DIRS["regime"])
    plt.show()

for asset in ["BTCUSDT", "ETHUSDT"]:
    plot_price_regime_overlay(quantile_labels, "quantile", asset, "15m", f"study_regime_{asset}_15m_quantile_close_by_regime")
    plot_price_regime_overlay(hmm_labels, "HMM", asset, "15m", f"study_regime_{asset}_15m_hmm_close_by_regime")
'''),
        md("""
### Interpretation
The overlay figures show where regime labels occur on the observed price path. Dense 15-minute data is sampled only for visualization clarity; the saved regime labels themselves are not changed.
"""),
        md("## Volatility Distribution by Regime"),
        code(r'''
def plot_volatility_by_regime(labels, method_name, asset, frequency, filename):
    scoped = filter_scope(labels, asset, frequency)
    if method_name.lower() == "quantile":
        selected = choose_quantile_method(scoped)
        if selected and "regime_method" in scoped.columns:
            scoped = scoped[scoped["regime_method"].astype(str) == selected]
    reg = regime_column(scoped)
    vol_col = next((c for c in ["volatility_value", "rolling_volatility_60", "rolling_vol"] if c in scoped.columns), None)
    if not vol_col:
        feature = load_feature_data(asset, frequency)
        ts = timestamp_column(scoped)
        fts = timestamp_column(feature)
        feature_vol = next((c for c in ["rolling_volatility_60", "rolling_vol"] if c in feature.columns), None)
        if ts and fts and feature_vol and reg:
            scoped = pd.merge(
                scoped[[ts, reg]].rename(columns={ts: "timestamp"}),
                feature[[fts, feature_vol]].rename(columns={fts: "timestamp"}),
                on="timestamp",
                how="inner",
            )
            vol_col = feature_vol
    if scoped.empty or not reg or not vol_col:
        print(f"No volatility column available for {asset} {frequency} {method_name}")
        return
    groups = [g[vol_col].dropna().to_numpy() for _, g in scoped.groupby(reg)]
    labels_order = [str(k) for k, _ in scoped.groupby(reg)]
    if not groups:
        print(f"No volatility observations available for {asset} {frequency} {method_name}")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(groups, labels=labels_order, showfliers=False)
    ax.set_title(f"Volatility distribution by regime - {asset} {frequency} {method_name}")
    ax.set_xlabel("Regime label")
    ax.set_ylabel(vol_col)
    fig.tight_layout()
    save_fig(fig, filename, FIGURE_DIRS["regime"])
    plt.show()

for asset in ["BTCUSDT", "ETHUSDT"]:
    plot_volatility_by_regime(quantile_labels, "quantile", asset, "15m", f"study_regime_{asset}_15m_quantile_volatility_by_regime")
    plot_volatility_by_regime(hmm_labels, "HMM", asset, "15m", f"study_regime_{asset}_15m_hmm_volatility_by_regime")
'''),
        md("## Transition Heatmaps"),
        code(r'''
q_method = choose_quantile_method(filter_scope(quantile_labels, "BTCUSDT", "15m"))
q_filters = {"asset": "BTCUSDT", "frequency": "15m"}
if q_method:
    q_filters["regime_method"] = q_method
plot_heatmap_from_table(
    quantile_transitions,
    "BTCUSDT 15m quantile transition probabilities",
    "study_regime_BTCUSDT_15m_quantile_transition_heatmap",
    FIGURE_DIRS["regime"],
    q_filters,
)
plot_heatmap_from_table(
    hmm_transitions,
    "BTCUSDT 15m HMM transition probabilities",
    "study_regime_BTCUSDT_15m_hmm_transition_heatmap",
    FIGURE_DIRS["regime"],
    {"asset": "BTCUSDT", "frequency": "15m"},
)
'''),
        md("## Regime Duration and Persistence"),
        code(r'''
def segment_durations(labels, method_name):
    rows = []
    if labels is None or labels.empty:
        return pd.DataFrame()
    ts = timestamp_column(labels)
    reg = regime_column(labels)
    if not ts or not reg:
        return pd.DataFrame()
    group_cols = [c for c in ["asset", "frequency", "regime_method"] if c in labels.columns]
    for keys, part in labels.dropna(subset=[ts, reg]).sort_values(ts).groupby(group_cols, dropna=False):
        part = part.sort_values(ts).copy()
        segment_id = (part[reg].astype(str) != part[reg].astype(str).shift()).cumsum()
        for sid, seg in part.groupby(segment_id):
            rec = {col: val for col, val in zip(group_cols, keys if isinstance(keys, tuple) else (keys,))}
            rec.update({
                "method_family": method_name,
                "regime_label": seg[reg].iloc[0],
                "duration_observations": len(seg),
                "start_timestamp": seg[ts].min(),
                "end_timestamp": seg[ts].max(),
            })
            rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    freq_minutes = {"15m": 15, "1h": 60}
    out["approx_duration_hours"] = out.apply(
        lambda r: r["duration_observations"] * freq_minutes.get(str(r.get("frequency")), np.nan) / 60,
        axis=1,
    )
    return out

durations = pd.concat([
    segment_durations(quantile_labels, "quantile"),
    segment_durations(hmm_labels, "HMM"),
], ignore_index=True)
display_table(durations, 20)
save_table(durations, "study_regime_segment_durations")

summary_cols = [c for c in ["method_family", "asset", "frequency", "regime_method", "regime_label"] if c in durations.columns]
duration_summary = durations.groupby(summary_cols, dropna=False)["duration_observations"].agg(["count", "mean", "median"]).reset_index() if not durations.empty else pd.DataFrame()
display_table(duration_summary, 30)
save_table(duration_summary, "study_regime_duration_summary")

if not durations.empty:
    focus = durations[durations.get("frequency", "").astype(str).eq("15m")] if "frequency" in durations.columns else durations
    fig, ax = plt.subplots(figsize=(12, 5))
    labels_box = []
    values = []
    for key, part in focus.groupby(["method_family", "regime_label"], dropna=False):
        labels_box.append(" / ".join(map(str, key)))
        values.append(part["duration_observations"].to_numpy())
    if values:
        ax.boxplot(values, labels=labels_box, showfliers=False)
        ax.set_title("Regime segment duration distribution, 15m labels")
        ax.set_ylabel("Consecutive observations")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        save_fig(fig, "study_regime_duration_distribution_15m", FIGURE_DIRS["regime"])
        plt.show()
'''),
        md("""
## Key findings
Use the tables and figures above to report which regime files are complete and which labels are available for each thesis-scope asset/frequency pair.

## Thesis-safe interpretation
Quantile regimes provide deterministic partitions of observed rolling volatility. HMM regimes provide latent probabilistic regimes inferred from multiple features when the corresponding HMM output files contain labels. Both families are used as alternative market-state labels for conditioned motif discovery.

## Limitations
The notebook only reports what exists in the result files. Missing files, empty files, or missing columns are displayed explicitly and should not be converted into numerical claims.

## Recommended figures for thesis
- `study_regime_BTCUSDT_15m_quantile_close_by_regime`
- `study_regime_BTCUSDT_15m_hmm_close_by_regime` when HMM labels are available
- `study_regime_BTCUSDT_15m_quantile_transition_heatmap`
- `study_regime_duration_distribution_15m`
"""),
    ]
    return cells


def notebook_2():
    cells = [
        md("""
# 02 Regime Comparison: Quantile vs HMM

## Objective
Compare quantile and HMM regimes using already-computed alignment, model-selection, persistence, and confusion-table outputs.

## Input files
- `results/regimes/hmm/hmm_model_selection.parquet`
- `results/regimes/hmm/hmm_persistence_metrics.parquet`
- `results/regimes/hmm/hmm_quantile_comparison.parquet`
- `results/regimes/hmm/hmm_quantile_confusion_table.parquet`
- HMM and quantile label files

## Output folder
`reports/study_notebooks/figures/regime_comparison` and `reports/study_notebooks/tables`.

## Thesis relevance
This notebook explains why the regime methods are complementary rather than expected to match exactly.

## Analysis-only safety
This notebook never imports or calls STUMPY, STUMP/MSTUMP, HMM fitting, LoCoMotif search, or any other expensive experiment algorithm. It only reads saved result files and thesis-scope feature parquet files, then produces derived tables and figures.

**Quantile caveat.** The quantile regime outputs store `rolling_volatility_60` as the actual volatility column across all quantile method identifiers. Therefore, quantile regimes are interpreted as 60-period rolling-volatility regimes with different regime-count granularities rather than as separate 30/60/240 volatility-horizon experiments.
"""),
        code(SETUP_CODE),
        md("## Load HMM-vs-Quantile Comparison Tables"),
        code(r'''
quantile_dir = result_path("regimes", "quantile")
hmm_dir = result_path("regimes", "hmm")

paths = {
    "hmm_labels": resolve_existing_file(hmm_dir, "hmm_regime_labels.parquet"),
    "quantile_labels": resolve_existing_file(quantile_dir, "quantile_regime_labels.parquet"),
    "hmm_model_selection": resolve_existing_file(hmm_dir, "hmm_model_selection.parquet"),
    "hmm_persistence": resolve_existing_file(hmm_dir, "hmm_persistence_metrics.parquet"),
    "hmm_quantile_comparison": resolve_existing_file(hmm_dir, "hmm_quantile_comparison.parquet"),
    "hmm_quantile_confusion": resolve_existing_file(hmm_dir, "hmm_quantile_confusion_table.parquet"),
    "quantile_transitions": resolve_existing_file(quantile_dir, "quantile_transition_matrix.parquet"),
}
loaded = {name: safe_read_parquet(path) for name, path in paths.items()}
for name in ["hmm_labels", "quantile_labels"]:
    loaded[name] = coerce_timestamp(loaded[name])

inventory = pd.DataFrame([
    {"name": name, "path": str(path), "exists": path.exists(), "rows": len(loaded[name]), "columns": len(loaded[name].columns)}
    for name, path in paths.items()
])
display_table(inventory)
save_table(inventory, "study_regime_comparison_file_inventory")
'''),
        md("## HMM Model Selection"),
        code(r'''
model_selection = loaded["hmm_model_selection"]
display_table(model_selection, 30)
save_table(model_selection, "study_regime_comparison_hmm_model_selection")

state_col = next((c for c in ["selected_n_states", "n_states", "states", "best_n_states"] if c in model_selection.columns), None)
if not model_selection.empty and state_col:
    label_cols = [c for c in ["asset", "frequency", "feature_set"] if c in model_selection.columns]
    labels = model_selection[label_cols].astype(str).agg(" ".join, axis=1) if label_cols else model_selection.index.astype(str)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, model_selection[state_col])
    ax.set_title("Selected HMM state count by thesis-scope dataset")
    ax.set_ylabel(state_col)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_fig(fig, "study_regime_comparison_hmm_selected_state_count", FIGURE_DIRS["regime_comparison"])
    plt.show()
else:
    print("HMM model-selection state column is not available.")
'''),
        md("## HMM Posterior Confidence"),
        code(r'''
hmm_labels = loaded["hmm_labels"]
confidence_col = next((c for c in ["regime_confidence", "posterior_probability", "max_posterior", "confidence"] if c in hmm_labels.columns), None)
reg = regime_column(hmm_labels)
if hmm_labels.empty or not confidence_col:
    print("No HMM posterior confidence column is available.")
else:
    fig, ax = plt.subplots(figsize=(10, 5))
    hmm_labels[confidence_col].dropna().plot(kind="hist", bins=40, ax=ax, color="#4C78A8")
    ax.set_title("HMM regime confidence distribution")
    ax.set_xlabel(confidence_col)
    fig.tight_layout()
    save_fig(fig, "study_regime_comparison_hmm_confidence_histogram", FIGURE_DIRS["regime_comparison"])
    plt.show()

    if reg:
        groups = [part[confidence_col].dropna().to_numpy() for _, part in hmm_labels.groupby(reg)]
        labels_order = [str(k) for k, _ in hmm_labels.groupby(reg)]
        if groups:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.boxplot(groups, labels=labels_order, showfliers=False)
            ax.set_title("HMM confidence by regime label")
            ax.set_xlabel("Regime label")
            ax.set_ylabel(confidence_col)
            fig.tight_layout()
            save_fig(fig, "study_regime_comparison_hmm_confidence_by_regime", FIGURE_DIRS["regime_comparison"])
            plt.show()
'''),
        md("## ARI and NMI Evaluation"),
        code(r'''
comparison = loaded["hmm_quantile_comparison"]
display_table(comparison, 30)
save_table(comparison, "study_regime_comparison_ari_nmi")

def plot_metric_bars(df, metric, filename):
    if df.empty or metric not in df.columns:
        print(f"{metric} is not available.")
        return
    label_cols = [c for c in ["asset", "frequency", "quantile_method", "regime_method"] if c in df.columns]
    labels = df[label_cols].astype(str).agg(" / ".join, axis=1) if label_cols else df.index.astype(str)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, df[metric])
    ax.set_title(f"{metric} by asset/frequency/quantile method")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_fig(fig, filename, FIGURE_DIRS["regime_comparison"])
    plt.show()

ari_col = next((c for c in comparison.columns if c.lower() in ["adjusted_rand_index", "ari"]), None) if not comparison.empty else None
nmi_col = next((c for c in comparison.columns if c.lower() in ["normalized_mutual_information", "nmi"]), None) if not comparison.empty else None
if ari_col:
    plot_metric_bars(comparison, ari_col, "study_regime_comparison_adjusted_rand_index")
if nmi_col:
    plot_metric_bars(comparison, nmi_col, "study_regime_comparison_normalized_mutual_information")
if not ari_col and not nmi_col:
    print("ARI/NMI columns are not available in the comparison table.")
'''),
        md("## Confusion Heatmaps"),
        code(r'''
confusion = loaded["hmm_quantile_confusion"]
display_table(confusion, 20)
save_table(confusion, "study_regime_comparison_confusion_table_raw")

for asset in THESIS_SCOPE_ASSETS:
    for frequency in THESIS_SCOPE_FREQUENCIES:
        filters = {"asset": asset, "frequency": frequency}
        if "quantile_method" in confusion.columns:
            methods = confusion["quantile_method"].dropna().astype(str).unique()
            preferred = "quantile_2_rolling_240"
            filters["quantile_method"] = preferred if preferred in methods else (methods[0] if len(methods) else preferred)
        plot_heatmap_from_table(
            confusion,
            f"{asset} {frequency} HMM vs quantile confusion",
            f"study_regime_comparison_confusion_{asset}_{frequency}",
            FIGURE_DIRS["regime_comparison"],
            filters,
        )
'''),
        md("## Regime Persistence Comparison"),
        code(r'''
def extract_self_transition(df, method_family):
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    from_col = next((c for c in ["from_regime", "source_regime", "regime_from"] if c in work.columns), None)
    to_col = next((c for c in ["to_regime", "target_regime", "regime_to"] if c in work.columns), None)
    prob_col = next((c for c in ["self_transition_probability", "probability", "transition_probability"] if c in work.columns and pd.api.types.is_numeric_dtype(work[c])), None)
    if "self_transition_probability" in work.columns:
        out = work.copy()
        out["self_transition_probability"] = out["self_transition_probability"]
    elif from_col and to_col and prob_col:
        out = work[work[from_col].astype(str) == work[to_col].astype(str)].copy()
        out["self_transition_probability"] = out[prob_col]
        out["regime_label"] = out[from_col]
    else:
        return pd.DataFrame()
    out["method_family"] = method_family
    keep = [c for c in ["method_family", "asset", "frequency", "regime_method", "regime_label", "self_transition_probability"] if c in out.columns]
    return out[keep]

self_transitions = pd.concat([
    extract_self_transition(loaded["quantile_transitions"], "quantile"),
    extract_self_transition(loaded["hmm_persistence"], "HMM"),
], ignore_index=True)
display_table(self_transitions, 30)
save_table(self_transitions, "study_regime_comparison_self_transition_probabilities")

if not self_transitions.empty and "self_transition_probability" in self_transitions.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = self_transitions[[c for c in ["method_family", "asset", "frequency", "regime_label"] if c in self_transitions.columns]].astype(str).agg(" / ".join, axis=1)
    ax.bar(labels, self_transitions["self_transition_probability"])
    ax.set_title("Self-transition probability comparison")
    ax.set_ylabel("Self-transition probability")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_fig(fig, "study_regime_comparison_self_transition_probabilities", FIGURE_DIRS["regime_comparison"])
    plt.show()
else:
    print("Self-transition probabilities could not be inferred from available files.")
'''),
        md("""
## Key findings
Quantile and HMM comparisons should be read from the ARI, NMI, confusion, confidence, and persistence tables above when available.

## Thesis-safe interpretation
Quantile and HMM regimes do not need to match exactly. Their disagreement is informative because quantile regimes partition observed volatility directly, while HMM regimes infer latent states from multiple features. For motif discovery, both provide alternative market-state partitions under nonstationarity.

## Limitations
Empty comparison or confusion tables mean no numerical alignment claim should be made from this notebook. Missing confidence columns mean posterior-confidence figures cannot be reported.

## Recommended figures for thesis
- HMM selected state count
- HMM confidence histogram if posterior columns are available
- ARI and NMI bar charts if comparison rows are available
- HMM-vs-quantile confusion heatmaps when the confusion table is populated
"""),
    ]
    return cells


def notebook_3():
    cells = [
        md("""
# 03 Matrix Profile Motif Visual Study

## Objective
Create the main Matrix Profile study notebook for visualizing motif distances, empirical thresholds, top motifs, motif overlays, motif timelines, evaluation metrics, and runtime evidence from already-computed files.

## Input files
- `results/motifs/matrix_profile/matrix_profile_motif_results.parquet`
- `results/motifs/matrix_profile/matrix_profile_evaluation.parquet`
- `results/motifs/matrix_profile/matrix_profile_runtime.parquet`
- `results/motifs/matrix_profile/matrix_profile_profiles.parquet`
- Thesis-scope feature parquet files under `final_dataset/features`

## Output folder
`reports/study_notebooks/figures/matrix_profile` and `reports/study_notebooks/tables`.

## Thesis relevance
This notebook explains what Matrix Profile motif distances mean, derives empirical thresholds from existing distances, and visualizes agnostic vs regime-conditioned motif discovery without rerunning Matrix Profile.

## Analysis-only safety
This notebook never imports or calls STUMPY, STUMP/MSTUMP, HMM fitting, LoCoMotif search, or any other expensive experiment algorithm. It only reads saved result files and thesis-scope feature parquet files, then produces derived tables and figures.
"""),
        code(SETUP_CODE),
        md("## Load Matrix Profile Results"),
        code(r'''
mp_dir = result_path("motifs", "matrix_profile")
mp_paths = {
    "motif_results": resolve_existing_file(mp_dir, "matrix_profile_motif_results.parquet"),
    "evaluation": resolve_existing_file(mp_dir, "matrix_profile_evaluation.parquet"),
    "runtime": resolve_existing_file(mp_dir, "matrix_profile_runtime.parquet"),
    "profiles": resolve_existing_file(mp_dir, "matrix_profile_profiles.parquet"),
}
mp = safe_read_parquet(mp_paths["motif_results"])
eval_df = safe_read_parquet(mp_paths["evaluation"])
runtime = safe_read_parquet(mp_paths["runtime"])
profiles = safe_read_parquet(mp_paths["profiles"])
for col in ["motif_timestamp_1", "motif_timestamp_2", "motif_end_timestamp_1", "motif_end_timestamp_2"]:
    if col in mp.columns:
        mp[col] = pd.to_datetime(mp[col], errors="coerce")
distance_col = find_distance_column(mp)
print("Selected distance column:", distance_col)
'''),
        md("## File Inventory and Schema"),
        code(r'''
inventory_rows = []
for name, path in mp_paths.items():
    df = {"motif_results": mp, "evaluation": eval_df, "runtime": runtime, "profiles": profiles}[name]
    inventory_rows.append({
        "file": name,
        "path": str(path),
        "exists": path.exists(),
        "rows": len(df),
        "columns": len(df.columns),
        "useful_columns": useful_columns(df),
    })
inventory = pd.DataFrame(inventory_rows)
display_table(inventory)
save_table(inventory, "study_mp_file_inventory")
'''),
        md("## Experiment Scope and Motif Result Overview"),
        code(r'''
overview_tables = {
    "study_mp_rows_by_asset_frequency_mode": group_count_table(mp, ["asset", "frequency", "mode"], "study_mp_rows_by_asset_frequency_mode"),
    "study_mp_rows_by_regime_method": group_count_table(mp, ["regime_method"], "study_mp_rows_by_regime_method"),
    "study_mp_rows_by_regime_label": group_count_table(mp, ["regime_label"], "study_mp_rows_by_regime_label"),
    "study_mp_rows_by_frequency_window": group_count_table(mp, ["frequency", "window_length"], "study_mp_rows_by_frequency_window"),
    "study_mp_rows_by_feature_set": group_count_table(mp, ["feature_set"], "study_mp_rows_by_feature_set"),
    "study_mp_rows_by_profile_type": group_count_table(mp, ["profile_type"], "study_mp_rows_by_profile_type"),
    "study_mp_rows_by_asset_frequency_regime": group_count_table(mp, ["asset", "frequency", "regime_label"], "study_mp_rows_by_asset_frequency_regime"),
}
for name, table in overview_tables.items():
    print("\n", name)
    display_table(table, 15)
summary = pd.DataFrame([{"metric": "total_motif_rows", "value": len(mp)}])
save_table(summary, "study_mp_total_motif_rows")
'''),
        md("## Motif Distances and Empirical Thresholds"),
        code(r'''
def plot_distance_hist(df, title, filename, by=None):
    if df.empty or not distance_col:
        print(f"No distance data for {title}")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    if by and by in df.columns:
        for key, part in df.groupby(by, dropna=False):
            values = pd.to_numeric(part[distance_col], errors="coerce").dropna()
            if len(values):
                ax.hist(values, bins=35, alpha=0.45, label=str(key), density=False)
        ax.legend(title=by)
    else:
        pd.to_numeric(df[distance_col], errors="coerce").dropna().plot(kind="hist", bins=50, ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(distance_col)
    ax.set_ylabel("Motif rows")
    fig.tight_layout()
    save_fig(fig, filename, FIGURE_DIRS["matrix_profile"])
    plt.show()

plot_distance_hist(mp, "Matrix Profile motif distance distribution, overall", "study_mp_distance_distribution_overall")
plot_distance_hist(mp, "Matrix Profile distances: agnostic vs conditioned", "study_mp_distance_distribution_by_mode", by="mode")
plot_distance_hist(mp, "Matrix Profile distances by regime label", "study_mp_distance_distribution_by_regime_label", by="regime_label")
plot_distance_hist(mp, "Matrix Profile distances by window length", "study_mp_distance_distribution_by_window_length", by="window_length")
plot_distance_hist(mp, "Matrix Profile distances by feature set", "study_mp_distance_distribution_by_feature_set", by="feature_set")
'''),
        code(r'''
def threshold_record(scope, df):
    if df.empty or not distance_col:
        return {"scope": scope, "rows": len(df)}
    values = pd.to_numeric(df[distance_col], errors="coerce").dropna()
    if values.empty:
        return {"scope": scope, "rows": len(df)}
    qs = values.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75])
    return {
        "scope": scope,
        "rows": len(values),
        "q01": qs.loc[0.01],
        "q05": qs.loc[0.05],
        "q10": qs.loc[0.10],
        "q25": qs.loc[0.25],
        "median": qs.loc[0.50],
        "q75": qs.loc[0.75],
        "min": values.min(),
        "max": values.max(),
    }

scopes = [("overall", mp)]
if "mode" in mp.columns:
    scopes.extend([(str(k), v) for k, v in mp.groupby("mode", dropna=False)])
if "regime_label" in mp.columns:
    for label in ["low_vol", "high_vol", "extreme_vol"]:
        scopes.append((label, mp[mp["regime_label"].astype(str).eq(label)]))

thresholds = pd.DataFrame([threshold_record(name, df) for name, df in scopes])
display_table(thresholds, 20)
save_table(thresholds, "study_mp_empirical_distance_thresholds")

if not mp.empty and distance_col:
    values = pd.to_numeric(mp[distance_col], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(values, bins=60, color="#4C78A8", alpha=0.75)
    overall = thresholds[thresholds["scope"].eq("overall")]
    for col, color in [("q01", "#D62728"), ("q05", "#FF7F0E"), ("q10", "#2CA02C"), ("median", "#000000")]:
        if col in overall.columns and not overall[col].isna().all():
            val = overall[col].iloc[0]
            ax.axvline(val, color=color, linestyle="--", linewidth=1.5, label=col)
    ax.set_title("Matrix Profile empirical distance thresholds")
    ax.set_xlabel(distance_col)
    ax.set_ylabel("Motif rows")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "study_mp_distance_threshold_lines", FIGURE_DIRS["matrix_profile"])
    plt.show()
'''),
        md("""
### Interpretation
Matrix Profile distances are lower-is-better for the selected distance column. Matrix Profile does not provide one universal intrinsic motif threshold, so this notebook reports empirical thresholds from the observed result distribution: top 1%, top 5%, top 10%, quartiles, median, minimum, and maximum.
"""),
        md("## Top 10 and Top 20 Motifs"),
        code(r'''
top_cols_preferred = [
    "asset", "frequency", "mode", "regime_method", "regime_label", "feature_set", "profile_type",
    "window_length", "motif_rank", "motif_start_1", "motif_start_2", "motif_timestamp_1",
    "motif_timestamp_2", "segment_id", "used_gpu",
]
if distance_col:
    top_cols_preferred.insert(-2, distance_col)
top_cols = [c for c in top_cols_preferred if c in mp.columns]

def top_table(df, n, name):
    if df.empty or not distance_col:
        print(f"No top table for {name}")
        return pd.DataFrame()
    out = df.sort_values(distance_col, ascending=True).head(n)[top_cols].copy()
    save_table(out, name)
    display_table(out, n)
    return out

top10_overall = top_table(mp, 10, "study_mp_top10_overall")
top20_overall = top_table(mp, 20, "study_mp_top20_overall")
top10_agnostic = top_table(mp[mp["mode"].astype(str).eq("agnostic")] if "mode" in mp.columns else pd.DataFrame(), 10, "study_mp_top10_agnostic")
top10_conditioned = top_table(mp[mp["mode"].astype(str).eq("conditioned")] if "mode" in mp.columns else pd.DataFrame(), 10, "study_mp_top10_conditioned")
for label in ["low_vol", "high_vol", "extreme_vol"]:
    subset = mp[mp["regime_label"].astype(str).eq(label)] if "regime_label" in mp.columns else pd.DataFrame()
    top_table(subset, 10, f"study_mp_top10_{label}")
for asset in ["BTCUSDT", "ETHUSDT"]:
    subset = mp[(mp["asset"].astype(str).eq(asset)) & (mp["frequency"].astype(str).eq("15m"))] if {"asset", "frequency"}.issubset(mp.columns) else pd.DataFrame()
    top_table(subset, 10, f"study_mp_top10_{asset}_15m")
'''),
        md("## Top Motif Overlay Plots"),
        code(r'''
def plot_motif_overlay(row, filename):
    asset = str(row.get("asset", ""))
    frequency = str(row.get("frequency", ""))
    feature_set = str(row.get("feature_set", "close"))
    window = int(row.get("window_length", 0)) if pd.notna(row.get("window_length", np.nan)) else 0
    s1 = int(row.get("motif_start_1", -1)) if pd.notna(row.get("motif_start_1", np.nan)) else -1
    s2 = int(row.get("motif_start_2", -1)) if pd.notna(row.get("motif_start_2", np.nan)) else -1
    if window <= 1 or s1 < 0 or s2 < 0:
        print("Cannot plot overlay: missing start indices or window length.")
        return False
    feature = load_feature_data(asset, frequency)
    value_col = find_feature_value_column(feature, feature_set)
    if feature.empty or not value_col:
        print(f"Cannot plot overlay for {asset} {frequency}: feature column unavailable.")
        return False
    if s1 + window > len(feature) or s2 + window > len(feature):
        print(f"Cannot plot overlay for {asset} {frequency}: motif windows exceed feature length.")
        return False
    y1 = z_normalize(feature[value_col].iloc[s1:s1 + window])
    y2 = z_normalize(feature[value_col].iloc[s2:s2 + window])
    x = np.arange(window)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, y1, label=f"window 1 @ {s1}", linewidth=2.0, color="#1F77B4")
    ax.plot(x, y2, label=f"nearest window @ {s2}", linewidth=2.0, color="#FF7F0E")
    dist_text = f"{distance_col}={row.get(distance_col):.4g}" if distance_col and pd.notna(row.get(distance_col, np.nan)) else "distance unavailable"
    title_bits = [asset, frequency, str(row.get("mode", "")), str(row.get("regime_label", "")), value_col, f"w={window}", dist_text]
    ax.set_title(" | ".join([b for b in title_bits if b and b != "nan"]))
    ax.set_xlabel("Window offset")
    ax.set_ylabel("z-normalized value")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, filename, FIGURE_DIRS["matrix_profile"])
    plt.show()
    return True

def overlay_batch(df, label, n=5):
    if df.empty or not distance_col:
        print(f"No rows available for overlay batch: {label}")
        return
    selected = df.sort_values(distance_col, ascending=True).head(n)
    successes = 0
    for idx, row in selected.iterrows():
        name = f"study_mp_overlay_{label}_{successes + 1}_{row.get('asset','asset')}_{row.get('frequency','freq')}_{row.get('feature_set','feature')}_w{row.get('window_length','w')}"
        successes += int(plot_motif_overlay(row, name))
    print(f"{label}: created {successes} overlay plots.")

overlay_batch(mp, "top5_overall")
if "mode" in mp.columns:
    overlay_batch(mp[mp["mode"].astype(str).eq("agnostic")], "top5_agnostic")
    overlay_batch(mp[mp["mode"].astype(str).eq("conditioned")], "top5_conditioned")
if "regime_label" in mp.columns:
    overlay_batch(mp[mp["regime_label"].astype(str).eq("high_vol")], "top5_high_vol")
    overlay_batch(mp[mp["regime_label"].astype(str).eq("low_vol")], "top5_low_vol")
'''),
        md("""
### Interpretation
Each overlay compares the motif window and its nearest-neighbour window after z-normalization. Similar shapes with low distances are candidate recurring subsequences under the selected feature, window length, and regime context.
"""),
        md("## Matrix Profile Curves or Ranked Distance Plots"),
        code(r'''
def plot_profile_for_top(row, filename):
    if profiles.empty:
        return False
    filters = {}
    for col in ["asset", "frequency", "mode", "regime_method", "regime_label", "segment_id", "window_length", "feature_set"]:
        if col in profiles.columns and col in row.index and pd.notna(row[col]):
            filters[col] = row[col]
    work = profiles.copy()
    for col, val in filters.items():
        work = work[work[col].astype(str) == str(val)]
    value_col = "matrix_profile" if "matrix_profile" in work.columns else find_distance_column(work)
    idx_col = "profile_index" if "profile_index" in work.columns else None
    if work.empty or not value_col:
        return False
    if len(work) > 20000:
        work = work.sample(20000, random_state=7).sort_values(idx_col) if idx_col else work.sample(20000, random_state=7)
    fig, ax = plt.subplots(figsize=(13, 5))
    x = work[idx_col] if idx_col else np.arange(len(work))
    ax.plot(x, work[value_col], color="#4C78A8", linewidth=1.0)
    for start_col, color in [("motif_start_1", "#D62728"), ("motif_start_2", "#FF7F0E")]:
        if start_col in row.index and pd.notna(row[start_col]):
            ax.axvline(float(row[start_col]), color=color, linestyle="--", label=start_col)
    if distance_col and not thresholds.empty:
        overall = thresholds[thresholds["scope"].eq("overall")]
        for q in ["q01", "q05", "q10"]:
            if q in overall.columns and not overall[q].isna().all():
                ax.axhline(overall[q].iloc[0], linestyle=":", linewidth=1.2, label=q)
    ax.set_title("Matrix Profile curve with top motif locations")
    ax.set_xlabel(idx_col or "Profile row")
    ax.set_ylabel(value_col)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, filename, FIGURE_DIRS["matrix_profile"])
    plt.show()
    return True

profile_done = False
if not top10_overall.empty:
    profile_done = plot_profile_for_top(top10_overall.iloc[0], "study_mp_profile_curve_top_motif")

if not profile_done:
    print("Raw matrix profile curve unavailable or not alignable; plotting ranked distances instead.")
    if not mp.empty and distance_col:
        ranked = mp.sort_values(distance_col, ascending=True).reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ranked["rank"], ranked[distance_col], marker=".", linewidth=1.0)
        ax.set_title("Ranked Matrix Profile motif distances")
        ax.set_xlabel("Rank, lowest distance first")
        ax.set_ylabel(distance_col)
        fig.tight_layout()
        save_fig(fig, "study_mp_ranked_distance_elbow", FIGURE_DIRS["matrix_profile"])
        plt.show()
'''),
        md("## Overlapping Time Motif Illustrations"),
        code(r'''
def overlap_table(df, n=50):
    if df.empty:
        return pd.DataFrame()
    rows = []
    for i, row in df.head(n).iterrows():
        w = int(row.get("window_length", 0)) if pd.notna(row.get("window_length", np.nan)) else 0
        s1 = int(row.get("motif_start_1", -1)) if pd.notna(row.get("motif_start_1", np.nan)) else -1
        s2 = int(row.get("motif_start_2", -1)) if pd.notna(row.get("motif_start_2", np.nan)) else -1
        if w <= 0 or s1 < 0 or s2 < 0:
            continue
        a1, a2 = s1, s1 + w
        b1, b2 = s2, s2 + w
        overlap_len = max(0, min(a2, b2) - max(a1, b1))
        rows.append({
            "motif_id": i,
            "asset": row.get("asset"),
            "frequency": row.get("frequency"),
            "mode": row.get("mode"),
            "regime_label": row.get("regime_label"),
            "window_length": w,
            "start_1": s1,
            "start_2": s2,
            "overlap_length": overlap_len,
            "overlap_ratio": overlap_len / w if w else np.nan,
        })
    return pd.DataFrame(rows)

mp_top50 = mp.sort_values(distance_col, ascending=True).head(50) if distance_col else mp.head(50)
overlaps = overlap_table(mp_top50)
display_table(overlaps, 20)
save_table(overlaps, "study_mp_top50_motif_window_overlap")

def plot_motif_timeline(df, asset, frequency, filename, title_extra=""):
    subset = filter_scope(df, asset, frequency)
    if subset.empty:
        print(f"No motif rows for timeline: {asset} {frequency}")
        return
    if distance_col:
        subset = subset.sort_values(distance_col, ascending=True).head(20)
    feature = load_feature_data(asset, frequency)
    ts = timestamp_column(feature)
    fig, ax = plt.subplots(figsize=(14, 5))
    if not feature.empty and ts and "close" in feature.columns:
        sampled = feature[[ts, "close"]].dropna().sort_values(ts)
        if len(sampled) > 8000:
            sampled = sampled.iloc[:: int(np.ceil(len(sampled) / 8000))]
        ax.plot(sampled[ts], sampled["close"], color="0.75", linewidth=0.8, label="close")
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for j, (_, row) in enumerate(subset.iterrows()):
        w = int(row.get("window_length", 1)) if pd.notna(row.get("window_length", np.nan)) else 1
        for start_col, time_col in [("motif_start_1", "motif_timestamp_1"), ("motif_start_2", "motif_timestamp_2")]:
            start_time = row.get(time_col)
            if pd.isna(start_time) and not feature.empty and ts and pd.notna(row.get(start_col, np.nan)):
                idx = int(row[start_col])
                if 0 <= idx < len(feature):
                    start_time = feature.iloc[idx][ts]
            if pd.notna(start_time):
                start_time = pd.to_datetime(start_time)
                end_time = start_time + pd.Timedelta(minutes=15 * w if frequency == "15m" else 60 * w)
                ax.axvspan(start_time, end_time, color=colors[j % len(colors)], alpha=0.18)
    ax.set_title(f"{asset} {frequency} top motif timeline {title_extra}".strip())
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Close price or motif spans")
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, filename, FIGURE_DIRS["matrix_profile"])
    plt.show()

plot_motif_timeline(mp, "BTCUSDT", "15m", "study_mp_timeline_BTCUSDT_15m_top_motifs")
plot_motif_timeline(mp, "ETHUSDT", "15m", "study_mp_timeline_ETHUSDT_15m_top_motifs")
plot_motif_timeline(mp, "BTCUSDT", "1h", "study_mp_timeline_BTCUSDT_1h_top_motifs")
if "regime_label" in mp.columns:
    plot_motif_timeline(mp[mp["regime_label"].astype(str).eq("high_vol")], "BTCUSDT", "15m", "study_mp_timeline_BTCUSDT_15m_high_vol_top_motifs", "high_vol")
'''),
        md("## Agnostic vs Conditioned Evidence"),
        code(r'''
def mode_metrics(df, mode):
    part = df[df["mode"].astype(str).eq(mode)] if "mode" in df.columns else pd.DataFrame()
    values = pd.to_numeric(part[distance_col], errors="coerce").dropna() if distance_col and not part.empty else pd.Series(dtype=float)
    return {
        "saved motif rows": len(part),
        "median distance": values.median() if len(values) else np.nan,
        "q10 distance": values.quantile(0.10) if len(values) else np.nan,
        "best distance": values.min() if len(values) else np.nan,
        "number of regimes represented": part["regime_label"].nunique(dropna=True) if "regime_label" in part.columns else np.nan,
        "number of windows represented": part["window_length"].nunique(dropna=True) if "window_length" in part.columns else np.nan,
        "number of feature sets represented": part["feature_set"].nunique(dropna=True) if "feature_set" in part.columns else np.nan,
        "runtime if available": part["runtime_seconds"].sum() if "runtime_seconds" in part.columns else np.nan,
        "figure examples available": "generated above when feature windows align",
    }

agnostic = mode_metrics(mp, "agnostic")
conditioned = mode_metrics(mp, "conditioned")
interpretations = {
    "saved motif rows": "Conditioned row counts are expected to be larger when searches are repeated across regimes, methods, and segments.",
    "median distance": "Lower values indicate closer motif-neighbour pairs within the saved results.",
    "q10 distance": "Lower top-decile distances indicate stronger best-candidate similarity.",
    "best distance": "Best observed saved motif distance.",
    "number of regimes represented": "Conditioned discovery attaches motifs to market states.",
    "number of windows represented": "Window-length coverage in the saved benchmark.",
    "number of feature sets represented": "Feature coverage in the saved benchmark.",
    "runtime if available": "Runtime is read from saved result rows when present.",
    "figure examples available": "Visual support depends on available feature data and start indices.",
}
comparison_rows = []
for metric in agnostic:
    comparison_rows.append({
        "Metric": metric,
        "Agnostic": agnostic[metric],
        "Conditioned": conditioned[metric],
        "Interpretation": interpretations.get(metric, ""),
    })
mode_comparison = pd.DataFrame(comparison_rows)
display_table(mode_comparison, 20)
save_table(mode_comparison, "study_mp_agnostic_vs_conditioned_evidence")
'''),
        md("""
### Interpretation
Conditioned discovery increases regime-specific interpretability by attaching motif candidates to market states. Row counts are larger because the search is repeated across regimes, regime methods, and segments.
"""),
        md("## Evaluation Metrics"),
        code(r'''
display_table(eval_df, 20)
save_table(eval_df, "study_mp_evaluation_raw")

numeric_metrics = [c for c in eval_df.columns if pd.api.types.is_numeric_dtype(eval_df[c])]
group_sets = [
    ["mode"],
    ["asset", "frequency", "mode"],
    ["regime_label"],
    ["window_length"],
    ["feature_set"],
    ["profile_type"],
]
for group_cols in group_sets:
    existing = [c for c in group_cols if c in eval_df.columns]
    if eval_df.empty or not existing or not numeric_metrics:
        continue
    summary = eval_df.groupby(existing, dropna=False)[numeric_metrics].agg(["mean", "median", "min", "max"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    name = "study_mp_evaluation_by_" + "_".join(existing)
    display_table(summary, 20)
    save_table(summary, name)
'''),
        md("## Runtime and CPU/GPU Audit"),
        code(r'''
display_table(runtime, 20)
save_table(runtime, "study_mp_runtime_raw")
for group_cols in [["asset", "frequency", "mode", "profile_type"], ["window_length"], ["profile_type"]]:
    existing = [c for c in group_cols if c in runtime.columns]
    if runtime.empty or "runtime_seconds" not in runtime.columns or not existing:
        continue
    out = runtime.groupby(existing, dropna=False)["runtime_seconds"].agg(["count", "sum", "mean", "median", "min", "max"]).reset_index()
    display_table(out, 20)
    save_table(out, "study_mp_runtime_by_" + "_".join(existing))

if "used_gpu" in mp.columns:
    gpu_audit = mp["used_gpu"].value_counts(dropna=False).rename_axis("used_gpu").reset_index(name="rows")
    display_table(gpu_audit)
    save_table(gpu_audit, "study_mp_gpu_audit")
    if len(gpu_audit) and gpu_audit["used_gpu"].astype(str).str.lower().isin(["false", "0"]).all():
        print("All reported Matrix Profile results were generated using CPU execution. GPU acceleration was allowed by the pipeline but was not used in the executed run.")
else:
    print("No used_gpu column is available in Matrix Profile motif results.")
'''),
        md("""
## Key findings
Use the generated top motif tables, threshold table, overlays, timelines, evaluation summaries, and runtime summaries to answer which Matrix Profile motifs were found and where the best distances occur.

## Thesis-safe interpretation
Matrix Profile found fixed-length motif candidates in the saved results when motif rows and distance columns are present. Lower distance values indicate stronger shape similarity under the selected feature and window length. Empirical thresholds are descriptive quantiles of the saved result distribution, not universal Matrix Profile constants.

## Limitations
This notebook does not rerun Matrix Profile and cannot recover missing motif windows, missing feature files, or empty profile tables. Agnostic and conditioned searches differ in scope, so row counts alone are not evidence of superiority.

## Recommended figures for thesis
- Distance distribution with empirical thresholds
- Top motif overlays for BTCUSDT 15m
- Top motif timelines for BTCUSDT 15m and ETHUSDT 15m
- Agnostic vs conditioned evidence table
- Runtime and CPU/GPU audit table
"""),
    ]
    return cells


def notebook_4():
    cells = [
        md("""
# 04 LoCoMotif Visual Study

## Objective
Study already-computed LoCoMotif outputs, visualize motif intervals when available, audit runtime/failures, and compare LoCoMotif with Matrix Profile for thesis presentation.

## Input files
- `results/motifs/locomotif/locomotif_motif_results.parquet`
- `results/motifs/locomotif/locomotif_evaluation.parquet`
- `results/motifs/locomotif/locomotif_runtime.parquet`
- `results/motifs/locomotif/04_locomotif_failures.parquet` if available
- Matrix Profile summary files for comparison

## Output folder
`reports/study_notebooks/figures/locomotif` and `reports/study_notebooks/tables`.

## Thesis relevance
LoCoMotif complements Matrix Profile by targeting variable/local-constrained interval motifs. This notebook treats LoCoMotif as a controlled subset experiment unless full-scale outputs are present.

## Analysis-only safety
This notebook never imports or calls STUMPY, STUMP/MSTUMP, HMM fitting, LoCoMotif search, or any other expensive experiment algorithm. It only reads saved result files and thesis-scope feature parquet files, then produces derived tables and figures.
"""),
        code(SETUP_CODE),
        md("## Load LoCoMotif Results"),
        code(r'''
loco_dir = result_path("motifs", "locomotif")
loco_paths = {
    "motif_results": resolve_existing_file(loco_dir, "locomotif_motif_results.parquet"),
    "evaluation": resolve_existing_file(loco_dir, "locomotif_evaluation.parquet"),
    "runtime": resolve_existing_file(loco_dir, "locomotif_runtime.parquet"),
    "failures": resolve_existing_file(loco_dir, "04_locomotif_failures.parquet"),
}
loco = safe_read_parquet(loco_paths["motif_results"])
loco_eval = safe_read_parquet(loco_paths["evaluation"])
loco_runtime = safe_read_parquet(loco_paths["runtime"])
loco_failures = safe_read_parquet(loco_paths["failures"])
for col in ["motif_start_timestamp", "motif_end_timestamp"]:
    if col in loco.columns:
        loco[col] = pd.to_datetime(loco[col], errors="coerce")
'''),
        md("## File Inventory"),
        code(r'''
fig_roots = [RESULTS_ROOT / "figures", RESULTS_ROOT / "figures_parallel"]
figure_files = []
for root in fig_roots:
    if root.exists():
        figure_files.extend(sorted(root.rglob("04_*")))

inventory = pd.DataFrame([
    {
        "file": name,
        "path": str(path),
        "exists": path.exists(),
        "rows": len({"motif_results": loco, "evaluation": loco_eval, "runtime": loco_runtime, "failures": loco_failures}.get(name, pd.DataFrame())),
        "columns": len({"motif_results": loco, "evaluation": loco_eval, "runtime": loco_runtime, "failures": loco_failures}.get(name, pd.DataFrame()).columns),
    }
    for name, path in loco_paths.items()
] + [{
    "file": "figures_04_prefix",
    "path": "; ".join(str(p) for p in fig_roots),
    "exists": bool(figure_files),
    "rows": len(figure_files),
    "columns": 0,
}])
display_table(inventory)
save_table(inventory, "study_locomotif_file_inventory")
'''),
        md("## Scope and Status"),
        code(r'''
scope_rows = []
for label, df in [("motif_results", loco), ("evaluation", loco_eval), ("runtime", loco_runtime), ("failures", loco_failures)]:
    scope_rows.append({
        "source": label,
        "rows": len(df),
        "assets": ", ".join(sorted(df["asset"].dropna().astype(str).unique())) if "asset" in df.columns and not df.empty else "",
        "frequencies": ", ".join(sorted(df["frequency"].dropna().astype(str).unique())) if "frequency" in df.columns and not df.empty else "",
        "modes": ", ".join(sorted(df["mode"].dropna().astype(str).unique())) if "mode" in df.columns and not df.empty else "",
        "regime_methods": ", ".join(sorted(df["regime_method"].dropna().astype(str).unique())) if "regime_method" in df.columns and not df.empty else "",
        "regime_labels": ", ".join(sorted(df["regime_label"].dropna().astype(str).unique())) if "regime_label" in df.columns and not df.empty else "",
        "feature_sets": ", ".join(sorted(df["feature_set"].dropna().astype(str).unique())) if "feature_set" in df.columns and not df.empty else "",
        "status_counts": json.dumps(df["status"].value_counts(dropna=False).to_dict(), default=str) if "status" in df.columns and not df.empty else "",
    })
scope = pd.DataFrame(scope_rows)
display_table(scope)
save_table(scope, "study_locomotif_scope_table")
'''),
        md("## Motif Interval Length Distribution"),
        code(r'''
length_col = next((c for c in ["motif_length", "interval_length", "length"] if c in loco.columns and pd.api.types.is_numeric_dtype(loco[c])), None)
if loco.empty or not length_col:
    print("No LoCoMotif interval-length rows are available.")
else:
    fig, ax = plt.subplots(figsize=(10, 5))
    loco[length_col].dropna().plot(kind="hist", bins=40, ax=ax, color="#4C78A8")
    ax.set_title("LoCoMotif interval length distribution")
    ax.set_xlabel(length_col)
    ax.set_ylabel("Intervals")
    fig.tight_layout()
    save_fig(fig, "study_locomotif_interval_length_distribution", FIGURE_DIRS["locomotif"])
    plt.show()

    for by in ["mode", "regime_label"]:
        if by in loco.columns:
            groups = [part[length_col].dropna().to_numpy() for _, part in loco.groupby(by, dropna=False)]
            labels = [str(k) for k, _ in loco.groupby(by, dropna=False)]
            if groups:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.boxplot(groups, labels=labels, showfliers=False)
                ax.set_title(f"LoCoMotif interval length by {by}")
                ax.set_xlabel(by)
                ax.set_ylabel(length_col)
                ax.tick_params(axis="x", rotation=45)
                fig.tight_layout()
                save_fig(fig, f"study_locomotif_interval_length_by_{by}", FIGURE_DIRS["locomotif"])
                plt.show()
'''),
        md("## Recurrence and Evaluation"),
        code(r'''
display_table(loco_eval, 20)
save_table(loco_eval, "study_locomotif_evaluation_raw")

metrics = [c for c in ["number_of_motifs", "recurrence_count", "mean_motif_length", "median_motif_length", "time_split_stability", "cross_regime_overlap"] if c in loco_eval.columns]
if not loco_eval.empty and metrics:
    group = "mode" if "mode" in loco_eval.columns else None
    if group:
        summary = loco_eval.groupby(group, dropna=False)[metrics].agg(["mean", "median", "min", "max"])
        summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
        summary = summary.reset_index()
        display_table(summary)
        save_table(summary, "study_locomotif_evaluation_by_mode")
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(9, 5))
            loco_eval.groupby(group, dropna=False)[metric].mean().plot(kind="bar", ax=ax, color="#4C78A8")
            ax.set_title(f"LoCoMotif {metric} by mode")
            ax.set_ylabel(metric)
            fig.tight_layout()
            save_fig(fig, f"study_locomotif_{metric}_by_mode", FIGURE_DIRS["locomotif"])
            plt.show()
else:
    print("No populated LoCoMotif evaluation metrics are available.")
'''),
        md("## Top LoCoMotif Examples"),
        code(r'''
def rank_locomotif(df):
    if df.empty:
        return df
    work = df.copy()
    if "motif_score" in work.columns and pd.api.types.is_numeric_dtype(work["motif_score"]):
        return work.sort_values("motif_score", ascending=False)
    for col in ["recurrence_count", "number_of_motifs", "time_split_stability"]:
        if col in work.columns and pd.api.types.is_numeric_dtype(work[col]):
            return work.sort_values(col, ascending=False)
    return work

def loco_top_table(df, name, n=10):
    ranked = rank_locomotif(df).head(n)
    if ranked.empty:
        print(f"No rows for {name}")
        return ranked
    cols = [c for c in [
        "asset", "frequency", "mode", "regime_method", "regime_label", "feature_set",
        "motif_set_rank", "motif_instance_id", "motif_start", "motif_end",
        "motif_start_timestamp", "motif_end_timestamp", "motif_length",
        "motif_score", "motif_set_size", "number_of_motifs", "recurrence_count",
        "mean_motif_length", "runtime_seconds", "status",
    ] if c in ranked.columns]
    out = ranked[cols]
    display_table(out, n)
    save_table(out, name)
    return out

source = loco if not loco.empty else loco_eval
loco_top_table(source[source["mode"].astype(str).eq("agnostic")] if "mode" in source.columns else pd.DataFrame(), "study_locomotif_top10_agnostic")
loco_top_table(source[source["mode"].astype(str).eq("conditioned")] if "mode" in source.columns else pd.DataFrame(), "study_locomotif_top10_conditioned")
for label in ["high_vol", "low_vol"]:
    loco_top_table(source[source["regime_label"].astype(str).eq(label)] if "regime_label" in source.columns else pd.DataFrame(), f"study_locomotif_top10_{label}")
'''),
        md("## Interval Illustrations"),
        code(r'''
def plot_locomotif_timeline(df, asset, frequency, regime_label=None, filename="study_locomotif_timeline"):
    subset = filter_scope(df, asset, frequency)
    if regime_label and "regime_label" in subset.columns:
        subset = subset[subset["regime_label"].astype(str).eq(regime_label)]
    if subset.empty:
        print(f"No LoCoMotif intervals for {asset} {frequency} {regime_label or ''}")
        return
    feature = load_feature_data(asset, frequency)
    ts = timestamp_column(feature)
    fig, ax = plt.subplots(figsize=(14, 5))
    if not feature.empty and ts and "close" in feature.columns:
        sampled = feature[[ts, "close"]].dropna().sort_values(ts)
        if len(sampled) > 8000:
            sampled = sampled.iloc[:: int(np.ceil(len(sampled) / 8000))]
        ax.plot(sampled[ts], sampled["close"], color="0.75", linewidth=0.8, label="close")
    draw = subset.head(50)
    for j, (_, row) in enumerate(draw.iterrows()):
        start = row.get("motif_start_timestamp")
        end = row.get("motif_end_timestamp")
        if pd.notna(start) and pd.notna(end):
            ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color=plt.cm.tab10(j % 10), alpha=0.2)
    ax.set_title(f"LoCoMotif intervals - {asset} {frequency} {regime_label or 'all'}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Close price or interval spans")
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, filename, FIGURE_DIRS["locomotif"])
    plt.show()

plot_locomotif_timeline(loco, "BTCUSDT", "15m", None, "study_locomotif_BTCUSDT_15m_agnostic_intervals")
plot_locomotif_timeline(loco, "BTCUSDT", "15m", "high_vol", "study_locomotif_BTCUSDT_15m_high_vol_intervals")
plot_locomotif_timeline(loco, "BTCUSDT", "15m", "low_vol", "study_locomotif_BTCUSDT_15m_low_vol_intervals")

if figure_files:
    fig_table = pd.DataFrame({"figure_path": [str(p) for p in figure_files]})
    display_table(fig_table, 20)
    save_table(fig_table, "study_locomotif_existing_figure_inventory")
else:
    print("No existing LoCoMotif figures with prefix 04_ were found.")
'''),
        md("## Runtime and Failure Audit"),
        code(r'''
display_table(loco_runtime, 30)
save_table(loco_runtime, "study_locomotif_runtime_raw")
if "status" in loco_runtime.columns and not loco_runtime.empty:
    status_counts = loco_runtime["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="rows")
    display_table(status_counts)
    save_table(status_counts, "study_locomotif_runtime_status_counts")

if not loco_paths["failures"].exists():
    print("LoCoMotif failure file is not available.")
elif loco_failures.empty:
    print("LoCoMotif failure file exists and contains no rows; no internal failures recorded in that file.")
else:
    display_table(loco_failures, 30)
    save_table(loco_failures, "study_locomotif_failures")

print("LoCoMotif is reported as a controlled subset experiment rather than a full-scale benchmark.")
'''),
        md("## LoCoMotif vs Matrix Profile"),
        code(r'''
mp_dir = result_path("motifs", "matrix_profile")
mp_results = safe_read_parquet(resolve_existing_file(mp_dir, "matrix_profile_motif_results.parquet"))
mp_eval = safe_read_parquet(resolve_existing_file(mp_dir, "matrix_profile_evaluation.parquet"))
mp_runtime = safe_read_parquet(resolve_existing_file(mp_dir, "matrix_profile_runtime.parquet"))

comparison = pd.DataFrame([
    {"Aspect": "Motif type", "Matrix Profile": "fixed-length subsequences", "LoCoMotif": "variable/local-constrained intervals"},
    {"Aspect": "Scale completed", "Matrix Profile": "full controlled benchmark", "LoCoMotif": "controlled subset"},
    {"Aspect": "Result rows", "Matrix Profile": len(mp_results), "LoCoMotif": len(loco)},
    {"Aspect": "Evaluation rows", "Matrix Profile": len(mp_eval), "LoCoMotif": len(loco_eval)},
    {"Aspect": "Runtime rows", "Matrix Profile": len(mp_runtime), "LoCoMotif": len(loco_runtime)},
    {"Aspect": "Strength", "Matrix Profile": "scalable baseline", "LoCoMotif": "flexible variable-length motif structure"},
    {"Aspect": "Limitation", "Matrix Profile": "fixed-length windows", "LoCoMotif": "more computationally constrained"},
])
display_table(comparison, 10)
save_table(comparison, "study_locomotif_vs_matrix_profile_comparison")
'''),
        md("""
## Key findings
LoCoMotif adds a variable/local-constrained interval perspective when populated interval or evaluation rows are available. Runtime and failure tables determine whether the run should be presented as completed, partial, or empty.

## Thesis-safe interpretation
LoCoMotif should be presented as a complementary controlled subset experiment unless the saved files show full-scale coverage. It supports the Matrix Profile results by testing a more flexible motif formulation, but it should not be framed as a like-for-like scalability benchmark unless the result files support that claim.

## Limitations
This notebook does not rerun LoCoMotif. Empty motif or evaluation outputs mean no motif-quality numerical claims can be made from LoCoMotif rows. Existing figures are inventoried but not treated as new evidence unless their source files are present.

## Recommended figures for thesis
- LoCoMotif interval timeline when interval rows are available
- Runtime status counts
- LoCoMotif vs Matrix Profile comparison table
- Existing selected `04_` figures from the controlled subset run
"""),
    ]
    return cells


def write_notebook(path: Path, cells) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


README = r'''
# Study Notebook Suite

These notebooks are for thesis illustration and result interpretation only. They read existing result files and create study figures/tables under `HPC workflow/HPC_Regime_and_motif_discovery/reports/study_notebooks`. They do not rerun regime detection, Matrix Profile, or LoCoMotif, and they do not modify original result files.

## Analysis-only safety

The notebooks never import or call STUMPY, STUMP/MSTUMP, HMM fitting, LoCoMotif search, or any other expensive experiment algorithm. They only use `pandas`, `numpy`, `matplotlib`, `pathlib`, `json`, and the local `study_helpers.py` module to read saved files and produce derived visualizations/tables.

## Notebooks

1. `01_regime_detection_visual_study.ipynb`
   - Loads quantile and HMM regime outputs.
   - Builds coverage tables, regime count charts, close/regime overlays, volatility-by-regime charts, transition heatmaps, and duration summaries.

2. `02_regime_comparison_quantile_vs_hmm.ipynb`
   - Loads HMM model selection, posterior confidence, quantile/HMM comparison, confusion, and persistence files.
   - Creates ARI/NMI charts, confusion heatmaps, confidence plots, and self-transition comparisons when the files are populated.

3. `03_matrix_profile_motif_visual_study.ipynb`
   - Loads Matrix Profile motif results, evaluation, runtime, and profile curves.
   - Creates file inventories, motif overview tables, distance threshold tables, top motif tables, motif overlays, timelines, overlap tables, evaluation summaries, and CPU/GPU runtime audits.

4. `04_locomotif_visual_study.ipynb`
   - Loads LoCoMotif result, evaluation, runtime, and failure files.
   - Creates scope tables, interval length plots, top motif-set tables, interval timelines, runtime/failure audits, and a LoCoMotif-vs-Matrix-Profile comparison.

## Outputs

The notebooks create these folders if needed:

- `reports/study_notebooks/figures/regime`
- `reports/study_notebooks/figures/regime_comparison`
- `reports/study_notebooks/figures/matrix_profile`
- `reports/study_notebooks/figures/locomotif`
- `reports/study_notebooks/tables`
- `reports/study_notebooks/html_exports`

Figures are saved as PNG and, where possible, PDF. Important tables are saved as CSV.

## Run Commands

Run from the repository root:

```bash
cd ~/Final_master_thesis
source .thesis-env/bin/activate
jupyter nbconvert --to notebook --execute \
  "HPC workflow/HPC_Regime_and_motif_discovery/notebooks/study/01_regime_detection_visual_study.ipynb" \
  --output "executed_01_regime_detection_visual_study.ipynb" \
  --output-dir "HPC workflow/HPC_Regime_and_motif_discovery/reports/study_notebooks/html_exports" \
  --ExecutePreprocessor.timeout=-1
```

```bash
jupyter nbconvert --to notebook --execute \
  "HPC workflow/HPC_Regime_and_motif_discovery/notebooks/study/02_regime_comparison_quantile_vs_hmm.ipynb" \
  --output "executed_02_regime_comparison_quantile_vs_hmm.ipynb" \
  --output-dir "HPC workflow/HPC_Regime_and_motif_discovery/reports/study_notebooks/html_exports" \
  --ExecutePreprocessor.timeout=-1
```

```bash
jupyter nbconvert --to notebook --execute \
  "HPC workflow/HPC_Regime_and_motif_discovery/notebooks/study/03_matrix_profile_motif_visual_study.ipynb" \
  --output "executed_03_matrix_profile_motif_visual_study.ipynb" \
  --output-dir "HPC workflow/HPC_Regime_and_motif_discovery/reports/study_notebooks/html_exports" \
  --ExecutePreprocessor.timeout=-1
```

```bash
jupyter nbconvert --to notebook --execute \
  "HPC workflow/HPC_Regime_and_motif_discovery/notebooks/study/04_locomotif_visual_study.ipynb" \
  --output "executed_04_locomotif_visual_study.ipynb" \
  --output-dir "HPC workflow/HPC_Regime_and_motif_discovery/reports/study_notebooks/html_exports" \
  --ExecutePreprocessor.timeout=-1
```

## Notes

- The notebooks resolve exact expected result filenames first, then fall back to suffixed files such as local smoke-test outputs if those are the only files present.
- Every numerical table or figure is derived from existing parquet result files or thesis-scope feature files.
- If a file or column is missing, the relevant cell prints that fact and continues.
- The quantile regime caveat is included in the regime notebooks: quantile outputs store `rolling_volatility_60` as the actual volatility column across quantile method identifiers, so they are interpreted as 60-period rolling-volatility regimes with different regime-count granularities.
'''


def main() -> None:
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    for path in [
        REPORT_ROOT / "figures" / "regime",
        REPORT_ROOT / "figures" / "regime_comparison",
        REPORT_ROOT / "figures" / "matrix_profile",
        REPORT_ROOT / "figures" / "locomotif",
        REPORT_ROOT / "tables",
        REPORT_ROOT / "html_exports",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    write_text(STUDY_DIR / "study_helpers.py", HELPERS.strip() + "\n")
    write_text(STUDY_DIR / "README_STUDY_NOTEBOOKS.md", README.strip() + "\n")
    notebooks = {
        "01_regime_detection_visual_study.ipynb": notebook_1(),
        "02_regime_comparison_quantile_vs_hmm.ipynb": notebook_2(),
        "03_matrix_profile_motif_visual_study.ipynb": notebook_3(),
        "04_locomotif_visual_study.ipynb": notebook_4(),
    }
    for filename, cells in notebooks.items():
        write_notebook(STUDY_DIR / filename, cells)

    summary = {
        "notebooks_created": list(notebooks),
        "helper_files_created": [
            str(STUDY_DIR / "study_helpers.py"),
            str(STUDY_DIR / "README_STUDY_NOTEBOOKS.md"),
        ],
        "expected_outputs": [
            str(REPORT_ROOT / "figures" / "regime"),
            str(REPORT_ROOT / "figures" / "regime_comparison"),
            str(REPORT_ROOT / "figures" / "matrix_profile"),
            str(REPORT_ROOT / "figures" / "locomotif"),
            str(REPORT_ROOT / "tables"),
            str(REPORT_ROOT / "html_exports"),
        ],
        "assumptions": [
            "Notebooks read existing parquet files only and do not rerun heavy experiments.",
            "Exact expected filenames are preferred; suffixed parquet files are used as fallback when exact files are absent.",
            "Feature overlays use thesis-scope crypto feature files under final_dataset/features/crypto.",
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
