from __future__ import annotations

import argparse
import math
import os
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "final_visual_evidence"
FIG = OUT / "figures"
TAB = OUT / "tables"
CFG = OUT / "configs"
LATEX = OUT / "latex"
LOG = OUT / "logs"

STUDY_TABLES = (
    ROOT
    / "HPC workflow"
    / "HPC_Regime_and_motif_discovery"
    / "reports"
    / "study_notebooks"
    / "tables"
)
CONTROL_TABLES = ROOT / "reports" / "locomotif_controlled_slice_comparison" / "tables"
CONTROL_FIG = ROOT / "reports" / "locomotif_controlled_slice_comparison" / "figures"


EVENTS = [
    ("2020-03-01", "COVID-19 crash / global risk-off"),
    ("2021-05-01", "crypto market drawdown"),
    ("2021-11-01", "crypto peak / reversal period"),
    ("2022-05-01", "Terra/Luna and crypto stress period"),
    ("2022-06-01", "crypto deleveraging / Celsius stress period"),
    ("2022-11-01", "FTX collapse period"),
    ("2023-03-01", "banking stress period"),
    ("2024-03-01", "crypto rally / ETF-flow period"),
    ("2025-06-01", "recent low-vol / post-rally observation period"),
]


def ensure_dirs() -> None:
    for path in [OUT, FIG, TAB, CFG, LATEX, LOG, OUT / "notebooks", OUT / "scripts"]:
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def slug(value: object) -> str:
    text = str(value).lower().strip()
    keep = []
    for char in text:
        keep.append(char if char.isalnum() else "_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "na"


def infer_role(path: Path) -> str:
    name = path.name.lower()
    if "vix" in name:
        return "vix_market_stress_context"
    if "locomotif" in name and "runtime" in name:
        return "locomotif_runtime"
    if "locomotif" in name and ("motif" in name or "occurrence" in name):
        return "locomotif_motif_result"
    if "matrix_profile" in name or name.startswith("mp_") or "_mp_" in name:
        if "runtime" in name:
            return "matrix_profile_runtime"
        return "matrix_profile_motif_result"
    if "runtime" in name:
        return "runtime_evidence"
    if path.suffix.lower() == ".parquet":
        return "feature_data"
    if path.suffix.lower() == ".png":
        return "existing_figure"
    return "supporting_input"


def table_shape(path: Path) -> tuple[object, object, str]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
            return len(df), len(df.columns), ""
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as pq

                meta = pq.ParquetFile(path).metadata
                return meta.num_rows, meta.num_columns, ""
            except Exception:
                df = pd.read_parquet(path)
                return len(df), len(df.columns), "shape read with pandas fallback"
    except Exception as exc:
        return "", "", f"shape unavailable: {exc}"
    return "", "", ""


def discover_inputs() -> pd.DataFrame:
    ensure_dirs()
    roots = [
        ROOT / "final_dataset" / "features",
        ROOT / "final_dataset" / "processed",
        ROOT / "reports" / "locomotif_controlled_slice_comparison",
        ROOT / "HPC workflow" / "HPC_Regime_and_motif_discovery" / "results",
        ROOT / "HPC workflow" / "HPC_Regime_and_motif_discovery" / "reports",
        ROOT / "Final Report" / "Final_Thesis" / "figures",
        ROOT / "matrix_profile_audit",
    ]
    terms = (
        "matrix_profile",
        "mp",
        "motif",
        "evaluation",
        "runtime",
        "distance",
        "overlay",
        "profile",
        "locomotif",
        "vix",
        "BTCUSDT_15m_features_2020_2025",
        "BTCUSDT_1h_features_2020_2025",
        "ETHUSDT_15m_features_2020_2025",
        "ETHUSDT_1h_features_2020_2025",
    )
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and any(term.lower() in path.name.lower() for term in terms):
                files.append(path)
    rows = []
    for path in sorted(set(files)):
        n_rows, n_cols, note = table_shape(path)
        rows.append(
            {
                "file_path": rel(path),
                "file_type": path.suffix.lower().lstrip("."),
                "role": infer_role(path),
                "rows": n_rows,
                "columns": n_cols,
                "notes": note,
            }
        )
    inventory = pd.DataFrame(rows)
    write_csv(inventory, TAB / "input_inventory.csv")
    return inventory


def load_feature(asset: str, frequency: str) -> pd.DataFrame:
    path = ROOT / "final_dataset" / "features" / "crypto" / f"{asset}_{frequency}_features_2020_2025.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_vix() -> pd.DataFrame:
    path = ROOT / "final_dataset" / "features" / "volatility" / "VIX_1d_features_2010_2025.parquet"
    if not path.exists():
        return pd.DataFrame()
    vix = pd.read_parquet(path)
    vix["timestamp"] = pd.to_datetime(vix["timestamp"], utc=True)
    vix = vix.sort_values("timestamp").reset_index(drop=True)
    vix["date"] = vix["timestamp"].dt.floor("D")
    vix["vix_close"] = pd.to_numeric(vix.get("close", vix.get("vix_level")), errors="coerce")
    vix["vix_rolling_20_mean"] = vix["vix_close"].rolling(20, min_periods=1).mean()
    vix["vix_percentile_full_sample"] = vix["vix_close"].rank(pct=True)
    pct = vix["vix_percentile_full_sample"]
    vix["vix_regime_label"] = np.select(
        [pct > 0.95, pct > 0.80, pct > 0.50],
        ["crisis", "stressed", "elevated"],
        default="calm",
    )
    return vix


def frequency_delta(freq: str) -> pd.Timedelta:
    if str(freq).endswith("m"):
        return pd.Timedelta(minutes=int(str(freq).replace("m", "")))
    if str(freq).endswith("h"):
        return pd.Timedelta(hours=int(str(freq).replace("h", "")))
    return pd.Timedelta(days=1)


def normalize_mp_rows() -> pd.DataFrame:
    frames = []
    for path in [
        CONTROL_TABLES / "mp_controlled_slice_motifs.csv",
        CONTROL_TABLES / "mp_controlled_slice_summary.csv",
    ]:
        df = read_csv(path)
        if not df.empty:
            df["source_file"] = rel(path)
            frames.append(df)
    for path in STUDY_TABLES.glob("study_mp_top10*.csv"):
        df = read_csv(path)
        if not df.empty:
            df["source_file"] = rel(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "motif_distance" in df.columns:
        if "best_motif_distance" in df.columns:
            df["best_motif_distance"] = df["best_motif_distance"].combine_first(df["motif_distance"])
            df = df.drop(columns=["motif_distance"])
        else:
            df = df.rename(columns={"motif_distance": "best_motif_distance"})
    for col in ["asset", "frequency", "regime_label", "window_length", "best_motif_distance"]:
        if col not in df.columns:
            df[col] = np.nan
    rows = []
    for idx, row in df.iterrows():
        for side in [1, 2]:
            ts_col = f"motif_timestamp_{side}"
            start_col = f"motif_start_{side}"
            if ts_col not in row or pd.isna(row.get(ts_col)):
                continue
            start_ts = pd.to_datetime(row.get(ts_col), utc=True, errors="coerce")
            if pd.isna(start_ts):
                continue
            w = int(row.get("window_length") if pd.notna(row.get("window_length")) else 24)
            freq = row.get("frequency", "1h")
            end_ts = start_ts + frequency_delta(freq) * max(w - 1, 1)
            rows.append(
                {
                    "method": "Matrix Profile",
                    "asset": row.get("asset", "BTCUSDT"),
                    "frequency": freq,
                    "regime_label": row.get("regime_label", row.get("mode", "agnostic")),
                    "mode": row.get("mode", ""),
                    "feature": row.get("feature", row.get("feature_set", "close")),
                    "motif_id": row.get("motif_rank", row.get("run_key", f"mp_{idx}")),
                    "occurrence_id": f"pair{idx + 1}_{side}",
                    "start_timestamp": start_ts,
                    "end_timestamp": end_ts,
                    "window_length_or_lmin_lmax": w,
                    "score_or_distance": row.get("best_motif_distance"),
                    "occurrence_count": 2,
                    "source_file": row.get("source_file", ""),
                    "notes": "fixed-length nearest-neighbour pair occurrence",
                }
            )
    occ = pd.DataFrame(rows)
    if occ.empty:
        return occ
    return occ.drop_duplicates(["method", "asset", "frequency", "start_timestamp", "end_timestamp", "source_file"])


def normalize_locomotif_rows() -> pd.DataFrame:
    frames = []
    controlled = read_csv(CONTROL_TABLES / "locomotif_controlled_occurrences.csv")
    if not controlled.empty:
        controlled["source_file"] = rel(CONTROL_TABLES / "locomotif_controlled_occurrences.csv")
        controlled = controlled.rename(
            columns={
                "occurrence_start_timestamp": "motif_start_timestamp",
                "occurrence_end_timestamp": "motif_end_timestamp",
                "occurrence_length": "motif_length",
            }
        )
        frames.append(controlled)
    for path in STUDY_TABLES.glob("study_locomotif_top10*.csv"):
        df = read_csv(path)
        if not df.empty:
            df["source_file"] = rel(path)
            frames.append(df)
    for path in (ROOT / "reports" / "results" / "03_year_2025_mp_vs_real_locomotif").glob("*_locomotif_motif_sets.csv"):
        df = read_csv(path)
        if not df.empty:
            df = df.rename(
                columns={
                    "start_time": "motif_start_timestamp",
                    "end_time": "motif_end_timestamp",
                    "length": "motif_length",
                    "feature_set_name": "feature_set",
                }
            )
            df["frequency"] = "1h"
            df["mode"] = "agnostic_2025"
            df["regime_label"] = "agnostic"
            df["source_file"] = rel(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    rows = []
    for idx, row in df.iterrows():
        if str(row.get("role", "occurrence")).lower() == "candidate":
            continue
        start_ts = pd.to_datetime(row.get("motif_start_timestamp"), utc=True, errors="coerce")
        end_ts = pd.to_datetime(row.get("motif_end_timestamp"), utc=True, errors="coerce")
        if pd.isna(start_ts) or pd.isna(end_ts):
            continue
        lmin = row.get("l_min", np.nan)
        lmax = row.get("l_max", np.nan)
        motif_length = row.get("motif_length", np.nan)
        if pd.notna(lmin) and pd.notna(lmax):
            length_label = f"{lmin}-{lmax}"
        elif pd.notna(motif_length):
            length_label = f"observed_length={motif_length}"
        else:
            length_label = ""
        rows.append(
            {
                "method": "LoCoMotif",
                "asset": row.get("asset", "BTCUSDT"),
                "frequency": row.get("frequency", "15m"),
                "regime_label": row.get("regime_label", row.get("mode", "agnostic")),
                "mode": row.get("mode", ""),
                "feature": row.get("feature", row.get("feature_set", "multivariate")),
                "motif_id": row.get("motif_set_rank", row.get("motif_set_id", f"loco_{idx}")),
                "occurrence_id": row.get("motif_instance_id", row.get("occurrence_id", idx + 1)),
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "window_length_or_lmin_lmax": length_label,
                "score_or_distance": row.get("motif_score", np.nan),
                "occurrence_count": row.get("motif_set_size", row.get("occurrence_count", np.nan)),
                "source_file": row.get("source_file", ""),
                "notes": "real LoCoMotif time-warped motif-set occurrence",
            }
        )
    occ = pd.DataFrame(rows)
    if occ.empty:
        return occ
    return occ.drop_duplicates(["method", "asset", "frequency", "start_timestamp", "end_timestamp", "motif_id"])


def all_occurrences() -> pd.DataFrame:
    frames = [normalize_mp_rows(), normalize_locomotif_rows()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def realized_vol(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    if "log_return" in df.columns:
        vals = pd.to_numeric(df["log_return"], errors="coerce")
    else:
        vals = np.log(pd.to_numeric(df["close"], errors="coerce")).diff()
    return float(vals.std(skipna=True) * math.sqrt(max(len(vals.dropna()), 1)))


def draw_candles(ax, df: pd.DataFrame) -> None:
    x = mdates.date2num(df["timestamp"].dt.to_pydatetime())
    width = 0.6 * np.median(np.diff(x)) if len(x) > 1 else 0.02
    for xi, row in zip(x, df.itertuples(index=False)):
        open_ = float(getattr(row, "open"))
        high = float(getattr(row, "high"))
        low = float(getattr(row, "low"))
        close = float(getattr(row, "close"))
        color = "#1a9850" if close >= open_ else "#d73027"
        ax.plot([xi, xi], [low, high], color=color, linewidth=0.8)
        y = min(open_, close)
        h = max(abs(close - open_), 1e-9)
        ax.add_patch(Rectangle((xi - width / 2, y), width, h, facecolor=color, edgecolor=color, alpha=0.8))
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.grid(True, axis="y", alpha=0.2)


def plot_occurrence(occ: pd.Series, figure_prefix: str) -> dict[str, str]:
    feature = load_feature(str(occ["asset"]), str(occ["frequency"]))
    if feature.empty:
        return {"notes": "feature parquet unavailable; figure skipped"}
    start = pd.Timestamp(occ["start_timestamp"])
    end = pd.Timestamp(occ["end_timestamp"])
    step = frequency_delta(str(occ["frequency"]))
    n = max(int((end - start) / step) + 1, 4)
    context_start = start - step * n * 2
    context_end = end + step * n * 2
    exact = feature[(feature["timestamp"] >= start) & (feature["timestamp"] <= end)].copy()
    context = feature[(feature["timestamp"] >= context_start) & (feature["timestamp"] <= context_end)].copy()
    if exact.empty or context.empty:
        return {"notes": "timestamp interval not found in feature parquet; figure skipped"}
    has_ohlc = all(col in feature.columns for col in ["open", "high", "low", "close"])
    out = {}

    pattern_path = FIG / f"{figure_prefix}_normal_pattern.png"
    y = pd.to_numeric(exact.get("close"), errors="coerce")
    z = (y - y.mean()) / (y.std() if y.std() else 1.0)
    plt.figure(figsize=(7, 3))
    plt.plot(exact["timestamp"], z, color="#2c7bb6", linewidth=1.6)
    plt.title(f"{occ['method']} {occ['asset']} {occ['frequency']} {occ['regime_label']} normalized motif")
    plt.ylabel("z-normalized close")
    plt.grid(True, alpha=0.25)
    savefig(pattern_path)
    out["normal_plot_path"] = rel(pattern_path)

    for label, data in [("zoom", exact), ("context", context)]:
        path = FIG / f"{figure_prefix}_{label}.png"
        rows = 2 if "volume" in data.columns else 1
        height = 5 if rows == 2 else 4
        fig, axes = plt.subplots(rows, 1, figsize=(10, height), sharex=True, gridspec_kw={"height_ratios": [3, 1]} if rows == 2 else None)
        ax = axes[0] if rows == 2 else axes
        if has_ohlc:
            draw_candles(ax, data)
            note = "OHLC candlestick"
        else:
            ax.plot(data["timestamp"], data["close"], color="#2c7bb6")
            note = "close-line fallback; OHLC unavailable"
        ax.axvspan(start, end, color="#fdae61", alpha=0.25, label="motif interval")
        ax.set_title(
            f"{occ['method']} | {occ['asset']} {occ['frequency']} | {occ['regime_label']} | "
            f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}"
        )
        ax.set_ylabel("price")
        ax.legend(loc="best", fontsize=8)
        if rows == 2:
            axes[1].bar(data["timestamp"], data["volume"], width=0.01, color="#636363")
            axes[1].set_ylabel("volume")
            axes[1].grid(True, axis="y", alpha=0.2)
        savefig(path)
        out[f"{label}_plot_path"] = rel(path)
        out["notes"] = note
    return out


def build_candlestick_and_vix() -> pd.DataFrame:
    ensure_dirs()
    CFG.joinpath("market_event_labels.yaml").write_text(
        "\n".join([f"- month: {date[:7]}\n  label: \"{label}\"\n  caveat: \"approximate context only; not a causal label\"" for date, label in EVENTS]),
        encoding="utf-8",
    )
    occ = all_occurrences()
    if occ.empty:
        write_csv(pd.DataFrame([{"reason": "No timestamped motif occurrences found."}]), TAB / "candlestick_motif_figure_index.csv")
        return pd.DataFrame()

    selected = (
        occ.sort_values(["method", "asset", "frequency", "regime_label", "score_or_distance"], na_position="last")
        .groupby(["method", "asset", "frequency", "regime_label"], dropna=False)
        .head(2)
        .head(36)
        .reset_index(drop=True)
    )
    figure_rows = []
    vix_rows = []
    for idx, row in selected.iterrows():
        prefix = f"motif_{idx + 1:02d}_{slug(row['method'])}_{slug(row['asset'])}_{slug(row['frequency'])}_{slug(row['regime_label'])}_{slug(row['motif_id'])}_{slug(row['occurrence_id'])}"
        paths = plot_occurrence(row, prefix)
        figure_rows.append(
            {
                "figure_path": paths.get("context_plot_path", ""),
                "method": row["method"],
                "asset": row["asset"],
                "frequency": row["frequency"],
                "regime_label": row["regime_label"],
                "motif_id": row["motif_id"],
                "occurrence_id": row["occurrence_id"],
                "start_timestamp": row["start_timestamp"],
                "end_timestamp": row["end_timestamp"],
                "window_length_or_lmin_lmax": row["window_length_or_lmin_lmax"],
                "source_table": row["source_file"],
                "notes": paths.get("notes", ""),
            }
        )
        vix_rows.append(row.to_dict())
    index = pd.DataFrame(figure_rows)
    write_csv(index, TAB / "candlestick_motif_figure_index.csv")
    build_vix_context(pd.DataFrame(vix_rows))
    return index


def nearest_event(date: pd.Timestamp) -> tuple[str, int, str]:
    if pd.isna(date):
        return "", np.nan, ""
    best_label = ""
    best_days = 10**9
    for event_date, label in EVENTS:
        days = abs((date.floor("D") - pd.Timestamp(event_date, tz="UTC")).days)
        if days < best_days:
            best_days = days
            best_label = label
    caveat = "near event window; contextual only, not causal" if best_days <= 45 else "not near configured event window"
    return best_label, int(best_days), caveat


def build_vix_context(occ: pd.DataFrame | None = None) -> pd.DataFrame:
    if occ is None or occ.empty:
        occ = all_occurrences().head(80)
    vix = load_vix()
    if occ.empty or vix.empty:
        write_csv(pd.DataFrame([{"reason": "Missing motif occurrences or VIX data."}]), TAB / "motif_occurrences_with_vix_context.csv")
        return pd.DataFrame()
    rows = []
    event_rows = []
    for _, row in occ.iterrows():
        start = pd.Timestamp(row["start_timestamp"]).tz_convert("UTC")
        end = pd.Timestamp(row["end_timestamp"]).tz_convert("UTC")
        date = start.floor("D")
        prior = vix[vix["date"] <= date]
        vix_row = prior.iloc[-1] if not prior.empty else pd.Series(dtype=object)
        feat = load_feature(str(row["asset"]), str(row["frequency"]))
        step = frequency_delta(str(row["frequency"]))
        n = max(int((end - start) / step) + 1, 4)
        pre = feat[(feat["timestamp"] >= start - step * n) & (feat["timestamp"] < start)] if not feat.empty else pd.DataFrame()
        during = feat[(feat["timestamp"] >= start) & (feat["timestamp"] <= end)] if not feat.empty else pd.DataFrame()
        post = feat[(feat["timestamp"] > end) & (feat["timestamp"] <= end + step * n)] if not feat.empty else pd.DataFrame()
        out = {
            "method": row["method"],
            "asset": row["asset"],
            "frequency": row["frequency"],
            "regime_label": row["regime_label"],
            "motif_id": row["motif_id"],
            "occurrence_id": row["occurrence_id"],
            "start_timestamp": start,
            "end_timestamp": end,
            "date": date.date().isoformat(),
            "vix_close": vix_row.get("vix_close", np.nan),
            "vix_percentile_full_sample": vix_row.get("vix_percentile_full_sample", np.nan),
            "vix_regime_label": vix_row.get("vix_regime_label", ""),
            "local_realized_vol_pre": realized_vol(pre),
            "local_realized_vol_during": realized_vol(during),
            "local_realized_vol_post": realized_vol(post),
            "notes": "VIX is used as a broad external market-stress proxy, not as a crypto-specific volatility label.",
        }
        rows.append(out)
        label, days, caveat = nearest_event(date)
        event_rows.append(
            {
                "motif_timestamp": start,
                "nearest_event_label": label,
                "days_from_event_month": days,
                "vix_close": out["vix_close"],
                "vix_regime_label": out["vix_regime_label"],
                "caveat": caveat,
            }
        )
    ctx = pd.DataFrame(rows)
    write_csv(ctx, TAB / "motif_occurrences_with_vix_context.csv")
    write_csv(pd.DataFrame(event_rows), TAB / "motif_event_context_table.csv")
    plot_vix_figures(ctx, vix)
    return ctx


def plot_vix_figures(ctx: pd.DataFrame, vix: pd.DataFrame) -> None:
    if ctx.empty or vix.empty:
        return
    plt.figure(figsize=(11, 4))
    plt.plot(vix["timestamp"], vix["vix_close"], color="#4d4d4d", linewidth=1.0)
    markers = {"Matrix Profile": "o", "LoCoMotif": "^"}
    colors = {"high_vol": "#d73027", "low_vol": "#1a9850", "agnostic": "#4575b4", "all": "#4575b4"}
    for method, grp in ctx.groupby("method"):
        dates = pd.to_datetime(grp["date"], utc=True)
        plt.scatter(
            dates,
            grp["vix_close"],
            marker=markers.get(method, "o"),
            s=34,
            alpha=0.8,
            label=method,
            c=[colors.get(str(x), "#756bb1") for x in grp["regime_label"]],
        )
    plt.title("VIX with motif occurrence markers")
    plt.ylabel("VIX close")
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=8)
    savefig(FIG / "vix_with_motif_occurrence_markers.png")

    ctx["category"] = ctx["method"].str.replace("Matrix Profile", "MP") + " " + ctx["regime_label"].astype(str)
    cats = list(ctx["category"].dropna().unique())
    data = [pd.to_numeric(ctx.loc[ctx["category"] == cat, "vix_close"], errors="coerce").dropna() for cat in cats]
    plt.figure(figsize=(max(8, len(cats) * 0.8), 4))
    plt.boxplot(data, labels=cats, showfliers=False)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("VIX close")
    plt.title("VIX distribution by motif category")
    plt.grid(True, axis="y", alpha=0.2)
    savefig(FIG / "vix_distribution_by_motif_category.png")

    counts = ctx["vix_regime_label"].value_counts().reindex(["calm", "elevated", "stressed", "crisis"]).fillna(0)
    plt.figure(figsize=(7, 4))
    plt.bar(counts.index, counts.values, color=["#91bfdb", "#fee090", "#fc8d59", "#d73027"])
    plt.title("Motif occurrences by VIX regime")
    plt.ylabel("occurrence count")
    savefig(FIG / "motif_occurrences_by_vix_regime.png")

    vol = ctx[["local_realized_vol_pre", "local_realized_vol_during", "local_realized_vol_post"]].apply(pd.to_numeric, errors="coerce")
    plt.figure(figsize=(7, 4))
    plt.boxplot([vol[c].dropna() for c in vol.columns], labels=["pre", "during", "post"], showfliers=False)
    plt.ylabel("local realized volatility")
    plt.title("Local volatility around motif occurrences")
    plt.grid(True, axis="y", alpha=0.2)
    savefig(FIG / "motif_local_volatility_pre_during_post.png")

    top = ctx.sort_values("vix_close", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    cols = ["method", "asset", "frequency", "regime_label", "date", "vix_close", "vix_regime_label"]
    table = ax.table(cellText=top[cols].round(3).values, colLabels=cols, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)
    ax.set_title("Top motif occurrences by VIX context")
    savefig(FIG / "vix_event_context_top_motifs.png")


def copy_if_exists(src: Path, dst_name: str) -> str:
    if src.exists():
        dst = FIG / dst_name
        shutil.copy2(src, dst)
        return rel(dst)
    return ""


def build_comparison() -> pd.DataFrame:
    ensure_dirs()
    discover_inputs()
    summary = read_csv(CONTROL_TABLES / "mp_vs_locomotif_controlled_comparison.csv")
    if summary.empty:
        summary = pd.DataFrame()
    runtime = read_csv(CONTROL_TABLES / "locomotif_controlled_runtime.csv")
    control_mp = read_csv(CONTROL_TABLES / "mp_controlled_slice_motifs.csv")
    study_loco = read_csv(STUDY_TABLES / "study_locomotif_evaluation_raw.csv")
    study_mp = read_csv(STUDY_TABLES / "study_mp_evaluation_raw.csv")
    frames = []
    if not summary.empty:
        frames.append(summary)
    if not study_loco.empty:
        frames.append(
            study_loco.rename(
                columns={
                    "number_of_motifs": "count",
                    "recurrence_count": "total_occurrences",
                    "mean_motif_distance_or_score": "mean_distance_or_score",
                }
            )
        )
    if not study_mp.empty:
        frames.append(
            study_mp.rename(
                columns={
                    "number_of_motifs": "count",
                    "recurrence_count": "total_occurrences",
                    "mean_motif_distance_or_score": "mean_distance_or_score",
                }
            )
        )
    final = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    write_csv(final, TAB / "final_mp_vs_locomotif_summary.csv")

    copies = {
        "final_mp_vs_locomotif_high_vol_side_by_side.png": "mp_vs_locomotif_high_vol_side_by_side.png",
        "final_mp_vs_locomotif_low_vol_side_by_side.png": "mp_vs_locomotif_low_vol_side_by_side.png",
        "final_mp_vs_locomotif_agnostic_1h_side_by_side.png": "micro_mp_vs_locomotif_btcusdt_1h_side_by_side.png",
        "final_locomotif_controlled_summary.png": "controlled_locomotif_experiment_summary.png",
        "final_mp_vs_locomotif_runtime.png": "mp_vs_locomotif_runtime.png",
        "final_mp_vs_locomotif_counts_and_occurrences.png": "mp_vs_locomotif_counts.png",
    }
    for dst, src in copies.items():
        copy_if_exists(CONTROL_FIG / src, dst)
    plot_method_object_diagram()
    plot_runtime_and_distributions(study_mp, study_loco, control_mp, runtime)
    return final


def plot_method_object_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.add_patch(Rectangle((0.05, 0.18), 0.38, 0.62, facecolor="#e8f4f8", edgecolor="#2c7bb6"))
    ax.add_patch(Rectangle((0.57, 0.18), 0.38, 0.62, facecolor="#f7f3e8", edgecolor="#d95f02"))
    ax.text(0.24, 0.68, "Matrix Profile", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.24, 0.51, "fixed-length nearest-neighbour pair", ha="center", fontsize=10)
    ax.plot([0.11, 0.20, 0.29, 0.38], [0.34, 0.46, 0.30, 0.42], color="#2c7bb6", linewidth=2)
    ax.plot([0.11, 0.20, 0.29, 0.38], [0.28, 0.40, 0.24, 0.36], color="#2c7bb6", linewidth=2, linestyle="--")
    ax.text(0.76, 0.68, "LoCoMotif", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.76, 0.51, "time-warped motif set with multiple occurrences", ha="center", fontsize=10)
    for off in [0.0, 0.05, -0.04]:
        ax.plot([0.62, 0.70, 0.79, 0.91], [0.32 + off, 0.45 + off, 0.29 + off, 0.40 + off], linewidth=1.8)
    ax.text(0.5, 0.08, "Counts are not directly equivalent: MP counts motif pairs; LoCoMotif counts motif sets and occurrences.", ha="center")
    savefig(FIG / "final_method_object_comparison_diagram.png")


def plot_runtime_and_distributions(study_mp: pd.DataFrame, study_loco: pd.DataFrame, control_mp: pd.DataFrame, loco_runtime: pd.DataFrame) -> None:
    source_rows = []
    if not study_mp.empty:
        source_rows.append({"figure": "mp_distance_distribution_by_regime.png", "source_file": rel(STUDY_TABLES / "study_mp_evaluation_raw.csv")})
        dist_col = "mean_motif_distance_or_score"
        groups = [g[dist_col].dropna() for _, g in study_mp.groupby("regime_label") if dist_col in g]
        labels = [str(k) for k, g in study_mp.groupby("regime_label") if dist_col in g and not g[dist_col].dropna().empty]
        if groups:
            plt.figure(figsize=(8, 4))
            plt.boxplot(groups, labels=labels, showfliers=False)
            plt.ylabel("mean motif distance")
            plt.title("MP distance distribution by regime")
            plt.grid(True, axis="y", alpha=0.2)
            savefig(FIG / "mp_distance_distribution_by_regime.png")
        groups = [g[dist_col].dropna() for _, g in study_mp.groupby("window_length") if dist_col in g]
        labels = [str(k) for k, g in study_mp.groupby("window_length") if dist_col in g and not g[dist_col].dropna().empty]
        if groups:
            plt.figure(figsize=(8, 4))
            plt.boxplot(groups, labels=labels, showfliers=False)
            plt.ylabel("mean motif distance")
            plt.xlabel("window length")
            plt.title("MP distance distribution by window")
            plt.grid(True, axis="y", alpha=0.2)
            savefig(FIG / "mp_distance_distribution_by_window.png")
        count = study_mp.groupby(["method", "regime_label"], dropna=False)["number_of_motifs"].sum().reset_index()
        plt.figure(figsize=(8, 4))
        labels = count["method"].astype(str) + " " + count["regime_label"].astype(str)
        plt.bar(labels, count["number_of_motifs"], color="#4575b4")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("motif count")
        plt.title("Motif count by regime and method")
        savefig(FIG / "motif_count_by_regime_and_method.png")
    runtime_frames = []
    mp_runtime = read_csv(STUDY_TABLES / "study_mp_runtime_raw.csv")
    if not mp_runtime.empty:
        tmp = mp_runtime.copy()
        tmp["method"] = "Matrix Profile " + tmp.get("profile_type", "").astype(str)
        runtime_frames.append(tmp[["method", "frequency", "runtime_seconds"]])
    if not study_loco.empty:
        tmp = study_loco.copy()
        tmp["method"] = "LoCoMotif"
        runtime_frames.append(tmp[["method", "frequency", "runtime_seconds"]])
    if runtime_frames:
        rt = pd.concat(runtime_frames, ignore_index=True)
        rt["label"] = rt["method"].astype(str) + " " + rt["frequency"].astype(str)
        groups = [pd.to_numeric(g["runtime_seconds"], errors="coerce").dropna() for _, g in rt.groupby("label")]
        labels = [str(k) for k, g in rt.groupby("label") if not pd.to_numeric(g["runtime_seconds"], errors="coerce").dropna().empty]
        if groups:
            plt.figure(figsize=(max(8, len(labels) * 0.8), 4))
            plt.boxplot(groups, labels=labels, showfliers=False)
            plt.yscale("log")
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("runtime seconds (log)")
            plt.title("Runtime distribution by frequency and method")
            savefig(FIG / "mp_runtime_distribution_by_frequency.png")
            plt.figure(figsize=(max(8, len(labels) * 0.8), 4))
            means = [g.mean() for g in groups]
            plt.bar(labels, means, color="#756bb1")
            plt.yscale("log")
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("mean runtime seconds (log)")
            plt.title("Runtime efficiency summary")
            savefig(FIG / "runtime_efficiency_summary_logscale.png")
    if not study_loco.empty:
        occ = study_loco.groupby("regime_label")["recurrence_count"].sum().reset_index()
        plt.figure(figsize=(7, 4))
        plt.bar(occ["regime_label"].astype(str), occ["recurrence_count"], color="#d95f02")
        plt.ylabel("LoCoMotif occurrence count")
        plt.title("LoCoMotif occurrence count by regime")
        savefig(FIG / "loco_occurrence_count_by_regime.png")
    write_csv(pd.DataFrame(source_rows), TAB / "final_distribution_plot_sources.csv")
    write_complexity()


def write_complexity() -> None:
    rows = [
        ["Matrix Profile univariate", "fixed-length motif pair", "n", "window length m", "optimized all-pairs subsequence similarity; scalable for fixed m", "study MP runtime tables", "strong baseline for fixed-length motif search"],
        ["Matrix Profile multivariate", "fixed-length multidimensional motif pair", "n, d", "window length m, dimension set", "more expensive because multiple channels and dimension combinations are evaluated", "study MP runtime tables show larger runtimes", "useful but costlier for rich feature spaces"],
        ["LoCoMotif", "time-warped motif set", "n", "lmin/lmax, rho, nb, overlap, warping", "runtime sensitive to slice length and motif-set search settings", "study LoCoMotif runtime tables and controlled timeout records", "more interpretable recurrence object when successful"],
        ["regime detection quantile", "regime labels", "n", "rolling window, quantile count", "cheap rolling-statistic thresholding", "feature/regime study outputs", "transparent conditioning mechanism"],
        ["HMM regime detection", "probabilistic state sequence", "n, states", "state count, covariance, fitting iterations", "requires fitting and model selection", "HMM study model-selection tables", "adds probabilistic regime modelling"],
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "method",
            "motif_object",
            "input_length_variable",
            "key_parameters",
            "theoretical_scaling_summary",
            "empirical_runtime_evidence",
            "thesis_interpretation",
        ],
    )
    write_csv(df, TAB / "algorithm_complexity_summary.csv")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")
    table_df = df[["method", "theoretical_scaling_summary", "thesis_interpretation"]]
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    ax.set_title("Algorithm complexity and thesis trade-off summary")
    savefig(FIG / "algorithm_complexity_tradeoff_summary.png")
    (LATEX / "algorithm_complexity_runtime_snippet.tex").write_text(
        "\\begin{table}[ht]\n\\centering\n\\caption{Algorithmic complexity and runtime evidence summary.}\n"
        "\\begin{tabular}{p{0.22\\linewidth}p{0.34\\linewidth}p{0.34\\linewidth}}\n\\hline\n"
        "Method & Scaling summary & Thesis interpretation \\\\\n\\hline\n"
        + "\n".join(
            f"{r.method} & {r.theoretical_scaling_summary} & {r.thesis_interpretation} \\\\"
            for r in df.itertuples(index=False)
        )
        + "\n\\hline\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )


def build_gallery_and_clustering() -> pd.DataFrame:
    ensure_dirs()
    occ = all_occurrences()
    if occ.empty:
        write_csv(pd.DataFrame([{"reason": "No timestamped occurrences available."}]), TAB / "top_motif_gallery_index.csv")
        return pd.DataFrame()
    occ["rank_metric"] = pd.to_numeric(occ["score_or_distance"], errors="coerce")
    mp = occ[occ["method"].eq("Matrix Profile")].sort_values("rank_metric", na_position="last")
    loco = occ[occ["method"].eq("LoCoMotif")].sort_values("occurrence_count", ascending=False, na_position="last")
    top = pd.concat([mp.head(10), loco.head(10)], ignore_index=True).drop_duplicates(
        ["method", "asset", "frequency", "start_timestamp", "end_timestamp"]
    ).head(18)
    rows = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        prefix = f"top_motif_{rank:02d}_{slug(row['method'])}_{slug(row['asset'])}_{slug(row['frequency'])}_{slug(row['regime_label'])}"
        paths = plot_occurrence(row, prefix)
        rows.append(
            {
                "rank": rank,
                "method": row["method"],
                "asset": row["asset"],
                "frequency": row["frequency"],
                "regime_label": row["regime_label"],
                "mode": row.get("mode", ""),
                "motif_id": row["motif_id"],
                "window_length_or_lmin_lmax": row["window_length_or_lmin_lmax"],
                "score_or_distance": row["score_or_distance"],
                "occurrence_count": row["occurrence_count"],
                "start_timestamp": row["start_timestamp"],
                "end_timestamp": row["end_timestamp"],
                "normal_plot_path": paths.get("normal_plot_path", ""),
                "candlestick_plot_path": paths.get("context_plot_path", ""),
                "source_file": row["source_file"],
            }
        )
    index = pd.DataFrame(rows)
    write_csv(index, TAB / "top_motif_gallery_index.csv")
    plot_gallery_pages(index)
    build_clustering(top)
    return index


def plot_gallery_pages(index: pd.DataFrame) -> None:
    if index.empty:
        return
    for page, start in enumerate(range(0, len(index), 6), start=1):
        subset = index.iloc[start : start + 6]
        fig, axes = plt.subplots(len(subset), 2, figsize=(10, 2.3 * len(subset)))
        if len(subset) == 1:
            axes = np.array([axes])
        for axrow, (_, row) in zip(axes, subset.iterrows()):
            for ax, col in zip(axrow, ["normal_plot_path", "candlestick_plot_path"]):
                ax.axis("off")
                path = ROOT / row[col] if row[col] else None
                if path and path.exists():
                    img = plt.imread(path)
                    ax.imshow(img)
            axrow[0].set_title(f"#{row['rank']} {row['method']} {row['asset']} {row['frequency']} {row['regime_label']}", fontsize=8)
        savefig(FIG / f"top_motif_gallery_page_{page:02d}.png")


def resample(values: np.ndarray, n: int = 64) -> np.ndarray:
    if len(values) < 2:
        return np.full(n, np.nan)
    x_old = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, values)


def build_clustering(occ: pd.DataFrame) -> None:
    seqs = []
    labels = []
    for _, row in occ.head(40).iterrows():
        feat = load_feature(str(row["asset"]), str(row["frequency"]))
        if feat.empty:
            continue
        start = pd.Timestamp(row["start_timestamp"])
        end = pd.Timestamp(row["end_timestamp"])
        win = feat[(feat["timestamp"] >= start) & (feat["timestamp"] <= end)]
        vals = pd.to_numeric(win["close"], errors="coerce").dropna().values
        if len(vals) < 8:
            continue
        z = (vals - vals.mean()) / (vals.std() if vals.std() else 1)
        seqs.append(resample(z, 64))
        labels.append(f"{row['method']} {row['asset']} {row['frequency']} {row['regime_label']}")
    if len(seqs) < 6:
        write_csv(pd.DataFrame([{"reason": "Fewer than six usable motif subsequences were available."}]), TAB / "motif_clustering_skipped_reason.csv")
        return
    try:
        from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
    except Exception as exc:
        write_csv(pd.DataFrame([{"reason": f"scipy unavailable: {exc}"}]), TAB / "motif_clustering_skipped_reason.csv")
        return
    X = np.vstack(seqs)
    Z = linkage(X, method="ward")
    clusters = fcluster(Z, t=min(4, len(seqs) - 1), criterion="maxclust")
    write_csv(pd.DataFrame({"label": labels, "cluster": clusters}), TAB / "motif_shape_cluster_assignments.csv")
    plt.figure(figsize=(10, 4))
    dendrogram(Z, labels=[f"M{i+1}" for i in range(len(labels))], leaf_rotation=90)
    plt.title("Motif shape dendrogram")
    savefig(FIG / "motif_shape_dendrogram.png")
    plt.figure(figsize=(8, 4))
    for cl in sorted(set(clusters)):
        avg = X[clusters == cl].mean(axis=0)
        plt.plot(avg, label=f"cluster {cl}")
    plt.title("Cluster-average motif shapes")
    plt.legend()
    plt.grid(True, alpha=0.2)
    savefig(FIG / "motif_cluster_average_shapes.png")
    plt.figure(figsize=(8, 4))
    for i in range(min(len(X), 12)):
        plt.plot(X[i], alpha=0.45)
    plt.title("Motif cluster examples")
    plt.grid(True, alpha=0.2)
    savefig(FIG / "motif_cluster_examples.png")


def write_shap_skip() -> None:
    candidates = list(ROOT.rglob("*model*")) + list(ROOT.rglob("*.pkl")) + list(ROOT.rglob("*.joblib"))
    supervised = [p for p in candidates if "supervised" in p.name.lower() or "classifier" in p.name.lower() or "regressor" in p.name.lower()]
    if not supervised:
        write_csv(
            pd.DataFrame(
                [
                    {
                        "reason": "SHAP is not central because the thesis pipeline is unsupervised motif discovery and regime detection, not supervised prediction. Adding SHAP without a supervised model would weaken methodological coherence."
                    }
                ]
            ),
            TAB / "shap_skipped_reason.csv",
        )


def write_latex_and_report() -> None:
    ensure_dirs()
    write_shap_skip()
    figs = sorted(FIG.glob("*.png"))
    tables = sorted(TAB.glob("*.csv"))
    summary = read_csv(TAB / "final_mp_vs_locomotif_summary.csv")
    vix_ctx = read_csv(TAB / "motif_occurrences_with_vix_context.csv")
    controlled_runtime = read_csv(CONTROL_TABLES / "locomotif_controlled_runtime.csv")
    timeout_note = ""
    if not controlled_runtime.empty and (controlled_runtime.get("success") == False).any():  # noqa: E712
        timeout_note = "The controlled LoCoMotif runtime CSV records timeouts and zero filtered motif sets for the controlled slice runs; no controlled LoCoMotif occurrences were fabricated."
    chapter = rf"""
\section{{Final Visual Evidence for Motif Interpretability}}

\subsection{{Candlestick Context for Discovered Motifs}}
Candlestick figures connect discovered motif intervals back to original OHLC price behaviour. Each shaded interval marks the motif window, while the surrounding context panel shows whether the pattern appears inside a local trend, consolidation, reversal, or high-range episode. Where OHLC columns are available, candlesticks are used; otherwise the workflow records a close-line fallback in the figure index.

\subsection{{VIX and Market-Stress Context}}
VIX is used as a broad external market-stress proxy, not as a crypto-specific volatility label. Crypto motifs are not trained on VIX. Motif timestamps are aligned to the nearest previous VIX daily close and summarized by full-sample VIX percentile regimes. {len(vix_ctx) if not vix_ctx.empty else 0} motif occurrences were written to the VIX context table.

\subsection{{Matrix Profile and LoCoMotif Visual Comparison}}
Matrix Profile and LoCoMotif output different motif objects. Matrix Profile returns fixed-length nearest-neighbour pairs with distances; LoCoMotif returns time-warped motif sets with multiple occurrences when the run succeeds. Counts are therefore not directly equivalent. {timeout_note}

\subsection{{Runtime Efficiency and Method Trade-Offs}}
The runtime and complexity figures compare empirical runtime evidence with qualitative scaling. Matrix Profile is scalable for fixed-length search using optimized algorithms; multivariate MP is more expensive because it evaluates multiple channels. LoCoMotif adds time-warping and motif-set recurrence but is sensitive to slice length and parameters such as lmin, lmax, rho, and the number of motif sets.
"""
    (LATEX / "final_visual_evidence_chapter5_snippet.tex").write_text(textwrap.dedent(chapter).strip() + "\n", encoding="utf-8")
    appendix = "\\section{Final Visual Evidence Appendix}\n\n"
    appendix += "This appendix collects the top motif gallery, candlestick context figures, VIX context table, clustering outputs when available, and the SHAP skip note when no supervised model is present.\n\n"
    for fig in figs:
        appendix += f"\\begin{{figure}}[ht]\n\\centering\n\\includegraphics[width=0.9\\linewidth]{{{rel(fig)}}}\n\\caption{{{caption_for(fig)}}}\n\\end{{figure}}\n\n"
    (LATEX / "final_visual_evidence_appendix_snippet.tex").write_text(appendix, encoding="utf-8")
    captions = "\n".join([f"\\newcommand{{\\caption{slug(fig.stem).title().replace('_', '')}}}{{{caption_for(fig)}}}" for fig in figs])
    (LATEX / "final_visual_evidence_captions.tex").write_text(captions + "\n", encoding="utf-8")
    report = [
        "# Final Visual Evidence Report",
        "",
        "VIX is used as a broad external market-stress proxy, not as a crypto-specific volatility label. No generated text treats VIX as a training label or causal driver of crypto motifs.",
        "",
        f"Figures generated: {len(figs)}",
        f"Tables generated: {len(tables)}",
        "",
        "## Controlled LoCoMotif status",
        timeout_note or "Controlled LoCoMotif machine-readable status did not record timeouts.",
        "",
        "## Key figures",
    ]
    report += [f"- `{rel(fig)}`" for fig in figs[:60]]
    report += ["", "## Key tables"]
    report += [f"- `{rel(table)}`" for table in tables]
    report += [
        "",
        "## Methodological caveats",
        "- MP and LoCoMotif counts are not directly equivalent.",
        "- Event labels are approximate context windows, not causal explanations.",
        "- No trading predictability or profitability claim is made.",
    ]
    (OUT / "FINAL_VISUAL_EVIDENCE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def caption_for(path: Path) -> str:
    name = path.stem.replace("_", " ")
    if "vix" in path.stem:
        return f"{name}. VIX is external market-stress context only and is not used as a motif-discovery label."
    if "locomotif" in path.stem or "mp_vs" in path.stem:
        return f"{name}. The figure compares real available Matrix Profile and LoCoMotif outputs while preserving object-type differences."
    if "candlestick" in path.stem or "motif" in path.stem:
        return f"{name}. The shaded interval marks the discovered motif occurrence in original price context."
    return f"{name}."


def write_hpc_instructions() -> None:
    text = """
# Final Visual Evidence HPC Run Instructions

## Local
git add notebooks scripts slurm reports/final_visual_evidence
git commit -m "Add final visual evidence HPC workflow"
git push

## HPC
cd ~/Final_master_thesis
git pull
source .thesis-env/bin/activate
python -m py_compile scripts/run_final_motif_candlestick_event_context.py
python -m py_compile scripts/run_final_mp_locomotif_visual_comparison.py
python -m py_compile scripts/run_final_top_motif_gallery_and_clustering.py
python -m py_compile scripts/run_final_visual_evidence_summary.py

## Submit
sbatch slurm/run_final_mp_locomotif_visual_comparison.slurm
sbatch slurm/run_final_motif_candlestick_event_context.slurm
sbatch slurm/run_final_top_motif_gallery_and_clustering.slurm
sbatch slurm/run_final_visual_evidence_summary.slurm

## Check
squeue -u $USER
ls -lt logs | head
tail -f logs/final_visual_*.out

## After completion
python scripts/run_final_visual_evidence_summary.py
cat reports/final_visual_evidence/FINAL_VISUAL_EVIDENCE_REPORT.md

## Commit HPC outputs
git add reports/final_visual_evidence logs/final_visual_*.out logs/final_visual_*.err
git commit -m "Add final visual evidence outputs"
git push

## Local pull
git pull
"""
    (OUT / "HPC_FINAL_VISUAL_RUN_INSTRUCTIONS.md").write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def run_all() -> None:
    ensure_dirs()
    build_comparison()
    build_candlestick_and_vix()
    build_gallery_and_clustering()
    write_latex_and_report()
    write_hpc_instructions()
    print("FINAL VISUAL EVIDENCE WORKFLOW READY")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["inventory", "comparison", "candlestick", "gallery", "summary", "all"])
    args = parser.parse_args(list(argv) if argv is not None else None)
    ensure_dirs()
    if args.task == "inventory":
        discover_inputs()
    elif args.task == "comparison":
        build_comparison()
    elif args.task == "candlestick":
        build_candlestick_and_vix()
    elif args.task == "gallery":
        build_gallery_and_clustering()
    elif args.task == "summary":
        discover_inputs()
        write_latex_and_report()
        write_hpc_instructions()
    elif args.task == "all":
        run_all()


if __name__ == "__main__":
    main()
