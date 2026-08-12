from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[3]
HPC_ROOT = ROOT / "HPC workflow" / "HPC_Regime_and_motif_discovery"
HPC_SRC = HPC_ROOT / "src"
CONFIG_PATH = HPC_ROOT / "run_configs" / "hpc_mp_crypto_15m_1h.yaml"
DATA_PATH = ROOT / "final_dataset" / "features" / "crypto" / "BTCUSDT_15m_features_2020_2025.parquet"
STUDY_TABLES = HPC_ROOT / "reports" / "study_notebooks" / "tables"
OUT = ROOT / "reports" / "final_story"
UNI_FIG = OUT / "univariate" / "figures"
UNI_TAB = OUT / "univariate" / "tables"
MULTI_FIG = OUT / "multivariate" / "figures"
MULTI_TAB = OUT / "multivariate" / "tables"

ASSET = "BTCUSDT"
FREQUENCY = "15m"
PERIOD_START = pd.Timestamp("2020-05-01 00:00:00", tz="UTC")
PERIOD_END = pd.Timestamp("2020-08-20 23:45:00", tz="UTC")
UNIVARIATE_FEATURE = "rolling_volatility_60"
WINDOW = 32
WINDOWS = [32, 64, 96]
TOP_K = 5
EXCLUSION_ZONE_FACTOR = 0.5
MULTIVARIATE_FEATURES = ["log_return", "rolling_volatility_60", "hl_range", "volume_zscore"]
PALETTE = {
    "blue": "#2c7fb8",
    "orange": "#f28e2b",
    "green": "#59a14f",
    "red": "#d95f02",
    "gray": "#5f6b6d",
    "light_blue": "#d9ecf2",
    "light_orange": "#fee6ce",
}


def ensure_project_imports() -> None:
    if str(HPC_SRC) not in sys.path:
        sys.path.insert(0, str(HPC_SRC))


def ensure_dirs() -> None:
    for path in [UNI_FIG, UNI_TAB, MULTI_FIG, MULTI_TAB, OUT / "executed_notebooks", OUT / "scripts"]:
        path.mkdir(parents=True, exist_ok=True)


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, path: Path, save_pdf: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    ensure_project_imports()
    from data_loading import load_feature_file
    from feature_selection import ensure_core_features

    df = load_feature_file(DATA_PATH)
    df = ensure_core_features(df, rolling_window=int(load_config()["quantile"]["default_rolling_window"]))
    return df


def selected_period(df: pd.DataFrame) -> pd.DataFrame:
    out = df[(df["timestamp"] >= PERIOD_START) & (df["timestamp"] <= PERIOD_END)].copy()
    out = out.reset_index(drop=True)
    if out.empty:
        raise ValueError("Selected period is empty.")
    return out


def clock_time(window: int) -> str:
    minutes = 15 * int(window)
    if minutes % 60 == 0:
        return f"{minutes // 60} hours"
    return f"{minutes} minutes"


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std == 0:
        std = 1.0
    return (arr - np.nanmean(arr)) / std


def timestamp_at(df: pd.DataFrame, index: int) -> pd.Timestamp:
    return pd.Timestamp(df["timestamp"].iloc[int(index)])


def window_frame(df: pd.DataFrame, start_index: int, window_length: int) -> pd.DataFrame:
    end = int(start_index) + int(window_length)
    if start_index < 0 or end > len(df):
        raise IndexError(f"Motif window [{start_index}, {end}) exceeds data length {len(df)}.")
    return df.iloc[int(start_index) : end].copy()


def run_univariate(df: pd.DataFrame, window_length: int = WINDOW, top_k: int = TOP_K) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_project_imports()
    from matrix_profile_utils import run_univariate_matrix_profile

    series = pd.to_numeric(df[UNIVARIATE_FEATURE], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(float)
    context = {
        "asset": ASSET,
        "frequency": FREQUENCY,
        "mode": "agnostic",
        "regime_method": "none",
        "regime_label": "all",
        "feature_set": UNIVARIATE_FEATURE,
        "profile_type": "univariate",
        "segment_id": "final_story_selected_period",
    }
    motifs, profile = run_univariate_matrix_profile(
        series,
        df["timestamp"],
        window_length=window_length,
        top_k=top_k,
        context=context,
        use_gpu=False,
        exclusion_zone_factor=EXCLUSION_ZONE_FACTOR,
    )
    profile["timestamp"] = df["timestamp"].iloc[: len(profile)].to_numpy()
    return motifs, profile


def build_univariate_context() -> dict:
    ensure_dirs()
    configure_plots()
    df_full = load_data()
    df = selected_period(df_full)
    motifs, profile = run_univariate(df)
    selected = motifs.iloc[0].to_dict()
    metadata = pd.DataFrame(
        [
            {
                "asset": ASSET,
                "frequency": FREQUENCY,
                "data_source": str(DATA_PATH),
                "selection_rule": "Bounded period selected because the saved study top-motif table contains a verified BTCUSDT 15m rolling_volatility_60 w32 pair inside this interval (2020-05-09 and 2020-07-21), giving a reproducible non-trivial recurrence without rerunning full history.",
                "period_start": PERIOD_START.isoformat(),
                "period_end": PERIOD_END.isoformat(),
                "n_observations": len(df),
                "feature": UNIVARIATE_FEATURE,
                "window_length": WINDOW,
                "window_clock_time": clock_time(WINDOW),
                "exclusion_zone_factor": EXCLUSION_ZONE_FACTOR,
                "exclusion_zone": int(WINDOW * EXCLUSION_ZONE_FACTOR),
                "top_k": TOP_K,
                "motif_start_1": int(selected["motif_start_1"]),
                "motif_start_2": int(selected["motif_start_2"]),
                "motif_timestamp_1": selected["motif_timestamp_1"],
                "motif_timestamp_2": selected["motif_timestamp_2"],
                "motif_distance": selected["motif_distance"],
                "runtime_seconds": selected["runtime_seconds"],
            }
        ]
    )
    write_csv(metadata, UNI_TAB / "01_univariate_selected_motif_metadata.csv")
    profile.to_csv(UNI_TAB / "02_univariate_matrix_profile_values.csv", index=False)
    motifs.to_csv(UNI_TAB / "06_univariate_top5_motifs.csv", index=False)
    write_csv(
        pd.DataFrame(
            [
                {
                    "metric": "univariate_mp_runtime_seconds",
                    "value": float(selected["runtime_seconds"]),
                    "window_length": WINDOW,
                    "n_observations": len(df),
                    "feature": UNIVARIATE_FEATURE,
                }
            ]
        ),
        UNI_TAB / "univariate_runtime_metadata.csv",
    )
    return {"df_full": df_full, "df": df, "motifs": motifs, "profile": profile, "selected": selected, "metadata": metadata}


def shade_motif(ax, df: pd.DataFrame, motif: dict, side: int, label: str, color: str) -> None:
    start = int(motif[f"motif_start_{side}"])
    end = start + int(motif["window_length"]) - 1
    ax.axvspan(timestamp_at(df, start), timestamp_at(df, end), color=color, alpha=0.22, label=label)


def univariate_overview(ctx: dict) -> Path:
    df = ctx["df"]
    selected = ctx["selected"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(df["timestamp"], df["close"], color=PALETTE["blue"], linewidth=0.85)
    shade_motif(ax, df, selected, 1, "Motif occurrence A", PALETTE["orange"])
    shade_motif(ax, df, selected, 2, "Motif occurrence B", PALETTE["green"])
    ax.set_title("BTCUSDT 15m price context with selected Matrix Profile motif windows")
    ax.set_ylabel("close price (USDT)")
    ax.set_xlabel("timestamp (UTC)")
    ax.legend(loc="best")
    return save_figure(fig, UNI_FIG / "01_univariate_full_series_overview.png")


def univariate_series_profile(ctx: dict) -> Path:
    df = ctx["df"]
    profile = ctx["profile"]
    selected = ctx["selected"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(df["timestamp"], df[UNIVARIATE_FEATURE], color=PALETTE["blue"], linewidth=0.8)
    axes[0].set_ylabel(UNIVARIATE_FEATURE)
    axes[0].set_title("Input feature and Matrix Profile over the selected period")
    axes[1].plot(profile["timestamp"], profile["matrix_profile"], color=PALETTE["gray"], linewidth=0.8)
    axes[1].set_ylabel("MP distance")
    axes[1].set_xlabel("timestamp (UTC)")
    for ax in axes:
        shade_motif(ax, df, selected, 1, "Motif A" if ax is axes[0] else None, PALETTE["orange"])
        shade_motif(ax, df, selected, 2, "Motif B" if ax is axes[0] else None, PALETTE["green"])
    axes[0].legend(loc="best")
    return save_figure(fig, UNI_FIG / "02_univariate_series_and_matrix_profile.png")


def univariate_overlay(ctx: dict) -> Path:
    df = ctx["df"]
    motif = ctx["selected"]
    i = int(motif["motif_start_1"])
    j = int(motif["motif_start_2"])
    m = int(motif["window_length"])
    a = window_frame(df, i, m)
    b = window_frame(df, j, m)
    values = pd.DataFrame(
        {
            "step": np.arange(m),
            "timestamp_a": a["timestamp"].astype(str).to_numpy(),
            "timestamp_b": b["timestamp"].astype(str).to_numpy(),
            "feature": UNIVARIATE_FEATURE,
            "occurrence_a_raw": a[UNIVARIATE_FEATURE].to_numpy(float),
            "occurrence_b_raw": b[UNIVARIATE_FEATURE].to_numpy(float),
            "occurrence_a_z": zscore(a[UNIVARIATE_FEATURE].to_numpy(float)),
            "occurrence_b_z": zscore(b[UNIVARIATE_FEATURE].to_numpy(float)),
        }
    )
    write_csv(values, UNI_TAB / "03_univariate_top_motif_overlay_values.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4))
    ax.plot(values["step"], values["occurrence_a_z"], label=f"A: {motif['motif_timestamp_1']}", color=PALETTE["orange"], linewidth=1.8)
    ax.plot(values["step"], values["occurrence_b_z"], label=f"B: {motif['motif_timestamp_2']}", color=PALETTE["green"], linewidth=1.8)
    ax.set_title(f"Top univariate motif overlay, rank 1, distance {float(motif['motif_distance']):.4f}")
    ax.set_xlabel("15-minute step within window")
    ax.set_ylabel("z-normalized feature value")
    ax.legend(loc="best")
    return save_figure(fig, UNI_FIG / "03_univariate_top_motif_normalized_overlay.png")


def univariate_original_scale(ctx: dict) -> Path:
    df = ctx["df"]
    motif = ctx["selected"]
    i = int(motif["motif_start_1"])
    j = int(motif["motif_start_2"])
    m = int(motif["window_length"])
    a = window_frame(df, i, m)
    b = window_frame(df, j, m)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    axes[0].plot(a["timestamp"], a[UNIVARIATE_FEATURE], color=PALETTE["orange"], linewidth=1.6)
    axes[0].set_title(f"Occurrence A\n{motif['motif_timestamp_1']}")
    axes[1].plot(b["timestamp"], b[UNIVARIATE_FEATURE], color=PALETTE["green"], linewidth=1.6)
    axes[1].set_title(f"Occurrence B\n{motif['motif_timestamp_2']}")
    for ax in axes:
        ax.set_ylabel(UNIVARIATE_FEATURE)
        ax.tick_params(axis="x", rotation=30)
    return save_figure(fig, UNI_FIG / "04_univariate_top_motif_original_scale.png")


def draw_candles(ax, df: pd.DataFrame) -> None:
    x = mdates.date2num(df["timestamp"].dt.to_pydatetime())
    width = 0.62 * np.median(np.diff(x)) if len(x) > 1 else 0.006
    for xi, row in zip(x, df.itertuples(index=False)):
        open_ = float(getattr(row, "open"))
        high = float(getattr(row, "high"))
        low = float(getattr(row, "low"))
        close = float(getattr(row, "close"))
        color = PALETTE["green"] if close >= open_ else PALETTE["red"]
        ax.plot([xi, xi], [low, high], color=color, linewidth=0.8)
        body_low = min(open_, close)
        body_height = max(abs(close - open_), 1e-9)
        ax.add_patch(Rectangle((xi - width / 2, body_low), width, body_height, facecolor=color, edgecolor=color, alpha=0.82))
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.grid(True, axis="y", alpha=0.25)


def univariate_candlesticks(ctx: dict) -> Path:
    df = ctx["df"]
    motif = ctx["selected"]
    m = int(motif["window_length"])
    a = window_frame(df, int(motif["motif_start_1"]), m)
    b = window_frame(df, int(motif["motif_start_2"]), m)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    draw_candles(axes[0], a)
    draw_candles(axes[1], b)
    axes[0].set_title(f"Occurrence A\n{motif['motif_timestamp_1']}")
    axes[1].set_title(f"Occurrence B\n{motif['motif_timestamp_2']}")
    for ax in axes:
        ax.set_ylabel("price (USDT)")
    return save_figure(fig, UNI_FIG / "05_univariate_top_motif_candlesticks.png")


def univariate_gallery(ctx: dict) -> Path:
    df = ctx["df"]
    motifs = ctx["motifs"].head(5)
    fig, axes = plt.subplots(len(motifs), 1, figsize=(9, 1.85 * len(motifs)), sharex=True)
    if len(motifs) == 1:
        axes = [axes]
    for ax, (_, motif) in zip(axes, motifs.iterrows()):
        m = int(motif["window_length"])
        a = window_frame(df, int(motif["motif_start_1"]), m)[UNIVARIATE_FEATURE].to_numpy(float)
        b = window_frame(df, int(motif["motif_start_2"]), m)[UNIVARIATE_FEATURE].to_numpy(float)
        ax.plot(zscore(a), color=PALETTE["orange"], linewidth=1.4, label="A")
        ax.plot(zscore(b), color=PALETTE["green"], linewidth=1.4, label="B")
        ax.set_ylabel(f"rank {int(motif['motif_rank'])}")
        ax.set_title(
            f"d={float(motif['motif_distance']):.4f} | {motif['motif_timestamp_1']} vs {motif['motif_timestamp_2']}",
            fontsize=8.5,
        )
    axes[-1].set_xlabel("15-minute step within window")
    axes[0].legend(loc="upper right")
    return save_figure(fig, UNI_FIG / "06_univariate_top5_motif_gallery.png")


def univariate_similarity(ctx: dict) -> Path:
    df = ctx["df"]
    motifs = ctx["motifs"].head(5)
    seqs = []
    labels = []
    for _, motif in motifs.iterrows():
        seq = window_frame(df, int(motif["motif_start_1"]), int(motif["window_length"]))[UNIVARIATE_FEATURE].to_numpy(float)
        seqs.append(zscore(seq))
        labels.append(f"M{int(motif['motif_rank'])}")
    matrix = np.zeros((len(seqs), len(seqs)))
    for i, a in enumerate(seqs):
        for j, b in enumerate(seqs):
            matrix[i, j] = float(np.linalg.norm(a - b))
    sim = pd.DataFrame(matrix, index=labels, columns=labels)
    sim.to_csv(UNI_TAB / "07_univariate_top5_similarity_matrix.csv")
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Auxiliary distance matrix among Top-5 motif representatives")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="z-normalized Euclidean distance")
    return save_figure(fig, UNI_FIG / "07_univariate_top5_similarity_matrix.png")


def univariate_window_sensitivity(ctx: dict) -> Path:
    df = ctx["df"]
    rows = []
    fig, axes = plt.subplots(len(WINDOWS), 1, figsize=(9, 2.0 * len(WINDOWS)), sharex=False)
    for ax, window in zip(axes, WINDOWS):
        motifs, _profile = run_univariate(df, window_length=window, top_k=1)
        motif = motifs.iloc[0]
        a = window_frame(df, int(motif["motif_start_1"]), window)[UNIVARIATE_FEATURE].to_numpy(float)
        b = window_frame(df, int(motif["motif_start_2"]), window)[UNIVARIATE_FEATURE].to_numpy(float)
        ax.plot(zscore(a), color=PALETTE["orange"], linewidth=1.4)
        ax.plot(zscore(b), color=PALETTE["green"], linewidth=1.4)
        ax.set_title(f"m={window} ({clock_time(window)}), d={float(motif['motif_distance']):.4f}", fontsize=9)
        rows.append(
            {
                "window_length": window,
                "clock_time": clock_time(window),
                "motif_start_1": int(motif["motif_start_1"]),
                "motif_start_2": int(motif["motif_start_2"]),
                "motif_timestamp_1": motif["motif_timestamp_1"],
                "motif_timestamp_2": motif["motif_timestamp_2"],
                "motif_distance": float(motif["motif_distance"]),
                "runtime_seconds": float(motif["runtime_seconds"]),
            }
        )
    axes[-1].set_xlabel("15-minute step within motif window")
    write_csv(pd.DataFrame(rows), UNI_TAB / "08_univariate_window_length_sensitivity.csv")
    return save_figure(fig, UNI_FIG / "08_univariate_window_length_sensitivity.png")


def run_all_univariate() -> dict:
    ctx = build_univariate_context()
    paths = [
        univariate_overview(ctx),
        univariate_series_profile(ctx),
        univariate_overlay(ctx),
        univariate_original_scale(ctx),
        univariate_candlesticks(ctx),
        univariate_gallery(ctx),
        univariate_similarity(ctx),
        univariate_window_sensitivity(ctx),
    ]
    summary = ctx["metadata"].copy()
    summary["generated_figures"] = ";".join(str(p) for p in paths)
    write_csv(summary, UNI_TAB / "univariate_results_summary.csv")
    return {"context": ctx, "figures": paths, "summary": summary}


def multivariate_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    ensure_project_imports()
    from feature_selection import clean_feature_frame, scale_feature_frame

    available = [feature for feature in MULTIVARIATE_FEATURES if feature in df.columns]
    cleaned, kept, diagnostics = clean_feature_frame(df, available, max_nan_fraction=0.40, min_non_constant_values=3)
    scaled, stats = scale_feature_frame(cleaned, scaler="robust")
    diagnostics = diagnostics.merge(stats, on="feature", how="left")
    diagnostics["retained"] = diagnostics["kept"]
    return cleaned, scaled, kept, diagnostics


def run_multivariate(df: pd.DataFrame, scaled: pd.DataFrame, features: list[str], window_length: int = WINDOW, top_k: int = TOP_K):
    ensure_project_imports()
    from matrix_profile_utils import run_multivariate_matrix_profile

    context = {
        "asset": ASSET,
        "frequency": FREQUENCY,
        "mode": "agnostic",
        "regime_method": "none",
        "regime_label": "all",
        "profile_type": "multivariate",
        "segment_id": "final_story_selected_period",
    }
    motifs, profile = run_multivariate_matrix_profile(
        scaled[features].to_numpy(float),
        df["timestamp"],
        feature_columns=features,
        window_length=window_length,
        top_k=top_k,
        context=context,
        exclusion_zone_factor=EXCLUSION_ZONE_FACTOR,
    )
    profile["timestamp"] = df["timestamp"].iloc[: len(profile)].to_numpy()
    return motifs, profile


def build_multivariate_context() -> dict:
    ensure_dirs()
    configure_plots()
    df = selected_period(load_data())
    cleaned, scaled, features, diagnostics = multivariate_features(df)
    write_csv(diagnostics, MULTI_TAB / "01_multivariate_feature_selection.csv")
    scaled_out = scaled[features].copy()
    scaled_out.insert(0, "timestamp", df["timestamp"].astype(str).to_numpy())
    write_csv(scaled_out, MULTI_TAB / "02_multivariate_scaled_feature_matrix.csv")
    motifs, profile = run_multivariate(df, scaled, features)
    selected = motifs.iloc[0].to_dict()
    profile.to_csv(MULTI_TAB / "03_multivariate_matrix_profile_values.csv", index=False)
    write_csv(motifs, MULTI_TAB / "07_multivariate_top5_motifs.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "asset": ASSET,
                    "frequency": FREQUENCY,
                    "data_source": str(DATA_PATH),
                    "period_start": PERIOD_START.isoformat(),
                    "period_end": PERIOD_END.isoformat(),
                    "n_observations": len(df),
                    "selected_features": ",".join(features),
                    "scaling": "robust",
                    "window_length": WINDOW,
                    "window_clock_time": clock_time(WINDOW),
                    "exclusion_zone_factor": EXCLUSION_ZONE_FACTOR,
                    "mstump_dimension_row": selected["mstump_dimension_row"],
                    "motif_start_1": int(selected["motif_start_1"]),
                    "motif_start_2": int(selected["motif_start_2"]),
                    "motif_timestamp_1": selected["motif_timestamp_1"],
                    "motif_timestamp_2": selected["motif_timestamp_2"],
                    "motif_distance": selected["motif_distance"],
                    "runtime_seconds": selected["runtime_seconds"],
                }
            ]
        ),
        MULTI_TAB / "03_multivariate_selected_motif_metadata.csv",
    )
    write_csv(
        pd.DataFrame(
            [{"metric": "multivariate_mstump_runtime_seconds", "value": float(selected["runtime_seconds"]), "window_length": WINDOW, "n_observations": len(df), "features": ",".join(features)}]
        ),
        MULTI_TAB / "multivariate_runtime_metadata.csv",
    )
    return {"df": df, "cleaned": cleaned, "scaled": scaled, "features": features, "diagnostics": diagnostics, "motifs": motifs, "profile": profile, "selected": selected}


def multivariate_feature_panel(ctx: dict) -> Path:
    df = ctx["df"]
    features = ctx["features"]
    fig, axes = plt.subplots(len(features), 1, figsize=(11, 1.8 * len(features)), sharex=True)
    for ax, feature in zip(axes, features):
        ax.plot(df["timestamp"], df[feature], linewidth=0.75, color=PALETTE["blue"])
        ax.set_ylabel(feature)
    axes[0].set_title("Selected multivariate feature channels before scaling")
    axes[-1].set_xlabel("timestamp (UTC)")
    return save_figure(fig, MULTI_FIG / "01_multivariate_feature_panel.png")


def multivariate_scaled_panel(ctx: dict) -> Path:
    df = ctx["df"]
    scaled = ctx["scaled"]
    features = ctx["features"]
    fig, axes = plt.subplots(len(features), 1, figsize=(11, 1.8 * len(features)), sharex=True)
    for ax, feature in zip(axes, features):
        ax.plot(df["timestamp"], scaled[feature], linewidth=0.75, color=PALETTE["gray"])
        ax.set_ylabel(feature)
    axes[0].set_title("Robust-scaled feature channels passed to MSTUMP")
    axes[-1].set_xlabel("timestamp (UTC)")
    return save_figure(fig, MULTI_FIG / "02_multivariate_scaled_feature_panel.png")


def multivariate_profile_context(ctx: dict) -> Path:
    df = ctx["df"]
    profile = ctx["profile"]
    motif = ctx["selected"]
    scaled = ctx["scaled"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.4, 1]})
    axes[0].plot(df["timestamp"], df["close"], color=PALETTE["blue"], linewidth=0.8)
    axes[0].set_ylabel("close")
    axes[0].set_title("Multivariate Matrix Profile context")
    composite = np.sqrt(np.nanmean(np.square(scaled[ctx["features"]].to_numpy(float)), axis=1))
    axes[1].plot(df["timestamp"], composite, color=PALETTE["gray"], linewidth=0.75)
    axes[1].set_ylabel("scaled RMS")
    axes[2].plot(profile["timestamp"], profile["matrix_profile"], color=PALETTE["gray"], linewidth=0.8)
    axes[2].set_ylabel("MP distance")
    axes[2].set_xlabel("timestamp (UTC)")
    for ax in axes:
        shade_motif(ax, df, motif, 1, "Motif A" if ax is axes[0] else None, PALETTE["orange"])
        shade_motif(ax, df, motif, 2, "Motif B" if ax is axes[0] else None, PALETTE["green"])
    axes[0].legend(loc="best")
    return save_figure(fig, MULTI_FIG / "03_multivariate_matrix_profile_context.png")


def multivariate_top_across_features(ctx: dict) -> Path:
    df = ctx["df"]
    scaled = ctx["scaled"]
    motif = ctx["selected"]
    features = ctx["features"]
    m = int(motif["window_length"])
    i = int(motif["motif_start_1"])
    j = int(motif["motif_start_2"])
    rows = []
    fig, axes = plt.subplots(len(features), 1, figsize=(8.5, 1.95 * len(features)), sharex=True)
    for ax, feature in zip(axes, features):
        a = scaled[feature].iloc[i : i + m].to_numpy(float)
        b = scaled[feature].iloc[j : j + m].to_numpy(float)
        ax.plot(a, color=PALETTE["orange"], linewidth=1.5, label="A")
        ax.plot(b, color=PALETTE["green"], linewidth=1.5, label="B")
        ax.set_ylabel(feature)
        for step in range(m):
            rows.append(
                {
                    "feature": feature,
                    "step": step,
                    "timestamp_a": str(df["timestamp"].iloc[i + step]),
                    "timestamp_b": str(df["timestamp"].iloc[j + step]),
                    "occurrence_a_scaled": a[step],
                    "occurrence_b_scaled": b[step],
                }
            )
    axes[0].set_title(f"Strongest multivariate motif across robust-scaled channels, d={float(motif['motif_distance']):.4f}")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("15-minute step within window")
    write_csv(pd.DataFrame(rows), MULTI_TAB / "04_multivariate_top_motif_feature_values.csv")
    return save_figure(fig, MULTI_FIG / "04_multivariate_top_motif_across_features.png")


def univariate_vs_multivariate(ctx: dict) -> Path:
    df = ctx["df"]
    uni_motifs, _ = run_univariate(df, top_k=1)
    uni = uni_motifs.iloc[0]
    multi = ctx["selected"]
    same_pair = {int(uni["motif_start_1"]), int(uni["motif_start_2"])} == {int(multi["motif_start_1"]), int(multi["motif_start_2"])}
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, motif, title, feature in [
        (axes[0], uni, "Univariate top pair", UNIVARIATE_FEATURE),
        (axes[1], multi, "Multivariate top pair", ctx["features"][0]),
    ]:
        m = int(motif["window_length"])
        i = int(motif["motif_start_1"])
        j = int(motif["motif_start_2"])
        if title.startswith("Univariate"):
            a = zscore(df[feature].iloc[i : i + m].to_numpy(float))
            b = zscore(df[feature].iloc[j : j + m].to_numpy(float))
            feat_label = feature
            method = "univariate"
        else:
            a = ctx["scaled"][feature].iloc[i : i + m].to_numpy(float)
            b = ctx["scaled"][feature].iloc[j : j + m].to_numpy(float)
            feat_label = ",".join(ctx["features"])
            method = "multivariate"
        ax.plot(a, color=PALETTE["orange"], linewidth=1.5)
        ax.plot(b, color=PALETTE["green"], linewidth=1.5)
        ax.set_title(f"{title}\n{motif['motif_timestamp_1']} vs {motif['motif_timestamp_2']}", fontsize=9)
        ax.set_xlabel("step")
        rows.append(
            {
                "method": method,
                "profile_type": method,
                "feature_set": feat_label,
                "window_length": m,
                "motif_start_1": i,
                "motif_start_2": j,
                "motif_timestamp_1": motif["motif_timestamp_1"],
                "motif_timestamp_2": motif["motif_timestamp_2"],
                "motif_distance": float(motif["motif_distance"]),
                "runtime_seconds": float(motif["runtime_seconds"]),
                "pair_matches_other_method_exactly": same_pair,
                "temporal_separation": str(abs(pd.Timestamp(motif["motif_timestamp_2"]) - pd.Timestamp(motif["motif_timestamp_1"]))),
            }
        )
    write_csv(pd.DataFrame(rows), MULTI_TAB / "05_univariate_vs_multivariate_top_motif_comparison.csv")
    return save_figure(fig, MULTI_FIG / "05_univariate_vs_multivariate_top_motif.png")


def multivariate_information_case(ctx: dict) -> Path:
    df = ctx["df"]
    uni_motifs, _ = run_univariate(df, top_k=20)
    features = ctx["features"]
    scaled = ctx["scaled"]
    rows = []
    for _, motif in uni_motifs.iterrows():
        m = int(motif["window_length"])
        i = int(motif["motif_start_1"])
        j = int(motif["motif_start_2"])
        per_feature = {}
        for feature in features:
            a = scaled[feature].iloc[i : i + m].to_numpy(float)
            b = scaled[feature].iloc[j : j + m].to_numpy(float)
            per_feature[feature] = float(np.linalg.norm(a - b) / math.sqrt(m))
        non_uni = [v for k, v in per_feature.items() if k != UNIVARIATE_FEATURE]
        rows.append({**motif.to_dict(), **{f"scaled_rms_distance_{k}": v for k, v in per_feature.items()}, "non_univariate_max_distance": max(non_uni) if non_uni else np.nan})
    cases = pd.DataFrame(rows).sort_values(["motif_distance", "non_univariate_max_distance"], ascending=[True, False])
    case = cases.iloc[0]
    write_csv(cases, MULTI_TAB / "06_when_multivariate_information_changes_similarity_case_candidates.csv")
    m = int(case["window_length"])
    i = int(case["motif_start_1"])
    j = int(case["motif_start_2"])
    fig, axes = plt.subplots(len(features), 1, figsize=(8.5, 1.95 * len(features)), sharex=True)
    for ax, feature in zip(axes, features):
        ax.plot(scaled[feature].iloc[i : i + m].to_numpy(float), color=PALETTE["orange"], linewidth=1.4, label="A")
        ax.plot(scaled[feature].iloc[j : j + m].to_numpy(float), color=PALETTE["green"], linewidth=1.4, label="B")
        ax.set_ylabel(feature)
    axes[0].set_title("Univariate-similar pair inspected across additional scaled dimensions")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("15-minute step within window")
    return save_figure(fig, MULTI_FIG / "06_when_multivariate_information_changes_similarity.png")


def multivariate_gallery(ctx: dict) -> Path:
    df = ctx["df"]
    scaled = ctx["scaled"]
    motifs = ctx["motifs"].head(5)
    features = ctx["features"]
    fig, axes = plt.subplots(len(motifs), 1, figsize=(9, 2.0 * len(motifs)), sharex=True)
    for ax, (_, motif) in zip(axes, motifs.iterrows()):
        m = int(motif["window_length"])
        i = int(motif["motif_start_1"])
        j = int(motif["motif_start_2"])
        a = scaled[features].iloc[i : i + m].to_numpy(float).mean(axis=1)
        b = scaled[features].iloc[j : j + m].to_numpy(float).mean(axis=1)
        ax.plot(a, color=PALETTE["orange"], linewidth=1.4, label="A mean channel")
        ax.plot(b, color=PALETTE["green"], linewidth=1.4, label="B mean channel")
        ax.set_ylabel(f"rank {int(motif['motif_rank'])}")
        ax.set_title(f"d={float(motif['motif_distance']):.4f} | {motif['motif_timestamp_1']} vs {motif['motif_timestamp_2']}", fontsize=8.5)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("15-minute step within window")
    return save_figure(fig, MULTI_FIG / "07_multivariate_top5_motif_gallery.png")


def multivariate_similarity(ctx: dict) -> Path:
    scaled = ctx["scaled"]
    motifs = ctx["motifs"].head(5)
    features = ctx["features"]
    seqs = []
    labels = []
    for _, motif in motifs.iterrows():
        seq = scaled[features].iloc[int(motif["motif_start_1"]) : int(motif["motif_start_1"]) + int(motif["window_length"])].to_numpy(float)
        seqs.append(seq)
        labels.append(f"M{int(motif['motif_rank'])}")
    matrix = np.zeros((len(seqs), len(seqs)))
    for i, a in enumerate(seqs):
        for j, b in enumerate(seqs):
            per_feature = [np.linalg.norm(zscore(a[:, k]) - zscore(b[:, k])) for k in range(a.shape[1])]
            matrix[i, j] = float(np.sqrt(np.mean(np.square(per_feature))))
    sim = pd.DataFrame(matrix, index=labels, columns=labels)
    sim.to_csv(MULTI_TAB / "08_multivariate_top5_similarity_matrix.csv")
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Auxiliary multivariate distance matrix among motif representatives")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="RMS per-feature z-normalized distance")
    return save_figure(fig, MULTI_FIG / "08_multivariate_top5_similarity_matrix.png")


def feature_set_comparison() -> Path:
    path = STUDY_TABLES / "study_mp_evaluation_by_feature_set.csv"
    df = pd.read_csv(path)
    subset = df[["feature_set", "runtime_seconds_mean", "runtime_seconds_median", "time_split_stability_mean", "recurrence_count_mean"]].copy()
    subset["source_table"] = str(path)
    write_csv(subset, MULTI_TAB / "09_feature_set_comparison.csv")
    labels = ["full multivariate" if "," in x else x for x in subset["feature_set"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(labels, subset["runtime_seconds_median"], color=PALETTE["blue"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median runtime seconds (log)")
    axes[0].set_title("Historical feature-set runtime")
    axes[1].bar(labels, subset["time_split_stability_mean"], color=PALETTE["green"])
    axes[1].set_ylabel("mean time-split stability")
    axes[1].set_title("Historical recurrence stability")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    return save_figure(fig, MULTI_FIG / "09_feature_set_runtime_and_stability_comparison.png")


def dimensionality_profile(ctx: dict) -> Path:
    stumpy = importlib.import_module("stumpy")
    matrix = ctx["scaled"][ctx["features"]].to_numpy(float).T
    t0 = time.perf_counter()
    profile_matrix, index_matrix = stumpy.mstump(matrix, WINDOW)
    runtime = time.perf_counter() - t0
    rows = []
    for row in range(profile_matrix.shape[0]):
        vals = np.asarray(profile_matrix[row], dtype=float)
        idx = int(np.nanargmin(vals))
        rows.append(
            {
                "mstump_row": row,
                "subspace_cardinality": row + 1,
                "min_profile_value": float(vals[idx]),
                "motif_start_1": idx,
                "motif_start_2": int(index_matrix[row, idx]),
                "motif_timestamp_1": str(ctx["df"]["timestamp"].iloc[idx]),
                "motif_timestamp_2": str(ctx["df"]["timestamp"].iloc[int(index_matrix[row, idx])]),
                "runtime_seconds_full_mstump_call": runtime,
                "semantic_note": "STUMPY mSTUMP row k stores the best profile for subsequences constrained to k+1 dimensions after subspace selection; the historical pipeline uses the last available row for the selected feature count.",
            }
        )
    dim = pd.DataFrame(rows)
    write_csv(dim, MULTI_TAB / "10_multivariate_dimensionality_profile.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dim["subspace_cardinality"], dim["min_profile_value"], marker="o", color=PALETTE["blue"])
    ax.set_xlabel("required agreeing dimensions")
    ax.set_ylabel("minimum MSTUMP profile value")
    ax.set_title("MSTUMP profile minimum by dimensionality row")
    return save_figure(fig, MULTI_FIG / "10_multivariate_dimensionality_profile.png")


def run_all_multivariate() -> dict:
    ctx = build_multivariate_context()
    paths = [
        multivariate_feature_panel(ctx),
        multivariate_scaled_panel(ctx),
        multivariate_profile_context(ctx),
        multivariate_top_across_features(ctx),
        univariate_vs_multivariate(ctx),
        multivariate_information_case(ctx),
        multivariate_gallery(ctx),
        multivariate_similarity(ctx),
        feature_set_comparison(),
        dimensionality_profile(ctx),
    ]
    summary = pd.read_csv(MULTI_TAB / "03_multivariate_selected_motif_metadata.csv")
    summary["generated_figures"] = ";".join(str(p) for p in paths)
    write_csv(summary, MULTI_TAB / "multivariate_results_summary.csv")
    return {"context": ctx, "figures": paths, "summary": summary}


def package_versions() -> pd.DataFrame:
    packages = ["python", "numpy", "pandas", "matplotlib", "stumpy", "yaml", "nbformat"]
    rows = [{"package": "python", "version": sys.version.split()[0]}]
    for package in packages[1:]:
        try:
            mod = importlib.import_module("pyyaml" if package == "yaml" else package)
        except Exception:
            try:
                mod = importlib.import_module(package)
            except Exception as exc:
                rows.append({"package": package, "version": f"unavailable: {exc}"})
                continue
        rows.append({"package": package, "version": str(getattr(mod, "__version__", "available"))})
    return pd.DataFrame(rows)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def write_run_record(kind: str) -> Path:
    table_dir = UNI_TAB if kind == "univariate" else MULTI_TAB
    record = pd.DataFrame(
        [
            {
                "kind": kind,
                "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_commit(),
                "data_source": str(DATA_PATH),
                "config_source": str(CONFIG_PATH),
                "asset": ASSET,
                "frequency": FREQUENCY,
                "period_start": PERIOD_START.isoformat(),
                "period_end": PERIOD_END.isoformat(),
                "window_length": WINDOW,
                "exclusion_zone_factor": EXCLUSION_ZONE_FACTOR,
            }
        ]
    )
    write_csv(record, table_dir / f"{kind}_execution_record.csv")
    write_csv(package_versions(), table_dir / f"{kind}_package_versions.csv")
    return table_dir / f"{kind}_execution_record.csv"


def write_manifest() -> Path:
    def rows_for(root: Path) -> list[Path]:
        return sorted(root.glob("*.csv"))

    uni_figs = sorted(UNI_FIG.glob("*.png"))
    multi_figs = sorted(MULTI_FIG.glob("*.png"))
    uni_meta = pd.read_csv(UNI_TAB / "01_univariate_selected_motif_metadata.csv") if (UNI_TAB / "01_univariate_selected_motif_metadata.csv").exists() else pd.DataFrame()
    multi_meta = pd.read_csv(MULTI_TAB / "03_multivariate_selected_motif_metadata.csv") if (MULTI_TAB / "03_multivariate_selected_motif_metadata.csv").exists() else pd.DataFrame()
    lines = [
        "# Final Story Manifest",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Git commit: {git_commit()}",
        "",
        "## Notebook 1",
        "",
        f"- exact notebook path: `{ROOT / 'notebooks' / 'final_story' / '01_univariate_matrix_profile_visual_story.ipynb'}`",
        f"- execution status: {'executed outputs present' if not uni_meta.empty else 'not executed'}",
        f"- data source: `{DATA_PATH}`",
        f"- asset: {ASSET}",
        f"- frequency: {FREQUENCY}",
        f"- date range: {PERIOD_START.isoformat()} to {PERIOD_END.isoformat()}",
        f"- feature: {UNIVARIATE_FEATURE}",
        f"- windows: {WINDOWS}",
        "",
        "### Notebook 1 Figures",
    ]
    for fig in uni_figs:
        section = fig.stem.split("_", 1)[0]
        lines.append(f"- `{fig}` | section {section} | demonstrates univariate motif evidence | source tables in `{UNI_TAB}` | recommended thesis usage: {'main text' if section in {'01','02','03','05'} else 'appendix'}")
    lines += ["", "### Notebook 1 CSVs"]
    lines += [f"- `{path}`" for path in rows_for(UNI_TAB)]
    lines += [
        "",
        "## Notebook 2",
        "",
        f"- exact notebook path: `{ROOT / 'notebooks' / 'final_story' / '02_multivariate_matrix_profile_visual_story.ipynb'}`",
        f"- execution status: {'executed outputs present' if not multi_meta.empty else 'not executed'}",
        f"- data source: `{DATA_PATH}`",
        f"- asset: {ASSET}",
        f"- frequency: {FREQUENCY}",
        f"- selected features: {', '.join(MULTIVARIATE_FEATURES)}",
        "- scaling: robust",
        f"- window: {WINDOW}",
        "",
        "### Notebook 2 Figures",
    ]
    for fig in multi_figs:
        section = fig.stem.split("_", 1)[0]
        recommendation = "main text" if section in {"01", "03", "04", "05", "06", "09"} else "appendix"
        lines.append(f"- `{fig}` | section {section} | demonstrates multivariate motif evidence | source tables in `{MULTI_TAB}` | recommended thesis usage: {recommendation}")
    lines += ["", "### Notebook 2 CSVs"]
    lines += [f"- `{path}`" for path in rows_for(MULTI_TAB)]
    lines += [
        "",
        "## Scientific Caveats",
        "",
        "- The selected period is bounded for reproducible visual analysis and does not replace the historical full-series benchmark.",
        "- Univariate and multivariate motif distances are not interpreted on a common quality scale.",
        "- The multivariate notebook preserves the historical `run_multivariate_matrix_profile` behavior: it uses `dimension_row = min(n_features - 1, profile_matrix_rows - 1)` from STUMPY MSTUMP.",
        "- Motifs are descriptive repeated subsequences, not trading signals or predictability claims.",
    ]
    path = OUT / "FINAL_STORY_MANIFEST.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def validate_outputs(kind: str) -> pd.DataFrame:
    fig_dir = UNI_FIG if kind == "univariate" else MULTI_FIG
    tab_dir = UNI_TAB if kind == "univariate" else MULTI_TAB
    rows = []
    rows.append({"check": "data_path_exists", "status": DATA_PATH.exists(), "detail": str(DATA_PATH)})
    try:
        ensure_project_imports()
        import_stumpy = importlib.import_module("matrix_profile_utils").import_stumpy
        import_stumpy()
        rows.append({"check": "stumpy_imports", "status": True, "detail": "stumpy imported through project utility"})
    except Exception as exc:
        rows.append({"check": "stumpy_imports", "status": False, "detail": str(exc)})
    for path in sorted(tab_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path)
            rows.append({"check": f"csv_nonzero_rows:{path.name}", "status": len(df) > 0, "detail": len(df)})
        except Exception as exc:
            rows.append({"check": f"csv_readable:{path.name}", "status": False, "detail": str(exc)})
    for path in sorted(fig_dir.glob("*.png")):
        rows.append({"check": f"png_nonempty:{path.name}", "status": path.stat().st_size > 0, "detail": path.stat().st_size})
    out = pd.DataFrame(rows)
    write_csv(out, tab_dir / f"{kind}_validation_checks.csv")
    return out

