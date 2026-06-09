from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from io_utils import ensure_dir, safe_name


REGIME_COLORS = {
    "low_vol": "#2C7BB6",
    "medium_vol": "#ABD9E9",
    "high_vol": "#F46D43",
    "extreme_vol": "#A50026",
}


def save_figure(fig: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _sample_for_plot(df: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = max(1, int(np.ceil(len(df) / max_points)))
    return df.iloc[::step].copy()


def plot_price_by_regime(
    df: pd.DataFrame,
    label_col: str,
    title: str,
    output_path: str | Path,
    max_points: int = 5000,
) -> Path | None:
    if "close" not in df.columns or label_col not in df.columns or df.empty:
        return None
    plot_df = _sample_for_plot(df, max_points=max_points)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for regime_label, group in plot_df.groupby(label_col):
        ax.scatter(
            group["timestamp"],
            group["close"],
            s=5,
            alpha=0.75,
            label=str(regime_label),
            color=REGIME_COLORS.get(str(regime_label), None),
        )
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Close")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.grid(True, alpha=0.25)
    return save_figure(fig, output_path)


def plot_volatility_by_regime(
    df: pd.DataFrame,
    vol_col: str,
    label_col: str,
    title: str,
    output_path: str | Path,
    max_points: int = 5000,
) -> Path | None:
    if vol_col not in df.columns or label_col not in df.columns or df.empty:
        return None
    plot_df = _sample_for_plot(df, max_points=max_points)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for regime_label, group in plot_df.groupby(label_col):
        ax.scatter(
            group["timestamp"],
            group[vol_col],
            s=5,
            alpha=0.75,
            label=str(regime_label),
            color=REGIME_COLORS.get(str(regime_label), None),
        )
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(vol_col)
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.grid(True, alpha=0.25)
    return save_figure(fig, output_path)


def plot_regime_distribution(summary: pd.DataFrame, title: str, output_path: str | Path) -> Path | None:
    if summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = summary["regime_label"].astype(str)
    ax.bar(labels, summary["share"], color=[REGIME_COLORS.get(label, "#666666") for label in labels])
    ax.set_title(title)
    ax.set_xlabel("Regime")
    ax.set_ylabel("Share")
    ax.set_ylim(0, max(0.05, float(summary["share"].max()) * 1.2))
    ax.grid(True, axis="y", alpha=0.25)
    return save_figure(fig, output_path)


def plot_transition_heatmap(transitions: pd.DataFrame, title: str, output_path: str | Path) -> Path | None:
    if transitions.empty:
        return None
    matrix = transitions.pivot_table(index="from_regime", columns="to_regime", values="probability", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=max(1e-9, matrix.to_numpy().max()))
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return save_figure(fig, output_path)


def plot_posterior_confidence(df: pd.DataFrame, title: str, output_path: str | Path, max_points: int = 5000) -> Path | None:
    if "regime_confidence" not in df.columns or df.empty:
        return None
    plot_df = _sample_for_plot(df, max_points=max_points)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(plot_df["timestamp"], plot_df["regime_confidence"], linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Posterior confidence")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    return save_figure(fig, output_path)


def plot_matrix_profile(profile: pd.DataFrame, motifs: pd.DataFrame, title: str, output_path: str | Path) -> Path | None:
    if profile.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(profile["profile_index"], profile["matrix_profile"], linewidth=0.9)
    if not motifs.empty:
        ax.scatter(motifs["motif_start_1"], motifs["motif_distance"], color="#D7191C", s=28, label="motif")
        ax.scatter(motifs["motif_start_2"], motifs["motif_distance"], color="#2C7BB6", s=28, label="nearest neighbor")
        ax.legend(frameon=False)
    ax.set_title(title)
    ax.set_xlabel("Subsequence start index")
    ax.set_ylabel("Matrix Profile distance")
    ax.grid(True, alpha=0.25)
    return save_figure(fig, output_path)


def plot_univariate_motif_overlay(
    series: np.ndarray,
    motif_row: pd.Series,
    title: str,
    output_path: str | Path,
) -> Path | None:
    if motif_row is None or len(series) == 0:
        return None
    window = int(motif_row["window_length"])
    i = int(motif_row["motif_start_1"])
    j = int(motif_row["motif_start_2"])
    if i + window > len(series) or j + window > len(series):
        return None
    a = np.asarray(series[i : i + window], dtype=float)
    b = np.asarray(series[j : j + window], dtype=float)
    a = (a - np.nanmean(a)) / (np.nanstd(a) or 1.0)
    b = (b - np.nanmean(b)) / (np.nanstd(b) or 1.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(a, linewidth=2, label="motif window")
    ax.plot(b, linewidth=2, linestyle="--", label="nearest neighbor")
    ax.set_title(title)
    ax.set_xlabel("Within-window step")
    ax.set_ylabel("Z-normalized value")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    return save_figure(fig, output_path)


def plot_locomotif_intervals(
    df: pd.DataFrame,
    value_series: pd.Series,
    timestamps: pd.Series,
    title: str,
    output_path: str | Path,
    max_intervals: int = 50,
) -> Path | None:
    if df.empty or len(value_series) == 0:
        return None
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(timestamps, value_series, linewidth=0.8, color="#333333")
    for _, row in df.head(max_intervals).iterrows():
        color = "#D7191C" if row.get("role") == "representative" else "#2C7BB6"
        ax.axvspan(
            pd.Timestamp(row["motif_start_timestamp"]),
            pd.Timestamp(row["motif_end_timestamp"]),
            color=color,
            alpha=0.18,
        )
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Close/log return")
    ax.grid(True, alpha=0.25)
    return save_figure(fig, output_path)

