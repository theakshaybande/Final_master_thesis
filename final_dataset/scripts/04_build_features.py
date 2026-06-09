"""Build feature parquet files for motif discovery and regime detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
OVERWRITE = False
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

PROCESSED_DIR = BASE_DIR / "processed"
FEATURES_DIR = BASE_DIR / "features"

CRYPTO_FREQUENCIES = ["1m", "5m", "15m", "1h", "1d"]
CRYPTO_VOLUME_WINDOWS = {
    "1m": 1440,
    "5m": 288,
    "15m": 96,
    "1h": 168,
    "1d": 30,
}


def safe_log_return(close: pd.Series) -> pd.Series:
    """Compute log returns with invalid values converted to missing."""
    close = pd.to_numeric(close, errors="coerce")
    return np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)


def rolling_percentile_last(values: np.ndarray) -> float:
    """Percentile rank of the latest value inside a rolling window."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    return float((values <= values[-1]).mean())


def add_return_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add common return and absolute-return fields."""
    result = frame.copy()
    result["log_return"] = safe_log_return(result["close"])
    result["abs_log_return"] = result["log_return"].abs()
    result["pct_return"] = result["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    return result


def add_ohlcv_features(frame: pd.DataFrame, volume_window: int) -> pd.DataFrame:
    """Add OHLCV regime features."""
    result = add_return_features(frame)
    result["hl_range"] = ((result["high"] - result["low"]) / result["close"]).replace([np.inf, -np.inf], np.nan)
    result["rolling_volatility_30"] = result["log_return"].rolling(30, min_periods=10).std()
    result["rolling_volatility_60"] = result["log_return"].rolling(60, min_periods=20).std()
    result["rolling_volatility_240"] = result["log_return"].rolling(240, min_periods=60).std()

    if "volume" in result.columns:
        rolling_mean = result["volume"].rolling(volume_window, min_periods=max(5, volume_window // 5)).mean()
        rolling_std = result["volume"].rolling(volume_window, min_periods=max(5, volume_window // 5)).std()
        result["volume_zscore"] = ((result["volume"] - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan)
    return result


def add_daily_nonvolume_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add daily features for non-volume market datasets."""
    result = add_return_features(frame)
    if {"high", "low", "close"}.issubset(result.columns):
        result["hl_range"] = ((result["high"] - result["low"]) / result["close"]).replace([np.inf, -np.inf], np.nan)
    result["rolling_volatility_20"] = result["log_return"].rolling(20, min_periods=10).std()
    result["rolling_volatility_60"] = result["log_return"].rolling(60, min_periods=20).std()
    return result


def add_vix_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add VIX-specific validation features."""
    result = add_daily_nonvolume_features(frame)
    result["vix_level"] = result["close"]
    result["vix_change"] = result["close"].diff()
    result["vix_percentile_252d"] = result["close"].rolling(252, min_periods=60).apply(rolling_percentile_last, raw=True)
    threshold = result["close"].rolling(252, min_periods=60).quantile(0.80)
    result["high_vix_flag"] = (result["close"] >= threshold).astype("Int64")
    result.loc[threshold.isna(), "high_vix_flag"] = pd.NA
    return result


def has_real_volume(frame: pd.DataFrame) -> bool:
    """Return true when a frame has a usable nonzero volume column."""
    return "volume" in frame.columns and frame["volume"].fillna(0).abs().sum() > 0


def write_features(input_path: Path, output_path: Path, builder, *args) -> None:
    """Read a processed parquet, build features, and write a feature parquet."""
    if output_path.exists() and not OVERWRITE and feature_matches_source(input_path, output_path):
        print(f"[skip] {output_path} exists")
        return
    if output_path.exists() and not OVERWRITE:
        print(f"[repair] {output_path} does not match source coverage; rebuilding")
    if not input_path.exists():
        print(f"[missing] {input_path}")
        return

    frame = pd.read_parquet(input_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    result = builder(frame, *args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"[saved] {output_path} ({len(result):,} rows)")


def feature_matches_source(input_path: Path, output_path: Path) -> bool:
    """Return true when an existing feature file has the same timestamp coverage as its source."""
    try:
        if not input_path.exists() or not output_path.exists():
            return False
        source = pd.read_parquet(input_path, columns=["timestamp"])
        feature = pd.read_parquet(output_path, columns=["timestamp"])
        if source.empty or feature.empty:
            return False
        source_ts = pd.to_datetime(source["timestamp"], utc=True, errors="coerce")
        feature_ts = pd.to_datetime(feature["timestamp"], utc=True, errors="coerce")
        return source_ts.min() == feature_ts.min() and source_ts.max() == feature_ts.max() and len(source) == len(feature)
    except Exception:
        return False


def build_crypto_features() -> None:
    """Build features for all crypto frequencies."""
    for symbol in SYMBOLS:
        for frequency in CRYPTO_FREQUENCIES:
            input_path = PROCESSED_DIR / "crypto" / frequency / f"{symbol}_{frequency}_{START_DATE[:4]}_{END_DATE[:4]}.parquet"
            output_path = FEATURES_DIR / "crypto" / f"{symbol}_{frequency}_features_{START_DATE[:4]}_{END_DATE[:4]}.parquet"
            write_features(input_path, output_path, add_ohlcv_features, CRYPTO_VOLUME_WINDOWS[frequency])


def build_market_features() -> None:
    """Build FX, equity index, and VIX feature files."""
    market_dirs = [
        ("fx", FEATURES_DIR / "fx"),
        ("equity_indices", FEATURES_DIR / "equity_indices"),
    ]
    for asset_class, output_dir in market_dirs:
        for input_path in sorted((PROCESSED_DIR / asset_class).glob("*.parquet")):
            output_path = output_dir / input_path.name.replace("_1d_", "_1d_features_")
            frame = pd.read_parquet(input_path)
            builder = add_ohlcv_features if has_real_volume(frame) else add_daily_nonvolume_features
            args = (30,) if builder is add_ohlcv_features else ()
            write_features(input_path, output_path, builder, *args)

    for input_path in sorted((PROCESSED_DIR / "volatility").glob("VIX_*.parquet")):
        output_path = FEATURES_DIR / "volatility" / input_path.name.replace("_1d_", "_1d_features_")
        write_features(input_path, output_path, add_vix_features)


def main() -> None:
    """Build all feature files."""
    build_crypto_features()
    build_market_features()


if __name__ == "__main__":
    main()
