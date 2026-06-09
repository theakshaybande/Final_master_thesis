"""Run the complete reproducible dataset pipeline in dependency order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
OVERWRITE = False
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

SCRIPTS = [
    "01_download_crypto_binance.py",
    "02_resample_crypto.py",
    "03_download_market_data_yfinance.py",
    "04_build_features.py",
    "05_validate_dataset.py",
]


def run_script(script_name: str) -> None:
    """Run one pipeline script with the current Python interpreter."""
    script_path = BASE_DIR / "scripts" / script_name
    print(f"\n[run] {script_path}")
    completed = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {completed.returncode}")


def main() -> None:
    """Run all dataset pipeline steps."""
    for script_name in SCRIPTS:
        run_script(script_name)
    print("\n[done] Dataset pipeline completed.")


if __name__ == "__main__":
    main()
