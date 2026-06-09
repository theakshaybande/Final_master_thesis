from __future__ import annotations

import json
import logging
import re
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_workflow_dirs(paths: Any) -> None:
    for value in vars(paths).values():
        if isinstance(value, Path) and value.suffix == "":
            value.mkdir(parents=True, exist_ok=True)


def safe_name(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def setup_stage_logger(log_dir: str | Path, stage_name: str, suffix: str = "") -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger(f"hpc_workflow.{stage_name}{suffix}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_path = Path(log_dir) / f"{stage_name}{suffix}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Log file: %s", log_path)
    return logger


def save_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    return path


def save_table(
    df: pd.DataFrame,
    csv_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    index: bool = False,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "rows": int(len(df)),
        "csv_path": str(csv_path) if csv_path is not None else None,
        "parquet_path": str(parquet_path) if parquet_path is not None else None,
        "csv_saved": False,
        "parquet_saved": False,
        "parquet_error": None,
    }
    if csv_path is not None:
        csv_path = Path(csv_path)
        ensure_dir(csv_path.parent)
        df.to_csv(csv_path, index=index)
        status["csv_saved"] = True
    if parquet_path is not None:
        parquet_path = Path(parquet_path)
        ensure_dir(parquet_path.parent)
        try:
            df.to_parquet(parquet_path, index=index)
            status["parquet_saved"] = True
        except Exception as exc:
            status["parquet_error"] = repr(exc)
            error_path = parquet_path.with_suffix(parquet_path.suffix + ".error.txt")
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
    return status


def read_table_if_exists(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table extension: {path}")


def failure_record(
    stage: str,
    exc: BaseException,
    asset: str | None = None,
    frequency: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    return {
        "stage": stage,
        "asset": asset,
        "frequency": frequency,
        "context": json.dumps(context, default=str),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "status": "failed",
    }


def save_failure_log(failures: list[dict[str, Any]], path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(failures)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "stage",
                "asset",
                "frequency",
                "context",
                "error_type",
                "error_message",
                "traceback",
                "status",
            ]
        )
    path = Path(path)
    save_table(df, csv_path=path, parquet_path=path.with_suffix(".parquet"), index=False)
    return df

