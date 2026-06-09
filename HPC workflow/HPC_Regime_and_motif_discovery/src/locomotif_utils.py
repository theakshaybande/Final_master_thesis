from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


MODULE_CANDIDATES = ["locomotif.locomotif", "locomotif"]
INSTALL_HINT = "Install the real package with: pip install dtai-locomotif"


def import_locomotif_module():
    errors: dict[str, str] = {}
    for module_name in MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "apply_locomotif"):
                return module_name, module
            errors[module_name] = "Imported, but apply_locomotif is missing."
        except Exception as exc:
            errors[module_name] = repr(exc)
    raise RuntimeError(f"Real LoCoMotif API is unavailable. {INSTALL_HINT}. Import attempts: {errors}")


def _subprocess_api_status(timeout_seconds: int = 30) -> dict[str, Any]:
    check_code = r"""
import importlib
import inspect
import json

module_candidates = ["locomotif.locomotif", "locomotif"]
errors = {}
for module_name in module_candidates:
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, "apply_locomotif"):
            try:
                signature = str(inspect.signature(module.apply_locomotif))
            except Exception:
                signature = None
            print(json.dumps({
                "available": True,
                "module": module_name,
                "apply_locomotif_signature": signature,
            }))
            raise SystemExit(0)
        errors[module_name] = "Imported, but apply_locomotif is missing."
    except Exception as exc:
        errors[module_name] = repr(exc)
print(json.dumps({"available": False, "errors": errors}))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", check_code],
            capture_output=True,
            text=True,
            check=False,
            timeout=int(timeout_seconds),
        )
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "import_timeout": True,
            "timeout_seconds": int(timeout_seconds),
            "error": "LoCoMotif import check timed out.",
            "install_hint": INSTALL_HINT,
        }
    if result.returncode != 0 and not result.stdout.strip():
        return {
            "available": False,
            "import_timeout": False,
            "error": result.stderr.strip() or f"Import check returned {result.returncode}",
            "install_hint": INSTALL_HINT,
        }
    try:
        status = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            "available": False,
            "import_timeout": False,
            "error": f"Could not parse LoCoMotif import check output: {exc}",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "install_hint": INSTALL_HINT,
        }
    status["install_hint"] = INSTALL_HINT
    status["import_timeout"] = False
    status["timeout_seconds"] = int(timeout_seconds)
    if result.stderr.strip():
        status["stderr"] = result.stderr.strip()
    return status


def locomotif_api_status(timeout_seconds: int = 30) -> dict[str, Any]:
    return _subprocess_api_status(timeout_seconds=timeout_seconds)


def locomotif_api_status_in_process() -> dict[str, Any]:
    try:
        module_name, module = import_locomotif_module()
        signature = None
        try:
            signature = str(inspect.signature(module.apply_locomotif))
        except Exception:
            pass
        return {
            "available": True,
            "module": module_name,
            "apply_locomotif_signature": signature,
            "install_hint": INSTALL_HINT,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc), "install_hint": INSTALL_HINT}


def _validate_interval(interval: Any, label: str) -> tuple[int, int]:
    if not isinstance(interval, (tuple, list)) or len(interval) != 2:
        raise ValueError(f"{label} must be a (start, end) interval; got {interval!r}")
    start, end = int(interval[0]), int(interval[1])
    if end <= start:
        raise ValueError(f"{label} end must be greater than start; got {interval!r}")
    return start, end


def parse_locomotif_result(
    motif_sets: Any,
    timestamps: pd.Series,
    context: dict[str, Any],
    params: dict[str, Any],
    runtime_seconds: float,
) -> pd.DataFrame:
    if motif_sets is None:
        raise ValueError("LoCoMotif returned None.")
    if not isinstance(motif_sets, list):
        raise ValueError(f"Expected LoCoMotif result to be a list; got {type(motif_sets)}.")

    rows: list[dict[str, Any]] = []
    for motif_set_rank, item in enumerate(motif_sets, start=1):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"Motif set {motif_set_rank} must be (representative, occurrences); got {item!r}")
        representative, occurrences = item
        representative_start, representative_end = _validate_interval(representative, f"representative {motif_set_rank}")
        if not isinstance(occurrences, list):
            raise ValueError(f"Motif set {motif_set_rank} occurrences must be a list; got {type(occurrences)}.")

        all_intervals = [("representative", representative_start, representative_end)]
        for occurrence in occurrences:
            occurrence_start, occurrence_end = _validate_interval(occurrence, f"occurrence {motif_set_rank}")
            all_intervals.append(("occurrence", occurrence_start, occurrence_end))

        motif_set_size = len(all_intervals)
        for occurrence_id, (role, start, end) in enumerate(all_intervals, start=1):
            safe_end = min(end - 1, len(timestamps) - 1)
            rows.append(
                {
                    **context,
                    "method": "locomotif",
                    "motif_set_rank": int(motif_set_rank),
                    "motif_instance_id": int(occurrence_id),
                    "role": role,
                    "motif_start": int(start),
                    "motif_end": int(end),
                    "motif_length": int(end - start),
                    "motif_start_timestamp": pd.Timestamp(timestamps.iloc[start]).isoformat(),
                    "motif_end_timestamp": pd.Timestamp(timestamps.iloc[safe_end]).isoformat(),
                    "motif_score": np.nan,
                    "motif_set_size": int(motif_set_size),
                    "l_min": int(params.get("l_min")),
                    "l_max": int(params.get("l_max")),
                    "rho": float(params.get("rho")),
                    "nb": int(params.get("nb")),
                    "overlap": float(params.get("overlap")),
                    "warping": bool(params.get("warping", True)),
                    "runtime_seconds": float(runtime_seconds),
                    "n_observations": int(len(timestamps)),
                    "status": "success",
                }
            )
    return pd.DataFrame(rows)


def run_locomotif_discovery(
    X: np.ndarray,
    timestamps: pd.Series,
    params: dict[str, Any],
    context: dict[str, Any],
    max_points: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    module_name, module = import_locomotif_module()
    matrix = np.asarray(X, dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.ndim != 2:
        raise ValueError(f"LoCoMotif expects a time x channel matrix; got shape {matrix.shape}.")
    if max_points is not None:
        matrix = matrix[: int(max_points)]
        timestamps = timestamps.iloc[: int(max_points)].reset_index(drop=True)
    if len(matrix) < int(params.get("l_max", 1)) + 2:
        raise ValueError(f"Segment length {len(matrix)} is too short for LoCoMotif l_max={params.get('l_max')}.")

    t0 = time.perf_counter()
    motif_sets = module.apply_locomotif(
        matrix,
        l_min=int(params.get("l_min")),
        l_max=int(params.get("l_max")),
        rho=float(params.get("rho")),
        nb=int(params.get("nb")),
        overlap=float(params.get("overlap", 0.2)),
        warping=bool(params.get("warping", True)),
    )
    runtime = time.perf_counter() - t0
    parsed = parse_locomotif_result(motif_sets, timestamps, context, params, runtime)
    status = {
        "module": module_name,
        "raw_motif_sets_count": len(motif_sets),
        "runtime_seconds": runtime,
        "rows": len(parsed),
    }
    return parsed, status
