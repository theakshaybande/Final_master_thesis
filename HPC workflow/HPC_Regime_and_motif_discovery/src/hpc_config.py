from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - handled at runtime in notebooks
    yaml = None


WORKFLOW_NAME = "HPC_Regime_and_motif_discovery"


@dataclass(frozen=True)
class WorkflowPaths:
    workflow_root: Path
    project_root: Path
    data_root: Path
    notebooks: Path
    results: Path
    regimes_quantile: Path
    regimes_hmm: Path
    motifs_matrix_profile: Path
    motifs_locomotif: Path
    comparisons: Path
    figures: Path
    tables: Path
    logs: Path
    configs: Path

    def as_string_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 20260609,
    "local_smoke_suffix": "_LOCAL_SMOKE_TEST",
    "data": {
        "feature_root": "final_dataset/features",
        "feature_glob": "**/*_features_*.parquet",
        "allowed_assets": None,
        "allowed_frequencies": None,
        "max_files": None,
        "local_allowed_frequencies": ["1h"],
        "local_max_assets": 2,
        "local_max_files": 2,
        "local_start_date": "2021-01-01",
        "local_end_date": "2021-03-31",
        "local_max_rows": 2500,
    },
    "feature_selection": {
        "preferred_features": [
            "log_return",
            "abs_log_return",
            "rolling_volatility_60",
            "rolling_volatility_30",
            "hl_range",
            "volume_zscore",
            "quote_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "spread_proxy",
        ],
        "hmm_features": [
            "log_return",
            "absolute_return",
            "rolling_vol",
            "squared_return",
            "volume_zscore",
            "hl_range",
        ],
        "max_nan_fraction": 0.40,
        "scaler": "robust",
        "min_non_constant_values": 3,
    },
    "quantile": {
        "rolling_windows": [60],
        "regime_counts": [3],
        "default_regime_count": 3,
        "default_rolling_window": 60,
        "volatility_columns": [
            "rolling_volatility_60",
            "rolling_volatility_30",
            "rolling_volatility_240",
            "rolling_vol",
            "realized_vol",
        ],
    },
    "hmm": {
        "states": [2, 3],
        "n_iter": 60,
        "covariance_type": "diag",
        "tol": 0.001,
        "min_rows": 250,
        "comparison_quantile_regime_count": 3,
    },
    "matrix_profile": {
        "top_k": 5,
        "min_segment_length": 96,
        "exclusion_zone_factor": 0.5,
        "univariate_channels": ["close", "log_return", "rolling_volatility_60"],
        "default_windows": [32, 64],
        "frequency_windows": {
            "1m": [60, 120],
            "5m": [48, 96],
            "15m": [32, 64],
            "1h": [24, 48],
            "1d": [20, 60],
            "daily": [20, 60],
        },
        "run_univariate": True,
        "run_multivariate": True,
        "use_gpu_if_available": True,
        "local_max_conditioned_segments_per_group": 2,
    },
    "locomotif": {
        "min_segment_length": 160,
        "feature_sets": ["core"],
        "parameter_grid": [
            {"l_min": 24, "l_max": 72, "rho": 0.65, "nb": 3, "overlap": 0.2, "warping": True}
        ],
        "local_max_points": 450,
        "hpc_max_points": None,
        "local_max_conditioned_segments_per_group": 2,
    },
    "plotting": {
        "max_points_per_figure": 5000,
        "max_figures_per_notebook": 40,
    },
}


def workflow_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_project_root(start: str | Path | None = None) -> Path:
    start_path = Path.cwd() if start is None else Path(start).resolve()
    candidates: list[Path] = [start_path, *start_path.parents]
    wf_root = workflow_root()
    candidates.extend([wf_root, *wf_root.parents])

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "final_dataset" / "features").exists():
            return candidate

    windows_default = Path(r"C:\Users\learn\OneDrive\Desktop\Final Masters Thesis")
    if (windows_default / "final_dataset" / "features").exists():
        return windows_default
    return wf_root.parent


def detect_execution_mode(override: str = "auto", project_root: str | Path | None = None) -> str:
    override_clean = str(override or "auto").strip().lower()
    if override_clean in {"local", "hpc"}:
        return override_clean
    if override_clean != "auto":
        raise ValueError("EXECUTION_MODE must be one of: auto, local, hpc")

    system = platform.system().lower()
    if system == "windows":
        return "local"

    hpc_env_vars = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "PBS_JOBID",
        "LSB_JOBID",
        "CUDA_VISIBLE_DEVICES",
    ]
    if any(os.environ.get(name) for name in hpc_env_vars):
        return "hpc"

    hostname = socket.gethostname().lower()
    hpc_host_tokens = ["hpc", "slurm", "login", "compute", "gpu", "node"]
    if any(token in hostname for token in hpc_host_tokens):
        return "hpc"

    root = Path(project_root) if project_root is not None else find_project_root()
    if system == "linux" and (root / "final_dataset" / "features").exists():
        return "hpc"

    return "local"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError(
            f"Cannot read {path}: PyYAML is not installed. Install pyyaml or use DEFAULT_CONFIG."
        )
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def load_workflow_config(mode: str, workflow_dir: str | Path | None = None) -> dict[str, Any]:
    wf_root = Path(workflow_dir).resolve() if workflow_dir else workflow_root()
    config_name = "local_smoke_test.yaml" if mode == "local" else "hpc_full_run.yaml"
    file_config = _load_yaml_config(wf_root / "run_configs" / config_name)
    config = _deep_merge(DEFAULT_CONFIG, file_config)
    config["active_mode"] = mode
    config["config_file"] = str(wf_root / "run_configs" / config_name)
    return config


def build_workflow_paths(
    workflow_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> WorkflowPaths:
    wf_root = Path(workflow_dir).resolve() if workflow_dir else workflow_root()
    proj_root = Path(project_root).resolve() if project_root else find_project_root(wf_root)
    results = wf_root / "results"
    return WorkflowPaths(
        workflow_root=wf_root,
        project_root=proj_root,
        data_root=proj_root / "final_dataset" / "features",
        notebooks=wf_root / "notebooks",
        results=results,
        regimes_quantile=results / "regimes" / "quantile",
        regimes_hmm=results / "regimes" / "hmm",
        motifs_matrix_profile=results / "motifs" / "matrix_profile",
        motifs_locomotif=results / "motifs" / "locomotif",
        comparisons=results / "comparisons",
        figures=results / "figures",
        tables=results / "tables",
        logs=results / "logs",
        configs=results / "configs",
    )


def is_local_mode(config: dict[str, Any]) -> bool:
    return config.get("active_mode") == "local"


def output_suffix(config: dict[str, Any]) -> str:
    return str(config.get("local_smoke_suffix", "_LOCAL_SMOKE_TEST")) if is_local_mode(config) else ""


def get_window_lengths(frequency: str, config: dict[str, Any]) -> list[int]:
    mp_cfg = config.get("matrix_profile", {})
    mapping = mp_cfg.get("frequency_windows", {}) or {}
    frequency_key = str(frequency).lower()
    windows = mapping.get(frequency_key, mp_cfg.get("default_windows", [32, 64]))
    return [int(window) for window in windows]


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    try:
        import numpy as np
        import pandas as pd

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()
    except Exception:
        pass
    return value


def package_versions(package_names: list[str] | None = None) -> dict[str, str | None]:
    package_names = package_names or [
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "hmmlearn",
        "stumpy",
        "numba",
        "matplotlib",
        "seaborn",
        "pyarrow",
        "fastparquet",
        "psutil",
        "pyyaml",
        "dtai-locomotif",
    ]
    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except Exception:
            versions[package_name] = None
    return versions


def detect_gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_available": False,
        "nvidia_smi_output": None,
        "cupy_available": False,
    }
    try:
        import cupy  # noqa: F401

        info["cupy_available"] = True
    except Exception:
        info["cupy_available"] = False

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        info["nvidia_smi_available"] = result.returncode == 0
        info["nvidia_smi_output"] = result.stdout.strip() if result.stdout else result.stderr.strip()
    except Exception as exc:
        info["nvidia_smi_output"] = f"nvidia-smi check failed: {exc}"
    return info


def environment_info(paths: WorkflowPaths | None = None) -> dict[str, Any]:
    memory_total_gb = None
    memory_available_gb = None
    try:
        import psutil

        mem = psutil.virtual_memory()
        memory_total_gb = round(mem.total / (1024**3), 3)
        memory_available_gb = round(mem.available / (1024**3), 3)
    except Exception:
        pass

    info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "memory_total_gb": memory_total_gb,
        "memory_available_gb": memory_available_gb,
        "working_directory": str(Path.cwd()),
        "package_versions": package_versions(),
        "gpu": detect_gpu_info(),
    }
    if paths is not None:
        info["paths"] = paths.as_string_dict()
    return info


def save_config_snapshot(
    config: dict[str, Any],
    paths: WorkflowPaths,
    stage_name: str,
    execution_mode: str,
) -> Path:
    paths.configs.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "stage": stage_name,
        "execution_mode": execution_mode,
        "config": config,
        "environment": environment_info(paths),
    }
    output_path = paths.configs / f"{stage_name}_config_snapshot{output_suffix(config)}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(snapshot), handle, indent=2)
    return output_path

