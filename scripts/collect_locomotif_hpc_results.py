from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "locomotif_controlled_slice_comparison"
TABLES = OUT / "tables"
CONFIGS = OUT / "configs"
REPORT_LOGS = OUT / "logs"
ROOT_LOGS = ROOT / "logs"
SUMMARY = OUT / "HPC_RESULTS_SUMMARY.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def collect_logs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in [REPORT_LOGS, ROOT_LOGS]:
        if not base.exists():
            continue
        for path in sorted(base.glob("locomotif_*.out")) + sorted(base.glob("locomotif_*.err")) + sorted(base.glob("*.log")):
            text = path.read_text(encoding="utf-8", errors="replace")
            lower = text.lower()
            rows.append(
                {
                    "path": rel(path),
                    "bytes": path.stat().st_size,
                    "mentions_complete": "complete" in lower or "completion date" in lower,
                    "mentions_timeout": "timeout" in lower or "timed out" in lower,
                    "mentions_error": "error" in lower or "traceback" in lower or "failed" in lower,
                    "tail": "\n".join(text.splitlines()[-12:]),
                }
            )
    return rows


def summarize_locomotif_runtime() -> list[str]:
    path = TABLES / "locomotif_controlled_runtime.csv"
    df = read_csv(path)
    lines = [f"- Runtime table: `{rel(path)}`"]
    if df.empty:
        lines.append("- No LoCoMotif runtime rows found.")
        return lines
    for _, row in df.iterrows():
        status = "success" if str(row.get("success", "")).lower() == "true" else "failed_or_timeout"
        lines.append(
            f"- {row.get('run_key')}: {status}, runtime={row.get('runtime_seconds')}, "
            f"motif_sets={row.get('filtered_motif_sets_count')}, occurrences={row.get('occurrence_count')}, "
            f"error={row.get('error_message', '')}"
        )
    return lines


def summarize_mp() -> list[str]:
    path = TABLES / "mp_controlled_slice_motifs.csv"
    df = read_csv(path)
    lines = [f"- Matrix Profile table: `{rel(path)}`"]
    if df.empty:
        lines.append("- No Matrix Profile motif rows found.")
        return lines
    for _, row in df.iterrows():
        lines.append(
            f"- {row.get('slice_id')} m={row.get('window_length')}: "
            f"best_distance={row.get('best_motif_distance')}, runtime={row.get('runtime_seconds')}"
        )
    return lines


def summarize_configs() -> list[str]:
    lines: list[str] = []
    if not CONFIGS.exists():
        return ["- No configs directory found."]
    for path in sorted(CONFIGS.glob("*.json")):
        payload = read_json(path)
        if isinstance(payload, list):
            lines.append(f"- `{rel(path)}`: {len(payload)} entries")
        elif isinstance(payload, dict):
            lines.append(f"- `{rel(path)}`: keys={', '.join(list(payload.keys())[:8])}")
        else:
            lines.append(f"- `{rel(path)}`")
    return lines


def summarize_figures() -> list[str]:
    figure_dir = OUT / "figures"
    if not figure_dir.exists():
        return ["- No figures directory found."]
    figures = sorted(figure_dir.glob("*.png"))
    if not figures:
        return ["- No PNG figures found."]
    return [f"- `{rel(path)}` ({path.stat().st_size} bytes)" for path in figures]


def status_recommendation() -> str:
    runtime = read_csv(TABLES / "locomotif_controlled_runtime.csv")
    if runtime.empty:
        return "No LoCoMotif runtime evidence was found. Do not claim LoCoMotif completed on HPC."
    success_count = (runtime["success"].astype(str).str.lower() == "true").sum()
    timeout_count = runtime["error_message"].astype(str).str.contains("timeout|exceeded", case=False, na=False).sum()
    if success_count:
        return "At least one real LoCoMotif run succeeded. Report successful motif-set counts and keep timeouts separate."
    if timeout_count:
        return "LoCoMotif jobs timed out or failed. Use Matrix Profile as the completed benchmark and discuss LoCoMotif as bounded-runtime evidence."
    return "Review runtime logs before making thesis claims."


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    logs = collect_logs()
    lines = [
        "# HPC Results Summary",
        "",
        "## Jobs Found",
    ]
    if logs:
        for log in logs:
            lines.append(
                f"- `{log['path']}`: bytes={log['bytes']}, complete={log['mentions_complete']}, "
                f"timeout={log['mentions_timeout']}, error={log['mentions_error']}"
            )
    else:
        lines.append("- No `logs/locomotif_*.out` or `logs/locomotif_*.err` files found.")

    lines.extend(["", "## LoCoMotif Runtime and Motif Sets"])
    lines.extend(summarize_locomotif_runtime())
    lines.extend(["", "## Matrix Profile Results"])
    lines.extend(summarize_mp())
    lines.extend(["", "## Configs"])
    lines.extend(summarize_configs())
    lines.extend(["", "## Generated Figures"])
    lines.extend(summarize_figures())
    lines.extend(["", "## Thesis Interpretation Recommendation", status_recommendation()])

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
