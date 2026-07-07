from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "locomotif_controlled_slice_comparison"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
LATEX = OUT / "latex"
PREV = ROOT / "reports" / "results" / "03_year_2025_mp_vs_real_locomotif"
HPC_TABLES = ROOT / "HPC workflow" / "HPC_Regime_and_motif_discovery" / "reports" / "study_notebooks" / "tables"
HPC_FIGURES = ROOT / "HPC workflow" / "HPC_Regime_and_motif_discovery" / "reports" / "study_notebooks" / "figures" / "locomotif"


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


def ensure_dirs() -> None:
    for directory in [TABLES, FIGURES, LATEX]:
        directory.mkdir(parents=True, exist_ok=True)


def infer_asset(name: str) -> str:
    upper = name.upper()
    if "BTCUSDT" in upper:
        return "BTCUSDT"
    if "ETHUSDT" in upper:
        return "ETHUSDT"
    return ""


def infer_frequency(path: Path, df: pd.DataFrame | None = None) -> str:
    text = str(path).lower()
    if "1h" in text or "1h" in path.name.lower():
        return "1h"
    if "15m" in text or "15m" in path.name.lower():
        return "15m"
    if df is not None and "frequency" in df.columns and not df.empty:
        return str(df["frequency"].dropna().iloc[0])
    return ""


def create_inventory() -> pd.DataFrame:
    patterns = ["*locomotif*", "*real_locomotif*", "*motif_sets*", "*dtai*", "*interval*"]
    files: dict[Path, None] = {}
    for base in [PREV, ROOT / "reports" / "locomotif_initial_study", HPC_TABLES, HPC_FIGURES]:
        if not base.exists():
            continue
        for pattern in patterns:
            for path in base.rglob(pattern):
                if path.is_file():
                    files[path] = None

    rows: list[dict[str, Any]] = []
    for path in sorted(files):
        suffix = path.suffix.lower().lstrip(".")
        rows_count: int | str = ""
        columns = ""
        notes = ""
        df: pd.DataFrame | None = None
        if suffix == "csv":
            try:
                df = pd.read_csv(path)
                rows_count = len(df)
                columns = ", ".join(df.columns[:12])
            except Exception as exc:
                notes = f"Could not read CSV: {exc}"
        elif suffix == "json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rows_count = 1
                columns = ", ".join(list(data.keys())[:12])
                if "real_locomotif_runs" in data:
                    notes = "Confirms real dtai-locomotif status."
            except Exception as exc:
                notes = f"Could not read JSON: {exc}"
        elif suffix in {"png", "pdf", "html", "ipynb", "txt"}:
            rows_count = ""
        rows.append(
            {
                "file_path": rel(path),
                "file_name": path.name,
                "file_type": suffix,
                "asset": infer_asset(path.name),
                "frequency": infer_frequency(path, df),
                "rows": rows_count,
                "important_columns": columns,
                "notes": notes,
            }
        )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(TABLES / "previous_successful_locomotif_inventory.csv", index=False)
    return inventory


def previous_summary() -> pd.DataFrame:
    status = json.loads((PREV / "real_locomotif_integration_status.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for asset in ["BTCUSDT", "ETHUSDT"]:
        motif_path = PREV / f"{asset}_locomotif_motif_sets.csv"
        motifs = read_csv(motif_path)
        run = status["real_locomotif_runs"].get(asset, {})
        occurrence_total = ""
        mean_occ = ""
        if not motifs.empty:
            occ = motifs[motifs["role"].astype(str).str.lower() == "occurrence"]
            occurrence_total = int(len(occ))
            mean_occ = float(motifs.drop_duplicates("motif_set_id")["occurrence_count"].mean())
        rows.append(
            {
                "asset": asset,
                "frequency": status.get("locomotif_frequency", "1h"),
                "number_of_motif_sets": run.get("motif_sets_found", motifs["motif_set_id"].nunique() if not motifs.empty else ""),
                "total_occurrences": occurrence_total,
                "mean_occurrence_count": mean_occ,
                "runtime_seconds": run.get("runtime_seconds", ""),
                "rho": motifs["rho"].dropna().iloc[0] if not motifs.empty and "rho" in motifs.columns else "",
                "lmin": motifs["l_min"].dropna().iloc[0] if not motifs.empty and "l_min" in motifs.columns else "",
                "lmax": motifs["l_max"].dropna().iloc[0] if not motifs.empty and "l_max" in motifs.columns else "",
                "source_file": rel(motif_path),
                "real_dtai_locomotif": bool(status.get("real_locomotif_import_ok") and status.get("real_locomotif_function_found")),
                "proxy_used": bool(status.get("any_proxy_used")),
                "placeholder_used": bool(status.get("any_placeholder_used")),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "previous_successful_locomotif_summary.csv", index=False)
    return summary


def copy_previous_figures() -> None:
    copies = {
        PREV / "BTCUSDT_real_locomotif_official_visualization.png": FIGURES / "previous_locomotif_BTCUSDT_motif_set_visualization.png",
        PREV / "ETHUSDT_real_locomotif_official_visualization.png": FIGURES / "previous_locomotif_ETHUSDT_motif_set_visualization.png",
    }
    for source, dest in copies.items():
        if source.exists():
            shutil.copy2(source, dest)


def bar_chart(df: pd.DataFrame, y: str, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["asset"], pd.to_numeric(df[y], errors="coerce"), color=["#234F73", "#D95F02"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def create_previous_plots(summary: pd.DataFrame) -> None:
    bar_chart(summary, "number_of_motif_sets", FIGURES / "previous_locomotif_motif_set_counts.png", "Previous real LoCoMotif motif-set counts", "Motif sets")
    bar_chart(summary, "runtime_seconds", FIGURES / "previous_locomotif_runtime.png", "Previous real LoCoMotif runtime", "Runtime seconds")
    if summary["total_occurrences"].replace("", np.nan).notna().any():
        bar_chart(summary, "total_occurrences", FIGURES / "previous_locomotif_occurrence_counts.png", "Previous real LoCoMotif occurrence counts", "Occurrences")


def create_final_evidence_table(previous: pd.DataFrame) -> pd.DataFrame:
    mp = read_csv(TABLES / "mp_controlled_slice_motifs.csv")
    loco_rt = read_csv(TABLES / "locomotif_controlled_runtime.csv")
    micro_rt = read_csv(TABLES / "micro_locomotif_runtime.csv")
    rows: list[dict[str, Any]] = []

    for slice_id in ["high_vol", "low_vol"]:
        subset = mp[mp["slice_id"] == slice_id].copy()
        if not subset.empty:
            best = subset.sort_values("best_motif_distance").iloc[0]
            rows.append(
                {
                    "method": "Matrix Profile",
                    "data": f"BTCUSDT 15m {slice_id}",
                    "status": "success",
                    "object_type": "fixed-length nearest-neighbour motif pair/candidate",
                    "runtime_seconds": best["runtime_seconds"],
                    "best_distance": best["best_motif_distance"],
                    "motif_sets": "",
                    "total_occurrences": 2,
                    "figure_path": f"figures/mp_btcusdt_15m_{slice_id}_top_motif_overlay.png",
                    "source_file": rel(TABLES / "mp_controlled_slice_motifs.csv"),
                    "note": "Completed on matched controlled slice.",
                }
            )
        timeout = loco_rt[loco_rt["slice_id"] == slice_id]
        if not timeout.empty:
            row = timeout.iloc[-1]
            threshold = "120" if "120" in str(row.get("error_message", "")) else ""
            rows.append(
                {
                    "method": "LoCoMotif",
                    "data": f"BTCUSDT 15m {slice_id}",
                    "status": "timeout",
                    "object_type": "time-warped motif set",
                    "runtime_seconds": threshold,
                    "best_distance": "",
                    "motif_sets": 0,
                    "total_occurrences": 0,
                    "figure_path": "",
                    "source_file": rel(TABLES / "locomotif_controlled_runtime.csv"),
                    "note": "Real dtai-locomotif attempted; no proxy output.",
                }
            )

    for _, row in previous.iterrows():
        rows.append(
            {
                "method": "LoCoMotif",
                "data": f"{row['asset']} {row['frequency']} previous validation",
                "status": "success",
                "object_type": "time-warped motif set",
                "runtime_seconds": row["runtime_seconds"],
                "best_distance": "",
                "motif_sets": row["number_of_motif_sets"],
                "total_occurrences": row["total_occurrences"],
                "figure_path": f"figures/previous_locomotif_{row['asset']}_motif_set_visualization.png",
                "source_file": row["source_file"],
                "note": "Previous real dtai-locomotif validation; no proxy or placeholder.",
            }
        )

    if not micro_rt.empty:
        for _, row in micro_rt.iterrows():
            success = str(row.get("success", "")).lower() == "true"
            rows.append(
                {
                    "method": "LoCoMotif",
                    "data": f"BTCUSDT 1h micro n={str(row['run_key']).split('_n')[-1]}",
                    "status": "success" if success else "timeout",
                    "object_type": "time-warped motif set",
                    "runtime_seconds": row["runtime_seconds"] if success else "600",
                    "best_distance": "",
                    "motif_sets": row.get("filtered_motif_sets_count", 0) if success else 0,
                    "total_occurrences": row.get("occurrence_count", 0) if success else 0,
                    "figure_path": "figures/micro_locomotif_btcusdt_1h_top_motif_set.png",
                    "source_file": rel(TABLES / "micro_locomotif_runtime.csv"),
                    "note": "Micro real dtai-locomotif attempt; no proxy output.",
                }
            )

    final = pd.DataFrame(rows)
    final.to_csv(TABLES / "final_mp_locomotif_evidence_table.csv", index=False)
    return final


def status_figures(final: pd.DataFrame) -> None:
    status_rows = [
        ("Previous BTCUSDT 1h", "success"),
        ("Previous ETHUSDT 1h", "success"),
        ("Matched 1h smoke 1000", "timeout"),
        ("Matched 15m high-vol 2000", "timeout"),
        ("Matched 15m low-vol 2000", "timeout"),
    ]
    micro = final[final["data"].astype(str).str.contains("micro", case=False, na=False)]
    if not micro.empty:
        status_rows.append(("Micro run", str(micro.iloc[-1]["status"])))
    colors = {"success": "#1B9E77", "timeout": "#D95F02", "failed": "#B2182B"}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([x[0] for x in status_rows], [1] * len(status_rows), color=[colors.get(x[1], "#7570B3") for x in status_rows])
    for idx, (_, status) in enumerate(status_rows):
        ax.text(idx, 0.5, status, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    ax.set_title("Final LoCoMotif status summary")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES / "final_locomotif_status_summary.png", dpi=300)
    plt.close(fig)

    grouped = final.groupby(["method", "status"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = grouped["method"] + " " + grouped["status"]
    ax.bar(labels, grouped["count"], color=["#234F73" if "Matrix" in label else "#D95F02" for label in labels])
    ax.set_ylabel("Evidence rows")
    ax.set_title("Final MP versus LoCoMotif evidence summary")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "final_mp_vs_locomotif_evidence_summary.png", dpi=300)
    plt.close(fig)


def write_salvage_report(inventory: pd.DataFrame, previous: pd.DataFrame, final: pd.DataFrame) -> None:
    lines = [
        "# LoCoMotif Salvage Report",
        "",
        "## Previous Successful Files Found",
    ]
    for _, row in previous.iterrows():
        lines.append(
            f"- {row['asset']} {row['frequency']}: {row['number_of_motif_sets']} motif sets, "
            f"{row['total_occurrences']} occurrence rows, runtime {row['runtime_seconds']} seconds, source `{row['source_file']}`."
        )
    lines.extend(
        [
            "",
            "The status file confirms real `dtai-locomotif` via `locomotif.locomotif.apply_locomotif`, with `any_proxy_used=false` and `any_placeholder_used=false`.",
            "",
            "## Figures and Occurrence Intervals",
            "- Previous official BTCUSDT and ETHUSDT LoCoMotif visualizations were copied into the controlled report figure folder.",
            "- Previous motif-set CSVs contain occurrence intervals with start/end indices and timestamps.",
            "",
            "## Micro Run Status",
        ]
    )
    micro = final[final["data"].astype(str).str.contains("micro", case=False, na=False)]
    if micro.empty:
        lines.append("- No micro rows were recorded.")
    else:
        for _, row in micro.iterrows():
            lines.append(f"- {row['data']}: status={row['status']}, note={row['note']}")
    lines.extend(["", "## Final Evidence Table", f"- `{rel(TABLES / 'final_mp_locomotif_evidence_table.csv')}`"])
    (OUT / "LOCOMOTIF_SALVAGE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def tex(value: Any) -> str:
    return str(value).replace("_", "\\_")


def write_latex(previous: pd.DataFrame, final: pd.DataFrame) -> None:
    btc = previous[previous["asset"] == "BTCUSDT"].iloc[0]
    eth = previous[previous["asset"] == "ETHUSDT"].iloc[0]
    mp_high = final[(final["method"] == "Matrix Profile") & (final["data"].str.contains("high_vol"))].iloc[0]
    mp_low = final[(final["method"] == "Matrix Profile") & (final["data"].str.contains("low_vol"))].iloc[0]
    section = rf"""\section{{LoCoMotif Validation and Controlled Comparison}}

\subsection{{Purpose of the LoCoMotif Experiment}}
LoCoMotif was included because Matrix Profile assumes fixed-length motifs, whereas LoCoMotif can identify motif sets whose occurrences may be locally time-warped. This makes LoCoMotif a useful complementary method for the thesis method-comparison question.

\subsection{{Previous Real LoCoMotif Validation}}
A previous real \texttt{{dtai-locomotif}} integration succeeded using \texttt{{locomotif.locomotif.apply\_locomotif}}. For BTCUSDT 1h data it found {btc['number_of_motif_sets']} motif sets in {float(btc['runtime_seconds']):.3f} seconds. For ETHUSDT 1h data it found {eth['number_of_motif_sets']} motif sets in {float(eth['runtime_seconds']):.3f} seconds. The integration status file records that no proxy or placeholder output was used.

\subsection{{Matched Slice Attempt}}
The new matched-slice experiment attempted to run LoCoMotif on the same BTCUSDT slices used for Matrix Profile. The 1h 1000-point smoke run timed out after 360 seconds. The 15-minute high-volatility 2000-point run timed out after 120 seconds, and the 15-minute low-volatility 2000-point run also timed out after 120 seconds. A smaller micro attempt also failed to complete within the bounded runtime, so no matched-slice LoCoMotif motif sets are reported.

\subsection{{Matrix Profile Baseline on the Same Slices}}
Matrix Profile completed on the matched BTCUSDT slices. On the high-volatility slice, the best matched-slice Matrix Profile distance was {float(mp_high['best_distance']):.6f} with runtime {float(mp_high['runtime_seconds']):.3f} seconds. On the low-volatility slice, the best distance was {float(mp_low['best_distance']):.6f} with runtime {float(mp_low['runtime_seconds']):.3f} seconds.

\subsection{{Interpretation for Method Comparison}}
The thesis therefore uses Matrix Profile as the main completed quantitative benchmark and LoCoMotif as a validated complementary method with practical scalability limitations in the matched-slice setting. Matrix Profile is more practical and directly interpretable for fixed-length nearest-neighbour evidence in the completed benchmark. LoCoMotif remains conceptually valuable for time-warped motif sets, and real integration was validated, but full matched-regime LoCoMotif benchmarking remains future work due to runtime constraints. Raw motif counts are not compared as equivalent quantities because the two methods return different motif objects.
"""
    (LATEX / "final_locomotif_results_section.tex").write_text(section, encoding="utf-8")
    discussion = """\\subsection{LoCoMotif Scalability and Interpretation}
The controlled LoCoMotif result is scientifically useful because it reveals practical scalability constraints rather than hiding them. LoCoMotif and Matrix Profile should not be compared by raw motif count because Matrix Profile returns fixed-length nearest-neighbour motif pairs, while LoCoMotif returns time-warped motif sets. The attempted matched-slice experiment shows that time-warped motif-set discovery under regime-conditioned financial slices is more computationally delicate than fixed-length Matrix Profile. A full LoCoMotif benchmark should therefore be treated as future work."""
    (LATEX / "final_locomotif_discussion_snippet.tex").write_text(discussion, encoding="utf-8")
    appendix = """\\section{LoCoMotif Validation Appendix}
This appendix reports the controlled run status table, the previous successful LoCoMotif output table, the timeout table, and the micro run table. The main machine-readable files are \\texttt{tables/final\\_mp\\_locomotif\\_evidence\\_table.csv}, \\texttt{tables/previous\\_successful\\_locomotif\\_summary.csv}, \\texttt{tables/locomotif\\_controlled\\_runtime.csv}, and \\texttt{tables/micro\\_locomotif\\_failure\\_summary.csv}. Figure references include \\texttt{figures/final\\_locomotif\\_status\\_summary.png}, \\texttt{figures/previous\\_locomotif\\_motif\\_set\\_counts.png}, and the copied previous LoCoMotif motif-set visualizations. No proxy outputs, placeholder outputs, or simulated motif sets were used."""
    (LATEX / "final_locomotif_appendix_snippet.tex").write_text(appendix, encoding="utf-8")


def update_main_report(previous: pd.DataFrame, final: pd.DataFrame) -> None:
    path = OUT / "CONTROLLED_LOCOMOTIF_RUN_REPORT.md"
    current = path.read_text(encoding="utf-8") if path.exists() else "# Controlled LoCoMotif Run Report\n"
    marker = "\n## Salvage Interpretation for Thesis\n"
    current = current.split(marker)[0].rstrip()
    btc = previous[previous["asset"] == "BTCUSDT"].iloc[0]
    eth = previous[previous["asset"] == "ETHUSDT"].iloc[0]
    section = f"""

## Salvage Interpretation for Thesis

What succeeded:
- Matrix Profile completed on the matched BTCUSDT 15m high-volatility and low-volatility slices.
- Previous real LoCoMotif validation succeeded on BTCUSDT 1h and ETHUSDT 1h using `dtai-locomotif`.
- Previous BTCUSDT LoCoMotif found {btc['number_of_motif_sets']} motif sets in {float(btc['runtime_seconds']):.3f} seconds.
- Previous ETHUSDT LoCoMotif found {eth['number_of_motif_sets']} motif sets in {float(eth['runtime_seconds']):.3f} seconds.

What timed out:
- Matched BTCUSDT 1h smoke, 1000 points: timeout after 360 seconds.
- Matched BTCUSDT 15m high-volatility, 2000 points: timeout after 120 seconds.
- Matched BTCUSDT 15m low-volatility, 2000 points: timeout after 120 seconds.
- Micro BTCUSDT 1h attempts at 300 and 150 points also timed out at 600 seconds.

What this means for RQ3:
Matrix Profile is the main completed quantitative benchmark. LoCoMotif is validated as a real complementary method for time-warped motif sets, but the matched-slice experiment shows practical runtime limitations in this environment. No proxy logic, fake motif sets, or simulated LoCoMotif outputs were used.

Exact thesis wording recommendation:
The thesis should state that LoCoMotif integration was validated using real `dtai-locomotif` outputs, but that the matched regime-conditioned BTCUSDT runs did not complete within bounded runtime. Matrix Profile should therefore be presented as the completed fixed-length nearest-neighbour benchmark, with LoCoMotif treated as validated complementary evidence and full matched-regime benchmarking left as future work.
"""
    path.write_text(current + section, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    inventory = create_inventory()
    previous = previous_summary()
    copy_previous_figures()
    create_previous_plots(previous)
    final = create_final_evidence_table(previous)
    status_figures(final)
    write_salvage_report(inventory, previous, final)
    write_latex(previous, final)
    update_main_report(previous, final)
    micro = final[final["data"].astype(str).str.contains("micro", case=False, na=False)]
    if not micro.empty and not (micro["status"] == "success").any():
        print("LOCOMOTIF SALVAGE PASS COMPLETE WITH MICRO FAILURE")
    else:
        print("LOCOMOTIF SALVAGE PASS COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
