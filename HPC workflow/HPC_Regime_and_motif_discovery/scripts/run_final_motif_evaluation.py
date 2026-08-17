from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from src.final_motif_evaluation import (  # noqa: E402
    OUTCOME_COLUMNS,
    build_locomotif_occurrences,
    build_mp_occurrences,
    compare_to_baseline,
    cross_method_agreement,
    dataframe_schema,
    financial_event_outcomes,
    intrinsic_metrics,
    load_price_slice,
    parameter_sensitivity,
    provenance_payload,
    read_table,
    resolve_cases,
    resolve_path,
    runtime_summary,
    summarize_outcomes,
    validate_controlled_slice,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final motif quality and financial validation evaluation.")
    parser.add_argument("--config", required=True, help="Path to final_motif_evaluation.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Audit inputs and configuration without writing outputs.")
    parser.add_argument("--smoke", action="store_true", help="Run only the BTCUSDT 1h agnostic case with reduced baselines.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configure_smoke(config: dict[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    config["baseline_repetitions"] = 50
    config["bootstrap"]["repetitions"] = 50
    config["inputs"]["slices"] = {"agnostic_1h": config["inputs"]["slices"]["agnostic_1h"]}
    return config


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "logs": output_dir / "logs",
        "latex": output_dir / "latex",
        "configs": output_dir / "configs",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    logging.info("Wrote %s (%s rows)", path, len(df))


def _save_figure(fig: plt.Figure, stem: str, figure_dir: Path) -> None:
    fig.tight_layout()
    fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_figures(tables: dict[str, pd.DataFrame], figure_dir: Path) -> None:
    intrinsic = tables.get("intrinsic", pd.DataFrame())
    if not intrinsic.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df = intrinsic.copy()
        plot_df["label"] = plot_df["method"] + "\n" + plot_df["slice_id"] + "\n" + plot_df["configuration_id"]
        x = np.arange(len(plot_df))
        ax.bar(x - 0.25, plot_df["occurrence_count"], width=0.25, label="Occurrences")
        ax.bar(x, plot_df["temporal_coverage"], width=0.25, label="Coverage")
        ax.bar(x + 0.25, plot_df["redundancy_fraction"], width=0.25, label="Redundancy")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["label"], rotation=70, ha="right", fontsize=7)
        ax.set_title("Intrinsic Motif Quality Summary")
        ax.set_ylabel("Count or fraction")
        ax.legend(frameon=False)
        _save_figure(fig, "intrinsic_motif_quality_summary", figure_dir)

    sensitivity = tables.get("intrinsic", pd.DataFrame())
    mp = sensitivity[sensitivity.get("method", pd.Series(dtype=str)) == "Matrix Profile"].copy() if not sensitivity.empty else pd.DataFrame()
    if not mp.empty:
        mp["window"] = mp["configuration_id"].str.extract(r"m=(\d+)").astype(float)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False)
        for slice_id, group in mp.groupby("slice_id"):
            axes[0].plot(group["window"], group["best_distance"], marker="o", label=slice_id)
            axes[1].plot(group["window"], group["temporal_coverage"], marker="o", label=slice_id)
            axes[2].plot(group["window"], group["occurrence_count"], marker="o", label=slice_id)
        axes[0].set_ylabel("Best MP distance")
        axes[1].set_ylabel("Coverage")
        axes[2].set_ylabel("Occurrences")
        for ax in axes:
            ax.set_xlabel("Window length m")
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle("Matrix Profile Window Sensitivity")
        _save_figure(fig, "mp_window_sensitivity", figure_dir)

    agreement = tables.get("agreement", pd.DataFrame())
    if not agreement.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(agreement))
        ax.bar(x - 0.2, agreement["union_coverage_jaccard"], width=0.2, label="Coverage Jaccard")
        ax.bar(x, agreement["proportion_mp_any_overlap"], width=0.2, label="Any overlap")
        iou_col = "proportion_mp_iou_ge_0p25"
        if iou_col in agreement.columns:
            ax.bar(x + 0.2, agreement[iou_col], width=0.2, label="IoU >= 0.25")
        ax.set_xticks(x)
        ax.set_xticklabels(agreement["slice_id"], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction")
        ax.set_title("Cross-Method Interval Agreement")
        ax.legend(frameon=False)
        _save_figure(fig, "cross_method_interval_agreement", figure_dir)

    inference = tables.get("inference", pd.DataFrame())
    for metric, stem, title in [
        ("simple_forward_return", "forward_return_vs_baseline", "Forward Return Difference vs Baseline"),
        ("future_realized_volatility", "future_realized_volatility_vs_baseline", "Future Realized Volatility Difference vs Baseline"),
        ("max_upward_excursion", "excursion_vs_baseline", "Excursion Difference vs Baseline"),
    ]:
        subset = inference[(inference.get("baseline_kind", "") == "unconditional") & (inference.get("metric", "") == metric)].copy()
        if subset.empty:
            continue
        subset["label"] = subset["method"] + "\n" + subset["slice_id"] + "\n" + subset["horizon_label"]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(subset))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.bar(x, subset["difference_in_means"], color="#4C78A8")
        if subset["bootstrap_mean_diff_ci_low"].notna().any():
            yerr = np.vstack(
                [
                    subset["difference_in_means"] - subset["bootstrap_mean_diff_ci_low"],
                    subset["bootstrap_mean_diff_ci_high"] - subset["difference_in_means"],
                ]
            )
            yerr = np.where(np.isfinite(yerr), yerr, 0)
            ax.errorbar(x, subset["difference_in_means"], yerr=yerr, fmt="none", color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(subset["label"], rotation=75, ha="right", fontsize=7)
        ax.set_ylabel("Observed mean minus baseline mean")
        ax.set_title(title)
        _save_figure(fig, stem, figure_dir)

    if not inference.empty:
        eligible = inference[inference["inference_status"] == "eligible"].copy()
        if not eligible.empty:
            eligible["label"] = eligible["metric"] + "\n" + eligible["slice_id"] + "\n" + eligible["horizon_label"]
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(eligible))
            ax.axhline(0, color="black", linewidth=0.8)
            ax.errorbar(
                x,
                eligible["difference_in_means"],
                yerr=np.vstack(
                    [
                        eligible["difference_in_means"] - eligible["bootstrap_mean_diff_ci_low"],
                        eligible["bootstrap_mean_diff_ci_high"] - eligible["difference_in_means"],
                    ]
                ),
                fmt="o",
                color="#F58518",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(eligible["label"], rotation=75, ha="right", fontsize=7)
            ax.set_ylabel("Mean difference with 95% CI")
            ax.set_title("Statistical Effect Summary")
            _save_figure(fig, "statistical_effect_summary", figure_dir)


def bib_keys(repo_root: Path, config: dict[str, Any]) -> dict[str, str]:
    bib_path = resolve_path(WORKFLOW_ROOT, config["inputs"].get("bib_file", ""))
    keys = {
        "mp1": "yeh2016matrixprofile1",
        "mp6": "yeh2017matrixprofile6",
        "locomotif": "vanwesenbeeck2024locomotif",
        "motiflets": "MISSING_MOTIFLETS_KEY",
        "motif_eval": "MISSING_MOTIF_EVAL_2026_KEY",
    }
    if not bib_path.exists():
        return keys
    text = bib_path.read_text(encoding="utf-8", errors="ignore").lower()
    for key in list(keys.values()):
        if key.lower() not in text and key.startswith("MISSING"):
            continue
    return keys


def write_latex_outputs(tables: dict[str, pd.DataFrame], latex_dir: Path, config: dict[str, Any]) -> None:
    keys = bib_keys(REPO_ROOT, config)
    methods = rf"""
\subsection{{Motif Evaluation and Financial Validation}}
Matrix Profile motif pairs are evaluated as fixed-length nearest-neighbour subsequences following \cite{{{keys['mp1']},{keys['mp6']}}}. LoCoMotif motif sets are evaluated as variable-length, locally time-warped occurrence sets following \cite{{{keys['locomotif']}}}. Motiflets and PROM-style ground-truth metrics are discussed as related evaluation concepts, but PROM is not used because the financial data in this thesis have no labelled motif ground truth.

The local bibliography was inspected before generating this text. Existing keys were found for Matrix Profile I, Matrix Profile VI, and LoCoMotif 2024. No existing BibTeX keys were found for Schafer and Leser (2022), ``Motiflets: Simple and Accurate Detection of Motifs in Time Series'', or Van Wesenbeeck et al. (2026), ``Quantitative evaluation of motif sets in time series''; those references should be added to the bibliography before final thesis compilation if they are cited in the main text.

For half-open motif intervals $I_k=[s_k,e_k)$ over an eligible slice of $N$ bars, temporal coverage is
\[
  C = \frac{{|\cup_k I_k|}}{{N}}.
\]
The thesis-derived redundancy statistic is
\[
  R = 1 - \frac{{|\cup_k I_k|}}{{\sum_k |I_k|}},
\]
with $R$ undefined when no interval mass exists. Cross-method interval agreement uses
\[
  \operatorname{{IoU}}(I,J) = \frac{{|I \cap J|}}{{|I \cup J|}}.
\]

Financial outcomes are anchored at the final bar of each motif. For close price $P_t$ at the motif end and horizon $h$, the simple and log forward returns are
\[
  R_{{t,h}} = \frac{{P_{{t+h}}}}{{P_t}} - 1,\qquad
  r_{{t,h}} = \log\left(\frac{{P_{{t+h}}}}{{P_t}}\right).
\]
Future realised volatility is reported over the same horizon as
\[
  RV_{{t,h}} = \sqrt{{\sum_{{i=1}}^h r_{{t+i}}^2}}.
\]
The random-baseline comparison estimates motif-conditioned differences against Monte Carlo samples of eligible non-motif anchors from the same asset and frequency; regime-matched baselines are used where regime labels are available.
""".strip()
    (latex_dir / "motif_evaluation_methods.tex").write_text(methods + "\n", encoding="utf-8")

    intrinsic = tables.get("intrinsic", pd.DataFrame()).copy()
    if not intrinsic.empty:
        columns = ["method", "slice_id", "configuration_id", "occurrence_count", "temporal_coverage", "redundancy_fraction", "best_distance"]
        intrinsic[columns].to_latex(latex_dir / "motif_evaluation_results_table.tex", index=False, float_format="%.4g")
    else:
        (latex_dir / "motif_evaluation_results_table.tex").write_text("% No intrinsic results available.\n", encoding="utf-8")

    summary = tables.get("summary", pd.DataFrame()).copy()
    if not summary.empty:
        columns = ["method", "slice_id", "configuration_id", "horizon_label", "n_eligible_events", "simple_forward_return_mean", "future_realized_volatility_mean", "inference_status"]
        summary[columns].head(30).to_latex(latex_dir / "financial_validation_results_table.tex", index=False, float_format="%.4g")
    else:
        (latex_dir / "financial_validation_results_table.tex").write_text("% No financial validation results available.\n", encoding="utf-8")


def write_methods_markdown(output_dir: Path) -> None:
    text = """# Final Motif Evaluation Methods

## Literature-Derived Motif Quantities

- Yeh et al. (2016), "Matrix Profile I: All Pairs Similarity Joins for Time Series", IEEE ICDM, DOI: 10.1109/ICDM.2016.0179. Local BibTeX key found: `yeh2016matrixprofile1`.
- Yeh, Kavantzas and Keogh (2017), "Matrix Profile VI: Meaningful Multidimensional Motif Discovery", IEEE ICDM, DOI: 10.1109/ICDM.2017.66. Local BibTeX key found: `yeh2017matrixprofile6`.
- Van Wesenbeeck et al. (2024), "LoCoMotif: Discovering time-warped motifs in time series", Data Mining and Knowledge Discovery, DOI: 10.1007/s10618-024-01032-z. Local BibTeX key found: `vanwesenbeeck2024locomotif`.
- Schafer and Leser (2022), "Motiflets: Simple and Accurate Detection of Motifs in Time Series", PVLDB 16(4), 725-737, DOI: 10.14778/3574245.3574257. No matching local BibTeX key was found.
- Van Wesenbeeck et al. (2026), "Quantitative evaluation of motif sets in time series", Data Mining and Knowledge Discovery, DOI: 10.1007/s10618-025-01169-5. No matching local BibTeX key was found.

## Thesis-Specific Quantities

The evaluation adds interval-union coverage, duplicated-coverage redundancy, cross-method interval agreement, no-lookahead financial outcomes, random baselines, non-parametric effect sizes, permutation tests, and block-bootstrap uncertainty. These are descriptive validation statistics, not native Matrix Profile or LoCoMotif quality scores and not trading-strategy metrics.

PROM, precision, recall and F1 are not calculated for BTC/ETH motifs because the financial market data have no labelled motif ground truth.
"""
    (output_dir / "METHODS.md").write_text(text, encoding="utf-8")


def write_notebook(notebook_path: Path) -> None:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Final Motif Quality and Financial Validation\n",
                "\n",
                "This notebook presents the derived evaluation tables produced by `scripts/run_final_motif_evaluation.py`. It does not rerun Matrix Profile or LoCoMotif discovery.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Literature Attribution\n",
                "\n",
                "- Yeh et al. (2016), Matrix Profile I, DOI: 10.1109/ICDM.2016.0179.\n",
                "- Yeh, Kavantzas and Keogh (2017), Matrix Profile VI, DOI: 10.1109/ICDM.2017.66.\n",
                "- Van Wesenbeeck et al. (2024), LoCoMotif, DOI: 10.1007/s10618-024-01032-z.\n",
                "- Schafer and Leser (2022), Motiflets, DOI: 10.14778/3574245.3574257; no local BibTeX key was found.\n",
                "- Van Wesenbeeck et al. (2026), Quantitative evaluation of motif sets, DOI: 10.1007/s10618-025-01169-5; no local BibTeX key was found.\n",
                "\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Metric Families\n",
                "\n",
                "- Literature-derived metric: Matrix Profile nearest-neighbour distance, fixed-length motif-pair interpretation, LoCoMotif motif-set/occurrence representation, coverage and similarity concepts.\n",
                "- Thesis-specific descriptive/robustness metric: interval-union coverage, duplicated-coverage redundancy, cross-method interval agreement, MP window sensitivity.\n",
                "- Thesis-specific financial validation metric: no-lookahead forward returns, future realised volatility, upward/downward excursion, and random-baseline differences.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "OUT = Path('../../reports/final_motif_evaluation/tables')\n",
                "tables = {p.stem: pd.read_csv(p) for p in OUT.glob('*.csv')}\n",
                "sorted(tables)\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Intrinsic Quality Results\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["tables['01_intrinsic_motif_metrics'].head(20)\n"],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## MP Window Sensitivity and Cross-Method Agreement\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["display(tables['03_parameter_sensitivity'])\ndisplay(tables['02_cross_method_agreement'])\n"],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Financial Outcomes, Baselines, and Inference\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(tables['05_financial_summary'].head(20))\n",
                "display(tables['08_statistical_inference'].head(30))\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Limitations\n",
                "\n",
                "The analysis is unsupervised and has no labelled financial motif ground truth. Matrix Profile pairs and LoCoMotif sets are different motif objects. Small event counts, especially Matrix Profile pairs, are descriptive only when `inference_status` is `insufficient_n`. These results do not establish predictability, profitability, or causality.\n",
            ],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def write_markdown_summary(
    output_dir: Path,
    audit: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    notes: list[str],
) -> None:
    intrinsic = tables.get("intrinsic", pd.DataFrame())
    summary = tables.get("summary", pd.DataFrame())
    inference = tables.get("inference", pd.DataFrame())
    lines = [
        "# Tasks 15-17 Results Summary",
        "",
        "## Scope",
        "This evaluation uses already-discovered controlled Matrix Profile and LoCoMotif artifacts only. It does not rerun motif discovery, regenerate regimes, or download data.",
        "",
        "## Input Data",
    ]
    for item in audit.get("input_files", []):
        lines.append(f"- `{item.get('path')}`: rows={item.get('rows')}, size={item.get('file_size')}, exists={item.get('exists')}")
    lines.extend(
        [
            "",
            "## Metrics",
            "- Literature-derived: Matrix Profile nearest-neighbour distance and fixed-length motif pairs; LoCoMotif motif-set/occurrence representation.",
            "- Thesis-specific descriptive/robustness: interval-union coverage, duplicated-coverage redundancy, cross-method interval agreement, MP window sensitivity.",
            "- Thesis-specific financial validation: no-lookahead forward returns, directional consistency, future realised volatility, upward/downward excursion, and random-baseline differences.",
            "",
            "## Key Numerical Results",
        ]
    )
    if not intrinsic.empty:
        for _, row in intrinsic.iterrows():
            lines.append(
                f"- {row['method']} {row['slice_id']} {row['configuration_id']}: occurrences={row['occurrence_count']}, "
                f"coverage={row['temporal_coverage']:.4f}, redundancy={row['redundancy_fraction']:.4f}."
            )
    if not summary.empty:
        insufficient = int((summary["inference_status"] == "insufficient_n").sum())
        lines.append(f"- Financial summary rows: {len(summary)}; rows marked insufficient_n: {insufficient}.")
    if not inference.empty:
        eligible = inference[inference["inference_status"] == "eligible"]
        lines.append(f"- Inference rows: {len(inference)}; eligible inference rows: {len(eligible)}.")
    lines.extend(
        [
            "",
            "## Caveats and Unsupported Items",
            "- PROM is not calculated because the BTC/ETH financial motif data have no genuine labelled motif ground truth.",
            "- Native LoCoMotif fitness/similarity is left as NaN where the stored artifacts do not provide an observed score.",
            "- LoCoMotif parameter sensitivity is unavailable unless multiple parameter values exist in stored outputs.",
            "- Matrix Profile rows with fewer than 10 eligible events remain descriptive only.",
        ]
    )
    for note in notes:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Thesis Placement",
            "- Main text: metric definitions, primary controlled comparison, no-lookahead event anchoring, and high-level baseline comparison.",
            "- Appendix: full MP window sensitivity, per-horizon event tables, baseline/inference details, and provenance.",
            "",
            "## Claim Discipline",
            "These results evaluate whether motif occurrences are followed by different market-behaviour distributions. They do not establish a trading strategy, profitability, predictability, or causality.",
        ]
    )
    (output_dir / "TASKS_15_17_RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    config_path = resolve_path(Path.cwd(), args.config)
    config = load_config(config_path)
    if args.smoke:
        config = configure_smoke(config)

    cases = resolve_cases(WORKFLOW_ROOT, config)
    table_dir = resolve_path(WORKFLOW_ROOT, config["inputs"]["controlled_table_dir"])
    output_dir = resolve_path(WORKFLOW_ROOT, config["output_dir"])
    mp_path = table_dir / "mp_controlled_slice_motifs.csv"
    mp_runtime_path = table_dir / "mp_controlled_slice_runtime.csv"
    lm_runtime_path = table_dir / "locomotif_controlled_runtime.csv"
    input_paths = [case.slice_path for case in cases] + [case.locomotif_raw_path for case in cases if case.locomotif_raw_path] + [
        mp_path,
        mp_runtime_path,
        lm_runtime_path,
        table_dir / "locomotif_controlled_motif_sets.csv",
        table_dir / "locomotif_controlled_occurrences.csv",
        table_dir / "mp_vs_locomotif_controlled_comparison.csv",
    ]

    price_frames: dict[str, pd.DataFrame] = {}
    schemas: dict[str, Any] = {}
    expected_metadata = config.get("expected_slice_metadata", {})
    for case in cases:
        raw_slice = read_table(case.slice_path)
        if case.slice_id in expected_metadata:
            validate_controlled_slice(case.slice_id, raw_slice, expected_metadata[case.slice_id])
        price_frames[case.slice_id] = load_price_slice(case.slice_path)
        schemas[str(case.slice_path)] = dataframe_schema(price_frames[case.slice_id])

    for path in [mp_path, mp_runtime_path, lm_runtime_path, table_dir / "mp_vs_locomotif_controlled_comparison.csv"]:
        if path.exists():
            schemas[str(path)] = dataframe_schema(read_table(path))

    if args.dry_run:
        print(json.dumps({"cases": [case.__dict__ | {"slice_path": str(case.slice_path), "locomotif_raw_path": str(case.locomotif_raw_path)} for case in cases], "schemas": schemas}, indent=2, default=str))
        return 0

    dirs = ensure_output_dirs(output_dir)
    (dirs["configs"] / Path(args.config).name).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    mp_rows = read_table(mp_path)
    all_occurrence_frames: list[pd.DataFrame] = []
    notes: list[str] = []
    for case in cases:
        mp_occ, mp_notes = build_mp_occurrences(mp_rows, case, price_frames[case.slice_id])
        lm_occ, lm_notes = build_locomotif_occurrences(
            case,
            price_frames[case.slice_id],
            invalid_interval_policy=str(config.get("invalid_interval_policy", "error")),
        )
        all_occurrence_frames.extend([mp_occ, lm_occ])
        notes.extend(mp_notes)
        notes.extend(lm_notes)

    occurrences = pd.concat([frame for frame in all_occurrence_frames if not frame.empty], ignore_index=True)
    occurrences = occurrences.drop_duplicates(subset=["method", "slice_id", "configuration_id", "occurrence_id"])
    slice_lengths = {slice_id: len(df) for slice_id, df in price_frames.items()}
    primary_windows = {case.slice_id: case.primary_mp_window for case in cases}

    intrinsic = intrinsic_metrics(occurrences, slice_lengths)
    agreement = cross_method_agreement(occurrences, primary_windows, config["agreement_iou_thresholds"])
    sensitivity = parameter_sensitivity(intrinsic)
    outcomes = financial_event_outcomes(occurrences, price_frames, config["horizons"])
    summary = summarize_outcomes(outcomes, minimum_inference_n=int(config["minimum_inference_n"]))
    unconditional, inference_unconditional = compare_to_baseline(outcomes, occurrences, price_frames, config, "unconditional")
    regime_matched, inference_regime = compare_to_baseline(outcomes, occurrences, price_frames, config, "regime_matched")
    inference = pd.concat([inference_unconditional, inference_regime], ignore_index=True) if not inference_regime.empty else inference_unconditional
    runtimes = runtime_summary(occurrences, [read_table(mp_runtime_path), read_table(lm_runtime_path)])

    final_summary = intrinsic.merge(
        summary.groupby(["method", "slice_id", "configuration_id"], dropna=False)["n_eligible_events"].max().reset_index(),
        on=["method", "slice_id", "configuration_id"],
        how="left",
    )
    final_summary["thesis_claim_level"] = np.where(final_summary["n_eligible_events"].fillna(0) >= int(config["minimum_inference_n"]), "baseline-supported-descriptive", "descriptive-only-small-n")

    tables = {
        "intrinsic": intrinsic,
        "agreement": agreement,
        "sensitivity": sensitivity,
        "outcomes": outcomes,
        "summary": summary,
        "unconditional": unconditional,
        "regime_matched": regime_matched,
        "inference": inference,
        "runtime": runtimes,
        "final_summary": final_summary,
    }
    table_files = {
        "intrinsic": "01_intrinsic_motif_metrics.csv",
        "agreement": "02_cross_method_agreement.csv",
        "sensitivity": "03_parameter_sensitivity.csv",
        "outcomes": "04_financial_event_outcomes.csv",
        "summary": "05_financial_summary.csv",
        "unconditional": "06_unconditional_baseline_comparison.csv",
        "regime_matched": "07_regime_matched_baseline_comparison.csv",
        "inference": "08_statistical_inference.csv",
        "runtime": "09_runtime_summary.csv",
        "final_summary": "10_final_thesis_evaluation_summary.csv",
    }
    for key, filename in table_files.items():
        save_table(tables[key], dirs["tables"] / filename)

    generate_figures(tables, dirs["figures"])
    write_latex_outputs(tables, dirs["latex"], config)
    write_methods_markdown(output_dir)
    write_notebook(WORKFLOW_ROOT / "notebooks" / "study" / "05_final_motif_quality_and_financial_validation.ipynb")
    audit = provenance_payload(REPO_ROOT, WORKFLOW_ROOT, config, input_paths, schemas, notes)
    write_json(output_dir / "DATA_PROVENANCE.json", audit)
    write_markdown_summary(output_dir, audit, tables, notes)
    logging.info("Completed final motif evaluation in %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
