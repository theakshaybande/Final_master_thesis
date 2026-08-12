from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = ROOT / "notebooks" / "final_story"


COMMON_SETUP = r"""
from pathlib import Path
import sys

ROOT = Path.cwd()
while not (ROOT / "reports" / "final_story" / "scripts" / "final_story_core.py").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("Could not locate thesis repository root.")
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT / "reports" / "final_story" / "scripts"))
from final_story_core import *

ensure_dirs()
configure_plots()
print(f"Repository root: {ROOT}")
print(f"Data source: {DATA_PATH}")
print(f"Selected period: {PERIOD_START} to {PERIOD_END}")
"""


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def write_notebook(path: Path, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def univariate_notebook() -> list:
    return [
        md("# Univariate Matrix Profile Motif Discovery Visual Story\n\nThis notebook turns the existing Matrix Profile implementation into a thesis-facing empirical narrative for one financial signal. It uses the repository implementation of `run_univariate_matrix_profile`; no Matrix Profile algorithm is reimplemented here."),
        code(COMMON_SETUP),
        md("## 1. Experimental Context\n\nThe selected BTCUSDT 15m period is bounded for visual reproducibility. It contains a previously verified strong BTCUSDT 15m `rolling_volatility_60` motif pair from the saved study tables, while avoiding a full-history rerun."),
        code("uni = build_univariate_context()\nwrite_run_record('univariate')\nuni['metadata']"),
        md("## 2. Full Financial Time-Series Overview\n\nThe shaded windows mark the two objectively selected motif occurrences on the original price context."),
        code("display(univariate_overview(uni))"),
        md("## 3. Price/Feature Plus Matrix Profile\n\nThe lower panel is the actual Matrix Profile returned by the project utility. Low values indicate subsequences with close nearest neighbours under the chosen univariate representation."),
        code("display(univariate_series_profile(uni))"),
        md("## 4. Top Motif Normalized Overlay\n\nEach subsequence is z-normalized independently for visual shape comparison only. The saved CSV contains the raw and normalized values."),
        code("display(univariate_overlay(uni))\npd.read_csv(UNI_TAB / '03_univariate_top_motif_overlay_values.csv').head()"),
        md("## 5. Original-Scale Motif Comparison\n\nThe same motif windows are shown on their original feature scale to distinguish normalized shape similarity from absolute market level."),
        code("display(univariate_original_scale(uni))"),
        md("## 6. Candlestick Comparison\n\nThe OHLC candles use exactly the motif start and end timestamps, with no external indicators added."),
        code("display(univariate_candlesticks(uni))"),
        md("## 7. Top-5 Motif Gallery\n\nThe gallery uses the existing non-overlapping top-pair extraction logic from the project utility."),
        code("display(univariate_gallery(uni))\npd.read_csv(UNI_TAB / '06_univariate_top5_motifs.csv')"),
        md("## 8. Pairwise Similarity Matrix\n\nThis is an auxiliary distance matrix among selected motif representatives, not the Matrix Profile. Distances are z-normalized Euclidean distances between equal-length representative subsequences."),
        code("display(univariate_similarity(uni))\npd.read_csv(UNI_TAB / '07_univariate_top5_similarity_matrix.csv', index_col=0)"),
        md("## 9. Window-Length Sensitivity\n\nThe definition of a recurring pattern depends on the subsequence horizon. This is a methodological sensitivity, not a flaw."),
        code("display(univariate_window_sensitivity(uni))\npd.read_csv(UNI_TAB / '08_univariate_window_length_sensitivity.csv')"),
        md("## 10. Concise Findings\n\nThe executed evidence supports four limited claims: repeated local volatility structures are detectable in BTCUSDT 15m data; the strongest pair in this bounded period occurs at distinct timestamps; normalized shape similarity can coexist with different original price and volatility levels; and changing the window changes the temporal scale of the discovered motif."),
        code("validation = validate_outputs('univariate')\nmanifest = write_manifest()\nprint('Manifest:', manifest)\nprint('Output figures:')\nfor p in sorted(UNI_FIG.glob('*.png')):\n    print(p)\nprint('Output CSVs:')\nfor p in sorted(UNI_TAB.glob('*.csv')):\n    print(p)\nvalidation"),
    ]


def multivariate_notebook() -> list:
    return [
        md("# Multivariate Matrix Profile Motif Discovery Visual Story\n\nThis notebook shows what changes when repeated subsequences must agree jointly across several financial variables. It uses the repository feature pipeline and `run_multivariate_matrix_profile`, preserving the historical STUMPY MSTUMP behavior."),
        code(COMMON_SETUP),
        md("## 1. Multivariate Representation\n\nThe visual example uses a compact feature subset drawn from available and historically used features: return, realized volatility, high-low range, and volume activity. The historical full-feature benchmark remains referenced for runtime comparison."),
        code("multi = build_multivariate_context()\nwrite_run_record('multivariate')\nmulti['diagnostics']"),
        code("display(multivariate_feature_panel(multi))\npd.read_csv(MULTI_TAB / '01_multivariate_feature_selection.csv')"),
        md("## 2. Scaled Representation\n\nMSTUMP receives robust-scaled features from the existing feature-selection methodology. This panel is not the raw market representation."),
        code("display(multivariate_scaled_panel(multi))"),
        md("## 3. Multivariate Matrix Profile\n\nThe historical implementation computes `stumpy.mstump(matrix.T, window_length)` and selects `dimension_row = min(n_features - 1, profile_matrix_rows - 1)`. For four selected features, this corresponds to the row requiring agreement across the full selected dimensionality after MSTUMP's subspace ordering."),
        code("display(multivariate_profile_context(multi))\npd.read_csv(MULTI_TAB / '03_multivariate_selected_motif_metadata.csv')"),
        md("## 4. Strongest Multivariate Motif Across Channels\n\nThe motif pair is shown feature-by-feature using the same robust-scaled values used for discovery."),
        code("display(multivariate_top_across_features(multi))\npd.read_csv(MULTI_TAB / '04_multivariate_top_motif_feature_values.csv').head()"),
        md("## 5. Univariate Versus Multivariate Motif\n\nThe two methods are compared at the same window length and period. Distances are reported but not interpreted on a common scale."),
        code("display(univariate_vs_multivariate(multi))\npd.read_csv(MULTI_TAB / '05_univariate_vs_multivariate_top_motif_comparison.csv')"),
        md("## 6. Case Study Where Additional Dimensions Matter\n\nThe selection rule searches the top univariate candidates for a pair with low univariate distance but relatively larger disagreement in at least one other scaled feature channel."),
        code("display(multivariate_information_case(multi))\npd.read_csv(MULTI_TAB / '06_when_multivariate_information_changes_similarity_case_candidates.csv').head()"),
        md("## 7. Multivariate Top-5 Motif Gallery\n\nEach panel summarizes a top multivariate pair using the mean of selected scaled channels for compact visual comparison."),
        code("display(multivariate_gallery(multi))\npd.read_csv(MULTI_TAB / '07_multivariate_top5_motifs.csv')"),
        md("## 8. Similarity Structure\n\nThis auxiliary matrix uses RMS aggregation of per-feature z-normalized Euclidean distances. It is not the MSTUMP profile itself."),
        code("display(multivariate_similarity(multi))\npd.read_csv(MULTI_TAB / '08_multivariate_top5_similarity_matrix.csv', index_col=0)"),
        md("## 9. Feature-Set Comparison\n\nThe saved historical benchmark is used here. Its central message is computational: the full multivariate representation costs much more runtime than simple univariate feature sets. This does not imply universal motif-quality improvement."),
        code("display(feature_set_comparison())\npd.read_csv(MULTI_TAB / '09_feature_set_comparison.csv')"),
        md("## 10. Feature-Number / Dimensionality Consideration\n\nSTUMPY MSTUMP returns rows associated with increasing subspace dimensionality. The figure below records the minimum profile value by row for the selected compact feature set and documents the row semantics used by the historical pipeline."),
        code("display(dimensionality_profile(multi))\npd.read_csv(MULTI_TAB / '10_multivariate_dimensionality_profile.csv')"),
        md("## 11. Concise Findings\n\nThe executed evidence supports limited descriptive claims: multivariate motifs can require coherence across return, volatility, range, and volume activity; the strongest multivariate pair may differ from the univariate pair; adding dimensions changes the geometry of subsequence similarity; and historical feature-set benchmarks show a major runtime cost for broad multivariate representations. Nonstationarity and regimes are examined later in the thesis."),
        code("validation = validate_outputs('multivariate')\nmanifest = write_manifest()\nprint('Manifest:', manifest)\nprint('Output figures:')\nfor p in sorted(MULTI_FIG.glob('*.png')):\n    print(p)\nprint('Output CSVs:')\nfor p in sorted(MULTI_TAB.glob('*.csv')):\n    print(p)\nvalidation"),
    ]


def main() -> None:
    write_notebook(NOTEBOOK_DIR / "01_univariate_matrix_profile_visual_story.ipynb", univariate_notebook())
    write_notebook(NOTEBOOK_DIR / "02_multivariate_matrix_profile_visual_story.ipynb", multivariate_notebook())
    print(NOTEBOOK_DIR / "01_univariate_matrix_profile_visual_story.ipynb")
    print(NOTEBOOK_DIR / "02_multivariate_matrix_profile_visual_story.ipynb")


if __name__ == "__main__":
    main()

