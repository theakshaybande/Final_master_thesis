from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.final_motif_evaluation import (  # noqa: E402
    EvaluationCase,
    benjamini_hochberg,
    build_locomotif_occurrences,
    compute_future_outcome,
    coverage,
    eligible_baseline_anchors,
    interval_iou,
    interval_union,
    redundancy_fraction,
    sample_random_baseline,
    validate_controlled_slice,
)


def price_frame(n: int = 8) -> pd.DataFrame:
    close = np.arange(100.0, 100.0 + n)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "regime_label": ["low_vol", "low_vol", "high_vol", "high_vol", "low_vol", "low_vol", "high_vol", "high_vol"][:n],
        }
    )
    df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    df["close_z"] = (df["close"] - df["close"].mean()) / df["close"].std(ddof=0)
    return df


class FinalMotifEvaluationTests(unittest.TestCase):
    def test_interval_union_merges_overlaps_and_touches(self) -> None:
        self.assertEqual(interval_union([(5, 7), (1, 3), (2, 4), (4, 5)]), [(1, 7)])

    def test_coverage_uses_union_mass(self) -> None:
        self.assertEqual(coverage([(0, 3), (2, 5)], eligible_observations=10), 0.5)

    def test_redundancy_fraction_detects_duplicate_coverage(self) -> None:
        value = redundancy_fraction([(0, 3), (2, 5)])
        self.assertTrue(math.isclose(value, 1.0 - 5.0 / 6.0))

    def test_interval_iou(self) -> None:
        self.assertTrue(math.isclose(interval_iou((0, 4), (2, 6)), 2.0 / 6.0))

    def test_forward_return_calculation(self) -> None:
        df = price_frame()
        outcome = compute_future_outcome(df, anchor_idx=2, horizon_bars=2)
        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertTrue(math.isclose(outcome["simple_forward_return"], 104.0 / 102.0 - 1.0))
        self.assertTrue(math.isclose(outcome["log_forward_return"], math.log(104.0 / 102.0)))

    def test_no_lookahead_event_anchoring_starts_after_motif_end(self) -> None:
        df = price_frame()
        outcome = compute_future_outcome(df, anchor_idx=2, horizon_bars=1)
        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertTrue(math.isclose(outcome["simple_forward_return"], 103.0 / 102.0 - 1.0))

    def test_end_of_sample_exclusion(self) -> None:
        df = price_frame()
        self.assertIsNone(compute_future_outcome(df, anchor_idx=6, horizon_bars=2))

    def test_deterministic_random_baseline_with_seed(self) -> None:
        df = price_frame()
        first = sample_random_baseline(df, horizon_bars=1, n_events=3, repetitions=5, seed=42, motif_intervals=[(1, 3)])
        second = sample_random_baseline(df, horizon_bars=1, n_events=3, repetitions=5, seed=42, motif_intervals=[(1, 3)])
        pd.testing.assert_frame_equal(first, second)

    def test_regime_matched_baseline_filters_anchors(self) -> None:
        df = price_frame()
        anchors = eligible_baseline_anchors(df, horizon_bars=1, motif_intervals=[], regime_label="high_vol")
        self.assertTrue(set(anchors.tolist()).issubset({2, 3, 6}))

    def test_bh_correction_behaviour(self) -> None:
        q = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
        self.assertTrue(math.isclose(q[0], 0.03))
        self.assertTrue(math.isclose(q[1], 0.04))
        self.assertTrue(math.isclose(q[2], 0.04))
        self.assertTrue(math.isnan(q[3]))

    def test_wrong_controlled_slice_row_count_raises(self) -> None:
        df = price_frame(5)
        expected = {
            "rows": 6,
            "start": "2020-01-01 00:00:00+00:00",
            "end": "2020-01-01 04:00:00+00:00",
        }
        with self.assertRaisesRegex(ValueError, "expected 6 rows, observed 5"):
            validate_controlled_slice("synthetic", df, expected)

    def test_wrong_controlled_slice_timestamp_range_raises(self) -> None:
        df = price_frame(5)
        expected = {
            "rows": 5,
            "start": "2020-01-02 00:00:00+00:00",
            "end": "2020-01-01 04:00:00+00:00",
        }
        with self.assertRaisesRegex(ValueError, "expected start"):
            validate_controlled_slice("synthetic", df, expected)

    def test_out_of_range_locomotif_interval_raises(self) -> None:
        df = price_frame(5)
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "locomotif_raw.txt"
            raw_path.write_text("# stdout\n\n# stderr\n\n# repr(motif_sets)\n[((1, 3), [(1, 3), (4, 7)])]\n", encoding="utf-8")
            case = EvaluationCase(
                slice_id="synthetic",
                asset="BTCUSDT",
                frequency="1h",
                regime_label="agnostic",
                slice_path=Path("unused.csv"),
                primary_mp_window=2,
                locomotif_raw_path=raw_path,
            )
            with self.assertRaisesRegex(ValueError, "LoCoMotif interval mismatch"):
                build_locomotif_occurrences(case, df, invalid_interval_policy="error")


if __name__ == "__main__":
    unittest.main()
