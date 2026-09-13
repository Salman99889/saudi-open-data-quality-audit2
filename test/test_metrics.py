"""
Unit tests for src/metrics.py — the four data-quality dimensions.

Each metric is a pure function with a mathematically known output, so these
tests use small, hand-constructed DataFrames whose scores can be verified by
inspection. This is the correctness check referenced in the dissertation's
reproducibility section.

Run from the project root:
    pytest                       # run all tests
    pytest -v                    # verbose, one line per test
    pytest --cov=src.metrics     # with coverage
"""
import numpy as np
import pandas as pd
import pytest

from src import metrics


# --------------------------------------------------------------------------- #
# Completeness: 1 - (null cells / total cells)
# --------------------------------------------------------------------------- #
class TestCompleteness:
    def test_no_nulls_scores_one(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        score, per_col = metrics.completeness(df)
        assert score == 1.0
        assert per_col["a"] == 1.0 and per_col["b"] == 1.0

    def test_all_nulls_scores_zero(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        score, _ = metrics.completeness(df)
        assert score == 0.0

    def test_half_missing_scores_half(self):
        # 2 of 4 cells null -> 0.5
        df = pd.DataFrame({"a": [1, np.nan], "b": [np.nan, 4]})
        score, _ = metrics.completeness(df)
        assert score == pytest.approx(0.5)

    def test_per_column_breakdown(self):
        # column a: 1 of 4 null -> 0.75 ; column b: fully populated -> 1.0
        df = pd.DataFrame({"a": [1, 2, 3, np.nan], "b": [1, 2, 3, 4]})
        _, per_col = metrics.completeness(df)
        assert per_col["a"] == pytest.approx(0.75)
        assert per_col["b"] == 1.0

    def test_empty_frame_returns_zero(self):
        score, per_col = metrics.completeness(pd.DataFrame())
        assert score == 0.0
        assert per_col == {}


# --------------------------------------------------------------------------- #
# Uniqueness: 1 - (duplicate rows / total rows)
# --------------------------------------------------------------------------- #
class TestUniqueness:
    def test_all_unique_scores_one(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        score, meta = metrics.uniqueness(df)
        assert score == 1.0
        assert meta["duplicate_rows"] == 0

    def test_one_duplicate(self):
        # 4 rows, last duplicates the first -> 1 dup / 4 = 0.75
        df = pd.DataFrame({"a": [1, 2, 3, 1], "b": [4, 5, 6, 4]})
        score, meta = metrics.uniqueness(df)
        assert score == pytest.approx(0.75)
        assert meta["duplicate_rows"] == 1
        assert meta["total_rows"] == 4

    def test_all_duplicates(self):
        # 3 identical rows -> 2 counted as duplicates -> 1 - 2/3
        df = pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
        score, meta = metrics.uniqueness(df)
        assert score == pytest.approx(1 - 2 / 3)
        assert meta["duplicate_rows"] == 2

    def test_empty_frame_returns_zero(self):
        score, meta = metrics.uniqueness(pd.DataFrame())
        assert score == 0.0
        assert meta["total_rows"] == 0


# --------------------------------------------------------------------------- #
# Consistency: proportion of columns conforming to their inferred type
# at >= parse_threshold (default 0.95)
# --------------------------------------------------------------------------- #
class TestConsistency:
    def test_clean_numeric_column_passes(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        score, per_col = metrics.consistency(df)
        assert score == 1.0
        assert per_col["a"] == 1.0

    def test_all_columns_clean_scores_one(self):
        df = pd.DataFrame({"num": [1, 2, 3], "cat": ["red", "blue", "red"]})
        score, _ = metrics.consistency(df)
        assert score == 1.0

    def test_column_dominated_by_sentinels_fails(self):
        # A column fails consistency only when it is neither parseable as a
        # type NOR "clean" categorical -- i.e. when sentinel junk dominates.
        # Here 3 of 4 values are "N/A" sentinels -> clean_rate 0.25, well
        # below the 0.95 threshold, so the column (and thus the score) fails.
        df = pd.DataFrame({"a": ["real", "N/A", "N/A", "N/A"]})
        score, per_col = metrics.consistency(df)
        assert per_col["a"] < 0.95
        assert score == 0.0  # the only column fails

    def test_numeric_column_with_minor_text_still_passes(self):
        # Documents a deliberate leniency: a numeric column contaminated with
        # a little free text still passes, because the categorical fallback
        # (no sentinels present) scores it as "clean". To fail consistency a
        # column needs sentinel junk, not just type contamination.
        vals = list(range(18)) + ["oops", "bad"]  # 90% numeric, 0% sentinels
        df = pd.DataFrame({"a": vals})
        _, per_col = metrics.consistency(df)
        assert per_col["a"] == 1.0

    def test_sentinel_values_lower_categorical_conformance(self):
        # half the categorical values are "N/A" sentinels
        df = pd.DataFrame({"cat": ["a", "N/A", "b", "N/A"]})
        _, per_col = metrics.consistency(df)
        assert per_col["cat"] == pytest.approx(0.5)

    def test_empty_frame_returns_zero(self):
        score, per_col = metrics.consistency(pd.DataFrame())
        assert score == 0.0
        assert per_col == {}


# --------------------------------------------------------------------------- #
# Timeliness: 1 - min(1, days_lag / (2 * stated_frequency_days))
# --------------------------------------------------------------------------- #
class TestTimeliness:
    def test_updated_today_scores_one(self):
        audit = pd.Timestamp("2026-05-14")
        score, meta = metrics.timeliness(audit, audit, 365)
        assert score == 1.0
        assert meta["days_lag"] == 0

    def test_lag_equal_to_one_frequency_scores_half(self):
        audit = pd.Timestamp("2026-05-14")
        last = audit - pd.Timedelta(days=365)
        score, _ = metrics.timeliness(last, audit, 365)
        assert score == pytest.approx(0.5)

    def test_lag_beyond_two_frequencies_floors_at_zero(self):
        audit = pd.Timestamp("2026-05-14")
        last = audit - pd.Timedelta(days=3 * 365)
        score, meta = metrics.timeliness(last, audit, 365)
        assert score == 0.0
        assert meta["stale"] is True

    def test_none_last_update_scores_zero(self):
        score, meta = metrics.timeliness(None, pd.Timestamp("2026-05-14"))
        assert score == 0.0
        assert meta["stale"] is True

    def test_default_frequency_is_365(self):
        audit = pd.Timestamp("2026-05-14")
        last = audit - pd.Timedelta(days=100)
        _, meta = metrics.timeliness(last, audit)  # no frequency passed
        assert meta["stated_frequency_days"] == 365


# --------------------------------------------------------------------------- #
# Composite: weighted mean of available (non-None) scores
# --------------------------------------------------------------------------- #
class TestComposite:
    def test_simple_average(self):
        assert metrics.composite({"a": 1.0, "b": 0.0}) == pytest.approx(0.5)

    def test_ignores_none_values(self):
        # None should be dropped, average of the remaining two
        assert metrics.composite({"a": 1.0, "b": None, "c": 0.5}) == pytest.approx(0.75)

    def test_all_none_returns_zero(self):
        assert metrics.composite({"a": None, "b": None}) == 0.0

    def test_custom_weights(self):
        # weight a=3, b=1 -> (0.8*3 + 0.4*1) / 4 = 0.7
        score = metrics.composite({"a": 0.8, "b": 0.4}, weights={"a": 3, "b": 1})
        assert score == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# DRL band mapping (Lawrence, 2017): A >= 0.85, B >= 0.70, else C
# --------------------------------------------------------------------------- #
class TestDrlBand:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (1.00, "Band A"),
            (0.85, "Band A"),   # boundary is inclusive
            (0.84, "Band B"),
            (0.70, "Band B"),   # boundary is inclusive
            (0.69, "Band C"),
            (0.00, "Band C"),
        ],
    )
    def test_band_boundaries(self, value, expected):
        assert metrics.to_drl_band(value) == expected
