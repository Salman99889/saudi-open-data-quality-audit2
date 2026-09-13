"""
Four data-quality dimensions, each as a pure function returning
(per-dataset score in [0,1], per-column breakdown dict).
Implements the formulas of §3.4 of the dissertation.
"""
import numpy as np
import pandas as pd


# ---------- Completeness ----------
def completeness(df: pd.DataFrame) -> tuple[float, dict]:
    """1 - (sum null cells / total cells), with per-column breakdown."""
    if df.size == 0:
        return 0.0, {}
    per_col = {}
    for col in df.columns:
        per_col[col] = float(1 - df[col].isna().mean())
    overall = float(1 - df.isna().sum().sum() / df.size)
    return overall, per_col


# ---------- Uniqueness (a sub-component of consistency) ----------
def uniqueness(df: pd.DataFrame) -> tuple[float, dict]:
    """1 - (duplicate rows / total rows)."""
    if len(df) == 0:
        return 0.0, {"duplicate_rows": 0, "total_rows": 0}
    n_dup = int(df.duplicated().sum())
    n = len(df)
    return float(1 - n_dup / n), {"duplicate_rows": n_dup, "total_rows": n}


# ---------- Consistency ----------
def _try_parse_numeric(s: pd.Series) -> float:
    """Share of non-null values that parse as numeric."""
    non_null = s.dropna()
    if len(non_null) == 0:
        return 1.0
    parsed = pd.to_numeric(non_null, errors="coerce")
    return float(parsed.notna().mean())


def _try_parse_date(s: pd.Series) -> float:
    """Share of non-null values that parse as date."""
    non_null = s.dropna().astype(str)
    if len(non_null) == 0:
        return 1.0
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def consistency(df: pd.DataFrame, parse_threshold: float = 0.95) -> tuple[float, dict]:
    """
    Per the draft §3.4: share of columns whose values parse uniformly to
    their inferred type at the parse_threshold rate.
    A column "passes" if its conformance >= parse_threshold; the overall
    score is the proportion of columns that pass.
    """
    if df.shape[1] == 0:
        return 0.0, {}
    per_col = {}
    for col in df.columns:
        s = df[col]
        # Already numeric?
        if pd.api.types.is_numeric_dtype(s):
            conformance = 1.0
        else:
            num_rate = _try_parse_numeric(s)
            date_rate = _try_parse_date(s)
            best_rate = max(num_rate, date_rate)
            non_null = s.dropna()
            if len(non_null) == 0:
                conformance = 0.0
            else:
                # categorical baseline: count "clean" categorical values (no sentinels)
                str_vals = non_null.astype(str)
                sentinels = ["N/A", "n/a", "NA", "-", "--", "غير متاح", "Not Available", "?", ""]
                clean_rate = float((~str_vals.isin(sentinels)).mean())
                conformance = max(best_rate, clean_rate)
        per_col[col] = float(conformance)
    # Binary aggregation: column passes iff conformance >= threshold
    pass_count = sum(1 for v in per_col.values() if v >= parse_threshold)
    overall = float(pass_count / len(per_col))
    return overall, per_col


# ---------- Timeliness ----------
def timeliness(
    last_update: pd.Timestamp | None,
    audit_date: pd.Timestamp,
    stated_frequency_days: int | None = None,
) -> tuple[float, dict]:
    """
    1 - min(1, days-since-update / 2*stated-frequency).
    If frequency is None, default to 365 days (Vetrò et al. 2016 heuristic).
    """
    if last_update is None:
        return 0.0, {"days_lag": None, "stale": True}
    if stated_frequency_days is None:
        stated_frequency_days = 365
    days_lag = max(0, (audit_date - last_update).days)
    score = float(max(0.0, 1.0 - min(1.0, days_lag / (2 * stated_frequency_days))))
    return score, {
        "days_lag": days_lag,
        "stated_frequency_days": stated_frequency_days,
        "stale": days_lag > 2 * stated_frequency_days,
    }


# ---------- Composite ----------
def composite(scores: dict, weights: dict | None = None) -> float:
    """Weighted aggregate of available scores."""
    valid = {k: v for k, v in scores.items() if v is not None}
    if not valid:
        return 0.0
    if weights is None:
        weights = {k: 1.0 for k in valid}
    w_sum = sum(weights.get(k, 0) for k in valid)
    if w_sum == 0:
        return 0.0
    return float(sum(valid[k] * weights.get(k, 0) for k in valid) / w_sum)


def to_drl_band(composite_score: float) -> str:
    """Map composite score to Lawrence (2017) Data Readiness Level band."""
    if composite_score >= 0.85:
        return "Band A"
    elif composite_score >= 0.70:
        return "Band B"
    else:
        return "Band C"
