"""
SDAIA NDMO Domain 3 mapping module.
Converts metric outputs to per-specification compliance verdicts:
pass / borderline / fail per §4.6 of the dissertation.
"""
import pandas as pd

# Domain 3 specifications mapped to computed dimensions and thresholds.
NDMO_MAPPING = {
    "3.1": {
        "name": "Accuracy",
        "dimension": "accuracy",  # 1 - IF anomaly rate
        "threshold": 0.95,
        "domain": 3,
    },
    "3.2": {
        "name": "Completeness",
        "dimension": "completeness",
        "threshold": 0.95,
        "domain": 3,
    },
    "3.4": {
        "name": "Consistency",
        "dimension": "consistency",
        "threshold": 0.95,
        "domain": 3,
    },
    "3.5": {
        "name": "Validity",
        "dimension": "consistency",  # operationalised via consistency proxy
        "threshold": 0.95,
        "domain": 3,
    },
    "3.7": {
        "name": "Currency / Timeliness",
        "dimension": "timeliness",
        "threshold": 0.80,
        "domain": 3,
    },
    "3.13": {
        "name": "Uniqueness",
        "dimension": "uniqueness",
        "threshold": 0.99,
        "domain": 3,
    },
}

WINDOW_PP = 0.05  # 5 percentage-point band for "borderline"


def verdict_for(value: float | None, threshold: float, window: float = WINDOW_PP) -> str:
    """Three-band verdict.

    pass: value >= threshold
    borderline: threshold - window <= value < threshold
    fail: value < threshold - window
    """
    if value is None:
        return "n/a"
    if value >= threshold:
        return "pass"
    if value < threshold - window:
        return "fail"
    return "borderline"


def per_dataset_gap_report(dataset_scores: dict) -> dict:
    """
    dataset_scores must have keys: completeness, consistency, uniqueness,
    accuracy, timeliness — any may be None.
    Returns {spec_id: {value, threshold, verdict, ndmo_name}}
    """
    report = {}
    for spec_id, info in NDMO_MAPPING.items():
        val = dataset_scores.get(info["dimension"])
        verdict = verdict_for(val, info["threshold"])
        report[spec_id] = {
            "ndmo_name": info["name"],
            "dimension": info["dimension"],
            "value": val,
            "threshold": info["threshold"],
            "verdict": verdict,
        }
    return report


def sector_gap_matrix(per_dataset_reports: dict) -> pd.DataFrame:
    """
    Roll dataset-level reports up to sector-level.
    per_dataset_reports: {dataset_id: {sector: ..., report: {spec: {...}}}}
    Returns: DataFrame indexed by spec_id, columns are sectors,
    with values = majority verdict in that sector (counts: pass/borderline/fail).
    """
    spec_ids = list(NDMO_MAPPING.keys())
    # group by sector
    by_sector = {}
    for ds_id, payload in per_dataset_reports.items():
        sector = payload["sector"]
        by_sector.setdefault(sector, []).append(payload["report"])
    sectors = sorted(by_sector.keys())
    rows = []
    for spec in spec_ids:
        row = {"specification": spec, "name": NDMO_MAPPING[spec]["name"]}
        for sector in sectors:
            verdicts = [r[spec]["verdict"] for r in by_sector[sector]]
            valid = [v for v in verdicts if v != "n/a"]
            if not valid:
                row[sector] = "n/a"
                row[f"{sector}_n"] = 0
                row[f"{sector}_pct_pass"] = None
                continue
            pass_n = sum(1 for v in valid if v == "pass")
            fail_n = sum(1 for v in valid if v == "fail")
            borderline_n = len(valid) - pass_n - fail_n
            if pass_n >= fail_n and pass_n >= borderline_n:
                majority = "pass"
            elif fail_n >= pass_n and fail_n >= borderline_n:
                majority = "fail"
            else:
                majority = "borderline"
            row[sector] = majority
            row[f"{sector}_n"] = len(valid)
            row[f"{sector}_pct_pass"] = round(100 * pass_n / len(valid), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def narrative_gap_register(per_dataset_reports: dict) -> pd.DataFrame:
    """
    Long-form gap register: one row per (dataset, specification) where
    verdict != 'pass'. This is what populates Table 5.6 of the dissertation.
    """
    rows = []
    for ds_id, payload in per_dataset_reports.items():
        for spec, r in payload["report"].items():
            if r["verdict"] in ("fail", "borderline"):
                rows.append({
                    "dataset_id": ds_id,
                    "sector": payload["sector"],
                    "specification": spec,
                    "ndmo_name": r["ndmo_name"],
                    "dimension": r["dimension"],
                    "observed": r["value"],
                    "threshold": r["threshold"],
                    "verdict": r["verdict"],
                })
    return pd.DataFrame(rows).sort_values(
        ["verdict", "sector", "specification"], ascending=[True, True, True]
    ) if rows else pd.DataFrame()
