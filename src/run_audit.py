"""
Master audit runner — produces the JSON+CSV outputs that Chapter 5 needs.
Run: python -m src.run_audit
"""
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

_TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TOOL_ROOT))
from src.ingestion import build_inventory, load_dataset
from src import metrics
from src import ml_engine
from src import ndmo_mapping

# Edit DATASET_ROOT to point at your dataset directory.
DATASET_ROOT = str(_TOOL_ROOT / "dataset")
RESULTS_DIR = _TOOL_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Fixed audit date reported in Chapter 5 of the dissertation
# (capture: 14 May 2026; audit run: 3 June 2026)
AUDIT_DATE = pd.Timestamp("2026-06-03")


def main():
    print("[1/7] Building inventory...")
    if not Path(DATASET_ROOT).exists():
        raise SystemExit(
            f"Dataset directory not found: {DATASET_ROOT}\n"
            "Set DATASET_ROOT in src/run_audit.py to the folder containing "
            "Investment/, health/, education/, and energy/ sub-folders."
        )
    inv = build_inventory(DATASET_ROOT)
    if "load_ok" not in inv.columns or inv.empty:
        raise SystemExit(
            "No CSV files found under the dataset directory. "
            "Ensure each sector sub-folder contains at least one CSV file."
        )
    ok = inv[inv["load_ok"]].reset_index(drop=True)
    print(f"  loaded {len(ok)} of {len(inv)} files")
    if len(ok) == 0:
        raise SystemExit("All files failed to load — check encoding and CSV format.")

    print("[2/7] Computing per-dataset metrics...")
    per_ds = {}
    rows = []
    if_label_store = {}
    db_label_store = {}
    for _, r in ok.iterrows():
        path = r["file_path"]
        try:
            df, meta = load_dataset(path)
        except Exception as e:
            print(f"  skipped {r['dataset_id']}: {e}")
            continue

        comp_score, _ = metrics.completeness(df)
        cons_score, _ = metrics.consistency(df)
        uniq_score, uniq_meta = metrics.uniqueness(df)
        # timeliness: we lack a stated last_update per file, so use file mtime as proxy
        try:
            mtime = pd.Timestamp(datetime.fromtimestamp(Path(path).stat().st_mtime))
            time_score, time_meta = metrics.timeliness(mtime, AUDIT_DATE, 365)
        except Exception:
            time_score, time_meta = None, {}

        # ML accuracy proxy
        if_res = ml_engine.isolation_forest_anomalies(df, contamination=0.05)
        acc_score = if_res.get("accuracy_proxy") if if_res.get("applicable") else None
        # store labels for agreement
        if if_res.get("applicable"):
            if_label_store[r["dataset_id"]] = if_res["labels"]

        # DBSCAN
        db_res = ml_engine.dbscan_noise(df)
        if db_res.get("applicable"):
            db_label_store[r["dataset_id"]] = db_res["labels"]

        # composite (3 inherent dimensions: completeness, consistency, uniqueness)
        three_dim = metrics.composite({
            "completeness": comp_score,
            "consistency": cons_score,
            "uniqueness": uniq_score,
        })
        # full 4-dim composite where ML & timeliness available
        all_dim_scores = {
            "completeness": comp_score,
            "consistency": cons_score,
            "uniqueness": uniq_score,
            "accuracy": acc_score,
            "timeliness": time_score,
        }
        full_composite = metrics.composite(all_dim_scores)
        drl_band = metrics.to_drl_band(full_composite)

        per_ds[r["dataset_id"]] = {
            "sector": r["sector"],
            "file_name": r["file_name"],
            "completeness": comp_score,
            "consistency": cons_score,
            "uniqueness": uniq_score,
            "accuracy": acc_score,
            "timeliness": time_score,
            "three_dim_composite": three_dim,
            "full_composite": full_composite,
            "drl_band": drl_band,
            "if_applicable": if_res.get("applicable", False),
            "if_anomaly_rate": if_res.get("anomaly_rate"),
            "db_applicable": db_res.get("applicable", False),
            "db_noise_rate": db_res.get("noise_rate"),
            "rows": len(df),
            "cols": len(df.columns),
            "total_cells": df.size,
            "n_nulls": int(df.isna().sum().sum()),
        }
        rows.append({"dataset_id": r["dataset_id"], **per_ds[r["dataset_id"]]})

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(RESULTS_DIR / "dataset_scores.csv", index=False)
    print(f"  wrote dataset_scores.csv with {len(scores_df)} rows")

    # ---- IF parameter sweep ----
    print("[3/7] Isolation Forest contamination sweep...")
    sweep_rows = []
    for ds_id, payload in per_ds.items():
        path = ok[ok["dataset_id"] == ds_id].iloc[0]["file_path"]
        try:
            df, _ = load_dataset(path)
        except Exception:
            continue
        for c in [0.02, 0.05, 0.075, 0.10, 0.15]:
            r = ml_engine.isolation_forest_anomalies(df, contamination=c)
            sweep_rows.append({
                "dataset_id": ds_id,
                "sector": payload["sector"],
                "contamination": c,
                "anomaly_rate": r.get("anomaly_rate"),
                "applicable": r.get("applicable"),
            })
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(RESULTS_DIR / "if_contamination_sweep.csv", index=False)

    # rank stability across contamination
    spearman = {}
    if len(sweep_df) > 0:
        pivot = sweep_df.pivot(index="dataset_id", columns="contamination", values="anomaly_rate")
        pivot = pivot.dropna()
        baseline = 0.05
        for c in [0.02, 0.05, 0.075, 0.10, 0.15]:
            if c == baseline:
                spearman[c] = 1.0
                continue
            if c in pivot.columns and baseline in pivot.columns and len(pivot) > 1:
                spearman[c] = float(pivot[c].rank().corr(pivot[baseline].rank(), method="spearman"))
    print(f"  Spearman ρ vs c=0.05: {spearman}")

    # ---- Cross-algorithm agreement ----
    print("[4/7] Cross-algorithm (IF vs DBSCAN) agreement...")
    agree_rows = []
    for ds_id in if_label_store:
        if ds_id in db_label_store:
            r = ml_engine.isolation_dbscan_agreement(
                if_label_store[ds_id], db_label_store[ds_id]
            )
            if r.get("applicable"):
                r["dataset_id"] = ds_id
                r["sector"] = per_ds[ds_id]["sector"]
                r["kappa_interp"] = ml_engine.interpret_kappa(r.get("kappa"))
                agree_rows.append(r)
    agree_df = pd.DataFrame(agree_rows)
    if len(agree_df) > 0:
        agree_df.to_csv(RESULTS_DIR / "if_dbscan_agreement.csv", index=False)
        print(f"  mean κ = {agree_df['kappa'].dropna().mean():.3f}")

    # ---- KMeans missingness clustering ----
    print("[5/7] KMeans on missingness/quality profile vectors...")
    km_input = scores_df.copy()
    km_input = km_input.merge(
        ok[["dataset_id", "numeric_cols", "categorical_cols", "date_cols", "freetext_cols"]],
        on="dataset_id",
    )
    km_res = ml_engine.kmeans_missingness(km_input)
    if km_res.get("applicable"):
        km_input["cluster"] = km_res["labels"]
        km_input.to_csv(RESULTS_DIR / "kmeans_clusters.csv", index=False)
        with open(RESULTS_DIR / "kmeans_meta.json", "w") as f:
            json.dump({
                "k_selected": km_res["k_selected"],
                "silhouette": km_res["silhouette"],
                "silhouettes": km_res["silhouettes"],
                "inertias": km_res["inertias"],
                "centroids": km_res["centroids"],
                "feature_names": km_res["feature_names"],
            }, f, indent=2)
        print(f"  k* = {km_res['k_selected']}, silhouette = {km_res['silhouette']:.3f}")

    # ---- NDMO gap mapping ----
    print("[6/7] NDMO gap mapping...")
    per_ds_reports = {}
    for ds_id, payload in per_ds.items():
        scores = {
            "completeness": payload["completeness"],
            "consistency": payload["consistency"],
            "uniqueness": payload["uniqueness"],
            "accuracy": payload["accuracy"],
            "timeliness": payload["timeliness"],
        }
        per_ds_reports[ds_id] = {
            "sector": payload["sector"],
            "report": ndmo_mapping.per_dataset_gap_report(scores),
        }
    sector_matrix = ndmo_mapping.sector_gap_matrix(per_ds_reports)
    sector_matrix.to_csv(RESULTS_DIR / "ndmo_sector_matrix.csv", index=False)
    narrative = ndmo_mapping.narrative_gap_register(per_ds_reports)
    if len(narrative) > 0:
        narrative.to_csv(RESULTS_DIR / "ndmo_gap_register.csv", index=False)

    # ---- Headline summary JSON ----
    print("[7/7] Computing headline KPIs...")
    portfolio_cells = scores_df["total_cells"].sum()
    portfolio_nulls = scores_df["n_nulls"].sum()
    portfolio_completeness = float(1 - portfolio_nulls / portfolio_cells) if portfolio_cells else 0

    headline = {
        "audit_date": AUDIT_DATE.isoformat(),
        "n_datasets": int(len(scores_df)),
        "portfolio_total_rows": int(scores_df["rows"].sum()),
        "portfolio_total_cells": int(portfolio_cells),
        "portfolio_completeness_aggregate": portfolio_completeness,
        "mean_per_dataset_completeness": float(scores_df["completeness"].mean()),
        "sd_per_dataset_completeness": float(scores_df["completeness"].std()),
        "datasets_meeting_95pct_completeness": int((scores_df["completeness"] >= 0.95).sum()),
        "mean_per_dataset_consistency": float(scores_df["consistency"].mean()),
        "sd_per_dataset_consistency": float(scores_df["consistency"].std()),
        "datasets_meeting_95pct_consistency": int((scores_df["consistency"] >= 0.95).sum()),
        "mean_per_dataset_uniqueness": float(scores_df["uniqueness"].mean()),
        "datasets_lt_1pct_duplicates": int((scores_df["uniqueness"] >= 0.99).sum()),
        "if_applicable_count": int(scores_df["if_applicable"].sum()),
        "mean_if_anomaly_rate": float(scores_df["if_anomaly_rate"].dropna().mean()),
        "db_applicable_count": int(scores_df["db_applicable"].sum()),
        "mean_db_noise_rate": float(scores_df["db_noise_rate"].dropna().mean()),
        "drl_band_distribution": scores_df["drl_band"].value_counts().to_dict(),
        "if_sweep_spearman": spearman,
    }

    if len(agree_df) > 0:
        headline["mean_if_dbscan_kappa"] = float(agree_df["kappa"].dropna().mean())
        headline["mean_if_dbscan_jaccard"] = float(agree_df["jaccard"].dropna().mean())

    # ------------------------------------------------------------------ #
    # Band sensitivity to composite composition (dissertation section 5.6)
    #
    # The reported composite averages whichever dimensions are available for
    # a given dataset. Two properties make that composition uneven: the
    # timeliness proxy resolves to 1.00 for every file, and accuracy is
    # defined only where Isolation Forest was applicable. The bands are
    # therefore recomputed over restricted dimension sets so the reported
    # Band A share can be quoted as an interval rather than a point estimate.
    # ------------------------------------------------------------------ #
    band_specs = {
        "as_reported_available_dims": None,  # the full_composite column
        "completeness_consistency": ["completeness", "consistency"],
        "completeness_consistency_uniqueness": ["completeness", "consistency", "uniqueness"],
        "excl_timeliness_incl_accuracy": ["completeness", "consistency", "uniqueness", "accuracy"],
    }
    band_rows = []
    for name, dims in band_specs.items():
        series = scores_df["full_composite"] if dims is None else scores_df[dims].mean(axis=1, skipna=True)
        bands = series.apply(metrics.to_drl_band)
        counts = bands.value_counts()
        band_rows.append({
            "specification": name,
            "dimensions": "available" if dims is None else "+".join(dims),
            "n": int(len(series)),
            "band_a": int(counts.get("Band A", 0)),
            "band_b": int(counts.get("Band B", 0)),
            "band_c": int(counts.get("Band C", 0)),
            "pct_band_a": round(float(counts.get("Band A", 0)) / len(series) * 100, 1),
        })
    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(RESULTS_DIR / "band_sensitivity.csv", index=False)

    # per-dataset band under each specification, so movers are inspectable
    movers = scores_df[["dataset_id", "sector", "completeness", "consistency",
                        "uniqueness", "accuracy", "timeliness", "full_composite"]].copy()
    for name, dims in band_specs.items():
        series = scores_df["full_composite"] if dims is None else scores_df[dims].mean(axis=1, skipna=True)
        movers[f"score_{name}"] = series.round(4)
        movers[f"band_{name}"] = series.apply(metrics.to_drl_band)
    movers.to_csv(RESULTS_DIR / "band_sensitivity_per_dataset.csv", index=False)

    headline["band_sensitivity"] = band_rows
    headline["band_a_share_range_pct"] = [
        min(r["pct_band_a"] for r in band_rows),
        max(r["pct_band_a"] for r in band_rows),
    ]

    print("\n=== BAND SENSITIVITY (composite composition) ===")
    print(band_df.to_string(index=False))

    # by sector
    sector_summary = scores_df.groupby("sector").agg(
        n=("dataset_id", "count"),
        completeness_mean=("completeness", "mean"),
        completeness_sd=("completeness", "std"),
        completeness_min=("completeness", "min"),
        completeness_max=("completeness", "max"),
        consistency_mean=("consistency", "mean"),
        consistency_sd=("consistency", "std"),
        uniqueness_mean=("uniqueness", "mean"),
        if_anomaly_mean=("if_anomaly_rate", "mean"),
    ).round(4)
    sector_summary.to_csv(RESULTS_DIR / "sector_summary.csv")
    headline["sector_summary"] = sector_summary.to_dict(orient="index")

    with open(RESULTS_DIR / "headline.json", "w") as f:
        json.dump(headline, f, indent=2, default=str)

    print("\n=== HEADLINE ===")
    print(json.dumps({
        k: v for k, v in headline.items()
        if k not in ("sector_summary", "if_sweep_spearman", "drl_band_distribution")
    }, indent=2, default=str))
    print("\n=== SECTOR SUMMARY ===")
    print(sector_summary)
    print("\n=== DRL BAND DISTRIBUTION ===")
    print(headline["drl_band_distribution"])


if __name__ == "__main__":
    main()
