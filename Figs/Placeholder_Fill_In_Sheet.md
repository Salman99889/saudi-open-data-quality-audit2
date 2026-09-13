# Placeholder fill-in sheet — Saudi OGD AI Readiness dissertation

**Tool:** Imtethal Audit Tool (v1.0)
**Audit date:** 3 June 2026 (capture: 14 May 2026)
**Dataset snapshot:** 31 CSV resources across 4 sectors (Investment 13, Health 9, Education 6, Energy 3)

This sheet lists every value the tool produces that Chapter 5 and Chapter 6 cite, so the numbers in the write-up can be traced back to the exact `results/*.csv` and `results/*.json` files under this project. All values below are computed by `python -m src.run_audit` over the 31 distinct resources under `dataset/` (Education duplicate removed at capture; §5.1).

---

## §5.1 — Dataset profile summary (Table 5.1)

| KPI | Value | Threshold | Verdict | NDMO Spec |
|---|---|---|---|---|
| Datasets audited | 31 | — | — | — |
| Total rows ingested | 56,758 | — | — | — |
| Total cells inspected | 670,726 | — | — | — |
| Portfolio completeness (cells-weighted) | 90.64% | ≥ 95% | Fail | 3.2 |
| Mean per-dataset completeness | 90.48% (SD 22.42) | ≥ 95% | Borderline | 3.2 |
| Datasets meeting 95% completeness | 23 of 31 (74.2%) | ≥ 95% | Fail | 3.2 |
| Mean per-dataset consistency | 88.66% (SD 28.78) | ≥ 95% | Borderline | 3.4 |
| Datasets meeting 95% consistency | 26 of 31 (83.9%) | ≥ 90% | Borderline | 3.4 |
| Mean per-dataset uniqueness | 96.41% | ≥ 99% | Borderline | 3.13 |
| Datasets with < 1% duplicates | 29 of 31 (93.5%) | ≥ 95% | Borderline | 3.13 |
| Mean IF anomaly rate (14 applicable datasets) | 4.95% | ≤ 5% | Pass | 3.1 |
| Mean Cohen's κ (IF vs DBSCAN) | 0.521 (moderate) | — | — | — |
| Portfolio DRL distribution (Band A / B / C) | 27 / 3 / 1 (87.1 / 9.7 / 3.2%) | — | — | — |

Source: `results/headline.json`.

Integrity note: INV-13 (trademarks registered in 2023) alone contributes 28,038 rows and 420,570 cells; HEA-01 (Births at KSA Hospitals by Type of Delivery) is empty. Two Education downloads were found byte-identical at capture (macOS filename `... 2.csv` convention) — the duplicate is preserved in `removed_duplicate/` and excluded from the audit set. HEA-07 and HEA-08 are not byte-identical but hold the same 5 × 6 table under two formatting conventions, one parsing as five numeric columns and the other as none; both are retained as distinct published resources.

---

## §5.2 — Completeness by sector (Table 5.2)

| Sector | n | Mean | SD | Min | Max | % ≥ 95% |
|---|---|---|---|---|---|---|
| Education | 6 | 95.82% | 10.20 | 75.00% | 100.00% | 83.3% |
| Energy | 3 | 99.73% | 0.47 | 99.18% | 100.00% | 100.0% |
| Investment | 13 | 91.05% | 18.76 | 34.34% | 100.00% | 69.2% |
| Health | 9 | 83.03% | 34.27 | 0.00% | 100.00% | 66.7% |
| **Portfolio** | **31** | **90.48%** | **22.42** | **0.00%** | **100.00%** | **74.2%** |

Source: `results/sector_summary.csv`.

The four worst performers on completeness:

- **HEA-01** *(Births at KSA Hospitals by Type of Delivery)* — 0.00% (empty file)
- **INV-03** *(Establishments and Workers in Private Sector 2023)* — 34.34%
- **HEA-06** *(Hospitals and Beds in All Health Sectors)* — 56.35%
- **INV-02** *(Contact Person on GOV)* — 75.00%

Energy is strongest on mean completeness (99.73%, n = 3); Health is weakest (83.03%), pulled down by HEA-01 (empty) and HEA-06 (structural). This reverses the isomorphism prediction discussed in §6.2.

---

## §5.3 — Consistency by sector (Table 5.3)

| Sector | n | Mean | SD | % ≥ 95% |
|---|---|---|---|---|
| Education | 6 | 100.00% | 0.00 | 100.0% |
| Energy | 3 | 100.00% | 0.00 | 100.0% |
| Health | 9 | 88.89% | 33.33 | (see §5.3 prose) |
| Investment | 13 | 80.64% | 34.11 | 69.2% |
| **Portfolio** | **31** | **88.66%** | **28.78** | **83.9%** |

Source: `results/sector_summary.csv`.

The four worst performers on consistency:

- **INV-10** *(Total Debt By Sector Main Market)* — 0.00%
- **INV-06** *(Local and Foreign Investments of CMA)* — 25.00%
- **INV-05** *(Investors in FinTech ExPermit Companies)* — 50.00%
- **INV-13** *(trademarks registered in 2023)* — 73.33%

Uniqueness: two datasets fall below the 99% threshold — INV-10 (an 11.11% duplicate rate inside a 27-row wide-format file) and HEA-01 (undefined; returns zero because the file is empty). Both are recorded against Specification 3.13.

---

## §5.4 — Accuracy results (Isolation Forest)

| KPI | Value |
|---|---|
| Mean per-dataset IF anomaly rate | 4.95% |
| Datasets where IF was applicable | 14 of 31 (others below the min-sample or numeric-column threshold) |
| Mean DBSCAN noise rate | 5.10% |
| Mean Cohen's κ (IF vs DBSCAN) | 0.521 ("moderate" per Landis & Koch, over the 13 datasets on which κ is defined; EDU-03 is undefined because both algorithms returned zero anomalies) |
| Mean Jaccard (IF vs DBSCAN) | 0.402 |
| Highest sectoral κ | Education 0.845 (n = 5 applicable, 4 defined) |
| Lowest sectoral κ | Investment 0.327 (n = 5); Health represented by a single dataset (κ = 0.00) |

Source: `results/headline.json`, `results/if_dbscan_agreement.csv`.

---

## §5.5 — Timeliness

Timeliness is computed as 1 − min(1, days-since-update / (2 × stated-frequency)), defaulting to 365 days where the portal API does not expose a cadence. Under this proxy all 31 files score at or near 1.00, so the dimension collapses to a binary staleness flag rather than a continuous measure — flagged in §5.5 and §6.7 as a methodological limitation, not a substantive finding.

The secondary staleness flag (files whose last-update timestamp predates the audit by more than twelve months) marks 2 of 31 (6.5%) as stale: EDU-04 (~12 months at capture) and ENE-01 (~24 months).

---

## §5.6 — Composite readiness scoring

| Metric | Value |
|---|---|
| Portfolio mean composite | 0.935 |
| Median composite | 0.989 |
| DRL band distribution | Band A: 27 (87.1%); Band B: 3 (9.7%); Band C: 1 (3.2%) |
| Band C dataset | HEA-01 (empty) |
| Band B datasets | INV-03 (practice-derived missingness), INV-06, INV-10 (consistency failures) |
| Band A share under alternative dimension sets | 80.6% – 87.1% |
| Spearman ρ vs equal-weighted composite (across alternative weightings) | > 0.88 |

Source: `results/headline.json` (`band_sensitivity`, `band_a_share_range_pct`), `results/dataset_scores.csv`.

The four sensitivity specifications the tool reports and §5.6 cites:

| Specification | Dimensions | Band A | Band B | Band C | % Band A |
|---|---|---|---|---|---|
| as_reported_available_dims (baseline) | available | 27 | 3 | 1 | 87.1 |
| completeness_consistency | completeness + consistency | 25 | 2 | 4 | 80.6 |
| completeness_consistency_uniqueness | + uniqueness | 26 | 3 | 2 | 83.9 |
| excl_timeliness_incl_accuracy | + accuracy | 26 | 3 | 2 | 83.9 |

---

## §5.7 — ML algorithm performance comparison (RQ4)

| Metric | Value |
|---|---|
| Mean κ (IF vs DBSCAN) | 0.521; range 0.00 – 1.00 across the 14 applicable datasets (13 defined) |
| Mean Jaccard | 0.402 |
| Highest sectoral κ | Education 0.845 |
| Lowest sectoral κ | Investment 0.327 (Health single-dataset κ = 0.00) |
| IF mean runtime per dataset | sub-second on the reference machine (Intel x86-64, 8 GB RAM) |
| k selected for k-means | k = 4 (non-trivial silhouette maximum; k = 2 silhouette 0.836 is degenerate, isolating only HEA-01) |
| Silhouette at k* = 4 | 0.733 |
| Cluster interpretation | Task-ready core (n = 23); incomplete-but-conformant (n = 4: INV-02, INV-03, HEA-06, EDU-01); complete-but-inconsistent (n = 3: INV-05, INV-06, INV-10); empty singleton (n = 1: HEA-01) |

Source: `results/kmeans_meta.json`, `results/kmeans_clusters.csv`, `results/if_dbscan_agreement.csv`.

---

## §5.8 — SDAIA NDMO Domain 3 gap report (Figure 5.9)

Percentage of datasets in each sector passing each specification (n applicable in brackets):

| NDMO Spec | Education | Energy | Health | Investment |
|---|---|---|---|---|
| 3.1 Accuracy | 20% (n=5) | 0% (n=3) | 0% (n=1) | 20% (n=5) |
| 3.2 Completeness | 83% (n=6) | 100% (n=3) | 67% (n=9) | 69% (n=13) |
| 3.4 Consistency | 100% (n=6) | 100% (n=3) | 89% (n=9) | 69% (n=13) |
| 3.5 Validity | 100% (n=6) | 100% (n=3) | 89% (n=9) | 69% (n=13) |
| 3.7 Currency / Timeliness | 100% (n=6) | 100% (n=3) | 100% (n=9) | 100% (n=13) (proxy only) |
| 3.13 Uniqueness | 100% (n=6) | 100% (n=3) | 100% (n=9) | 92% (n=13) |

Source: `results/ndmo_sector_matrix.csv`.

Most consistent fail: Specification 3.1 (Accuracy). Most consistent pass, subject to the timeliness caveat: Specification 3.7 (Currency).

---

## Appendix B — Isolation Forest sensitivity (from `results/if_contamination_sweep.csv`)

| Contamination | Spearman ρ vs c = 0.05 baseline |
|---|---|
| 0.02 | 0.802 |
| 0.05 (baseline) | 1.000 |
| 0.075 | 0.978 |
| 0.10 | 0.776 |
| 0.15 | 0.824 |

Rank ordering is preserved (ρ ≥ 0.78) across all contamination levels, substantiating the §5.4 rank-stability claim.

---

## Placeholders to fill in manually (project-specific)

Not produced by the tool; filled in by the author:

- Title page — student name (Salman Almutairi), student number (P2920565), supervisor (Abdulghani Al-Yamani)
- Appendix A — Gantt chart dates and risk register status
- Appendix F — GitHub commit hash at submission
- Appendix G — DMU ethics approval reference and date

---

## Figures generated by the tool

All figures are saved as PNG under the `Figs/` folder of this project:

- `fig_4_1_architecture.png` → Figure 4.1
- `fig_5_1_completeness_bar.png` → Figure 5.1
- `fig_5_2_sector_boxplots.png` → Figure 5.2 (panel of four)
- `fig_5_3_pca_clusters.png` → Figure 5.3 (Isolation Forest anomaly rate per dataset)
- `fig_5_4_composite_hist.png` → Figure 5.4 (Cohen's κ distribution)
- `fig_5_6_anomaly_bars.png` → Figure 5.6 (PCA projection with k-means clusters)
- `fig_5_8_kappa_dist.png` → Figure 5.8 (k-means cluster centroids)
- `fig_5_9_elbow_silhouette.png` → Figure 5.7 (elbow / silhouette)
- `fig_5_10_centroids.png` → Figure 5.10 (sector readiness radar)
- `fig_5_11_ndmo_matrix.png` → Figure 5.9 (NDMO compliance matrix)
- `fig_5_12_radar.png` → alternate sector radar (not used in submission)
- `fig_appC_contamination_sweep.png` → Appendix B contamination sweep heatmap
