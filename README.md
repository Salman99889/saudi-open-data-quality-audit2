# Imtethal Audit Tool


## Pipeline

```
Ingestion -> Profiling -> DQ Metrics -> ML Engine -> NDMO Gap Mapping -> Dashboard
```

All thresholds and hyperparameters are module-level constants in `src/metrics.py`,
`src/ml_engine.py` and `src/ndmo_mapping.py`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.run_audit          # writes results/
python -m src.generate_figures   # writes figures/
streamlit run src/dashboard/streamlit_app.py
```

## Tests

```bash
pytest                              # 30 tests
pytest --cov=src.metrics            # 95% line coverage of src/metrics.py
```

`test/test_metrics.py` covers the four dimensional metrics, the uniqueness sub-metric,
the composite aggregator and the DRL band mapping, including boundary conditions
(empty frame, all-null, all-duplicate, sentinel-contaminated columns, band edges).

## Layout

| Path | Contents |
|---|---|
| `src/` | pipeline modules and the Streamlit dashboard sub-package |
| `test/` | pytest suite for the metrics module |
| `dataset/` | the 31 audited CSVs, by sector |
| `results/` | pipeline outputs (per-dataset scores, headline KPIs, agreement, clusters, NDMO gap register) |
| `figures/` | generated figures (Figure 4.1, 5.1-5.10, C.1) |
| `tables/` | machine-readable exports of the dissertation tables |
| `checksums.sha256` | SHA-256 digests of the captured datasets |

## Outputs

- `dataset_scores.csv` — per-dataset scores on all dimensions
- `sector_summary.csv` — per-sector aggregates
- `headline.json` — headline KPIs (Table 5.1)
- `if_contamination_sweep.csv` — Isolation Forest sensitivity sweep (Appendix C)
- `if_dbscan_agreement.csv` — Cohen's kappa per dataset (RQ4)
- `kmeans_clusters.csv`, `kmeans_meta.json` — clustering output
- `ndmo_sector_matrix.csv`, `ndmo_gap_register.csv` — NDMO Domain 3 compliance (RQ3)

## Algorithms

- **Isolation Forest** (Liu, Ting and Zhou, 2008, 2012) — primary anomaly engine,
  contamination 0.05, 100 estimators, swept 0.02-0.15.
- **DBSCAN** (Ester et al., 1996) — density-based cross-check, eps from the
  90th-percentile k-distance graph, minPts = max(2d, 4).
- **k-means** (MacQueen, 1967) — partitions datasets on their quality-profile vectors;
  k = 4 selected as the silhouette-maximising non-trivial partition.

Random seed fixed at 42 throughout; package versions pinned in `requirements.txt`.

## Licence

MIT (code). Underlying data: Open Data Commons Attribution Licence, per
open.data.gov.sa terms.
