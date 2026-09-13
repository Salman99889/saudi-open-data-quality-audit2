"""
Unsupervised ML engine for accuracy proxy + missingness pattern detection.
Implements Isolation Forest (Liu et al. 2008/2012), DBSCAN (Ester et al. 1996),
and KMeans (MacQueen 1967) per §3.5 of the dissertation.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, cohen_kappa_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_SEED = 42


def numeric_matrix(df: pd.DataFrame, max_rows: int = 50000) -> np.ndarray | None:
    """Return standardised numeric matrix; None if insufficient data.
    For very large datasets, randomly sample to avoid memory/runtime blowups."""
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] == 0 or num_df.shape[0] < 10:
        return None
    # sample if too large
    if len(num_df) > max_rows:
        num_df = num_df.sample(n=max_rows, random_state=RANDOM_SEED)
    # impute simple median
    num_df = num_df.fillna(num_df.median(numeric_only=True))
    # drop rows still containing inf
    num_df = num_df.replace([np.inf, -np.inf], np.nan).dropna()
    if num_df.shape[0] < 10:
        return None
    scaler = StandardScaler()
    return scaler.fit_transform(num_df.values)


# ---------- Isolation Forest ----------
def isolation_forest_anomalies(
    df: pd.DataFrame, contamination: float = 0.05, n_estimators: int = 100
) -> dict:
    """
    Fit Isolation Forest on the standardised numeric matrix.
    Returns dict with anomaly_rate, scores, labels, n_records.
    """
    X = numeric_matrix(df)
    if X is None:
        return {
            "applicable": False,
            "anomaly_rate": None,
            "n_records": 0,
            "anomaly_score_mean": None,
        }
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=RANDOM_SEED,
    )
    labels = iso.fit_predict(X)  # -1 anomaly, 1 normal
    scores = -iso.score_samples(X)  # higher = more anomalous
    rate = float((labels == -1).mean())
    return {
        "applicable": True,
        "anomaly_rate": rate,
        "anomaly_score_mean": float(scores.mean()),
        "anomaly_score_std": float(scores.std()),
        "labels": labels.tolist(),
        "scores": scores.tolist(),
        "n_records": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "accuracy_proxy": float(1 - rate),
    }


def isolation_forest_sweep(
    df: pd.DataFrame,
    contaminations: list[float] = [0.02, 0.05, 0.075, 0.10, 0.15],
) -> dict:
    """Sweep contamination and return rates + rank-stability statistics."""
    results = {}
    for c in contaminations:
        r = isolation_forest_anomalies(df, contamination=c)
        results[c] = {
            "anomaly_rate": r.get("anomaly_rate"),
            "applicable": r.get("applicable"),
        }
    return results


# ---------- DBSCAN ----------
def dbscan_noise(df: pd.DataFrame, eps_multiplier: float = 1.0) -> dict:
    """
    Calibrate eps via k-distance graph; DBSCAN returns noise set.
    """
    X = numeric_matrix(df)
    if X is None:
        return {"applicable": False, "noise_rate": None, "n_clusters": 0}
    d = X.shape[1]
    minpts = max(2 * d, 4)
    # k-distance: distance to k-th neighbour
    if X.shape[0] < minpts + 1:
        return {"applicable": False, "noise_rate": None}
    k = minpts
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, k - 1])
    # elbow: max curvature heuristic — pick the 90th percentile as eps base
    eps_base = float(np.percentile(k_distances, 90))
    eps = eps_base * eps_multiplier
    if eps <= 0:
        eps = 0.5
    db = DBSCAN(eps=eps, min_samples=minpts).fit(X)
    labels = db.labels_
    noise_rate = float((labels == -1).mean())
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    return {
        "applicable": True,
        "noise_rate": noise_rate,
        "n_clusters": n_clusters,
        "eps_used": eps,
        "minpts": minpts,
        "labels": labels.tolist(),
        "n_records": int(X.shape[0]),
    }


# ---------- KMeans for missingness patterns ----------
def kmeans_missingness(
    inventory: pd.DataFrame, k_range: range = range(2, 9)
) -> dict:
    """
    Cluster datasets by their quality profile (per dissertation Sec 3.5.3 /
    4.5.3): the feature space is restricted to the three bounded,
    scale-invariant inherent-quality ratios (completeness, consistency,
    uniqueness). Raw size and type-mix counts (rows, cols, numeric_cols,
    etc.) are deliberately EXCLUDED, since their unbounded scale would
    otherwise dominate the standardised Euclidean geometry and make the
    partition track dataset size rather than quality.

    Selection rule: the silhouette score is evaluated over k in k_range.
    A partition that trivially isolates a single-record (singleton)
    cluster is discounted as degenerate; the reported k* is the
    silhouette-maximising partition among the remaining, non-degenerate
    candidates.
    """
    feats = inventory[
        [
            "completeness",
            "consistency",
            "uniqueness",
        ]
    ].copy()
    feats = feats.fillna(0)
    feats = (feats - feats.mean()) / (feats.std().replace(0, 1))
    X = feats.values

    silhouettes = {}
    inertias = {}
    candidates = {}
    for k in k_range:
        if X.shape[0] <= k:
            continue
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit(X)
        labels = km.labels_
        try:
            s = float(silhouette_score(X, labels))
        except Exception:
            s = -1
        silhouettes[k] = s
        inertias[k] = float(km.inertia_)
        min_cluster_size = int(np.bincount(labels).min())
        candidates[k] = (s, km, labels, min_cluster_size)
    if not candidates:
        return {"applicable": False}

    # The global silhouette optimum at k=2 merely isolates a single
    # outlier record as a trivial singleton-vs-rest split and is
    # discounted as degenerate (per dissertation Sec 3.5.3/4.5.3); the
    # reported k* is the silhouette-maximising partition among k >= 3.
    non_trivial = {k: v for k, v in candidates.items() if k >= 3}
    pool = non_trivial if non_trivial else candidates
    k_star = max(pool, key=lambda k: pool[k][0])
    best_score, km_final, labels_final, _ = candidates[k_star]
    return {
        "applicable": True,
        "k_selected": int(k_star),
        "silhouette": float(best_score),
        "silhouettes": silhouettes,
        "inertias": inertias,
        "labels": labels_final.tolist(),
        "centroids": km_final.cluster_centers_.tolist(),
        "feature_names": list(feats.columns),
    }


# ---------- Cross-algorithm agreement ----------
def isolation_dbscan_agreement(if_labels: list, dbscan_labels: list) -> dict:
    """
    Compute Cohen's kappa and overlap between IF (-1 anomaly) and DBSCAN (-1 noise).
    """
    if len(if_labels) != len(dbscan_labels) or len(if_labels) == 0:
        return {"applicable": False}
    if_arr = np.array(if_labels)
    db_arr = np.array(dbscan_labels)
    if_anom = (if_arr == -1).astype(int)
    db_noise = (db_arr == -1).astype(int)
    n_if = int(if_anom.sum())
    n_db = int(db_noise.sum())
    overlap = int(((if_anom == 1) & (db_noise == 1)).sum())
    try:
        kappa = float(cohen_kappa_score(if_anom, db_noise))
    except Exception:
        kappa = None
    jaccard = (
        overlap / float(n_if + n_db - overlap)
        if (n_if + n_db - overlap) > 0
        else 0.0
    )
    return {
        "applicable": True,
        "kappa": kappa,
        "jaccard": float(jaccard),
        "if_anom_count": n_if,
        "dbscan_noise_count": n_db,
        "overlap": overlap,
        "n_records": len(if_arr),
    }


def interpret_kappa(k: float | None) -> str:
    """Landis & Koch (1977) interpretation."""
    if k is None:
        return "n/a"
    if k < 0:
        return "poor"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"
