"""
Generate all figures referenced in Chapter 5 and Appendix H.
Writes PNGs to the tool's figures/ directory.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TOOL_ROOT))

RESULTS = _TOOL_ROOT / "results"
FIGS = _TOOL_ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")

SECTOR_COLORS = {
    "Investment": "#1f77b4",
    "Health": "#d62728",
    "Education": "#2ca02c",
    "Energy": "#ff7f0e",
}


def figure_5_1():
    """Per-dataset completeness, sorted, sector-coded."""
    df = pd.read_csv(RESULTS / "dataset_scores.csv").sort_values("completeness", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = [SECTOR_COLORS.get(s, "grey") for s in df["sector"]]
    ax.bar(df["dataset_id"], df["completeness"] * 100, color=colors)
    ax.axhline(95, color="red", linestyle="--", label="NDMO 95% threshold")
    ax.set_ylabel("Completeness (%)")
    ax.set_xlabel("Dataset ID")
    ax.set_title("Figure 5.1 — Per-dataset completeness, sorted, colour-coded by sector")
    plt.xticks(rotation=90, fontsize=8)
    # legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=s) for s, c in SECTOR_COLORS.items()]
    handles.append(plt.Line2D([0], [0], color="red", linestyle="--", label="95% threshold"))
    ax.legend(handles=handles, loc="lower left")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(FIGS / "fig_5_1_completeness_bar.png", dpi=120)
    plt.close()
    print("  fig_5_1_completeness_bar.png")


def figure_5_2_sector_boxplot():
    """Per-sector boxplot for four dimensions."""
    df = pd.read_csv(RESULTS / "dataset_scores.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    dims = [
        ("completeness", "Completeness", 0.95),
        ("consistency", "Consistency", 0.95),
        ("uniqueness", "Uniqueness", 0.99),
        ("if_anomaly_rate", "Anomaly rate (IF, c=0.05)", None),
    ]
    sector_order = ["Investment", "Health", "Education", "Energy"]
    for ax, (col, label, thr) in zip(axes.flatten(), dims):
        plot_df = df[["sector", col]].dropna()
        sns.boxplot(data=plot_df, x="sector", y=col, ax=ax, order=sector_order,
                    palette=[SECTOR_COLORS[s] for s in sector_order])
        sns.stripplot(data=plot_df, x="sector", y=col, ax=ax, order=sector_order,
                      color="black", size=3, alpha=0.5)
        if thr is not None:
            ax.axhline(thr, color="red", linestyle="--", alpha=0.7, label=f"Threshold {thr}")
            ax.legend()
        ax.set_title(label)
        ax.set_xlabel("")
    plt.suptitle("Figure 5.2 — Sector-level distribution across four quality dimensions", y=1.00)
    plt.tight_layout()
    plt.savefig(FIGS / "fig_5_2_sector_boxplots.png", dpi=120)
    plt.close()
    print("  fig_5_2_sector_boxplots.png")


def figure_5_3_pca_clusters():
    """PCA projection of dataset feature vectors with KMeans clusters."""
    if not (RESULTS / "kmeans_clusters.csv").exists():
        return
    km = pd.read_csv(RESULTS / "kmeans_clusters.csv")
    with open(RESULTS / "kmeans_meta.json") as f:
        meta = json.load(f)
    feat_cols = meta["feature_names"]
    if not all(c in km.columns for c in feat_cols):
        return
    X = km[feat_cols].fillna(0).values
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.cm.tab10
    for c in sorted(km["cluster"].unique()):
        mask = km["cluster"] == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=cmap(c), label=f"Cluster {c} (n={int(mask.sum())})", s=80, alpha=0.8)
        for i, idx in enumerate(km.index[mask]):
            ax.annotate(km.loc[idx, "dataset_id"], (coords[idx, 0], coords[idx, 1]),
                        fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)")
    ax.set_title(f"Figure 5.6 — PCA projection with KMeans clusters (k={meta['k_selected']}, silhouette={meta['silhouette']:.3f})")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(FIGS / "fig_5_3_pca_clusters.png", dpi=120)
    plt.close()
    print("  fig_5_3_pca_clusters.png")


def figure_5_4_composite_hist():
    df = pd.read_csv(RESULTS / "dataset_scores.csv")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(df["full_composite"], bins=20, edgecolor="black", color="#1f77b4", alpha=0.75)
    ax.axvline(0.85, color="green", linestyle="--", label="Band A threshold (0.85)")
    ax.axvline(0.70, color="orange", linestyle="--", label="Band B threshold (0.70)")
    ax.set_xlabel("Composite readiness score")
    ax.set_ylabel("Dataset count")
    ax.set_title("Figure 5.5 — Distribution of composite readiness scores")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "fig_5_4_composite_hist.png", dpi=120)
    plt.close()
    print("  fig_5_4_composite_hist.png")


def figure_5_6_anomaly_bars():
    """Per-dataset anomaly rates from Isolation Forest."""
    df = pd.read_csv(RESULTS / "dataset_scores.csv")
    df = df[df["if_anomaly_rate"].notna()].sort_values("if_anomaly_rate", ascending=False)
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    colors = [SECTOR_COLORS.get(s, "grey") for s in df["sector"]]
    ax.bar(df["dataset_id"], df["if_anomaly_rate"] * 100, color=colors)
    ax.axhline(5, color="red", linestyle="--", label="Contamination = 0.05")
    ax.set_ylabel("Anomaly rate (%)")
    ax.set_xlabel("Dataset ID")
    ax.set_title("Figure 5.3 — Isolation Forest anomaly rate per dataset")
    plt.xticks(rotation=90, fontsize=9)
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=s) for s, c in SECTOR_COLORS.items()]
    handles.append(plt.Line2D([0], [0], color="red", linestyle="--", label="Contamination=0.05"))
    ax.legend(handles=handles, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGS / "fig_5_6_anomaly_bars.png", dpi=120)
    plt.close()
    print("  fig_5_6_anomaly_bars.png")


def figure_5_8_kappa_dist():
    """Distribution of Cohen's kappa across datasets (IF vs DBSCAN)."""
    if not (RESULTS / "if_dbscan_agreement.csv").exists():
        return
    df = pd.read_csv(RESULTS / "if_dbscan_agreement.csv")
    if "kappa" not in df.columns or df["kappa"].dropna().empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(df["kappa"].dropna(), bins=12, edgecolor="black", color="#2ca02c", alpha=0.75)
    ax.axvline(df["kappa"].mean(), color="red", linestyle="--", label=f"Mean κ = {df['kappa'].mean():.3f}")
    for x, label in [(0.20, "Slight"), (0.40, "Fair"), (0.60, "Moderate"), (0.80, "Substantial")]:
        ax.axvline(x, color="grey", linestyle=":", alpha=0.5)
        ax.text(x, ax.get_ylim()[1] * 0.95, label, fontsize=8, rotation=90, va="top")
    ax.set_xlabel("Cohen's κ (IF vs DBSCAN per-record agreement)")
    ax.set_ylabel("Dataset count")
    ax.set_title("Figure 5.4 — Cross-algorithm agreement (Cohen's κ) distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "fig_5_8_kappa_dist.png", dpi=120)
    plt.close()
    print("  fig_5_8_kappa_dist.png")


def figure_5_9_elbow_silhouette():
    """Elbow + silhouette for KMeans k selection."""
    if not (RESULTS / "kmeans_meta.json").exists():
        return
    with open(RESULTS / "kmeans_meta.json") as f:
        meta = json.load(f)
    ks = sorted(int(k) for k in meta["silhouettes"].keys())
    sils = [meta["silhouettes"][str(k)] for k in ks]
    inertias = [meta["inertias"][str(k)] for k in ks]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(ks, inertias, marker="o", color="#1f77b4")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia (within-cluster SS)")
    ax1.set_title("Elbow plot")
    ax1.axvline(meta["k_selected"], color="red", linestyle="--", label=f"k* = {meta['k_selected']}")
    ax1.legend()
    ax2.plot(ks, sils, marker="o", color="#2ca02c")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette plot")
    ax2.axvline(meta["k_selected"], color="red", linestyle="--", label=f"k* = {meta['k_selected']}")
    ax2.legend()
    plt.suptitle("Figure 5.7 — KMeans k-selection via elbow and silhouette criteria")
    plt.tight_layout()
    plt.savefig(FIGS / "fig_5_9_elbow_silhouette.png", dpi=120)
    plt.close()
    print("  fig_5_9_elbow_silhouette.png")


def figure_5_10_centroids():
    if not (RESULTS / "kmeans_meta.json").exists():
        return
    with open(RESULTS / "kmeans_meta.json") as f:
        meta = json.load(f)
    centroids = np.array(meta["centroids"])
    features = meta["feature_names"]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(centroids, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=features, yticklabels=[f"Cluster {i}" for i in range(centroids.shape[0])],
                ax=ax, cbar_kws={"label": "Standardised mean"})
    ax.set_title("Figure 5.8 — KMeans cluster centroids (standardised features)")
    plt.tight_layout()
    plt.savefig(FIGS / "fig_5_10_centroids.png", dpi=120)
    plt.close()
    print("  fig_5_10_centroids.png")


def figure_5_11_ndmo_matrix():
    if not (RESULTS / "ndmo_sector_matrix.csv").exists():
        return
    df = pd.read_csv(RESULTS / "ndmo_sector_matrix.csv")
    sectors = [c for c in df.columns if c not in ("specification", "name") and not c.endswith("_n") and not c.endswith("_pct_pass")]
    if not sectors:
        return
    # Use proportion of datasets passing in each sector
    grid = np.zeros((len(df), len(sectors)))
    labels = np.empty(grid.shape, dtype=object)
    for i in range(len(df)):
        for j, s in enumerate(sectors):
            pct_col = f"{s}_pct_pass"
            n_col = f"{s}_n"
            pct = df.loc[i, pct_col] if pct_col in df.columns else None
            n = df.loc[i, n_col] if n_col in df.columns else 0
            if pd.isna(pct):
                grid[i, j] = 0
                labels[i, j] = "n/a"
            else:
                grid[i, j] = pct
                labels[i, j] = f"{pct:.0f}%\n(n={int(n)})"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{r['specification']} — {r['name']}" for _, r in df.iterrows()])
    ax.set_xticks(range(len(sectors)))
    ax.set_xticklabels(sectors, rotation=30)
    for i in range(len(df)):
        for j in range(len(sectors)):
            v = grid[i, j]
            txt = labels[i, j]
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if v < 50 else "black")
    cbar = plt.colorbar(im, ax=ax, label="% of datasets passing")
    ax.set_title("Figure 5.9 — NDMO Domain 3 compliance (% passing per sector)")
    plt.tight_layout()
    plt.savefig(FIGS / "fig_5_11_ndmo_matrix.png", dpi=120)
    plt.close()
    print("  fig_5_11_ndmo_matrix.png")


def figure_5_12_radar():
    """Per-sector radar against idealised polygon."""
    df = pd.read_csv(RESULTS / "dataset_scores.csv")
    sector_means = df.groupby("sector").agg(
        completeness=("completeness", "mean"),
        consistency=("consistency", "mean"),
        uniqueness=("uniqueness", "mean"),
        accuracy=("accuracy", "mean"),
        timeliness=("timeliness", "mean"),
    )
    # ideal polygon
    ideal = pd.Series({c: 1.0 for c in sector_means.columns})
    sectors = sector_means.index.tolist()
    dims = sector_means.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), subplot_kw=dict(polar=True))
    if len(sectors) == 1:
        axes = [axes]
    for ax, sector in zip(axes, sectors):
        vals = sector_means.loc[sector].fillna(0).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=SECTOR_COLORS.get(sector, "grey"), linewidth=2, label=sector)
        ax.fill(angles, vals, color=SECTOR_COLORS.get(sector, "grey"), alpha=0.25)
        ideal_vals = ideal.tolist() + [ideal.iloc[0]]
        ax.plot(angles, ideal_vals, color="black", linestyle="--", linewidth=1, label="Ideal (NDMO)")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dims, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(sector, fontweight="bold")
        ax.legend(loc="upper right", fontsize=7, bbox_to_anchor=(1.3, 1.1))
    plt.suptitle("Figure 5.10 — Sector readiness signatures against the idealised NDMO polygon", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGS / "fig_5_12_radar.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  fig_5_12_radar.png")


def figure_appendix_C_sweep():
    """Contamination sweep heatmap."""
    if not (RESULTS / "if_contamination_sweep.csv").exists():
        return
    df = pd.read_csv(RESULTS / "if_contamination_sweep.csv")
    df = df[df["applicable"]]
    if len(df) == 0:
        return
    pivot = df.pivot(index="dataset_id", columns="contamination", values="anomaly_rate")
    fig, ax = plt.subplots(figsize=(10, max(5, len(pivot) * 0.25)))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Anomaly rate (%)"})
    ax.set_title("Figure C.1 — Isolation Forest contamination sweep (anomaly rate %)")
    ax.set_xlabel("Contamination")
    plt.tight_layout()
    plt.savefig(FIGS / "fig_appC_contamination_sweep.png", dpi=120)
    plt.close()
    print("  fig_appC_contamination_sweep.png")


def figure_4_1_architecture():
    """Architecture diagram (boxes and arrows)."""
    fig, ax = plt.subplots(figsize=(13, 5))
    layers = [
        ("Ingestion", "#1f77b4"),
        ("Profiling", "#ff7f0e"),
        ("DQ Metrics", "#2ca02c"),
        ("ML Engine", "#d62728"),
        ("NDMO Gap", "#9467bd"),
        ("Visualisation", "#8c564b"),
    ]
    n = len(layers)
    box_w = 1.6
    gap = 0.25
    total = n * box_w + (n - 1) * gap
    start = -total / 2
    for i, (name, color) in enumerate(layers):
        x = start + i * (box_w + gap)
        ax.add_patch(plt.Rectangle((x, -0.5), box_w, 1.0, facecolor=color, edgecolor="black", alpha=0.7))
        ax.text(x + box_w / 2, 0, name, ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        if i < n - 1:
            ax.annotate("", xy=(x + box_w + gap, 0), xytext=(x + box_w, 0),
                        arrowprops=dict(arrowstyle="->", lw=2))
    # configuration plane
    ax.add_patch(plt.Rectangle((start, -2.2), total, 0.6, facecolor="#dddddd", edgecolor="black"))
    ax.text(start + total / 2, -1.9, "YAML configuration (thresholds, hyperparameters, NDMO mapping)",
            ha="center", va="center", fontsize=10, style="italic")
    for i in range(n):
        x = start + i * (box_w + gap) + box_w / 2
        ax.plot([x, x], [-0.5, -1.6], "k:", linewidth=0.8)
    ax.set_xlim(start - 0.5, start + total + 0.5)
    ax.set_ylim(-2.6, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Figure 4.1 — Layered architecture of the assessment pipeline", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGS / "fig_4_1_architecture.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  fig_4_1_architecture.png")


def main():
    print("Generating figures...")
    figure_4_1_architecture()
    figure_5_1()
    figure_5_2_sector_boxplot()
    figure_5_3_pca_clusters()
    figure_5_4_composite_hist()
    figure_5_6_anomaly_bars()
    figure_5_8_kappa_dist()
    figure_5_9_elbow_silhouette()
    figure_5_10_centroids()
    figure_5_11_ndmo_matrix()
    figure_5_12_radar()
    figure_appendix_C_sweep()
    print("Done.")


if __name__ == "__main__":
    main()
