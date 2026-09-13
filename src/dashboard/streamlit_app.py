"""
Saudi OGD AI Readiness Audit — Streamlit dashboard (dark intelligence theme).
Run: streamlit run streamlit_app_redesigned.py --server.port 5000
Place your result CSVs/JSONs in a results/ folder next to this script.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS = Path("results")

st.set_page_config(
    page_title="Saudi OGD AI Readiness Audit",
    page_icon="🛰️",
    layout="wide",
)

# ── palette ──────────────────────────────────────────────────────────────────
BG     = "#050A14"
PANEL  = "#0D1521"
PANEL2 = "#111D2E"
PANEL3 = "#162035"
BORDER = "#1E2D42"
GHOST  = "#1A2A3D"
TEAL   = "#00D4FF"
GOLD   = "#FFB800"
GREEN  = "#00E676"
AMBER  = "#FFA726"
RED    = "#FF3366"
WHITE  = "#F0F6FF"
SILVER = "#7A8FA6"

SECTOR_COLOR = {
    "Investment": TEAL,
    "Health":     RED,
    "Education":  GREEN,
    "Energy":     GOLD,
}
BAND_COLOR    = {"Band A": GREEN, "Band B": AMBER, "Band C": RED}
FONT = '"Space Grotesk","Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif'

# ── helpers ───────────────────────────────────────────────────────────────────
def _hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"

def _dark_layout(**kw):
    base = dict(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family=FONT, color=WHITE, size=12),
        margin=dict(l=12, r=12, t=44, b=12),
        xaxis=dict(gridcolor=GHOST, zerolinecolor=GHOST, linecolor=BORDER,
                   tickfont=dict(color=SILVER), title_font=dict(color=SILVER)),
        yaxis=dict(gridcolor=GHOST, zerolinecolor=GHOST, linecolor=BORDER,
                   tickfont=dict(color=SILVER), title_font=dict(color=SILVER)),
        legend=dict(bgcolor="rgba(13,17,23,0.9)", bordercolor=BORDER,
                    borderwidth=1, font=dict(color=SILVER)),
        hoverlabel=dict(bgcolor=PANEL3, bordercolor=TEAL,
                        font_family=FONT, font_color=WHITE),
        coloraxis_colorbar=dict(tickfont=dict(color=SILVER),
                                title_font=dict(color=SILVER),
                                bgcolor=PANEL, bordercolor=BORDER),
    )
    base.update(kw)
    return base

def svg_ring(pct, color, size=72):
    r = 28; cx = cy = size // 2
    circ = 2 * 3.14159 * r
    dash = circ * max(0.0, min(1.0, pct))
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1E2D42" stroke-width="7"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="7"'
        f' stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-linecap="round"'
        f' transform="rotate(-90 {cx} {cy})"/>'
        f'<text x="{cx}" y="{cy+5}" text-anchor="middle"'
        f' font-family="Space Grotesk,sans-serif" font-size="13"'
        f' font-weight="700" fill="{color}">{pct*100:.0f}%</text>'
        f'</svg>'
    )

def section_header(title, sub=""):
    sub_html = f'<p style="color:{SILVER};font-size:0.82rem;margin:2px 0 14px 0">{sub}</p>' if sub else ""
    st.markdown(
        f'<div style="margin:20px 0 4px 0">'
        f'<span style="font-size:1.1rem;font-weight:700;color:{WHITE}">{title}</span>'
        f'</div>{sub_html}',
        unsafe_allow_html=True,
    )

def kpi_card(col, label, value, pct, color, sub=""):
    ring = svg_ring(pct, color) if pct is not None else ""
    sub_html = f'<div style="font-size:0.75rem;color:{SILVER};margin-top:4px">{sub}</div>' if sub else ""
    col.markdown(
        f'<div style="background:{PANEL2};border:1px solid {BORDER};border-radius:16px;'
        f'padding:18px 20px;display:flex;align-items:center;gap:16px;height:100%">'
        f'<div>{ring}</div>'
        f'<div>'
        f'<div style="font-size:0.72rem;font-weight:600;color:{SILVER};'
        f'etext-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">{label}</div>'
        f'<div style="font-size:1.6rem;font-weight:800;color:{WHITE};'
        f'letter-spacing:-0.02em;line-height:1.1">{value}</div>'
        f'{sub_html}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

# ── data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    scores   = pd.read_csv(RESULTS / "dataset_scores.csv")
    with open(RESULTS / "headline.json") as f:
        hl = json.load(f)
    sector   = pd.read_csv(RESULTS / "sector_summary.csv")
    ndmo     = pd.read_csv(RESULTS / "ndmo_sector_matrix.csv")
    sweep    = pd.read_csv(RESULTS / "if_contamination_sweep.csv")
    try:    agr = pd.read_csv(RESULTS / "if_dbscan_agreement.csv")
    except FileNotFoundError: agr = pd.DataFrame()
    try:    reg = pd.read_csv(RESULTS / "ndmo_gap_register.csv")
    except FileNotFoundError: reg = pd.DataFrame()
    try:    km  = pd.read_csv(RESULTS / "kmeans_clusters.csv")
    except FileNotFoundError: km  = pd.DataFrame()
    return scores, hl, sector, ndmo, sweep, agr, reg, km

if not (RESULTS / "dataset_scores.csv").exists():
    st.error("No data found in `results/`. Drop your audit CSVs into `results/`.")
    st.stop()

scores, hl, sector_df, ndmo_df, sweep_df, agr_df, reg_df, km_df = load_data()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{{font-family:{FONT};background:{BG};color:{WHITE}}}
.block-container{{padding:1.5rem 2rem 2rem 2rem;max-width:1400px}}
h1,h2,h3{{font-family:{FONT};color:{WHITE};letter-spacing:-0.02em}}
[data-testid="stSidebar"]{{background:{PANEL};border-right:1px solid {BORDER}}}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label{{color:{SILVER}}}
[data-testid="stDataFrame"]{{background:{PANEL2};border-radius:12px;border:1px solid {BORDER}}}
[data-testid="stPlotlyChart"]{{background:{PANEL2};border-radius:16px;border:1px solid {BORDER};padding:8px}}
.stSelectbox>div>div{{background:{PANEL2};border:1px solid {BORDER};border-radius:10px;color:{WHITE}}}
#MainMenu,footer,[data-testid="stHeader"]{{visibility:hidden}}
</style>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="font-size:1.1rem;font-weight:800;color:{TEAL};padding:8px 0 16px 0">'
        f'🛰 OGD Readiness</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<hr style="border-color:{BORDER};margin:4px 0 12px 0">', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.78rem;color:{SILVER};line-height:1.9">'
        f'<b style="color:{WHITE}">Audit metadata</b><br>'
        f'Date &middot; {hl["audit_date"][:10]}<br>'
        f'Datasets &middot; {hl["n_datasets"]}<br>'
        f'Total rows &middot; {hl["portfolio_total_rows"]:,}<br>'
        f'Total cells &middot; {hl["portfolio_total_cells"]:,}<br><br>'
        f'<b style="color:{WHITE}">Algorithms</b><br>'
        f'Isolation Forest<br>DBSCAN<br>K-Means'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="background:linear-gradient(135deg,{PANEL} 0%,{PANEL2} 100%);'
    f'border:1px solid {BORDER};border-radius:20px;padding:28px 36px;margin-bottom:24px">'
    f'<div style="font-size:2rem;font-weight:800;letter-spacing:-0.03em;'
    f'background:linear-gradient(90deg,{TEAL},{GOLD});-webkit-background-clip:text;'
    f'-webkit-text-fill-color:transparent;line-height:1.2">'
    f'Saudi OGD AI Readiness Audit</div>'
    f'<div style="color:{SILVER};font-size:0.9rem;margin-top:8px">'
    f'ML-driven assessment against SDAIA NDMO Domain 3 &nbsp;&middot;&nbsp; '
    f'{hl["n_datasets"]} datasets &nbsp;&middot;&nbsp; {hl["portfolio_total_rows"]:,} rows</div>'
    f'<div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">'
    f'<span style="background:{PANEL3};border:1px solid {BORDER};border-radius:20px;'
    f'padding:3px 14px;font-size:0.75rem;color:{TEAL}">🛰 Isolation Forest</span>'
    f'<span style="background:{PANEL3};border:1px solid {BORDER};border-radius:20px;'
    f'padding:3px 14px;font-size:0.75rem;color:{GOLD}">🔬 DBSCAN</span>'
    f'<span style="background:{PANEL3};border:1px solid {BORDER};border-radius:20px;'
    f'padding:3px 14px;font-size:0.75rem;color:{GREEN}">📐 K-Means</span>'
    f'<span style="background:{PANEL3};border:1px solid {BORDER};border-radius:20px;'
    f'padding:3px 14px;font-size:0.75rem;color:{SILVER}">Vision 2030</span>'
    f'</div></div>',
    unsafe_allow_html=True,
)

# ── KPI row ───────────────────────────────────────────────────────────────────
comp  = hl["portfolio_completeness_aggregate"]
cons  = hl["mean_per_dataset_consistency"]
uniq  = hl.get("mean_per_dataset_uniqueness", 0)
kappa = hl.get("mean_if_dbscan_kappa", 0)

c1, c2, c3, c4 = st.columns(4)
kpi_card(c1, "Portfolio Completeness", f"{comp*100:.1f}%", comp, TEAL,
         f'{hl["datasets_meeting_95pct_completeness"]}/{hl["n_datasets"]} datasets meet 95%')
kpi_card(c2, "Mean Consistency", f"{cons*100:.1f}%", cons, GOLD,
         f'{hl["datasets_meeting_95pct_consistency"]}/{hl["n_datasets"]} meet 95% threshold')
kpi_card(c3, "Mean Uniqueness", f"{uniq*100:.1f}%", uniq, GREEN,
         f'{hl.get("datasets_lt_1pct_duplicates", 0)}/{hl["n_datasets"]} below 1% duplicates')
kpi_card(c4, "Mean Cohen\u2019s \u03ba", f"{kappa:.3f}", kappa, AMBER,
         "IF vs DBSCAN cross-algorithm agreement")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ── sector cards ──────────────────────────────────────────────────────────────
cols = st.columns(len(sector_df))
for col, (_, row) in zip(cols, sector_df.iterrows()):
    c = SECTOR_COLOR.get(row["sector"], SILVER)
    dims = [
        ("Completeness", float(row["completeness_mean"]), 0.95),
        ("Consistency",  float(row["consistency_mean"]),  0.95),
        ("Uniqueness",   float(row["uniqueness_mean"]),   0.99),
    ]
    bars = ""
    for dname, dval, dthresh in dims:
        pv   = dval * 100
        bc   = GREEN if dval >= dthresh else (AMBER if dval >= dthresh - 0.05 else RED)
        bars += (
            f'<div style="margin-top:8px">'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:0.7rem;color:{SILVER};margin-bottom:3px">'
            f'<span>{dname}</span><span style="color:{bc}">{pv:.1f}%</span></div>'
            f'<div style="background:{BORDER};border-radius:4px;height:5px">'
            f'<div style="background:{bc};width:{min(pv,100):.1f}%;'
            f'border-radius:4px;height:5px"></div></div></div>'
        )
    col.markdown(
        f'<div style="background:{PANEL2};border:1px solid {BORDER};'
        f'border-top:3px solid {c};border-radius:16px;padding:18px 20px">'
        f'<div style="font-size:1rem;font-weight:700;color:{c}">{row["sector"]}</div>'
        f'<div style="font-size:0.72rem;color:{SILVER};margin-bottom:4px">'
        f'n={int(row["n"])} datasets</div>'
        f'{bars}</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ═══════════════════════════ TABS ════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Portfolio", "🚨 Anomalies", "🏛 NDMO Compliance",
    "🔎 Dataset Inspector", "📐 Sensitivity",
])

# ─── TAB 1 — Portfolio ────────────────────────────────────────────────────────
with tab1:
    section_header("Composite readiness by dataset",
                   "Sorted descending · dashed lines = Band A (0.85) and Band B (0.70)")
    plot_df     = scores[["dataset_id","sector","full_composite","drl_band"]].sort_values(
        "full_composite", ascending=False)
    bar_colors  = [SECTOR_COLOR.get(s, SILVER) for s in plot_df["sector"]]
    fig = go.Figure(go.Bar(
        x=plot_df["dataset_id"], y=plot_df["full_composite"],
        marker_color=bar_colors, marker_line_width=0,
        customdata=plot_df[["sector","drl_band"]].values,
        hovertemplate=(
            "<b>%{x}</b><br>Score: %{y:.4f}<br>"
            "Sector: %{customdata[0]}<br>Band: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.add_hline(y=0.85, line_dash="dot", line_color=GREEN, line_width=1.5,
                  annotation_text="Band A", annotation_font_color=GREEN)
    fig.add_hline(y=0.70, line_dash="dot", line_color=AMBER, line_width=1.5,
                  annotation_text="Band B", annotation_font_color=AMBER)
    fig.update_layout(**_dark_layout(height=400, xaxis_tickangle=-45))
    st.plotly_chart(fig, use_container_width=True)

    ca, cb = st.columns(2)

    with ca:
        section_header("DRL band distribution", "Data Readiness Levels (Lawrence 2017)")
        bands = hl.get("drl_band_distribution", {})
        if bands:
            fig_b = go.Figure(go.Bar(
                x=list(bands.keys()), y=list(bands.values()),
                marker_color=[BAND_COLOR.get(k, SILVER) for k in bands],
                marker_line_width=0,
                text=list(bands.values()), textposition="outside",
                textfont=dict(color=WHITE),
            ))
            fig_b.update_layout(**_dark_layout(height=320, showlegend=False))
            st.plotly_chart(fig_b, use_container_width=True)

    with cb:
        section_header("Sector radar", "Mean quality dimensions per sector")
        d3      = ["completeness_mean", "consistency_mean", "uniqueness_mean"]
        dlabels = ["Completeness", "Consistency", "Uniqueness"]
        fig_r   = go.Figure()
        for _, row in sector_df.iterrows():
            c    = SECTOR_COLOR.get(row["sector"], SILVER)
            vals = [float(row[d]) for d in d3]
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=dlabels + [dlabels[0]],
                fill="toself", name=row["sector"],
                line=dict(color=c, width=2.5),
                fillcolor=_hex_rgba(c, 0.13),
                hovertemplate="%{theta}: %{r:.3f}<extra>" + row["sector"] + "</extra>",
            ))
        fig_r.update_layout(**_dark_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0.6, 1.05],
                                gridcolor=GHOST, linecolor=BORDER,
                                tickfont=dict(color=SILVER, size=9)),
                angularaxis=dict(tickfont=dict(color=SILVER),
                                 gridcolor=BORDER, linecolor=BORDER),
                bgcolor=PANEL,
            ),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                        xanchor="center", x=0.5),
        ))
        st.plotly_chart(fig_r, use_container_width=True)

    section_header("Portfolio hierarchy", "Portfolio → Sector → DRL Band → Dataset")
    sb = scores[["dataset_id","sector","drl_band","full_composite"]].copy()
    u_sec  = sb["sector"].unique().tolist()
    u_band = sb.drop_duplicates(["sector","drl_band"])[["sector","drl_band"]]

    ids     = (["OGD Portfolio"]
               + u_sec
               + (u_band["sector"] + " // " + u_band["drl_band"]).tolist()
               + sb["dataset_id"].tolist())
    labels  = (["OGD Portfolio"]
               + u_sec
               + u_band["drl_band"].tolist()
               + sb["dataset_id"].tolist())
    parents = ([""]
               + ["OGD Portfolio"] * len(u_sec)
               + u_band["sector"].tolist()
               + (sb["sector"] + " // " + sb["drl_band"]).tolist())
    values  = ([sb["full_composite"].sum()]
               + [sb[sb["sector"]==s]["full_composite"].sum() for s in u_sec]
               + [sb[(sb["sector"]==r.sector)&(sb["drl_band"]==r.drl_band)]["full_composite"].sum()
                  for _,r in u_band.iterrows()]
               + sb["full_composite"].tolist())
    colors  = ([TEAL]
               + [SECTOR_COLOR.get(s, SILVER) for s in u_sec]
               + [BAND_COLOR.get(b, SILVER) for b in u_band["drl_band"]]
               + [SECTOR_COLOR.get(s, SILVER) for s in sb["sector"]])

    fig_sun = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        branchvalues="total",
        marker=dict(colors=colors),
        hovertemplate="<b>%{label}</b><br>Score: %{value:.3f}<extra></extra>",
        textfont=dict(family=FONT, color=WHITE, size=11),
        insidetextorientation="radial",
    ))
    fig_sun.update_layout(**_dark_layout(height=480, margin=dict(l=0,r=0,t=10,b=0)))
    st.plotly_chart(fig_sun, use_container_width=True)

# ─── TAB 2 — Anomalies ───────────────────────────────────────────────────────
with tab2:
    try:
        section_header("Isolation Forest vs DBSCAN anomaly rates",
                       "IF contamination = 0.05 · grouped by dataset")
        # accept boolean True, string "True", or numeric 1
        _if_mask = scores["if_applicable"].apply(
            lambda v: str(v).strip().lower() not in ("false","nan","none","0","")
        )
        anom_df = scores[_if_mask].copy().sort_values("if_anomaly_rate", ascending=False)
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            name="Isolation Forest", x=anom_df["dataset_id"], y=anom_df["if_anomaly_rate"],
            marker_color=RED, marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>IF rate: %{y:.3f}<extra></extra>",
        ))
        if "db_noise_rate" in anom_df.columns:
            fig5.add_trace(go.Bar(
                name="DBSCAN noise", x=anom_df["dataset_id"], y=anom_df["db_noise_rate"],
                marker_color=TEAL, marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>DBSCAN rate: %{y:.3f}<extra></extra>",
            ))
        fig5.add_hline(y=0.05, line_dash="dot", line_color=GOLD, line_width=1.5,
                       annotation_text="contamination=0.05", annotation_font_color=GOLD)
        fig5.update_layout(**_dark_layout(height=400, barmode="group", xaxis_tickangle=-45))
        st.plotly_chart(fig5, use_container_width=True)
    except Exception as e:
        st.error(f"Anomaly rates chart error: {e}")

    if not agr_df.empty:
        try:
            ca, cb = st.columns(2)
            with ca:
                section_header("Cohen\u2019s \u03ba by dataset",
                               "substantial \u2265 0.6 \u00b7 almost perfect \u2265 0.8")
                agr_k = agr_df.dropna(subset=["kappa"]).sort_values("kappa", ascending=False)
                kap_c = [GREEN if k>=0.6 else (AMBER if k>=0.4 else RED) for k in agr_k["kappa"]]
                fig_k = go.Figure(go.Bar(
                    x=agr_k["dataset_id"], y=agr_k["kappa"],
                    marker_color=kap_c, marker_line_width=0,
                    text=[f"{k:.3f}" for k in agr_k["kappa"]],
                    textposition="outside", textfont=dict(color=WHITE, size=10),
                    hovertemplate="<b>%{x}</b><br>\u03ba = %{y:.3f}<extra></extra>",
                ))
                fig_k.update_layout(**_dark_layout(height=360, xaxis_tickangle=-45))
                st.plotly_chart(fig_k, use_container_width=True)

            with cb:
                section_header("\u03ba vs Jaccard scatter", "colour = sector")
                agr2 = agr_df.dropna(subset=["kappa", "jaccard"]).copy()
                # if sector is missing from agr_df, pull it from scores
                if "sector" not in agr2.columns:
                    agr2 = agr2.merge(
                        scores[["dataset_id", "sector"]].drop_duplicates(),
                        on="dataset_id", how="left"
                    )
                if "sector" not in agr2.columns:
                    agr2["sector"] = "Unknown"
                fig_sc = go.Figure()
                for sec in agr2["sector"].fillna("Unknown").unique():
                    sub = agr2[agr2["sector"].fillna("Unknown") == sec]
                    c   = SECTOR_COLOR.get(sec, SILVER)
                    fig_sc.add_trace(go.Scatter(
                        x=sub["kappa"], y=sub["jaccard"],
                        mode="markers+text", name=sec,
                        text=sub["dataset_id"], textposition="top center",
                        textfont=dict(color=c, size=9),
                        marker=dict(size=12, color=c, line=dict(color=WHITE, width=1)),
                        hovertemplate=(
                            "<b>%{text}</b><br>\u03ba=%{x:.3f}<br>J=%{y:.3f}<extra></extra>"
                        ),
                    ))
                fig_sc.add_vline(x=0.6, line_dash="dot", line_color=SILVER,
                                 annotation_text="substantial", annotation_font_color=SILVER)
                fig_sc.update_layout(**_dark_layout(
                    height=360, xaxis_title="Cohen\u2019s \u03ba", yaxis_title="Jaccard"))
                st.plotly_chart(fig_sc, use_container_width=True)
        except Exception as e:
            st.error(f"Agreement charts error: {e}")

        try:
            section_header("Cross-algorithm agreement register")
            want = ["dataset_id", "sector", "kappa", "jaccard", "if_anom_count",
                    "dbscan_noise_count", "overlap", "kappa_interp"]
            show = [c for c in want if c in agr_df.columns]
            st.dataframe(
                agr_df[show].sort_values("kappa", ascending=False, na_position="last"),
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Agreement table error: {e}")

# ─── TAB 3 — NDMO Compliance ─────────────────────────────────────────────────
with tab3:
    try:
        section_header("SDAIA NDMO Domain 3 compliance heatmap",
                       "Pass \u00b7 Borderline \u00b7 Fail per sector")
        # detect sector columns dynamically — avoid hardcoded names
        _skip = {"specification", "name", "id", "index"}
        secs_h = [
            c for c in ndmo_df.columns
            if c not in _skip
            and not c.endswith("_n")
            and not c.endswith("_pct_pass")
            and ndmo_df[c].dtype == object
        ]
        _label_col = next((c for c in ("name", "specification") if c in ndmo_df.columns), None)
        specs = ndmo_df[_label_col].tolist() if _label_col else list(range(len(ndmo_df)))
        if secs_h:
            zmap   = {"pass": 1.0, "borderline": 0.5, "fail": 0.0}
            z_vals = [[zmap.get(str(ndmo_df.iloc[i][s]).lower().strip(), 0.5)
                       for s in secs_h] for i in range(len(ndmo_df))]
            t_vals = [[str(ndmo_df.iloc[i][s]) for s in secs_h]
                      for i in range(len(ndmo_df))]
            fig_hm = go.Figure(go.Heatmap(
                z=z_vals, x=secs_h, y=specs,
                text=t_vals, texttemplate="%{text}",
                colorscale=[
                    [0.0, "rgba(255,51,102,0.45)"],
                    [0.5, "rgba(255,167,38,0.45)"],
                    [1.0, "rgba(0,230,118,0.45)"],
                ],
                showscale=False,
                hovertemplate="Sector: %{x}<br>Spec: %{y}<br>%{text}<extra></extra>",
                textfont=dict(family=FONT, size=11, color=WHITE),
            ))
            fig_hm.update_layout(**_dark_layout(
                height=max(300, len(ndmo_df) * 28),
                margin=dict(l=200, r=20, t=20, b=20),
                yaxis=dict(tickfont=dict(color=WHITE, size=10)),
            ))
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("No verdict columns detected in ndmo_sector_matrix.csv.")
    except Exception as e:
        st.error(f"NDMO heatmap error: {e}")

    ca, cb = st.columns(2)
    with ca:
        try:
            section_header("Pass rate by sector & specification")
            pct_cols = [c for c in ndmo_df.columns if c.endswith("_pct_pass")]
            _label_col2 = next(
                (c for c in ("name", "specification") if c in ndmo_df.columns), None
            )
            if pct_cols and _label_col2:
                fig_pr = go.Figure()
                for pc in pct_cols:
                    sn = pc.replace("_pct_pass", "")
                    c  = SECTOR_COLOR.get(sn, SILVER)
                    fig_pr.add_trace(go.Bar(
                        name=sn, x=ndmo_df[_label_col2], y=ndmo_df[pc],
                        marker_color=c, marker_line_width=0,
                        hovertemplate=f"<b>%{{x}}</b><br>{sn}: %{{y:.1f}}%<extra></extra>",
                    ))
                fig_pr.update_layout(**_dark_layout(
                    height=340, barmode="group", xaxis_tickangle=-30,
                    yaxis_title="Pass rate (%)"))
                st.plotly_chart(fig_pr, use_container_width=True)
            else:
                st.info("No *_pct_pass columns found.")
        except Exception as e:
            st.error(f"Pass rate chart error: {e}")

    with cb:
        section_header("Gap register", "Failing & borderline datasets")
        if not reg_df.empty:
            st.dataframe(reg_df, use_container_width=True)
        else:
            st.info("No gap register data available.")

# ─── TAB 4 — Dataset Inspector ───────────────────────────────────────────────
def _col(row, key, default=None):
    """Safe column accessor for a pandas Series."""
    try:
        v = row[key]
        return default if pd.isna(v) else v
    except KeyError:
        return default

with tab4:
    try:
        section_header("Dataset inspector")
        ds  = st.selectbox("Select dataset", scores["dataset_id"].tolist(), key="ds_sel")
        row = scores[scores["dataset_id"] == ds].iloc[0]
        sc  = SECTOR_COLOR.get(_col(row, "sector", ""), SILVER)

        fname = _col(row, "file_name", "")
        band  = _col(row, "drl_band", "—")
        st.markdown(
            f'<div style="background:{PANEL2};border:1px solid {BORDER};'
            f'border-radius:14px;padding:14px 20px;display:flex;'
            f'align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:14px">'
            f'<span style="background:{_hex_rgba(sc,0.12)};border:1px solid {_hex_rgba(sc,0.4)};'
            f'color:{sc};padding:4px 14px;border-radius:20px;font-weight:700">'
            f'{_col(row,"sector","—")}</span>'
            f'<span style="color:{SILVER};font-size:0.82rem">{fname}</span>'
            f'<span style="background:{PANEL3};padding:4px 12px;border-radius:20px;'
            f'color:{BAND_COLOR.get(band,SILVER)};font-size:0.8rem;'
            f'border:1px solid {BORDER}">{band}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Inspector header error: {e}")

    try:
        def _pct_kpi(col_key, label, color):
            v = _col(row, col_key)
            if v is None:
                return
            fv = float(v)
            kpi_card(st.columns(1)[0], label, f"{fv*100:.1f}%", fv, color)

        c1, c2, c3, c4 = st.columns(4)
        comp_v = float(_col(row, "completeness", 0))
        cons_v = float(_col(row, "consistency",  0))
        uniq_v = float(_col(row, "uniqueness",   0))
        kpi_card(c1, "Completeness", f"{comp_v*100:.1f}%", comp_v, TEAL)
        kpi_card(c2, "Consistency",  f"{cons_v*100:.1f}%", cons_v, GOLD)
        kpi_card(c3, "Uniqueness",   f"{uniq_v*100:.1f}%", uniq_v, GREEN)
        try:
            r_int = int(float(_col(row, "rows", 0)))
            c_int = int(float(_col(row, "cols", 0)))
        except (TypeError, ValueError):
            r_int = c_int = 0
        kpi_card(c4, "Shape", f"{r_int:,} \u00d7 {c_int}", None, SILVER)
    except Exception as e:
        st.error(f"KPI cards error: {e}")

    try:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        section_header("Quality dimensions breakdown")
        dnames = ["Completeness", "Consistency", "Uniqueness", "Accuracy", "Timeliness"]
        dkeys  = ["completeness", "consistency", "uniqueness", "accuracy", "timeliness"]
        dvals  = []
        for k in dkeys:
            try:
                raw = _col(row, k)
                v   = float(raw) if raw is not None else None
                if v is not None and pd.isna(v):
                    v = None
            except (TypeError, ValueError):
                v = None
            dvals.append(v)

        dcolors = [GREEN if (v is not None and v >= 0.95)
                   else (AMBER if (v is not None and v >= 0.80) else RED) for v in dvals]
        dplot   = [v if v is not None else 0.0 for v in dvals]
        dtext   = [f"{v*100:.1f}%" if v is not None else "N/A" for v in dvals]

        fig_ds = go.Figure(go.Bar(
            x=dnames, y=dplot, marker_color=dcolors, marker_line_width=0,
            text=dtext, textposition="outside", textfont=dict(color=WHITE),
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        ))
        fig_ds.add_hline(y=0.95, line_dash="dot", line_color=GREEN, line_width=1.5,
                         annotation_text="95% target", annotation_font_color=GREEN)
        fig_ds.add_hline(y=0.80, line_dash="dot", line_color=AMBER, line_width=1.5,
                         annotation_text="80% floor", annotation_font_color=AMBER)
        fig_ds.update_layout(**_dark_layout(height=360, yaxis_range=[0, 1.12], showlegend=False))
        st.plotly_chart(fig_ds, use_container_width=True)
    except Exception as e:
        st.error(f"Dimension chart error: {e}")

    try:
        _if_ok = str(_col(row, "if_applicable", "false")).strip().lower()
        if _if_ok not in ("false", "nan", "none", "0", ""):
            section_header("Anomaly detection")
            a1, a2 = st.columns(2)
            if_rate = _col(row, "if_anomaly_rate", 0)
            a1.markdown(
                f'<div style="background:{PANEL2};border:1px solid {BORDER};'
                f'border-radius:12px;padding:16px 20px">'
                f'<div style="color:{SILVER};font-size:0.75rem;text-transform:uppercase;'
                f'letter-spacing:.05em">IF Anomaly Rate</div>'
                f'<div style="font-size:1.8rem;font-weight:800;color:{RED};margin-top:4px">'
                f'{float(if_rate)*100:.2f}%</div></div>',
                unsafe_allow_html=True,
            )
            db_rate = _col(row, "db_noise_rate")
            if db_rate is not None:
                a2.markdown(
                    f'<div style="background:{PANEL2};border:1px solid {BORDER};'
                    f'border-radius:12px;padding:16px 20px">'
                    f'<div style="color:{SILVER};font-size:0.75rem;text-transform:uppercase;'
                    f'letter-spacing:.05em">DBSCAN Noise Rate</div>'
                    f'<div style="font-size:1.8rem;font-weight:800;color:{TEAL};margin-top:4px">'
                    f'{float(db_rate)*100:.2f}%</div></div>',
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.error(f"Anomaly section error: {e}")

# ─── TAB 5 — Sensitivity ─────────────────────────────────────────────────────
with tab5:
    try:
        section_header("IF contamination sweep",
                       "Mean anomaly rate across portfolio as contamination varies")
        swp_avg = sweep_df.groupby("contamination")["anomaly_rate"].mean().reset_index()
        fig_sw  = go.Figure(go.Scatter(
            x=swp_avg["contamination"], y=swp_avg["anomaly_rate"],
            mode="lines+markers",
            line=dict(color=TEAL, width=3, shape="spline"),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
            marker=dict(size=10, color=TEAL, line=dict(color=WHITE, width=2)),
            hovertemplate="contamination=%{x:.3f}<br>mean rate=%{y:.4f}<extra></extra>",
        ))
        fig_sw.add_vline(x=0.05, line_dash="dot", line_color=GOLD, line_width=2,
                         annotation_text="baseline c=0.05", annotation_font_color=GOLD)
        fig_sw.update_layout(**_dark_layout(
            height=380, xaxis_title="Contamination", yaxis_title="Mean anomaly rate"))
        st.plotly_chart(fig_sw, use_container_width=True)
    except Exception as e:
        st.error(f"Sweep line chart error: {e}")

    try:
        spearman = hl.get("if_sweep_spearman", {})
        if spearman:
            section_header("Rank stability \u2014 Spearman \u03c1 vs c=0.05",
                           "Higher \u03c1 = more stable ranking across contamination levels")
            df_sp = pd.DataFrame(
                [{"contamination": float(k), "rho": v} for k, v in spearman.items()]
            ).sort_values("contamination")
            sp_c  = [GREEN if r >= 0.9 else (AMBER if r >= 0.7 else RED) for r in df_sp["rho"]]
            fig_sp = go.Figure(go.Bar(
                x=[str(c) for c in df_sp["contamination"]], y=df_sp["rho"],
                marker_color=sp_c, marker_line_width=0,
                text=[f"{r:.3f}" for r in df_sp["rho"]],
                textposition="outside", textfont=dict(color=WHITE),
                hovertemplate="c=%{x}<br>\u03c1=%{y:.3f}<extra></extra>",
            ))
            fig_sp.add_hline(y=0.9, line_dash="dot", line_color=GREEN, line_width=1.5)
            fig_sp.update_layout(**_dark_layout(
                height=340, yaxis_range=[0, 1.12],
                xaxis_title="Contamination", yaxis_title="Spearman \u03c1",
                showlegend=False))
            st.plotly_chart(fig_sp, use_container_width=True)
    except Exception as e:
        st.error(f"Spearman chart error: {e}")

    try:
        section_header("Per-dataset sweep heatmap",
                       "Anomaly rate (%) at each contamination level")
        pivot  = sweep_df.pivot(
            index="dataset_id", columns="contamination", values="anomaly_rate"
        )
        fig_ph = go.Figure(go.Heatmap(
            z=(pivot * 100).values,
            x=[str(c) for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=[
                [0.0, PANEL3],
                [0.5, "rgba(255,184,0,0.45)"],
                [1.0, "rgba(255,51,102,0.65)"],
            ],
            hovertemplate="Dataset: %{y}<br>c=%{x}<br>rate=%{z:.2f}%<extra></extra>",
            colorbar=dict(tickfont=dict(color=SILVER),
                          title=dict(text="%", font=dict(color=SILVER))),
        ))
        fig_ph.update_layout(**_dark_layout(
            height=max(300, len(pivot) * 22),
            margin=dict(l=120, r=20, t=20, b=60),
            xaxis_title="Contamination",
            yaxis=dict(tickfont=dict(color=WHITE, size=10)),
        ))
        st.plotly_chart(fig_ph, use_container_width=True)
    except Exception as e:
        st.error(f"Sweep heatmap error: {e}")
