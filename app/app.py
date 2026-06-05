"""
app.py — Car Recommendation System
====================================
Streamlit web app untuk rekomendasi mobil spesifik (merk, tipe, harga)
berbasis Edmunds/Kaggle Car Features & MSRP Dataset.

Jalankan: python -m streamlit run app/app.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Path setup (HARUS sebelum import lokal) ───────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_loader import (
    NUMERIC_FEATURES, CAT_FEATURES, TARGET_COL,
    load_data, get_train_test_split, label_price_segment, PRICE_SEGMENTS
)
from recommend import get_recommendations, DISPLAY_COLS
from evaluate import evaluate_classifier, evaluate_regressor

MODELS_DIR = os.path.join(ROOT, "models")

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoMatch — Car Recommender",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Rajdhani:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 40px rgba(48,43,99,0.5);
    border: 1px solid rgba(255,255,255,0.07);
}
.main-header h1 {
    font-family: 'Rajdhani', sans-serif;
    color: #fff;
    font-size: 3rem;
    margin: 0;
    letter-spacing: 3px;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-header p { color: rgba(255,255,255,0.65); font-size: 1rem; margin-top: 0.4rem; }

.metric-card {
    background: linear-gradient(135deg, #0f0f1a, #1a1a2e);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(167,139,250,0.2);
}
.metric-value { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 0.82rem; color: #888; margin-top: 4px; }

.car-card {
    background: linear-gradient(135deg, #0d0d1a, #1a1a2e);
    border-left: 4px solid #7c3aed;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin: 0.5rem 0;
    transition: border-color 0.2s, transform 0.2s;
}
.car-card:hover { border-color: #a78bfa; transform: translateX(4px); }
.car-rank  { font-size: 1.5rem; font-weight: 700; color: #a78bfa; }
.car-make  { font-family: 'Rajdhani', sans-serif; font-size: 1.3rem;
             font-weight: 700; color: #fff; letter-spacing: 1px; }
.car-info  { font-size: 0.88rem; color: #bbb; }

.badge {
    display: inline-block;
    padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; margin-right: 4px;
}
.badge-budget   { background:#16a34a; color:#fff; }
.badge-mid      { background:#2563eb; color:#fff; }
.badge-premium  { background:#9333ea; color:#fff; }
.badge-luxury   { background:#b45309; color:#fff; }

.section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.6rem; font-weight: 700;
    color: #a78bfa;
    border-bottom: 2px solid rgba(167,139,250,0.3);
    padding-bottom: 0.4rem; margin-bottom: 1.2rem;
    letter-spacing: 1px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050510, #0d0d1a);
    border-right: 1px solid rgba(167,139,250,0.15);
}

div.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #7c3aed);
    color: #fff; border: none; border-radius: 8px;
    padding: 0.6rem 1.5rem; font-weight: 600; font-size: 1rem;
    width: 100%; transition: all 0.2s;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #7c3aed, #a78bfa);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(124,58,237,0.4);
}

.stTabs [data-baseweb="tab"] {
    background: rgba(124,58,237,0.1);
    border-radius: 8px; padding: 6px 18px;
    border: 1px solid rgba(124,58,237,0.2); color: #bbb;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6d28d9, #7c3aed) !important;
    color: #fff !important; border-color: #a78bfa !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def models_exist() -> bool:
    needed = ["rf_classifier.pkl", "rf_regressor.pkl",
              "knn_recommender.pkl", "scaler.pkl", "encoders.pkl", "df_encoded.pkl",
              "X_test.pkl", "y_test.pkl", "price_test.pkl"]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in needed)


@st.cache_resource(show_spinner="Memuat model...")
def load_all_models():
    rf_clf   = joblib.load(os.path.join(MODELS_DIR, "rf_classifier.pkl"))
    rf_reg   = joblib.load(os.path.join(MODELS_DIR, "rf_regressor.pkl"))
    scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    df_enc   = joblib.load(os.path.join(MODELS_DIR, "df_encoded.pkl"))
    X_test   = joblib.load(os.path.join(MODELS_DIR, "X_test.pkl"))
    y_test   = joblib.load(os.path.join(MODELS_DIR, "y_test.pkl"))
    price_t  = joblib.load(os.path.join(MODELS_DIR, "price_test.pkl"))
    return rf_clf, rf_reg, scaler, encoders, df_enc, X_test, y_test, price_t


@st.cache_data(show_spinner="Memuat dataset...")
def load_cached_data():
    return load_data()


def price_badge(segment: str) -> str:
    cls_map = {
        "Budget (< $15K)":     "badge-budget",
        "Mid-Range ($15-30K)": "badge-mid",
        "Premium ($30-60K)":   "badge-premium",
        "Luxury (> $60K)":     "badge-luxury",
    }
    cls = cls_map.get(segment, "badge-mid")
    return f'<span class="badge {cls}">{segment}</span>'


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>AUTO MATCH</h1>
    <p>Sistem Rekomendasi Mobil Cerdas &nbsp;·&nbsp; Kaggle Dataset &nbsp;·&nbsp; Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Panel Kontrol")
    st.markdown("---")

    if not models_exist():
        st.warning("⚙️ Model belum tersedia — memulai training otomatis...")
        with st.spinner("Melatih model... (~60 detik, harap tunggu)"):
            import subprocess
            res = subprocess.run(
                [sys.executable, os.path.join(ROOT, "src", "train_model.py")],
                capture_output=True, text=True, cwd=ROOT
            )
            if res.returncode == 0:
                st.success("✅ Training berhasil!")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error("❌ Training gagal!")
                st.code(res.stderr[-2000:])
                st.stop()
    else:
        st.success("Model siap digunakan")

    st.markdown("---")
    st.markdown("### Preferensi Anda")

    # ── Filter utama ────────────────────────────────────────────────────────
    body_style = st.selectbox(
        "Tipe Bodi", ["(Semua)", "sedan", "suv", "coupe", "hatchback", "pickup", "van", "wagon", "convertible"]
    )
    fuel_type = st.selectbox("Tipe BBM", ["(Semua)", "gas", "diesel", "electric", "flex-fuel", "natural gas"])
    drive_wheels = st.selectbox("Penggerak Roda", ["(Semua)", "fwd", "rwd", "awd", "4wd"])
    transmission = st.selectbox("Transmisi", ["(Semua)", "AUTOMATIC", "MANUAL", "AUTOMATED_MANUAL", "DIRECT_DRIVE"])
    num_doors = st.selectbox("Jumlah Pintu", ["(Semua)", "2", "3", "4"])
    n_recs = st.slider("Jumlah Rekomendasi", 3, 15, 5)

    st.markdown("---")
    st.markdown("### Spesifikasi Teknis")

    max_price   = st.number_input("Budget Maksimal ($)", min_value=2000, max_value=2500000, value=40000, step=1000)
    horsepower  = st.slider("Tenaga (HP)", 50, 1050, 200, 10)
    city_mpg    = st.slider("Konsumsi BBM Kota (mpg)", 5, 140, 25, 1)

    st.markdown("---")
    recommend_btn = st.button("Cari Rekomendasi", key="rec_btn")


# ─── Main tabs ────────────────────────────────────────────────────────────────
if not models_exist():
    st.info("⚙️ Training otomatis sedang berjalan... Halaman akan dimuat ulang setelah selesai.")
    st.stop()

rf_clf, rf_reg, scaler, encoders, df_enc, X_test, y_test, price_t = load_all_models()

tab1, tab2, tab3, tab4 = st.tabs(["Rekomendasi", "Evaluasi Model", "Eksplorasi Data", "Tentang"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — REKOMENDASI
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Temukan Mobil Terbaik untuk Anda</p>', unsafe_allow_html=True)

    # Susun preferensi user
    prefs = {
        "horsepower":  float(horsepower),
        "city-mpg":    float(city_mpg),
        # 'price' tidak dimasukkan ke prefs (bukan fitur model, dihandle lewat filter max_price)
        # highway-mpg & cylinders dihapus dari NUMERIC_FEATURES (multikolinearitas)
    }
    if body_style != "(Semua)":
        prefs["body-style"] = body_style
    if fuel_type != "(Semua)":
        prefs["fuel-type"] = fuel_type
    if drive_wheels != "(Semua)":
        prefs["drive-wheels"] = drive_wheels
    if transmission != "(Semua)":
        prefs["transmission"] = transmission
    if num_doors != "(Semua)":
        prefs["num-of-doors"] = float(num_doors)

    # Filter opsional
    bs_filter   = None if body_style == "(Semua)"  else body_style
    ft_filter   = None if fuel_type  == "(Semua)"  else fuel_type
    trans_filter = None if transmission == "(Semua)" else transmission
    dw_filter   = None if drive_wheels == "(Semua)" else drive_wheels

    # Prediksi merk & harga
    from recommend import _encode_input
    input_vec    = _encode_input(prefs, df_enc, encoders)
    input_scaled = scaler.transform(input_vec)

    pred_make  = rf_clf.predict(input_scaled)[0]
    pred_proba = rf_clf.predict_proba(input_scaled)[0]
    pred_price = float(rf_reg.predict(input_scaled)[0])
    classes    = rf_clf.classes_

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    seg = label_price_segment(pred_price)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{pred_make.upper()}</div>
            <div class="metric-label">Merk Paling Cocok</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">${pred_price:,.0f}</div>
            <div class="metric-label">Estimasi Harga</div></div>""", unsafe_allow_html=True)
    with c3:
        conf = pred_proba.max() * 100
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{conf:.1f}%</div>
            <div class="metric-label">Keyakinan Model</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{n_recs}</div>
            <div class="metric-label">Rekomendasi</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar chart probabilitas merk
    top10_idx  = np.argsort(pred_proba)[::-1][:10]
    prob_df    = pd.DataFrame({
        "Merk": [classes[i].title() for i in top10_idx],
        "Probabilitas (%)": [pred_proba[i] * 100 for i in top10_idx]
    })
    fig_prob = px.bar(
        prob_df, x="Probabilitas (%)", y="Merk", orientation="h",
        color="Probabilitas (%)",
        color_continuous_scale=[[0,"#1e1b4b"],[0.5,"#7c3aed"],[1.0,"#a78bfa"]],
        title="Top-10 Merk Paling Cocok dengan Preferensi Anda"
    )
    fig_prob.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", height=340, coloraxis_showscale=False,
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("---")

    # Rekomendasi
    with st.spinner("Mencari mobil terbaik..."):
        result_df, _, _ = get_recommendations(
            prefs, scaler, encoders, df_enc,
            classifier=rf_clf, regressor=None,
            n_recommendations=n_recs,
            body_style_filter=bs_filter,
            fuel_type_filter=ft_filter,
            transmission_filter=trans_filter,
            drive_wheels_filter=dw_filter,
            max_price=float(max_price),
        )

    # Cek apakah brand hasil prediksi masuk dalam daftar rekomendasi utama
    pred_make_lower = pred_make.lower()
    has_pred_make_in_recs = pred_make_lower in result_df["make"].str.lower().values if len(result_df) > 0 else False
    
    if not has_pred_make_in_recs:
        with st.expander("💡 Kenapa merk paling cocok (Prediksi) berbeda dengan daftar rekomendasi?", expanded=True):
            st.markdown(f"""
            **Analisis Penyelarasan Model:**
            - Model **Brand Classifier (Random Forest)** menyimpulkan bahwa spesifikasi umum Anda (Tenaga **{horsepower:.0f} HP**, Budget **${max_price:,.0f}**) secara historis paling mencerminkan brand **{pred_make.upper()}**.
            - Namun, Anda juga mengaktifkan filter spesifik pada panel kontrol (seperti Bodi: **{body_style}**, BBM: **{fuel_type}**, Penggerak Roda: **{drive_wheels}**, Transmisi: **{transmission}**).
            - Karena **{pred_make.upper()}** tidak memiliki mobil yang memenuhi seluruh filter spesifik tersebut secara sempurna (atau model yang ada memiliki kemiripan spesifikasi yang lebih rendah), sistem rekomendasi **KNN** memprioritaskan merk lain yang fiturnya cocok secara persis (seperti yang ditunjukkan pada daftar rekomendasi di bawah).
            - *Tips: Cobalah untuk mengatur filter bodi/penggerak roda ke **(Semua)** atau turunkan tingkat ke-spesifik-an filter untuk menyelaraskan kedua model.*
            """)

    st.markdown('<p class="section-title">Top Rekomendasi Mobil</p>', unsafe_allow_html=True)

    for rank, (_, row) in enumerate(result_df.iterrows(), 1):
        seg_badge = price_badge(row.get("price_segment", "-"))
        sim       = float(row.get("similarity (%)", 0))
        sim_bar   = "█" * int(sim / 10) + "░" * (10 - int(sim / 10))
        hp        = row.get("horsepower", "-")
        price_val = row.get("price", 0)
        body      = row.get("body-style", "-")
        ft        = row.get("fuel-type", "-")
        mpg       = row.get("city-mpg", "-")
        model_name = row.get("model", "-")
        year      = row.get("year", "-")
        trans     = row.get("transmission", "-")

        st.markdown(f"""
        <div class="car-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="car-rank">#{rank}</span>
                <span class="car-make">{row.get('make_display', row.get('make','?')).upper()} &nbsp;·&nbsp; {model_name.upper()} &nbsp;({year:.0f})</span>
                <span style="color:#a78bfa; font-weight:700; font-size:1.1rem;">{sim:.1f}% match</span>
            </div>
            <div style="margin-top:6px; font-size:0.8rem; color:#a78bfa;">{sim_bar}</div>
            <div style="margin-top:8px;">{seg_badge}</div>
            <div class="car-info" style="margin-top:8px;">
                Bodi: <b>{body}</b> &nbsp;|&nbsp;
                Transmisi: <b>{trans}</b> &nbsp;|&nbsp;
                Penggerak: <b>{row.get('drive-wheels', '-')}</b> &nbsp;|&nbsp;
                BBM: <b>{ft}</b> &nbsp;|&nbsp;
                HP: <b>{hp:.0f}</b> &nbsp;|&nbsp;
                MPG: <b>{mpg:.0f}</b> &nbsp;|&nbsp;
                Harga: <b>${price_val:,.0f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    # ── Rekomendasi Khusus Merk Prediksi ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<p class="section-title">Pilihan Terbaik Khusus Merk: {pred_make.upper()}</p>', unsafe_allow_html=True)
    st.caption(f"Menampilkan alternatif mobil dari brand **{pred_make.upper()}** di bawah budget ${max_price:,.0f} (mengabaikan filter bodi/bbm/penggerak yang terlalu ketat).")
    
    # Query khusus untuk brand prediksi
    with st.spinner(f"Mencari mobil terbaik {pred_make}..."):
        brand_result_df, _, _ = get_recommendations(
            prefs, scaler, encoders, df_enc,
            classifier=None, regressor=None,
            n_recommendations=3,
            max_price=float(max_price),
            make_filter=pred_make
        )
        
    if len(brand_result_df) == 0:
        st.info(f"Tidak ditemukan mobil dari merk {pred_make} di bawah budget ${max_price:,.0f}.")
    else:
        for b_rank, (_, b_row) in enumerate(brand_result_df.iterrows(), 1):
            b_seg_badge = price_badge(b_row.get("price_segment", "-"))
            b_sim       = float(b_row.get("similarity (%)", 0))
            b_sim_bar   = "█" * int(b_sim / 10) + "░" * (10 - int(b_sim / 10))
            b_hp        = b_row.get("horsepower", "-")
            b_price_val = b_row.get("price", 0)
            b_body      = b_row.get("body-style", "-")
            b_ft        = b_row.get("fuel-type", "-")
            b_mpg       = b_row.get("city-mpg", "-")
            b_model_name = b_row.get("model", "-")
            b_year      = b_row.get("year", "-")
            b_trans     = b_row.get("transmission", "-")
            b_dw        = b_row.get("drive-wheels", "-")
            
            st.markdown(f"""
            <div class="car-card" style="border-left-color: #10b981;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="car-rank" style="color:#10b981;">#{b_rank}</span>
                    <span class="car-make">{b_row.get('make_display', b_row.get('make','?')).upper()} &nbsp;·&nbsp; {b_model_name.upper()} &nbsp;({b_year:.0f})</span>
                    <span style="color:#10b981; font-weight:700; font-size:1.1rem;">{b_sim:.1f}% match</span>
                </div>
                <div style="margin-top:6px; font-size:0.8rem; color:#10b981;">{b_sim_bar}</div>
                <div style="margin-top:8px;">{b_seg_badge} <span class="badge" style="background:#0f172a; color:#10b981; border:1px solid #10b981;">Opsi Khusus {pred_make.upper()}</span></div>
                <div class="car-info" style="margin-top:8px;">
                    Bodi: <b>{b_body}</b> &nbsp;|&nbsp;
                    Transmisi: <b>{b_trans}</b> &nbsp;|&nbsp;
                    Penggerak: <b>{b_dw}</b> &nbsp;|&nbsp;
                    BBM: <b>{b_ft}</b> &nbsp;|&nbsp;
                    HP: <b>{b_hp:.0f}</b> &nbsp;|&nbsp;
                    MPG: <b>{b_mpg:.0f}</b> &nbsp;|&nbsp;
                    Harga: <b>${b_price_val:,.0f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Radar chart — hanya tampilkan jika ada rekomendasi
    if len(result_df) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Radar — Preferensi vs Rekomendasi #1")

        # Gunakan hanya fitur yang benar-benar digunakan model (NUMERIC_FEATURES)
        # dan yang tersedia di result_df (DISPLAY_COLS)
        # cylinders sudah dikeluarkan (multikolinearitas tinggi dengan horsepower)
        _radar_candidates = ["horsepower", "city-mpg", "year", "num-of-doors"]
        radar_feats = [f for f in _radar_candidates if f in result_df.columns and f in df_enc.columns]

        if len(radar_feats) >= 2:
            def norm_val(col, val):
                lo, hi = df_enc[col].min(), df_enc[col].max()
                return (float(val) - lo) / (hi - lo + 1e-9)

            top1   = result_df.iloc[0]
            user_r = [norm_val(f, prefs.get(f, df_enc[f].median())) for f in radar_feats]
            top_r  = [norm_val(f, top1[f]) for f in radar_feats]
            cats   = [f.replace("-", "\n").title() for f in radar_feats]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=user_r + [user_r[0]], theta=cats + [cats[0]],
                fill="toself", name="Preferensi Anda",
                line_color="#a78bfa", fillcolor="rgba(167,139,250,0.2)"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=top_r + [top_r[0]], theta=cats + [cats[0]],
                fill="toself", name=f"#{1} {top1.get('make_display','?')} {top1.get('model','')}",
                line_color="#34d399", fillcolor="rgba(52,211,153,0.2)"
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], color="#555", gridcolor="#333"),
                    angularaxis=dict(color="#aaa", gridcolor="#333"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(font_color="#ccc", bgcolor="rgba(0,0,0,0)"),
                height=420
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # Tabel detail
    st.markdown("#### Tabel Detail Rekomendasi")
    show_cols = ["make_display", "model", "year", "body-style", "fuel-type", "transmission", "drive-wheels",
                 "horsepower", "city-mpg", "price", "price_segment", "similarity (%)"]
    show_cols = [c for c in show_cols if c in result_df.columns]
    
    st.dataframe(
        result_df[show_cols].rename(columns={
            "make_display": "Merk", "model": "Model", "year": "Tahun", "body-style": "Bodi",
            "fuel-type": "BBM", "transmission": "Transmisi", "drive-wheels": "Penggerak",
            "horsepower": "HP", "city-mpg": "MPG Kota",
            "price": "Harga ($)", "price_segment": "Segmen", "similarity (%)": "Kemiripan (%)"
        }),
        use_container_width=True, hide_index=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUASI MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Evaluasi Performa Model ML</p>', unsafe_allow_html=True)

    rf_res  = evaluate_classifier(rf_clf,  X_test, y_test, "Random Forest")
    reg_res = evaluate_regressor(rf_reg,   X_test, price_t)

    # Brand Classifier metrics
    st.markdown("### Brand Classifier (Random Forest)")
    m1, m2, m3, m4 = st.columns(4)
    for col, (label, val) in zip([m1,m2,m3,m4], [
        ("Accuracy", rf_res["accuracy"]),
        ("Precision", rf_res["precision"]),
        ("Recall", rf_res["recall"]),
        ("F1-Score", rf_res["f1_score"]),
    ]):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val:.3f}</div>
                <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Price regressor metrics
    st.markdown("### Price Regressor (Random Forest)")
    pr1, pr2, pr3 = st.columns(3)
    for col, (label, val, fmt) in zip([pr1, pr2, pr3], [
        ("MAE",  reg_res["mae"],  "${:,.0f}"),
        ("RMSE", reg_res["rmse"], "${:,.0f}"),
        ("R²",   reg_res["r2"],   "{:.4f}"),
    ]):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{fmt.format(val)}</div>
                <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature importance
    st.markdown("#### Feature Importance — Random Forest Brand Classifier")
    feat_names  = NUMERIC_FEATURES + encoders
    importances = rf_clf.feature_importances_
    top20_idx   = np.argsort(importances)[::-1][:15]
    fi_df = pd.DataFrame({
        "Feature":    [feat_names[i].replace("_enc","") for i in top20_idx],
        "Importance": importances[top20_idx]
    })
    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=[[0,"#1e1b4b"],[0.5,"#7c3aed"],[1.0,"#a78bfa"]],
        title="Top-15 Feature Importance"
    )
    fig_fi.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", xaxis=dict(gridcolor="#333"),
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False, height=420
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Price actual vs predicted
    st.markdown("#### Harga Aktual vs Prediksi")
    price_df = pd.DataFrame({"Aktual ($)": reg_res["y_test"], "Prediksi ($)": reg_res["y_pred"]})
    fig_price = px.scatter(
        price_df, x="Aktual ($)", y="Prediksi ($)",
        opacity=0.75,
        color_discrete_sequence=["#60a5fa"],
        title=f"Actual vs Predicted Price  |  R² = {reg_res['r2']:.4f}"
    )
    mn = price_df.min().min(); mx = price_df.max().max()
    fig_price.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                        line=dict(color="#ef4444", dash="dash", width=2))
    fig_price.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        height=400
    )
    st.plotly_chart(fig_price, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EKSPLORASI DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Eksplorasi Dataset Mobil</p>', unsafe_allow_html=True)

    df_raw = load_cached_data()

    e1, e2, e3, e4 = st.columns(4)
    for col, (label, val) in zip([e1,e2,e3,e4], [
        ("Total Mobil", len(df_raw)),
        ("Jumlah Merk", df_raw["make"].nunique()),
        ("Harga Min ($)", f"{df_raw['price'].min():,.0f}"),
        ("Harga Max ($)", f"{df_raw['price'].max():,.0f}"),
    ]):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Distribusi merk
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        make_counts = df_raw["make"].value_counts().reset_index()
        make_counts.columns = ["Merk", "Jumlah"]
        # Ambil Top-15 untuk visualisasi agar tidak terlalu padat
        fig_make = px.bar(
            make_counts.head(15), x="Jumlah", y="Merk", orientation="h",
            color="Jumlah",
            color_continuous_scale=[[0,"#1e1b4b"],[1,"#a78bfa"]],
            title="Top-15 Distribusi Mobil per Merk"
        )
        fig_make.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc", yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False, height=500
        )
        st.plotly_chart(fig_make, use_container_width=True)

    with col_d2:
        seg_counts = df_raw["price_segment"].value_counts().reset_index()
        seg_counts.columns = ["Segmen", "Jumlah"]
        fig_seg = px.pie(
            seg_counts, values="Jumlah", names="Segmen",
            color="Segmen",
            color_discrete_map={
                "Budget (< $15K)": "#16a34a",
                "Mid-Range ($15-30K)": "#2563eb",
                "Premium ($30-60K)": "#9333ea",
                "Luxury (> $60K)": "#b45309",
            },
            title="Distribusi Segmen Harga"
        )
        fig_seg.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc", height=380)
        st.plotly_chart(fig_seg, use_container_width=True)

    # Harga per merk (Top 10 terpopuler)
    st.markdown("#### Distribusi Harga per Merk (Top 10 Terpopuler)")
    top10_makes = df_raw["make"].value_counts().head(10).index
    df_box = df_raw[df_raw["make"].isin(top10_makes)].copy()
    
    fig_box = px.box(
        df_box, x="make", y="price", color="price_segment",
        color_discrete_map={
            "Budget (< $15K)": "#16a34a",
            "Mid-Range ($15-30K)": "#2563eb",
            "Premium ($30-60K)": "#9333ea",
            "Luxury (> $60K)": "#b45309",
        },
        title="Distribusi Harga pada Top 10 Merk Mobil",
        labels={"make": "Merk", "price": "Harga ($)"}
    )
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", xaxis=dict(gridcolor="#333", tickangle=45),
        yaxis=dict(gridcolor="#333"), height=430,
        showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Scatter HP vs Price
    st.markdown("#### Hubungan Fitur Teknis vs Harga")
    sc1, sc2 = st.columns(2)
    with sc1:
        # Pengecualian kolom price
        features_x = [f for f in NUMERIC_FEATURES if f != "price"]
        x_col = st.selectbox("Fitur X", features_x, index=1, key="sc_x")
    with sc2:
        color_col = st.selectbox("Warna berdasarkan", ["make", "body-style", "fuel-type",
                                                        "drive-wheels", "price_segment"], key="sc_c")
    fig_sc = px.scatter(
        df_raw, x=x_col, y="price", color=color_col, opacity=0.8,
        size="horsepower", hover_data=["make", "model", "price"],
        title=f"{x_col.title()} vs Harga"
    )
    fig_sc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"), height=430
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Correlation Heatmap ──────────────────────────────────────────────────
    st.markdown("#### Korelasi Fitur Numerik")

    # Gunakan semua fitur numerik relevan (termasuk highway-mpg & popularity)
    ALL_NUM_COLS = ["year", "horsepower", "cylinders", "num-of-doors",
                    "highway-mpg", "city-mpg", "popularity", "price"]
    num_df = df_raw[[c for c in ALL_NUM_COLS if c in df_raw.columns]].copy()
    corr   = num_df.corr(numeric_only=True)

    # Warna diverging: merah negatif, putih 0, biru positif
    corr_scale = [
        [0.00, "#b91c1c"], [0.25, "#7f1d1d"],
        [0.50, "#1a1a2e"],
        [0.75, "#1e3a5f"], [1.00, "#2563eb"]
    ]
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale=corr_scale,
        zmin=-1, zmax=1,
        title="Correlation Matrix of Numerical Features"
    )
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc",
        height=560, title_font_size=14
    )
    fig_corr.update_traces(textfont=dict(size=11))
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Analisis pasangan berkorelasi tinggi ─────────────────────────────────
    st.markdown("##### Analisis Multikolinearitas — Pasangan Fitur Berkorelasi Tinggi")
    cols_list = list(corr.columns)
    high_pairs = []
    for i, r in enumerate(cols_list):
        for j, c in enumerate(cols_list):
            if i < j:
                v = abs(corr.loc[r, c])
                if v >= 0.70:
                    high_pairs.append({"Fitur A": r, "Fitur B": c, "|Korelasi|": round(v, 3)})

    if high_pairs:
        hp_df = pd.DataFrame(high_pairs).sort_values("|Korelasi|", ascending=False)
        st.dataframe(
            hp_df.style.background_gradient(subset=["|Korelasi|"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Tidak ada pasangan fitur dengan |korelasi| ≥ 0.70")

    # ── Info card pemilihan fitur ─────────────────────────────────────────────
    st.markdown("")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.error("""
**⛔ Fitur yang DIKELUARKAN dari model:**

- **`highway-mpg`** — Korelasi sangat tinggi dengan `city-mpg` (~0.89).
  Redundan karena mengukur hal yang sama (efisiensi BBM).
  Tetap tersedia di dataset untuk tampilan/filter.

- **`cylinders`** — Korelasi sangat tinggi dengan `horsepower` (~0.85).
  Informasi yang sama dikodekan dua kali → distorsi model.

- **`popularity`** — Nilai konstan per merk (make).
  Menyebabkan *target leakage* → akurasi model palsu 99%+.
        """)
    with col_info2:
        st.success("""
**✅ Fitur INPUT yang digunakan model:**

- **`year`** — Tahun produksi
- **`horsepower`** — Tenaga mesin (HP)
- **`num-of-doors`** — Jumlah pintu
- **`city-mpg`** — Konsumsi BBM kota (mewakili efisiensi)
- **+ Fitur kategorik** (body-style, fuel-type, transmission, drive-wheels, vehicle-size)
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TENTANG
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">Tentang Dataset & Proyek</p>', unsafe_allow_html=True)

    c_a, c_b = st.columns([1.6, 1])
    with c_a:
        st.markdown("""
        ### Edmunds / Kaggle Car Features & MSRP Dataset
 
        Dataset ini berisi spesifikasi teknis dan MSRP (harga pabrikan) mobil yang dijual di Amerika Serikat dari tahun **1990 hingga 2017**.
        Data ini disadur untuk melatih model rekomendasi berbasis konten yang tangguh.

        ---

        ### Model ML yang Digunakan

        | Model / Teknik | Tugas | Metrik |
        |---|---|---|
        | **Random Forest Classifier** | Prediksi merk mobil terbaik | F1-Score |
        | **Random Oversampling (NumPy)** | Menyeimbangkan class imbalance tanpa library eksternal | - |
        | **Random Forest Regressor** | Estimasi harga mobil sesuai spesifikasi | R², MAE, RMSE |
        | **KNN (cosine similarity)** | Sistem rekomendasi berbasis kemiripan | Cosine similarity |

        ---

        ### Fitur Dataset (16 Kolom)

        | Kelompok | Fitur |
        |---|---|
        | **Identitas** | make (merk), model (model), year (tahun) |
        | **Mesin** | horsepower (tenaga), cylinders* (info saja), fuel-type (BBM) |
        | **Konsumsi** | city-mpg, highway-mpg* |
        | **Bodi & Transmisi** | body-style, transmission, drive-wheels, num-of-doors, vehicle-size |
        | **Harga** | price (MSRP - target regresi) |

        *\*cylinders & highway-mpg tersedia di dataset namun tidak digunakan sebagai fitur input model (multikolinearitas)*
        """)
    with c_b:
        st.markdown("""
        ### 48 Merk Mobil Terdaftar
        Beberapa merk yang terdapat dalam dataset ini meliputi:
        * Toyota, Honda, Nissan, Mazda, Mitsubishi, Subaru
        * BMW, Mercedes-Benz, Audi, Porsche, Volkswagen, Volvo
        * Ford, Chevrolet, Dodge, GMC, Chrysler, Cadillac
        * Ferrari, Lamborghini, McLaren, Aston Martin, Bentley, Rolls-Royce
        * dan lain-lain.

        ---

        **Stack:**
        - Python 3.x + scikit-learn
        - Streamlit + Plotly
        - Pandas / NumPy
        """)
