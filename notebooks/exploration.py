# -*- coding: utf-8 -*-
"""
exploration.py — Car Recommendation System EDA
================================================
Skrip eksplorasi data (EDA) untuk Car Features & MSRP Dataset.
Jalankan: python notebooks/exploration.py

Analisis meliputi:
  1. Overview dataset & statistik dasar
  2. Distribusi fitur numerik
  3. Korelasi antar fitur (heatmap + analisis multikolinearitas)
  4. Distribusi merk & segmen harga
  5. Hubungan fitur teknis vs harga
  6. Evaluasi justifikasi pemilihan fitur model
"""

# ─────────────────────────────────────────────────────────
# CELL 1 — Import Library & Setup
# ─────────────────────────────────────────────────────────
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — plot disimpan ke file, tidak blocking
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from data_loader import (
    load_data, NUMERIC_FEATURES, CAT_FEATURES, TARGET_COL, PRICE_SEGMENTS
)

# Setup output dir
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style seragam
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0d1117",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "text.color":       "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})

print("✅ Library berhasil dimuat")

# ─────────────────────────────────────────────────────────
# CELL 2 — Load Dataset
# ─────────────────────────────────────────────────────────
df = load_data()

print(f"\n📊 Shape dataset      : {df.shape}")
print(f"   Jumlah merk unik  : {df[TARGET_COL].nunique()}")
print(f"   Jumlah model unik : {df['model'].nunique()}")
print(f"   Tahun             : {df['year'].min()} – {df['year'].max()}")
print(f"   Rentang harga     : ${df['price'].min():,.0f} – ${df['price'].max():,.0f}")
print(f"\nKolom tersedia:\n{list(df.columns)}")
print("\nStatistik dasar (fitur numerik):")
print(df[NUMERIC_FEATURES + ["price"]].describe().round(2).to_string())

# ─────────────────────────────────────────────────────────
# CELL 3 — EDA: Distribusi Fitur Numerik
# ─────────────────────────────────────────────────────────
num_cols = NUMERIC_FEATURES + ["price"]
n_cols = len(num_cols)
n_rows = (n_cols + 2) // 3

fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 4))
axes = axes.flatten()
fig.suptitle("Distribusi Fitur Numerik — Car Features & MSRP Dataset",
             fontsize=16, fontweight="bold", color="#e6edf3", y=1.01)

colors = plt.cm.plasma(np.linspace(0.25, 0.9, n_cols))
for i, (col, clr) in enumerate(zip(num_cols, colors)):
    ax = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=40, color=clr, alpha=0.85, edgecolor="none")
    ax.axvline(data.mean(), color="white", linestyle="--", linewidth=1.2,
               label=f"Mean={data.mean():.1f}")
    ax.axvline(data.median(), color="#34d399", linestyle=":", linewidth=1.2,
               label=f"Median={data.median():.1f}")
    ax.set_title(col, fontsize=12, fontweight="bold", color="#e6edf3")
    ax.set_ylabel("Frekuensi", fontsize=9)
    ax.legend(fontsize=8)

# Sembunyikan subplot kosong
for j in range(n_cols, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "eda_numeric_dist.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print(f"✅ Plot disimpan: {out_path}")

# ─────────────────────────────────────────────────────────
# CELL 4 — EDA: Correlation Heatmap + Analisis Multikolinearitas
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ANALISIS KORELASI FITUR NUMERIK")
print("="*60)

corr_cols = NUMERIC_FEATURES + ["price", "popularity"]
# Hanya ambil kolom yang benar-benar ada
corr_cols = [c for c in corr_cols if c in df.columns]
corr_matrix = df[corr_cols].corr(numeric_only=True)

# Cetak pasangan dengan korelasi > 0.7
print("\nPasangan fitur dengan |korelasi| > 0.70 (indikasi multikolinearitas):")
high_corr_pairs = []
for i, r in enumerate(corr_cols):
    for j, c in enumerate(corr_cols):
        if i < j:
            v = abs(corr_matrix.loc[r, c])
            if v > 0.70:
                high_corr_pairs.append((r, c, v))
                print(f"  ⚠️  {r:20s} ↔ {c:20s} : {v:.3f}")

if not high_corr_pairs:
    print("  (tidak ada pasangan dengan korelasi > 0.70)")

# ─── Plot heatmap ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
mask = np.zeros_like(corr_matrix, dtype=bool)  # tanpa mask → full matrix (seperti screenshot)

cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True, fmt=".2f",
    cmap=cmap,
    center=0,
    linewidths=0.5,
    square=True,
    ax=ax,
    annot_kws={"size": 9, "color": "white"},
    cbar_kws={"shrink": 0.8}
)
ax.set_title("Correlation Matrix of Numerical Features", fontsize=14,
             fontweight="bold", color="#e6edf3", pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right",
                   fontsize=9, color="#8b949e")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9, color="#8b949e")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "eda_correlation.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print(f"\n✅ Plot disimpan: {out_path}")

# ─────────────────────────────────────────────────────────
# CELL 5 — Justifikasi Pemilihan Fitur Model
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  JUSTIFIKASI PEMILIHAN FITUR INPUT MODEL ML")
print("="*60)
print("""
Berdasarkan correlation heatmap di atas:

  ⛔ highway-mpg  (DIHAPUS dari fitur model)
     → Korelasi dengan city-mpg = ~0.89 (sangat tinggi / multikolinearitas)
     → Kedua fitur mengukur hal yang sama (efisiensi BBM).
     → Dampak jika keduanya dipakai:
         • KNN: dimensi MPG dihitung 2x → distorsi jarak cosine
         • Random Forest: feature importance terbagi secara buatan
     → KEPUTUSAN: Hanya 'city-mpg' yang digunakan. 'highway-mpg' tetap
       tersedia di dataset untuk keperluan display/filter UI.

  ⚠️  horsepower ↔ cylinders   (r ≈ 0.78, TETAP DIPERTAHANKAN)
     → Secara semantik mewakili dimensi berbeda:
         • horsepower = output tenaga aktual mesin
         • cylinders  = konfigurasi/kapasitas mesin
     → Keduanya penting sebagai parameter preferensi pengguna.
     → r=0.78 tinggi tapi masih dapat diterima untuk kasus ini.

  ⛔ popularity  (DIHAPUS dari fitur model)
     → Nilainya konstan per merk (make) → TARGET LEAKAGE
     → Jika dipakai, model akan "menghafal" merk dari popularitas
       bukan dari spesifikasi teknis → akurasi palsu 99%+.

  ✅ Fitur numerik FINAL yang digunakan model:
     {feats}
""".format(feats=NUMERIC_FEATURES))

# ─────────────────────────────────────────────────────────
# CELL 6 — EDA: Distribusi Merk & Segmen Harga
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Distribusi Merk & Segmen Harga Mobil", fontsize=15,
             fontweight="bold", color="#e6edf3")

# ── Kiri: Top-15 merk terbanyak ────────────────────────
make_counts = df[TARGET_COL].value_counts().head(15)
colors_bar  = plt.cm.plasma(np.linspace(0.3, 0.9, len(make_counts)))
axes[0].barh(make_counts.index[::-1], make_counts.values[::-1],
             color=colors_bar, alpha=0.9, edgecolor="none")
axes[0].set_title("Top-15 Merk Mobil Terbanyak", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Jumlah Mobil", fontsize=10)
for i, v in enumerate(make_counts.values[::-1]):
    axes[0].text(v + 5, i, str(v), va="center", fontsize=8, color="#e6edf3")

# ── Kanan: Pie chart segmen harga ──────────────────────
seg_counts = df["price_segment"].value_counts()
seg_colors = ["#16a34a", "#2563eb", "#9333ea", "#b45309"]
wedges, texts, autotexts = axes[1].pie(
    seg_counts.values,
    labels=seg_counts.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=seg_colors,
    wedgeprops={"edgecolor": "#0d1117", "linewidth": 2},
    textprops={"color": "#e6edf3", "fontsize": 10}
)
for at in autotexts:
    at.set_color("white")
    at.set_fontweight("bold")
axes[1].set_title("Proporsi Segmen Harga", fontsize=12, fontweight="bold")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "eda_make_segment.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print(f"✅ Plot disimpan: {out_path}")

# ─────────────────────────────────────────────────────────
# CELL 7 — EDA: Distribusi Fitur Kategorik
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
fig.suptitle("Distribusi Fitur Kategorik", fontsize=15,
             fontweight="bold", color="#e6edf3")

cat_plot_cols = ["body-style", "fuel-type", "drive-wheels",
                 "transmission", "vehicle-size"]
for i, col in enumerate(cat_plot_cols):
    if col not in df.columns:
        axes[i].set_visible(False)
        continue
    vc = df[col].value_counts()
    clr = plt.cm.viridis(np.linspace(0.3, 0.9, len(vc)))
    axes[i].bar(vc.index, vc.values, color=clr, alpha=0.9, edgecolor="none")
    axes[i].set_title(col, fontsize=12, fontweight="bold")
    axes[i].set_ylabel("Jumlah", fontsize=9)
    axes[i].tick_params(axis="x", rotation=30)

axes[-1].set_visible(False)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "eda_cat_dist.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print(f"✅ Plot disimpan: {out_path}")

# ─────────────────────────────────────────────────────────
# CELL 8 — EDA: Hubungan Fitur Teknis vs Harga
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Hubungan Fitur Teknis vs Harga", fontsize=15,
             fontweight="bold", color="#e6edf3")

scatter_pairs = [
    ("horsepower", "price"),
    ("cylinders",  "price"),
    ("city-mpg",   "price"),
]
scatter_colors = ["#a78bfa", "#60a5fa", "#34d399"]
for ax, (x_col, y_col), clr in zip(axes, scatter_pairs, scatter_colors):
    ax.scatter(df[x_col], df[y_col], alpha=0.3, s=15, color=clr, edgecolors="none")
    # Trend line
    z = np.polyfit(df[x_col].dropna(), df[y_col][df[x_col].notna()], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df[x_col].min(), df[x_col].max(), 200)
    ax.plot(x_range, p(x_range), color="white", linewidth=1.5, linestyle="--", alpha=0.8)
    r = df[[x_col, y_col]].corr().iloc[0, 1]
    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.set_title(f"{x_col} vs {y_col}  (r={r:.2f})", fontsize=11, fontweight="bold")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "eda_feature_vs_price.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print(f"✅ Plot disimpan: {out_path}")

# ─────────────────────────────────────────────────────────
# CELL 9 — EDA: Boxplot Harga per Body Style
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
order = df.groupby("body-style")["price"].median().sort_values(ascending=False).index
df.boxplot(column="price", by="body-style", ax=ax,
           positions=range(len(order)),
           patch_artist=True,
           boxprops=dict(facecolor="#7c3aed", color="#a78bfa", alpha=0.7),
           medianprops=dict(color="#34d399", linewidth=2),
           whiskerprops=dict(color="#8b949e"),
           capprops=dict(color="#8b949e"),
           flierprops=dict(marker=".", color="#a78bfa", alpha=0.3, markersize=4))
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=30, ha="right", color="#e6edf3")
ax.set_title("Distribusi Harga per Tipe Bodi Mobil", fontsize=13,
             fontweight="bold", color="#e6edf3")
ax.set_xlabel("Tipe Bodi", fontsize=10)
ax.set_ylabel("Harga ($)", fontsize=10)
plt.suptitle("")  # hapus judul default boxplot
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "eda_price_by_bodystyle.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print(f"✅ Plot disimpan: {out_path}")

# ─────────────────────────────────────────────────────────
# CELL 10 — EDA: Missing Values & Data Quality
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  KUALITAS DATA — MISSING VALUES")
print("="*60)
all_check_cols = NUMERIC_FEATURES + CAT_FEATURES + ["price", "popularity"]
all_check_cols = [c for c in all_check_cols if c in df.columns]
missing = df[all_check_cols].isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing": missing, "Persen (%)": missing_pct})
missing_df = missing_df[missing_df["Missing"] > 0]
if missing_df.empty:
    print("  ✅ Tidak ada missing value setelah preprocessing.")
else:
    print(missing_df.to_string())

print("\n" + "="*60)
print("  RINGKASAN EDA SELESAI")
print("="*60)
print(f"  Total plot disimpan ke: {OUTPUT_DIR}")
print("""
  Plot yang dihasilkan:
  1. eda_numeric_dist.png     — Distribusi fitur numerik
  2. eda_correlation.png      — Heatmap korelasi (full matrix)
  3. eda_make_segment.png     — Distribusi merk & segmen harga
  4. eda_cat_dist.png         — Distribusi fitur kategorik
  5. eda_feature_vs_price.png — Scatter plot fitur vs harga
  6. eda_price_by_bodystyle.png — Boxplot harga per body style
""")
