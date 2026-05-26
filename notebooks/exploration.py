# Wine Quality Recommendation System — Jupyter Notebook (Script version)
# Jalankan cell per cell atau sebagai script: python notebooks/exploration.py

# ─────────────────────────────────────────────────────────
# CELL 1 — Import Library
# ─────────────────────────────────────────────────────────
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from data_loader import load_data, get_train_test_split, FEATURE_COLS, label_quality

print("✅ Library berhasil dimuat")

# ─────────────────────────────────────────────────────────
# CELL 2 — Load Dataset
# ─────────────────────────────────────────────────────────
df = load_data("both")
df["quality_label"] = df["quality"].apply(label_quality)

print(f"\n📊 Shape dataset: {df.shape}")
print(f"Kolom: {list(df.columns)}")
print("\nStatistik dasar:")
print(df.describe().round(3).to_string())

# ─────────────────────────────────────────────────────────
# CELL 3 — EDA: Distribusi Kualitas
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Distribusi Kualitas Wine", fontsize=16, fontweight="bold")

# Plot 1: Score distribution
quality_counts = df.groupby(["quality", "wine_type"]).size().unstack(fill_value=0)
quality_counts.plot(kind="bar", ax=axes[0], color=["#8b0000", "#d4a017"], edgecolor="white", width=0.7)
axes[0].set_title("Distribusi Skor per Jenis Wine")
axes[0].set_xlabel("Skor Kualitas")
axes[0].set_ylabel("Jumlah Sampel")
axes[0].legend(["Red Wine", "White Wine"])
axes[0].tick_params(axis="x", rotation=0)

# Plot 2: Pie chart kategori
label_counts = df["quality_label"].value_counts()
axes[1].pie(
    label_counts.values, labels=label_counts.index,
    autopct="%1.1f%%", startangle=140,
    colors=["#e74c3c", "#f39c12", "#27ae60"],
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
axes[1].set_title("Proporsi Kategori Kualitas")

plt.tight_layout()
output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "eda_quality_dist.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Plot disimpan: output/eda_quality_dist.png")

# ─────────────────────────────────────────────────────────
# CELL 4 — EDA: Distribusi Fitur
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
fig.suptitle("Distribusi Fitur Wine per Kategori Kualitas", fontsize=16, fontweight="bold")

palette = {"Low": "#e74c3c", "Medium": "#f39c12", "High": "#2ecc71"}

for i, feat in enumerate(FEATURE_COLS):
    for label, color in palette.items():
        subset = df[df["quality_label"] == label][feat]
        axes[i].hist(subset, alpha=0.55, bins=30, label=label, color=color, density=True)
    axes[i].set_title(feat, fontsize=11)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Density")
    if i == 0:
        axes[i].legend(fontsize=9)

# Hide last subplot (kita punya 11 fitur, 12 subplots)
axes[-1].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_feature_dist.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Plot disimpan: output/eda_feature_dist.png")

# ─────────────────────────────────────────────────────────
# CELL 5 — EDA: Correlation Heatmap
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 10))
corr_matrix = df[FEATURE_COLS + ["quality"]].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, linewidths=0.5,
    square=True, ax=ax, annot_kws={"size": 9}
)
ax.set_title("Heatmap Korelasi Fitur Wine", fontsize=15, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_correlation.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Plot disimpan: output/eda_correlation.png")

# ─────────────────────────────────────────────────────────
# CELL 6 — Boxplot: Alkohol vs Kualitas
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Fitur Kunci vs Kategori Kualitas", fontsize=15, fontweight="bold")

key_features = ["alcohol", "volatile acidity", "sulphates"]
for i, feat in enumerate(key_features):
    order = ["Low", "Medium", "High"]
    palette_list = [palette[k] for k in order]
    df_box = [df[df["quality_label"] == k][feat].values for k in order]
    bp = axes[i].boxplot(df_box, labels=order, patch_artist=True)
    for patch, color in zip(bp["boxes"], palette_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[i].set_title(feat.title(), fontsize=12)
    axes[i].set_ylabel(feat)
    axes[i].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_boxplot_key.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Plot disimpan: output/eda_boxplot_key.png")

# ─────────────────────────────────────────────────────────
# CELL 7 — Load Model & Evaluasi
# ─────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def check_model_exists(name):
    return os.path.exists(os.path.join(MODELS_DIR, f"{name}.pkl"))

if not check_model_exists("random_forest"):
    print("⚠️ Model belum ditraining. Jalankan: python src/train_model.py")
else:
    rf     = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    gb     = joblib.load(os.path.join(MODELS_DIR, "gradient_boosting.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    X_train, X_test, y_train, y_test, _, _ = get_train_test_split(df)

    for name, model in [("Random Forest", rf), ("Gradient Boosting", gb)]:
        y_pred = model.predict(X_test)
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred, target_names=["High", "Low", "Medium"], zero_division=0))

    # ─────────────────────────────────────────────────────────
    # CELL 8 — Feature Importance
    # ─────────────────────────────────────────────────────────
    importances = rf.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]
    sorted_feat = [FEATURE_COLS[i] for i in sorted_idx]
    sorted_imp  = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(sorted_imp / sorted_imp.max())
    bars = ax.barh(sorted_feat[::-1], sorted_imp[::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Feature Importance — Random Forest Classifier", fontsize=14, fontweight="bold", pad=10)
    ax.axvline(x=sorted_imp.mean(), color="navy", linestyle="--", alpha=0.7, label=f"Mean = {sorted_imp.mean():.3f}")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n📊 Feature terpenting: {sorted_feat[0]} ({sorted_imp[0]:.4f})")
    print(f"   Feature kedua      : {sorted_feat[1]} ({sorted_imp[1]:.4f})")

    # ─────────────────────────────────────────────────────────
    # CELL 9 — Confusion Matrix
    # ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrix", fontsize=15, fontweight="bold")
    class_order = ["Low", "Medium", "High"]

    for ax, (name, model) in zip(axes, [("Random Forest", rf), ("Gradient Boosting", gb)]):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=class_order)
        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax, cmap="Blues",
            xticklabels=class_order, yticklabels=class_order,
            linewidths=0.5, linecolor="gray"
        )
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # ─────────────────────────────────────────────────────────
    # CELL 10 — Demo Rekomendasi
    # ─────────────────────────────────────────────────────────
    df_labeled = joblib.load(os.path.join(MODELS_DIR, "df_labeled.pkl"))
    from recommend import get_recommendations

    print("\n" + "="*60)
    print("  DEMO REKOMENDASI WINE")
    print("="*60)

    user_prefs = {
        "alcohol":          13.5,   # tinggi
        "pH":               3.1,    # sedikit asam
        "volatile acidity": 0.25,   # rendah (lebih enak)
        "residual sugar":   2.0,    # kering
        "sulphates":        0.65,   # sedang
        "citric acid":      0.4,
    }

    print("\n🧑 Preferensi Pengguna:")
    for k, v in user_prefs.items():
        print(f"   {k:25s} = {v}")

    result = get_recommendations(
        user_preferences=user_prefs,
        knn_recommender=None,
        scaler=scaler,
        df_labeled=df_labeled,
        classifier=rf,
        n_recommendations=5,
        wine_type_filter="both"
    )

    print("\n🏆 Top 5 Rekomendasi Wine:")
    display_cols = ["wine_type", "quality", "quality_label", "similarity (%)", "alcohol", "pH", "sulphates"]
    print(result[display_cols].to_string(index=False))

    print("\n✅ Notebook selesai! Semua plot disimpan ke folder output/")
