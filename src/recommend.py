"""
recommend.py
============
Content-based car recommendation menggunakan KNN cosine similarity.
Mengembalikan daftar mobil spesifik (merk, model, tipe bodi, dll.)
yang paling mirip dengan preferensi user.
"""

import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import (
    NUMERIC_FEATURES, CAT_FEATURES, TARGET_COL, label_price_segment
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

# Kolom tampilan di hasil rekomendasi
DISPLAY_COLS = [
    "make", "make_display", "model", "year", "body-style", "fuel-type",
    "transmission", "drive-wheels", "num-of-doors", "horsepower",
    "city-mpg", "highway-mpg", "popularity", "price",
    "price_segment", "similarity (%)"
]


def _encode_input(user_prefs: dict, df_ref: pd.DataFrame,
                  one_hot_cols: list) -> np.ndarray:
    """
    Konversi preferensi user (dict) ke vektor numerik sesuai format model (One-Hot Encoded).

    Parameters
    ----------
    user_prefs   : dict  Preferensi user
    df_ref       : pd.DataFrame  Dataset referensi untuk median imputation
    one_hot_cols : list  List nama kolom biner hasil pd.get_dummies

    Returns
    -------
    np.ndarray shape (1, n_features)
    """
    vec = []

    # ── Numerik ────────────────────────────────────────────────────────────────
    for col in NUMERIC_FEATURES:
        if col in user_prefs:
            vec.append(float(user_prefs[col]))
        else:
            vec.append(float(df_ref[col].median()))

    # ── Kategorikal (One-Hot Encoded) ──────────────────────────────────────────
    for col in one_hot_cols:
        base_feature = None
        for cat in CAT_FEATURES:
            if col.startswith(cat + "_"):
                base_feature = cat
                break

        if base_feature is not None:
            user_val = user_prefs.get(base_feature, None)
            if user_val is not None:
                val_part = col[len(base_feature) + 1:]
                if str(user_val).lower() == val_part.lower():
                    vec.append(1.0)
                else:
                    vec.append(0.0)
            else:
                # Jika user memilih "(Semua)", kita isi dengan 0.0 agar netral
                vec.append(0.0)
        else:
            vec.append(0.0)

    return np.array(vec, dtype=float).reshape(1, -1)


def get_recommendations(
    user_prefs: dict,
    scaler,
    one_hot_cols: list,
    df_labeled: pd.DataFrame,
    classifier=None,
    regressor=None,
    n_recommendations: int = 5,
    body_style_filter: str = None,
    fuel_type_filter: str = None,
    transmission_filter: str = None,
    drive_wheels_filter: str = None,
    max_price: float = None,
    make_filter: str = None,
) -> pd.DataFrame:
    """
    Rekomendasikan mobil berdasarkan preferensi user.

    Parameters
    ----------
    user_prefs        : dict  Preferensi user (bisa sebagian kolom)
    scaler            : StandardScaler
    one_hot_cols      : list  List kolom biner hasil pd.get_dummies
    df_labeled        : pd.DataFrame  Dataset asli yang sudah diproses
    classifier        : model (opsional) prediksi merk terbaik
    regressor         : model (opsional) estimasi harga
    n_recommendations : int  Jumlah rekomendasi
    body_style_filter : str  Filter body style ("sedan", "suv", dll.)
    fuel_type_filter  : str  Filter tipe BBM ("gas" / "diesel" / "electric" dll.)
    transmission_filter: str Filter tipe transmisi ("MANUAL" / "AUTOMATIC" dll.)
    drive_wheels_filter: str Filter penggerak roda ("fwd" / "rwd" / "awd" / "4wd")
    max_price         : float  Filter harga maksimal
    make_filter       : str  Filter khusus merk/brand tertentu (misal: "Cadillac")

    Returns
    -------
    pd.DataFrame  — Top-N rekomendasi mobil dengan similarity score
    """
    # ── 1. Encode input user ──────────────────────────────────────────────────
    input_vec    = _encode_input(user_prefs, df_labeled, one_hot_cols)
    input_scaled = scaler.transform(input_vec)

    # ── 2. Prediksi merk & harga (opsional) ──────────────────────────────
    pred_make  = None
    pred_price = None

    if classifier is not None:
        pred_make  = classifier.predict(input_scaled)[0]
        pred_proba = classifier.predict_proba(input_scaled)[0]
        classes    = classifier.classes_
        print(f"\n[PREDICT] Merk terbaik untuk preferensi Anda: {pred_make.upper()}")
        print("  Probabilitas top-5 merk:")
        top5_idx = np.argsort(pred_proba)[::-1][:5]
        for i in top5_idx:
            print(f"  - {classes[i]:20s}: {pred_proba[i]*100:.1f}%")

    if regressor is not None:
        pred_price = float(regressor.predict(input_scaled)[0])
        print(f"[PREDICT] Estimasi harga: ${pred_price:,.0f}")

    # ── 3. Terapkan filter pada dataset ──────────────────────────────────────
    df_f = df_labeled.copy()
    if make_filter:
        df_f = df_f[df_f["make"].str.lower() == make_filter.lower()]

    df_brand_baseline = df_f.copy()

    if body_style_filter:
        df_f = df_f[df_f["body-style"] == body_style_filter]
    if fuel_type_filter:
        df_f = df_f[df_f["fuel-type"] == fuel_type_filter]
    if transmission_filter:
        df_f = df_f[df_f["transmission"] == transmission_filter]
    if drive_wheels_filter:
        df_f = df_f[df_f["drive-wheels"] == drive_wheels_filter]
    if max_price is not None:
        df_f = df_f[df_f["price"] <= max_price]

    if len(df_f) == 0:
        print("[WARN] Filter terlalu ketat, tidak ada data — reset filter.")
        if make_filter:
            df_f = df_brand_baseline.copy()
            if max_price is not None:
                df_f_price = df_f[df_f["price"] <= max_price]
                if len(df_f_price) > 0:
                    df_f = df_f_price
        else:
            df_f = df_labeled.copy()

    df_f = df_f.reset_index(drop=True)

    # ── 4. KNN lokal pada dataset yang sudah difilter ─────────────────────────
    all_cols = NUMERIC_FEATURES + one_hot_cols
    X_f = scaler.transform(df_f[all_cols].to_numpy(dtype=float))
    
    input_scaled_knn = input_scaled.copy()
    X_f_knn = X_f.copy()

    # Netralisasi kolom one-hot untuk kategori yang tidak ditentukan (user memilih "(Semua)")
    # Jika kategori (misal: 'body-style') tidak ada di user_prefs, set nilainya ke 0.0
    # pada query maupun database kandidat agar tidak membiaskan pencarian.
    for cat in CAT_FEATURES:
        if cat not in user_prefs:
            for i, col in enumerate(all_cols):
                if col.startswith(cat + "_"):
                    input_scaled_knn[0, i] = 0.0
                    X_f_knn[:, i] = 0.0

    # Netralisasi fitur numerik yang tidak ditentukan oleh user
    # Jika fitur numerik (misal: 'year', 'num-of-doors') tidak ada di user_prefs,
    # set nilainya ke 0.0 (netral dalam skala StandardScaler)
    for num in NUMERIC_FEATURES:
        if num not in user_prefs:
            for i, col in enumerate(all_cols):
                if col == num:
                    input_scaled_knn[0, i] = 0.0
                    X_f_knn[:, i] = 0.0

    # Ambil hingga 1000 tetangga terdekat agar bisa melakukan drop duplicate model
    # dan menyertakan merk yang diprediksi oleh classifier yang mungkin berada di luar top 100.
    k = min(1000, len(df_f))
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")
    knn.fit(X_f_knn)

    distances, indices = knn.kneighbors(input_scaled_knn)

    # ── 5. Susun hasil ─────────────────────────────────────────────────────────
    results = []
    dist_arr = distances[0]
    # Rescale similarity agar lebih bermakna dan informatif di UI:
    # cosine distance biasanya sangat kecil (0.001 - 0.1) setelah StandardScaler,
    # sehingga rumus lama (cos+1)/2 selalu ~100%.
    # Solusi: normalisasi terhadap rentang distance aktual dalam hasil ini.
    d_min = dist_arr.min()
    d_max = dist_arr.max() if dist_arr.max() > dist_arr.min() else d_min + 1e-9
    for dist, idx in zip(dist_arr, indices[0]):
        row = df_f.iloc[idx].copy()
        # Semakin kecil distance → semakin tinggi similarity
        # Normalisasi: 100% untuk distance terkecil, turun ke ~50% untuk distance terbesar
        normalized = 1.0 - (dist - d_min) / (d_max - d_min)  # range [0, 1]
        final_sim   = round(50.0 + normalized * 50.0, 1)       # range [50%, 100%]
        row["similarity (%)"] = final_sim
        results.append(row)

    result_df = pd.DataFrame(results)
    
    # Drop duplikat baris dengan kombinasi Merk + Model yang sama
    # (mencegah 1 tipe mobil mendominasi daftar rekomendasi)
    result_df = result_df.drop_duplicates(subset=["make", "model"])

    # Pastikan kolom display ada
    existing_display = [c for c in DISPLAY_COLS if c in result_df.columns]
    result_df = (
        result_df[existing_display]
        .sort_values("similarity (%)", ascending=False)
        .head(n_recommendations)
        .reset_index(drop=True)
    )

    return result_df, pred_make, pred_price
