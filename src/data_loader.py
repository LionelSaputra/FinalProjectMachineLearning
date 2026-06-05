"""
data_loader.py
==============
Unduh dan preprocess dataset mobil Edmunds/Kaggle (Car Features & MSRP).
Dataset: https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv

Kolom utama setelah diproses:
- make          : merk mobil (BMW, Chevrolet, Toyota, dll.) — target klasifikasi
- model         : model mobil (Corolla, Camry, 3 Series, dll.)
- year          : tahun pembuatan
- price         : harga mobil (MSRP)
- horsepower    : tenaga mesin (HP)
- cylinders     : jumlah silinder (tidak digunakan sebagai fitur model — korelasi tinggi dengan horsepower)
- transmission  : transmisi (AUTOMATIC, MANUAL, dll.)
- drive-wheels  : penggerak roda (fwd, rwd, awd, 4wd)
- fuel-type     : jenis bahan bakar (gas, diesel, electric, dll.)
- body-style    : bodi mobil (sedan, suv, coupe, dll.)
"""

import os
import sys
import requests
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Paksa stdout UTF-8 di Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ─── URL Dataset ───────────────────────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
LOCAL_FILE = os.path.join(DATA_DIR, "car_features_msrp.csv")

# ─── Fitur numerik untuk model ML ───────────────────────────────────────────────
# Fitur yang dikeluarkan (tidak digunakan sebagai input model):
#   - 'popularity'  : nilainya konstan per merk (make) → TARGET LEAKAGE (akurasi palsu 99%+)
#   - 'highway-mpg' : korelasi sangat tinggi dengan 'city-mpg' (r ≈ 0.89) → MULTIKOLINEARITAS
#                     Kedua fitur mengukur hal yang sama (efisiensi BBM).
#                     Mempertahankan keduanya akan mendistorsi jarak KNN (dimensi MPG
#                     dihitung 2x) dan membagi feature importance RF secara buatan.
#                     Kolom ini tetap ada di dataset untuk keperluan display/filter UI.
#   - 'cylinders'   : korelasi sangat tinggi dengan 'horsepower' (r ≈ 0.85) → MULTIKOLINEARITAS
#                     Mesin bertenaga tinggi hampir selalu memiliki lebih banyak silinder.
#                     Mempertahankan keduanya mendistorsi model karena informasi yang sama
#                     dikodekan dua kali.
NUMERIC_FEATURES = [
    "year", "horsepower", "num-of-doors",
    "city-mpg"
]

# ─── Fitur kategorikal (untuk encoding) ────────────────────────────────────────
CAT_FEATURES = [
    "fuel-type", "transmission", "drive-wheels",
    "vehicle-size", "body-style"
]

# ─── Fitur gabungan untuk KNN ──────────────────────────────────────────────────
ALL_FEATURES  = NUMERIC_FEATURES + CAT_FEATURES
TARGET_COL    = "make"

# ─── Segmen harga ──────────────────────────────────────────────────────────────
PRICE_SEGMENTS = {
    "Budget (< $15K)":     (0,      15000),
    "Mid-Range ($15-30K)": (15000,  30000),
    "Premium ($30-60K)":   (30000,  60000),
    "Luxury (> $60K)":     (60000,  99999999),
}


def download_dataset() -> None:
    """Unduh dataset jika belum ada secara lokal."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LOCAL_FILE):
        print("[INFO] Mengunduh dataset Edmunds Car Features & MSRP...")
        resp = requests.get(DATA_URL, timeout=30)
        resp.raise_for_status()
        with open(LOCAL_FILE, "wb") as f:
            f.write(resp.content)
        print("[INFO] Dataset berhasil diunduh.")
    else:
        print("[INFO] Dataset sudah ada secara lokal, melewati unduhan.")


def clean_fuel_type(val: str) -> str:
    """Sederhanakan kategori tipe bahan bakar."""
    val = str(val).lower()
    if "diesel" in val:
        return "diesel"
    elif "electric" in val:
        return "electric"
    elif "flex-fuel" in val:
        return "flex-fuel"
    elif "natural gas" in val:
        return "natural gas"
    else:
        return "gas"


def clean_drive_wheels(val: str) -> str:
    """Sederhanakan kategori penggerak roda."""
    val = str(val).lower()
    if "front" in val:
        return "fwd"
    elif "rear" in val:
        return "rwd"
    elif "all" in val:
        return "awd"
    elif "four" in val:
        return "4wd"
    return "fwd"


def clean_body_style(val: str) -> str:
    """Kelompokkan tipe bodi mobil ke kategori standar."""
    val = str(val).lower()
    if "sedan" in val:
        return "sedan"
    elif "coupe" in val:
        return "coupe"
    elif "convertible" in val:
        return "convertible"
    elif "suv" in val:
        return "suv"
    elif "hatchback" in val:
        return "hatchback"
    elif "wagon" in val:
        return "wagon"
    elif "pickup" in val:
        return "pickup"
    elif "van" in val or "minivan" in val:
        return "van"
    return "sedan"


def load_data() -> pd.DataFrame:
    """
    Muat dan bersihkan dataset Edmunds Car Features.

    Returns
    -------
    pd.DataFrame  DataFrame yang sudah bersih dengan nama kolom baru.
    """
    download_dataset()
    df = pd.read_csv(LOCAL_FILE)

    # ─── Mapping nama kolom agar pythonic ───────────────────────────────────────
    col_mapping = {
        "Make": "make",
        "Model": "model",
        "Year": "year",
        "Engine Fuel Type": "fuel-type",
        "Engine HP": "horsepower",
        "Engine Cylinders": "cylinders",
        "Transmission Type": "transmission",
        "Driven_Wheels": "drive-wheels",
        "Number of Doors": "num-of-doors",
        "Vehicle Size": "vehicle-size",
        "Vehicle Style": "body-style",
        "highway MPG": "highway-mpg",
        "city mpg": "city-mpg",
        "Popularity": "popularity",
        "MSRP": "price"
    }
    df = df.rename(columns=col_mapping)

    # Pastikan target kolom utama ada dan tidak bernilai kosong
    df = df.dropna(subset=["price", "make"]).copy()

    # ─── Sederhanakan kategori kategorikal ──────────────────────────────────────
    df["fuel-type"] = df["fuel-type"].apply(clean_fuel_type)
    df["drive-wheels"] = df["drive-wheels"].apply(clean_drive_wheels)
    df["body-style"] = df["body-style"].apply(clean_body_style)

    # ─── Imputasi nilai yang hilang ─────────────────────────────────────────────
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # ─── Tambah kolom segmentasi harga & tampilan merk ─────────────────────────
    df["price_segment"] = df["price"].apply(label_price_segment)
    df["make_display"]  = df["make"].str.replace("-", " ").str.title()

    df = df.reset_index(drop=True)
    return df


def label_price_segment(price: float) -> str:
    """Konversi nominal harga ke kategori segmen."""
    for label, (lo, hi) in PRICE_SEGMENTS.items():
        if lo <= price < hi:
            return label
    return "Luxury (> $60K)"


def preprocess(df: pd.DataFrame, scaler: StandardScaler = None,
               encoders = None):
    """
    Preprocessing data: one-hot encoding kategorikal + standard scaling numerik.

    Returns
    -------
    X_scaled   : np.ndarray
    y          : np.ndarray
    scaler     : StandardScaler
    encoders   : list (menyimpan list nama kolom one-hot untuk menjaga konsistensi)
    df_encoded : pd.DataFrame
    """
    df = df.copy()

    # ─── Encode fitur kategorikal (One-Hot Encoding) ────────────────────────────
    df_cat = pd.get_dummies(df[CAT_FEATURES], columns=CAT_FEATURES, dtype=float)

    if encoders is None:
        encoders = list(df_cat.columns)
    else:
        df_cat = df_cat.reindex(columns=encoders, fill_value=0.0)

    # Gabungkan kolom numerik, one-hot, dan kembalikan kolom kategorikal asli untuk UI
    df_encoded = pd.concat([df.drop(columns=CAT_FEATURES), df_cat], axis=1)
    for col in CAT_FEATURES:
        df_encoded[col] = df[col]

    all_cols = NUMERIC_FEATURES + encoders
    X        = df_encoded[all_cols].to_numpy(dtype=float)
    y        = df_encoded[TARGET_COL].to_numpy(dtype=str)

    if scaler is None:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler, encoders, df_encoded


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2,
                          random_state: int = 42):
    """
    Membagi dataset menjadi train & test set.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, encoders, df_encoded
    """
    # ─── Encode fitur kategorikal (One-Hot Encoding) tanpa scaling dulu ────────
    df_copy = df.copy()
    df_cat = pd.get_dummies(df_copy[CAT_FEATURES], columns=CAT_FEATURES, dtype=float)
    encoders = list(df_cat.columns)
    
    df_encoded = pd.concat([df_copy.drop(columns=CAT_FEATURES), df_cat], axis=1)
    for col in CAT_FEATURES:
        df_encoded[col] = df_copy[col]
        
    all_cols = NUMERIC_FEATURES + encoders
    X = df_encoded[all_cols].to_numpy(dtype=float)
    y = df_encoded[TARGET_COL].to_numpy(dtype=str)

    # Stratify jika jumlah sample tiap kelas mencukupi
    class_counts = np.unique(y, return_counts=True)[1]
    stratify = y if class_counts.min() >= 2 else None

    # Split index
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    
    # ─── FIT SCALER HANYA PADA DATA TRAINING (Mencegah Data Leakage) ───────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders, df_encoded


if __name__ == "__main__":
    df = load_data()
    print(f"Shape dataset baru: {df.shape}")
    print(f"Jumlah merk unik ({df['make'].nunique()}): {sorted(df['make'].unique()[:15])}...")
    print(f"\nDistribusi segmen harga:\n{df['price_segment'].value_counts()}")
    print(f"\nSampel data:\n{df[['make', 'model', 'year', 'body-style', 'horsepower', 'price']].head()}")
