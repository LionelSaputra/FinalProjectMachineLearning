"""
train_model.py
==============
Training model untuk Car Recommendation System:
- Random Forest Classifier  : prediksi merk mobil
- Gradient Boosting Classifier: model pembanding
- Random Forest Regressor   : estimasi harga
- KNN (NearestNeighbors)    : content-based recommendation
"""

import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor
)
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_score, train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import (
    load_data, get_train_test_split, preprocess,
    NUMERIC_FEATURES, CAT_FEATURES, TARGET_COL
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def _safe_cv(y_train: np.ndarray, max_cv: int = 5) -> int:
    """Tentukan jumlah fold yang aman (min 2, max sesuai ukuran kelas terkecil)."""
    counts = np.unique(y_train, return_counts=True)[1]
    return max(2, min(max_cv, int(counts.min())))


def _random_oversample(X: np.ndarray, y: np.ndarray, random_state: int = 42,
                       max_total: int = 15000) -> tuple:
    """Oversampling sederhana berbasis numpy — tanpa library eksternal.
    Duplikasi sampel dari kelas minoritas hingga setiap kelas memiliki
    jumlah sampel yang sama dengan kelas mayoritas.
    max_total: batas total sampel setelah oversampling (cegah OOM di Streamlit Cloud).
    """
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    # Batasi target per kelas agar total tidak meledak
    n_classes = max(len(classes), 1)
    target_per_class = min(counts.max(), max_total // n_classes)
    X_parts, y_parts = [], []
    for cls, cnt in zip(classes, counts):
        idx = np.where(y == cls)[0]
        if cnt < target_per_class:
            chosen = rng.choice(idx, size=target_per_class, replace=True)
        else:
            chosen = idx  # kelas mayoritas: pakai semua
        X_parts.append(X[chosen])
        y_parts.append(y[chosen])
    return np.vstack(X_parts), np.concatenate(y_parts)


def train_brand_classifier(X_train, y_train, cv: int = 5):
    """Train Random Forest Classifier untuk prediksi merk mobil.
    Menggunakan random oversampling (numpy) untuk mengatasi class imbalance
    tanpa bergantung pada library eksternal seperti imbalanced-learn.
    """
    print("\n[TRAIN] Random Forest Classifier (prediksi merk)...")
    X_arr = np.asarray(X_train, dtype=float)
    y_arr = np.asarray(y_train, dtype=str)

    # ─── Oversampling berbasis numpy (tanpa imblearn) ──────────────────────────
    counts = np.unique(y_arr, return_counts=True)[1]
    min_samples = counts.min()

    if min_samples > 1:
        print(f"[OVERSAMPLE] Menerapkan random oversampling untuk mengatasi data imbalance...")
        X_res, y_res = _random_oversample(X_arr, y_arr, random_state=42)
        print(f"             Ukuran data sebelum oversampling : {len(X_arr)}")
        print(f"             Ukuran data sesudah oversampling : {len(X_res)}")
    else:
        print("[WARN] Oversampling dilewati karena ada kelas dengan <2 sampel di data training.")
        X_res, y_res = X_arr, y_arr

    # ─── CV pada data ASLI (sebelum oversampling) agar tidak data leakage ─────
    # Oversampling hanya untuk training final, bukan evaluasi CV
    safe_orig = _safe_cv(y_arr, cv)
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    cv_scores = cross_val_score(model, X_arr, y_arr, cv=safe_orig, scoring="f1_weighted")
    print(f"[CV]  F1-Weighted (data asli): {cv_scores.round(4)} | mean={cv_scores.mean():.4f}")

    # ─── Fit final pada data OVERSAMPLED ──────────────────────────────────────
    safe  = _safe_cv(y_res, cv)
    model.fit(X_res, y_res)
    print("[TRAIN] Selesai!")
    return model, cv_scores





def train_price_regressor(X_train, y_price_train, cv: int = 5):
    """Train Random Forest Regressor untuk estimasi harga."""
    print("\n[TRAIN] Random Forest Regressor (estimasi harga)...")
    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    X_arr = np.asarray(X_train, dtype=float)
    y_arr = np.asarray(y_price_train, dtype=float)

    cv_scores = cross_val_score(model, X_arr, y_arr, cv=cv, scoring="r2")
    print(f"[CV]  R2: {cv_scores.round(4)} | mean={cv_scores.mean():.4f}")
    model.fit(X_arr, y_arr)
    print("[TRAIN] Selesai!")
    return model, cv_scores


def train_recommender(X_scaled: np.ndarray, n_neighbors: int = 10) -> NearestNeighbors:
    """Train KNN untuk content-based recommendation."""
    k = min(n_neighbors, len(X_scaled) - 1)
    print(f"\n[RECOMMENDER] KNN Recommender (k={k})...")
    knn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(np.asarray(X_scaled, dtype=float))
    print("[RECOMMENDER] Selesai!")
    return knn


def save_models(models_dict: dict) -> None:
    """Simpan semua model ke folder models/."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, obj in models_dict.items():
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(obj, path)
        print(f"[SAVE] {name}.pkl")


def main():
    print("=" * 60)
    print("  CAR RECOMMENDATION SYSTEM - TRAINING PIPELINE")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_data()
    print(f"\n[DATA] Total mobil  : {len(df)}")
    print(f"[DATA] Jumlah merk  : {df[TARGET_COL].nunique()}")
    print(f"[DATA] Merk: {sorted(df[TARGET_COL].unique())}")

    # ── 2. Split train / test ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler, encoders, df_enc = get_train_test_split(df)
    print(f"\n[SPLIT] Train: {len(X_train)}, Test: {len(X_test)}")

    # ── 3. Ambil harga sesuai indeks split ────────────────────────────────────
    # Stratify flag sama seperti di get_train_test_split
    counts = df_enc[TARGET_COL].value_counts()
    stratify = np.asarray(df_enc[TARGET_COL], dtype=str) if counts.min() >= 2 else None
    idx = np.arange(len(df_enc))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=stratify
    )
    price_train = df_enc.iloc[idx_train]["price"].to_numpy(dtype=float)
    price_test  = df_enc.iloc[idx_test]["price"].to_numpy(dtype=float)

    # ── 4. Train models ───────────────────────────────────────────────────────
    rf_clf,  rf_cv  = train_brand_classifier(X_train, y_train)
    rf_reg,  reg_cv = train_price_regressor(X_train, price_train)

    # KNN on full dataset (all samples)
    X_all, _, _, _, _ = preprocess(df, scaler, encoders)
    knn_rec = train_recommender(X_all, n_neighbors=10)

    # ── 5. Simpan semua ───────────────────────────────────────────────────────
    save_models({
        "rf_classifier":   rf_clf,
        "rf_regressor":    rf_reg,
        "knn_recommender": knn_rec,
        "scaler":          scaler,
        "encoders":        encoders,
        "df_encoded":      df_enc,
        "X_test":          X_test,
        "y_test":          y_test,
        "price_test":      price_test,
    })

    print("\n[DONE] Training selesai! Semua model tersimpan.")
    print("=" * 60)
    print(f"[SUMMARY] RF Classifier   CV F1 : {rf_cv.mean():.4f}")
    print(f"[SUMMARY] RF Regressor    CV R2 : {reg_cv.mean():.4f}")

    return X_test, y_test, rf_clf


if __name__ == "__main__":
    main()
