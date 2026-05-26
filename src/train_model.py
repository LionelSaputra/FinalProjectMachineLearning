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
from imblearn.over_sampling import SMOTE

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


def train_brand_classifier(X_train, y_train, cv: int = 5):
    """Train Random Forest Classifier untuk prediksi merk mobil dengan SMOTE."""
    print("\n[TRAIN] Random Forest Classifier (prediksi merk)...")
    X_arr = np.asarray(X_train, dtype=float)
    y_arr = np.asarray(y_train, dtype=str)
    
    # ─── Menerapkan SMOTE dengan aman ──────────────────────────────────────────
    counts = np.unique(y_arr, return_counts=True)[1]
    min_samples = counts.min()
    
    if min_samples > 1:
        k_neighbors = min(5, min_samples - 1)
        print(f"[SMOTE] Menerapkan SMOTE (k_neighbors={k_neighbors}) untuk mengatasi data imbalance...")
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_res, y_res = smote.fit_resample(X_arr, y_arr)
        print(f"        Ukuran data sebelum SMOTE : {len(X_arr)}")
        print(f"        Ukuran data sesudah SMOTE : {len(X_res)}")
    else:
        print("[WARN] SMOTE dilewati karena ada kelas dengan <2 sampel di data training.")
        X_res, y_res = X_arr, y_arr
        
    safe  = _safe_cv(y_res, cv)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    cv_scores = cross_val_score(model, X_res, y_res, cv=safe, scoring="f1_weighted")
    print(f"[CV]  F1-Weighted: {cv_scores.round(4)} | mean={cv_scores.mean():.4f}")
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
