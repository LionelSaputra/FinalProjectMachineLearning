"""
evaluate.py
===========
Evaluasi model Car Recommendation System:
- Brand Classifier: Accuracy, F1, Confusion Matrix
- Price Regressor : MAE, RMSE, R2
- Feature Importance
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import (
    load_data, get_train_test_split, NUMERIC_FEATURES, CAT_FEATURES, TARGET_COL
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")


def evaluate_classifier(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Evaluasi model klasifikasi merk mobil."""
    y_test_arr = np.asarray(y_test, dtype=str)
    y_pred_arr = np.asarray(model.predict(X_test), dtype=str)

    acc  = accuracy_score(y_test_arr, y_pred_arr)
    f1   = f1_score(y_test_arr, y_pred_arr, average="weighted", zero_division=0)
    prec = precision_score(y_test_arr, y_pred_arr, average="weighted", zero_division=0)
    rec  = recall_score(y_test_arr, y_pred_arr, average="weighted", zero_division=0)

    present = sorted(set(y_test_arr))

    print(f"\n{'='*55}")
    print(f"  {model_name} -- Hasil Evaluasi Klasifikasi")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        y_test_arr, y_pred_arr,
        labels=present, target_names=present, zero_division=0
    ))

    return {
        "model": model_name, "type": "classifier",
        "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
        "y_pred": y_pred_arr, "y_test": y_test_arr,
    }


def evaluate_regressor(model, X_test, y_price_test, model_name: str = "Price Regressor") -> dict:
    """Evaluasi model regresi harga."""
    y_true = np.asarray(y_price_test, dtype=float)
    y_pred = model.predict(np.asarray(X_test, dtype=float))

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n{'='*55}")
    print(f"  {model_name} -- Hasil Evaluasi Regresi")
    print(f"{'='*55}")
    print(f"  MAE  : ${mae:,.0f}")
    print(f"  RMSE : ${rmse:,.0f}")
    print(f"  R2   : {r2:.4f}")

    return {
        "model": model_name, "type": "regressor",
        "mae": mae, "rmse": rmse, "r2": r2,
        "y_pred": y_pred, "y_test": y_true,
    }


def plot_confusion_matrix(y_test, y_pred, model_name: str, save_path: str = None):
    """Plot confusion matrix untuk brand classifier."""
    y_t = np.asarray(y_test, dtype=str)
    y_p = np.asarray(y_pred, dtype=str)
    labels = sorted(set(y_t))

    cm = confusion_matrix(y_t, y_p, labels=labels)
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) - 2)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.4, linecolor="gray", ax=ax
    )
    ax.set_xlabel("Prediksi Merk", fontsize=11)
    ax.set_ylabel("Merk Aktual", fontsize=11)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=13, pad=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {save_path}")
    plt.close(fig)


def plot_feature_importance(rf_model, all_feature_names: list, save_path: str = None):
    """Plot feature importance dari Random Forest Classifier."""
    importances = rf_model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1][:20]  # Top 20
    sorted_feat = [all_feature_names[i] for i in sorted_idx]
    sorted_imp  = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(sorted_imp / sorted_imp.max())
    ax.barh(sorted_feat[::-1], sorted_imp[::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Top-20 Feature Importance - Random Forest", fontsize=13, pad=10)
    ax.axvline(x=sorted_imp.mean(), color="navy", linestyle="--", alpha=0.7, label="Mean")
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {save_path}")
    plt.close(fig)


def plot_price_prediction(y_true, y_pred, save_path: str = None):
    """Scatter plot harga aktual vs prediksi."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, color="#2980b9", edgecolors="white", s=60)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("Harga Aktual ($)", fontsize=12)
    ax.set_ylabel("Harga Prediksi ($)", fontsize=12)
    ax.set_title("Actual vs Predicted Car Price", fontsize=13, pad=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {save_path}")
    plt.close(fig)





def save_report(clf_results: list, reg_result: dict, output_path: str):
    """Simpan laporan evaluasi ke file teks."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("   CAR RECOMMENDATION - LAPORAN EVALUASI\n")
        f.write("=" * 60 + "\n\n")
        f.write("=== BRAND CLASSIFIER ===\n")
        for res in clf_results:
            f.write(f"Model     : {res['model']}\n")
            f.write(f"Accuracy  : {res['accuracy']:.4f}\n")
            f.write(f"Precision : {res['precision']:.4f}\n")
            f.write(f"Recall    : {res['recall']:.4f}\n")
            f.write(f"F1-Score  : {res['f1_score']:.4f}\n\n")
        if reg_result:
            f.write("=== PRICE REGRESSOR ===\n")
            f.write(f"Model : {reg_result['model']}\n")
            f.write(f"MAE   : ${reg_result['mae']:,.0f}\n")
            f.write(f"RMSE  : ${reg_result['rmse']:,.0f}\n")
            f.write(f"R2    : {reg_result['r2']:.4f}\n")
    print(f"[SAVE] Laporan: {output_path}")


def main():
    print("=" * 60)
    print("  CAR RECOMMENDATION - PIPELINE EVALUASI")
    print("=" * 60)

    # Load model
    rf_clf   = joblib.load(os.path.join(MODELS_DIR, "rf_classifier.pkl"))
    rf_reg   = joblib.load(os.path.join(MODELS_DIR, "rf_regressor.pkl"))
    X_test   = joblib.load(os.path.join(MODELS_DIR, "X_test.pkl"))
    y_test   = joblib.load(os.path.join(MODELS_DIR, "y_test.pkl"))
    price_t  = joblib.load(os.path.join(MODELS_DIR, "price_test.pkl"))

    # Evaluasi classifier
    rf_res = evaluate_classifier(rf_clf,  X_test, y_test, "Random Forest")
    reg_res = evaluate_regressor(rf_reg,  X_test, price_t)

    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    feature_names = NUMERIC_FEATURES + encoders

    # Plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_confusion_matrix(y_test, rf_res["y_pred"],  "Random Forest",
                          os.path.join(OUTPUT_DIR, "cm_rf.png"))
    plot_feature_importance(rf_clf, feature_names,
                            os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plot_price_prediction(reg_res["y_test"], reg_res["y_pred"],
                          os.path.join(OUTPUT_DIR, "price_prediction.png"))
    save_report([rf_res], reg_res,
                os.path.join(OUTPUT_DIR, "evaluation_report.txt"))

    print("\n[DONE] Evaluasi selesai!")


if __name__ == "__main__":
    main()
