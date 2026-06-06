# 🚗 Car Recommendation System

> **Final Project Machine Learning** — Edmunds/Kaggle Car Features & MSRP Dataset  
> Sistem rekomendasi mobil cerdas berbasis *content-based filtering* menggunakan algoritma Machine Learning klasik (non-deep learning) dengan prediksi merk terpopuler dan estimasi harga jual (MSRP).

### 👥 Tim Pengembang
* **Lionel Saputra Rusli** - 2802400780
* **Harris Kristanto** - 2802412515

---

## 📊 Dataset

**[Kaggle Car Features and MSRP Dataset](https://www.kaggle.com/datasets/CooperUnion/cardataset)**
* **Jumlah Sampel**: 11.914 mobil
* **Rentang Tahun**: 1990 s.d. 2017
* **Jumlah Merk**: 48 merk unik (Toyota, Honda, BMW, Chevrolet, Ferrari, Tesla, dll.)

---

## 🧹 Tahap Preprocessing Data

Sebelum data digunakan untuk melatih model Machine Learning, dilakukan serangkaian preprocessing berikut:
1. **Cleaning Kolom Kategorikal**: Menyederhanakan kategori tipe bahan bakar (`fuel-type`), penggerak roda (`drive-wheels`), dan bentuk bodi (`body-style`) agar tidak terlalu banyak kategori yang jarang.
2. **Imputasi Nilai Kosong (Missing Values)**:
   * Kolom numerik diimputasi menggunakan nilai **Median** dari masing-masing kolom.
   * Kolom kategorik diimputasi menggunakan nilai **Mode** (nilai terbanyak).
3. **Penyelesaian Multikolinearitas (Fitur Redundan)**:
   * Menghapus `highway-mpg` karena memiliki korelasi sangat tinggi ($r \approx 0.89$) dengan `city-mpg`.
   * Menghapus `cylinders` karena memiliki korelasi sangat tinggi ($r \approx 0.85$) dengan `horsepower`.
4. **Pencegahan Target Leakage**: Menghapus `popularity` dari dataset training karena nilai kepopuleran bersifat konstan per merk, yang menyebabkan model klasifikasi merk mengalami kebocoran data (mencapai akurasi palsu 99%+).
5. **One-Hot Encoding**: Mengubah data kategorikal menjadi representasi numerik biner (one-hot).
6. **Feature Scaling (Standard Scaling)**: Melakukan penskalaan standardisasi (`StandardScaler`) hanya pada data training untuk mencegah *data leakage*, kemudian menerapkannya pada data pengujian.
7. **Random Oversampling**: Menerapkan *oversampling* acak menggunakan NumPy pada data training untuk mengatasi masalah *class imbalance* yang parah pada kolom target `make` (merk mobil), dengan batas maksimal `max_total=15000` baris data.

---

## 🤖 Model Machine Learning & Hasil Evaluasi

Model ini dikembangkan **murni menggunakan algoritma Machine Learning klasik** (tanpa Deep Learning) untuk menjamin keringanan performa dan interpretabilitas fitur (*feature importance*).

| Model | Kegunaan | CV Score | Test Set Score |
|---|---|---|---|
| **Random Forest Classifier** | Prediksi merk terbaik | **82.26%** (F1-Weighted) | **89.30%** (Accuracy) |
| **Random Forest Regressor** | Estimasi harga mobil | **96.98%** ($R^2$ Score) | **99.15%** ($R^2$ Score) |
| **KNN** (k=10, cosine similarity) | Content-based recommendation | — | **Akurat berdasarkan atribut bodi & spesifikasi** |

### 🔬 Catatan Pencegahan Overfitting (Target Leakage)
Sebelumnya, model mencapai akurasi **99%+** secara instan. Hasil investigasi menemukan adanya **Target Leakage (kebocoran data)** pada fitur `popularity`. Fitur `popularity` memiliki nilai konstan untuk setiap merk (misalnya seluruh mobil BMW bernilai 3916, seluruh Toyota bernilai 2031). Model Random Forest dengan mudah melakukan klasifikasi hanya berdasarkan nilai ini.

**Solusi**: Fitur `popularity` telah dihapus dari training ML (hanya digunakan sebagai metadata tampilan). Hal ini menghasilkan akurasi yang lebih realistis dan aman dari overfitting, yaitu sekitar **~89.3%**.

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies
Pastikan Python 3.8+ terinstal, lalu jalankan:
```bash
pip install -r requirements.txt
```
*(Catatan: Aplikasi ini bebas dari dependensi `statsmodels` sehingga tidak akan memicu error `ModuleNotFoundError` saat menggambar grafik).*

### 2. Training Model
Latih ulang seluruh model klasifikasi, regresi, dan rekomendasi:
```bash
python src/train_model.py
```

### 3. Evaluasi Model
Hasilkan metrik detail beserta visualisasi plot (confusion matrix, feature importance, actual vs predicted price):
```bash
python src/evaluate.py
```
Plot visual akan disimpan ke folder `output/`.

### 4. Jalankan Web App (Streamlit)
Jalankan dashboard aplikasi interaktif:
```bash
python -m streamlit run app/app.py
```
Aplikasi akan otomatis berjalan pada alamat local: [http://localhost:8501](http://localhost:8501)

---

## 📁 Struktur Proyek

```
Final Project ML/
├── data/
│   └── car_features_msrp.csv      # Dataset Edmunds/Kaggle (diunduh otomatis)
├── models/
│   ├── rf_classifier.pkl          # Model Random Forest (Merk)
│   ├── rf_regressor.pkl           # Model Random Forest (Harga)
│   ├── knn_recommender.pkl        # KNN Recommender
│   ├── scaler.pkl                 # StandardScaler untuk fitur numerik
│   ├── encoders.pkl               # LabelEncoder untuk fitur kategorikal
│   └── df_encoded.pkl             # Dataset yang telah ter-encode
├── src/
│   ├── data_loader.py             # Download, cleaning & preprocessing data
│   ├── train_model.py             # Pipeline pelatihan model ML
│   ├── recommend.py               # Engine sistem rekomendasi
│   └── evaluate.py                # Evaluasi performa model & plotting
├── app/
│   └── app.py                     # Streamlit web app interaktif
├── notebooks/
│   └── exploration.py             # Analisis data eksploratif (EDA)
├── output/
│   ├── evaluation_report.txt      # Metrik evaluasi lengkap
│   ├── cm_rf.png                  # Confusion Matrix Random Forest
│   ├── feature_importance.png     # Feature Importance Random Forest
│   └── price_prediction.png       # Plot harga aktual vs prediksi
├── requirements.txt
└── README.md
```

---

## 🖥️ Fitur Web App

1. **🔮 Cari Rekomendasi**: Input filter bodi mobil, tipe BBM, transmisi, serta spesifikasi silinder, HP, dan budget maksimal. Sistem akan memprediksi merk terbaik, mengestimasi harga, dan memberikan daftar mobil rekomendasi yang paling mendekati preferensi.
2. **📊 Evaluasi Performa**: Lihat metrik evaluasi Random Forest beserta plot *Feature Importance* dan penyebaran deviasi prediksi harga.
3. **📊 Eksplorasi Data**: Dashboard analisis data interaktif untuk melihat sebaran tipe bodi mobil, korelasi antar-fitur teknis, dan perbandingan harga per merk terpopuler.
4. **ℹ️ Tentang**: Detail dataset, batasan model, dan visualisasi 48 merk mobil terdaftar.

---

## ⚠️ Batasan Sistem (Limitations)

Meskipun demikian, aplikasi ini masih memiliki beberapa batasan:
* **Pasar Indonesia**: Beberapa brand mobil yang ditampilkan belum tentu dipasarkan secara resmi di Indonesia.
* **Akurasi Harga Real-Time**: Sistem mengambil data dari harga eceran produsen (MSRP) saat model dilatih/diperbarui. Sistem tidak dapat menjamin harga real-time terkait diskon dealer lokal atau negosiasi spesifik.
* **Batasan Tahun**: Dataset yang digunakan hanya mencakup data mobil dari tahun **1990 hingga 2017**.

---

## 🔮 Rencana Pengembangan (Future Works)

* **Pembaruan Tahun**: Memperbarui dataset ke tahun **2018–2025** (termasuk kendaraan listrik/EV modern seperti Tesla, BYD, Hyundai Ioniq, dll.).
* **Adaptasi Pasar Lokal**: Menambahkan data mobil spesifik pasar Indonesia (seperti Avanza, Brio, Xpander, dll.) untuk relevansi lokal yang lebih baik.

---

## 📚 Referensi

* Cooper Union. *Car Features and MSRP Dataset on Kaggle.*
* [scikit-learn Documentation](https://scikit-learn.org)
* [Streamlit Documentation](https://docs.streamlit.io)
