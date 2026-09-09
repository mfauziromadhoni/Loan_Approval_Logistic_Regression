# Prediksi Persetujuan Pinjaman (Loan Approval Prediction) — Logistic Regression

Proyek ini membangun model klasifikasi **Logistic Regression** untuk memprediksi apakah suatu pengajuan pinjaman akan **disetujui** atau **ditolak**, berdasarkan karakteristik demografis, finansial, dan riwayat kredit pemohon.

## Latar Belakang Bisnis

Lembaga pemberi pinjaman perlu mengambil keputusan persetujuan kredit secara cepat, konsisten, dan berbasis data — bukan hanya berdasarkan penilaian manual yang rentan bias dan lambat. Model prediktif ini bertujuan untuk:

- Mengotomatisasi penilaian awal kelayakan pinjaman berdasarkan pola historis
- Mengidentifikasi faktor-faktor yang paling memengaruhi keputusan persetujuan
- Memberikan dasar kuantitatif untuk menyeimbangkan **risiko kredit macet** dengan **peluang bisnis yang hilang** akibat penolakan pemohon yang sebenarnya layak

## Dataset

| Detail | Keterangan |
|---|---|
| Jumlah observasi | 45.000 baris data pemohon pinjaman |
| Jumlah kolom awal | 14 kolom (13 fitur + 1 target) |
| Variabel target | `loan_status` (1 = Disetujui, 0 = Ditolak) |
| Missing values | Tidak ada (0 di seluruh kolom) |

**Fitur yang digunakan:**

| Kolom | Deskripsi |
|---|---|
| `person_age` | Usia pemohon |
| `person_gender` | Jenis kelamin pemohon |
| `person_education` | Tingkat pendidikan pemohon |
| `person_income` | Pendapatan tahunan pemohon |
| `person_emp_exp` | Lama pengalaman kerja (tahun) |
| `person_home_ownership` | Status kepemilikan rumah (RENT/OWN/MORTGAGE/OTHER) |
| `loan_amnt` | Jumlah pinjaman yang diajukan |
| `loan_intent` | Tujuan pinjaman (PERSONAL, EDUCATION, MEDICAL, dll.) |
| `loan_int_rate` | Suku bunga pinjaman |
| `loan_percent_income` | Rasio jumlah pinjaman terhadap pendapatan tahunan |
| `cb_person_cred_hist_length` | Panjang riwayat kredit (tahun) |
| `credit_score` | Skor kredit pemohon |
| `previous_loan_defaults_on_file` | Riwayat gagal bayar pinjaman sebelumnya (Yes/No) |

## Metodologi

Alur kerja proyek ini mengikuti pipeline machine learning klasik:

1. **Data Understanding** — Memahami arti dan tipe setiap kolom dalam dataset.
2. **Pemeriksaan Missing Values** — Memastikan tidak ada data yang hilang sebelum pemodelan.
3. **Encoding Variabel Kategorikal** — Lima kolom kategorikal (`person_gender`, `person_education`, `person_home_ownership`, `loan_intent`, `previous_loan_defaults_on_file`) diubah menjadi variabel numerik menggunakan **One-Hot Encoding** (`drop_first=True` untuk menghindari multikolinearitas). Jumlah kolom bertambah dari 14 menjadi 23 (22 fitur + 1 target).
4. **Pemisahan Fitur dan Target** — Data dipisah menjadi matriks fitur `X` (22 kolom) dan target `y` (`loan_status`).
5. **Pembagian Data Training/Testing** — Data dibagi 80:20 (36.000 data training, 9.000 data testing) menggunakan `train_test_split` dengan `stratify=y` agar proporsi kelas disetujui/ditolak tetap seimbang di kedua set.
6. **Feature Scaling** — Fitur numerik distandardisasi menggunakan `StandardScaler` (rata-rata 0, standar deviasi 1), karena Logistic Regression sensitif terhadap skala fitur yang berbeda-beda.
7. **Pelatihan Model** — Model `LogisticRegression` (scikit-learn, solver `liblinear`) dilatih pada data training yang sudah di-scale.
8. **Evaluasi Model** — Model diuji pada data testing menggunakan metrik Accuracy, Precision, Recall, F1-Score, ROC AUC, dan Confusion Matrix.

## Hasil Evaluasi Model

| Metrik | Nilai | Interpretasi |
|---|---|---|
| **Accuracy** | 0.83 | Model memprediksi status pinjaman dengan benar pada 83% kasus secara keseluruhan |
| **Precision** | 0.69 | Dari semua pinjaman yang diprediksi **disetujui**, 69% di antaranya benar-benar disetujui |
| **Recall** | 0.40 | Dari semua pinjaman yang **seharusnya** disetujui, model hanya berhasil mengenali 40% |
| **F1-Score** | 0.51 | Keseimbangan antara Precision dan Recall — mengindikasikan ada ruang perbaikan yang signifikan |
| **ROC AUC** | 0.84 | Kemampuan model membedakan kelas "disetujui" vs "ditolak" tergolong baik (jauh di atas 0.5 = tebakan acak) |

### Interpretasi Bisnis

- **Accuracy dan ROC AUC yang tinggi** (0.83 dan 0.84) menunjukkan model secara umum mampu membedakan pola antara pemohon yang layak dan tidak layak menerima pinjaman.
- **Recall yang rendah (0.40)** adalah titik perhatian utama: model melewatkan cukup banyak pemohon yang **seharusnya disetujui** (False Negative tinggi), yang berpotensi menghilangkan peluang bisnis yang menguntungkan.
- **Precision yang moderat (0.69)** menunjukkan bahwa ketika model merekomendasikan persetujuan, keputusan tersebut cukup dapat diandalkan — cocok untuk skenario bisnis yang mengutamakan **mitigasi risiko kredit macet** dibanding **maksimalisasi akuisisi nasabah**.

Trade-off Precision vs Recall ini perlu disesuaikan dengan prioritas bisnis: jika fokus utama adalah menghindari kerugian dari kredit macet, performa model saat ini sudah cukup memadai. Namun jika fokus utama adalah menjaring lebih banyak pemohon potensial yang layak, model perlu ditingkatkan lebih lanjut.

## Rekomendasi Pengembangan Lebih Lanjut

1. **Penanganan Ketidakseimbangan Kelas** — Menerapkan teknik resampling seperti SMOTE untuk meningkatkan Recall jika distribusi kelas target tidak seimbang.
2. **Eksplorasi Model Lain** — Mencoba algoritma yang lebih kompleks seperti Random Forest, XGBoost, atau LightGBM untuk menangkap pola non-linear.
3. **Hyperparameter Tuning** — Melakukan optimasi parameter model (baik Logistic Regression maupun model alternatif) untuk meningkatkan performa.
4. **Feature Engineering** — Mengeksplorasi kombinasi atau transformasi fitur baru yang berpotensi lebih prediktif terhadap status persetujuan pinjaman.

## Tools & Library

- **Python 3**
- `pandas` — manipulasi dan eksplorasi data
- `scikit-learn` — pembagian data (`train_test_split`), standardisasi (`StandardScaler`), pemodelan (`LogisticRegression`), dan evaluasi (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`)
- `seaborn` & `matplotlib` — visualisasi Confusion Matrix


## Struktur Notebook

| Tahap | Isi |
|---|---|
| Import Data | Memuat dataset `loan_data.csv` |
| Data Understanding | Penjelasan setiap kolom dataset |
| Pemeriksaan Missing Values | Validasi kelengkapan data |
| Cek Tipe Data | Pemeriksaan struktur dan tipe data per kolom |
| Encoding Variabel Kategorikal | One-Hot Encoding untuk 5 kolom kategorikal |
| Pemisahan Fitur & Target | Membentuk `X` dan `y` |
| Pembagian Data Training/Testing | Split 80:20 dengan stratifikasi |
| Feature Scaling | Standardisasi fitur numerik |
| Model Logistic Regression | Pelatihan model klasifikasi |
| Evaluasi Model | Perhitungan metrik dan visualisasi Confusion Matrix |
| Insight | Interpretasi bisnis dari hasil model |

## Batasan (Limitations)

- Recall yang rendah (0.40) berarti model ini **belum optimal** untuk digunakan sebagai satu-satunya dasar keputusan bisnis tanpa mitigasi lebih lanjut, terutama jika tujuan utamanya adalah memaksimalkan jumlah pinjaman yang disetujui secara tepat.
- Model belum diuji terhadap potensi ketidakseimbangan kelas (class imbalance) pada `loan_status`, yang bisa memengaruhi keandalan metrik Precision dan Recall.
- Sebagai model linear, Logistic Regression mengasumsikan hubungan log-odds yang linear antar fitur dan target — pola non-linear yang kompleks mungkin tidak tertangkap sepenuhnya.
