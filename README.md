# Predictive-Aanalytics
Machine Learning Terapan1


# Laporan Proyek Machine Learning: Prediksi Biaya Asuransi Kesehatan

## Domain Proyek

Industri asuransi kesehatan menghadapi tantangan dalam menentukan premi yang adil bagi pelanggan berdasarkan profil risiko mereka. Biaya klaim asuransi kesehatan dipengaruhi oleh berbagai faktor seperti usia, indeks massa tubuh (BMI), jumlah tanggungan, dan kebiasaan merokok.

### Latar Belakang

Penentuan premi asuransi kesehatan yang akurat sangat penting untuk:
1. Menjamin akses masyarakat terhadap layanan kesehatan yang terjangkau
2. Memastikan keberlanjutan finansial perusahaan asuransi
3. Memungkinkan pelanggan merencanakan keuangan pribadi dengan lebih baik

Menurut data American Medical Association, biaya perawatan kesehatan terus meningkat setiap tahun dengan rata-rata 4,5%. Faktor gaya hidup seperti kebiasaan merokok dapat meningkatkan risiko penyakit kronis dan biaya perawatan jangka panjang, sehingga menjadi pertimbangan penting dalam kalkulasi premi asuransi.

### Permasalahan

Bagaimana cara memprediksi biaya klaim asuransi kesehatan individu berdasarkan karakteristik demografis dan kesehatan mereka dengan akurasi yang dapat diandalkan?

### Tujuan

Membangun model machine learning untuk memprediksi biaya asuransi kesehatan berdasarkan faktor-faktor seperti usia, jenis kelamin, BMI, jumlah anak, status merokok, dan wilayah geografis. Model ini akan membantu perusahaan asuransi menentukan premi yang lebih adil dan memberikan transparansi kepada konsumen.

### Manfaat

- **Bagi perusahaan asuransi**: Penentuan premi yang lebih akurat dan pengelolaan risiko yang lebih baik
- **Bagi konsumen**: Transparansi dalam perhitungan premi dan perencanaan keuangan yang lebih baik
- **Bagi masyarakat**: Akses yang lebih adil terhadap asuransi kesehatan

## Data Understanding

Dataset yang digunakan berisi informasi tentang pemegang polis asuransi kesehatan dengan 1338 baris data dan 7 kolom.

### Variabel-variabel pada Dataset

1. **age**: Usia pemegang polis (numerik, rentang 18-64 tahun)
2. **sex**: Jenis kelamin pemegang polis (kategori: female/male)
3. **bmi**: Body Mass Index, indikator massa tubuh relatif terhadap tinggi dan berat badan (numerik, rentang 15.96-53.13)
4. **children**: Jumlah anak/tanggungan yang tercakup dalam asuransi (numerik, rentang 0-5)
5. **smoker**: Status merokok (kategori: yes/no)
6. **region**: Wilayah tempat tinggal di AS (kategori: northeast, northwest, southeast, southwest)
7. **charges**: Biaya medis yang dibebankan oleh asuransi kesehatan (numerik, rentang $1,121.87-$63,770.43) - variabel target

### Exploratory Data Analysis

Beberapa temuan penting dari analisis eksplorasi data:

1. **Distribusi Target Variable**:
   - Distribusi biaya asuransi menunjukkan skewness positif yang signifikan
   - Sebagian besar pelanggan memiliki biaya moderat, namun terdapat kasus ekstrem dengan biaya sangat tinggi

2. **Korelasi Antar Variabel**:
   - Status merokok memiliki korelasi sangat tinggi dengan biaya asuransi
   - Usia dan BMI juga menunjukkan korelasi positif moderat dengan biaya
   - Terdapat efek interaksi antara BMI dan status merokok

3. **Analisis Variabel Kategorikal**:
   - Perokok memiliki biaya asuransi rata-rata hampir 4 kali lebih tinggi dibanding non-perokok
   - Perbedaan biaya berdasarkan jenis kelamin relatif kecil
   - Variasi biaya antar region tidak signifikan

## Data Preparation

Beberapa teknik data preparation yang diterapkan:

1. **Penanganan Data Kategorikal**:
   - One-Hot Encoding untuk variabel sex, smoker, dan region

2. **Feature Engineering**:
   - Penambahan fitur interaksi: bmi_smoker (BMI × status merokok)
   - Penambahan fitur interaksi: age_smoker (usia × status merokok)

3. **Normalisasi Data**:
   - Standarisasi fitur numerik (age, bmi, children) menggunakan StandardScaler

4. **Data Splitting**:
   - Pembagian data menjadi training set (80%) dan testing set (20%)

## Modeling

Saya mengimplementasikan dan membandingkan empat algoritma regresi:

1. **Linear Regression**
   - Model dasar yang mengasumsikan hubungan linear antara variabel
   - Interpretasi mudah melalui koefisien

2. **Ridge Regression**
   - Linear regression dengan regularisasi L2
   - Membantu mengatasi multikolinearitas dan overfitting

3. **Random Forest Regression**
   - Ensemble learning berbasis decision tree
   - Mampu menangkap hubungan non-linear dan interaksi antar fitur

4. **Gradient Boosting Regression**
   - Ensemble learning dengan pendekatan boosting
   - Secara bertahap memperbaiki kesalahan prediksi

### Hyperparameter Tuning

- Ridge Regression: Pencarian alpha optimal menggunakan Grid Search
- Random Forest: Optimasi n_estimators, max_depth, dan min_samples_split
- Gradient Boosting: Optimasi n_estimators, learning_rate, dan max_depth

## Evaluation

Untuk evaluasi model regresi, saya menggunakan beberapa metrik:

1. **R-squared (R²)**
   - Mengukur proporsi variasi dalam variabel target yang dijelaskan oleh model
   - Nilai ideal: mendekati 1.0

2. **Root Mean Squared Error (RMSE)**
   - Mengukur akar rata-rata kesalahan kuadrat antara nilai prediksi dan aktual
   - Memberikan bobot lebih besar pada kesalahan besar

3. **Mean Absolute Error (MAE)**
   - Rata-rata dari nilai absolut selisih antara nilai prediksi dan nilai aktual
   - Lebih mudah diinterpretasi dalam konteks bisnis

### Hasil Evaluasi Model

| Model | R² | RMSE | MAE |
|-------|------|------|-----|
| Linear Regression | 0.783593 | $5796.284659 | $4181.1944741 |
| Ridge Regression | 0.783281 | $5800.464938 | $4193.195353 |
| Random Forest | 0.868229 | $4522.971696 | $2524.397531 |
| Gradient Boosting | 0.879117 | $4332.083812 | $2464.898853 |

Berdasarkan evaluasi, **Gradient Boosting Regression** memberikan performa terbaik dengan R² sekitar 0.87 dan RMSE sekitar $4332.083812.

### Analisis Feature Importance

Berdasarkan model terbaik, faktor-faktor yang paling berpengaruh terhadap biaya asuransi:
1. Status merokok (kontribusi sekitar 60%)
2. Interaksi BMI dan status merokok
3. Usia
4. BMI
5. Interaksi usia dan status merokok

### Analisis Residual

Analisis residual menunjukkan:
- Residual terdistribusi cukup merata di sekitar nol untuk nilai prediksi rendah hingga menengah
- Terdapat beberapa kasus dengan residual besar pada nilai prediksi tinggi
- Distribusi residual mendekati normal namun dengan beberapa outlier

## Conclusion

Proyek ini berhasil membangun model machine learning yang dapat memprediksi biaya asuransi kesehatan dengan akurasi yang cukup baik (R² = 0.879). Beberapa kesimpulan utama:

1. **Faktor Penentu Utama**:
   - Status merokok merupakan faktor yang paling signifikan dalam menentukan biaya asuransi
   - BMI dan usia juga memiliki pengaruh yang cukup signifikan, terutama ketika dikombinasikan dengan status merokok

2. **Implikasi Bisnis**:
   - Perusahaan asuransi dapat menyesuaikan premi berdasarkan faktor risiko utama
   - Program pencegahan dan penghentian merokok dapat membantu mengurangi biaya asuransi
   - Pelanggan dapat memperkirakan biaya asuransi mereka dengan lebih akurat

3. **Keterbatasan Model**:
   - Model mungkin kurang akurat untuk kasus-kasus ekstrem
   - Dataset relatif kecil (1338 sampel) dan mungkin tidak mewakili populasi yang lebih luas
   - Fitur lain yang mungkin relevan seperti riwayat penyakit dan gaya hidup tidak termasuk dalam dataset

4. **Perbaikan di Masa Depan**:
   - Mengumpulkan data tambahan tentang riwayat kesehatan
   - Menguji model non-linear yang lebih kompleks
   - Melakukan validasi eksternal dengan dataset dari sumber berbeda
