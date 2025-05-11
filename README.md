# Predictive-Analytics
Machine Learning Terapan1


# Laporan Proyek Machine Learning - Reinhart Jens Robert

## Domain Proyek

Industri asuransi kesehatan merupakan sektor penting dalam sistem kesehatan di berbagai negara. Salah satu tantangan utama bagi perusahaan asuransi adalah menetapkan premi yang sesuai untuk setiap pelanggan berdasarkan profil risiko mereka. Premi yang terlalu tinggi dapat membuat pelanggan mencari alternatif lain, sementara premi yang terlalu rendah dapat menyebabkan kerugian finansial bagi perusahaan [1].
Biaya klaim asuransi kesehatan dipengaruhi oleh berbagai faktor, seperti usia, jenis kelamin, indeks massa tubuh (BMI), jumlah tanggungan, status merokok, dan wilayah geografis tempat tinggal pelanggan [2]. Memahami pengaruh faktor-faktor ini terhadap biaya klaim menjadi sangat penting bagi perusahaan asuransi untuk menetapkan premi yang adil dan kompetitif.
Menurut penelitian yang dilakukan oleh The Kaiser Family Foundation, premi asuransi kesehatan di Amerika Serikat telah meningkat 22% dalam lima tahun terakhir [3]. Peningkatan ini menjadi beban bagi banyak keluarga. Dengan memahami faktor-faktor yang mempengaruhi biaya asuransi, perusahaan dapat mengembangkan produk yang lebih terjangkau dan desain program preventif yang efektif.
Penggunaan machine learning dalam industri asuransi telah menunjukkan peningkatan dalam beberapa tahun terakhir. Sebuah studi oleh Deloitte menunjukkan bahwa perusahaan yang mengadopsi analitik prediktif dalam penetapan harga mengalami peningkatan profitabilitas hingga 15% [4]. Dengan demikian, pengembangan model prediktif untuk biaya asuransi kesehatan tidak hanya bermanfaat bagi perusahaan tetapi juga dapat membantu konsumen dalam merencanakan keuangan mereka dengan lebih baik.

### Business Understanding

## Problem Statements
Berdasarkan latar belakang yang telah diuraikan, berikut adalah pernyataan masalah dalam proyek ini:

1. Faktor-faktor apa yang paling signifikan mempengaruhi biaya asuransi kesehatan individu?
2. Seberapa akurat model machine learning dapat memprediksi biaya asuransi kesehatan berdasarkan karakteristik demografis dan kesehatan pelanggan?
3. Bagaimana perusahaan asuransi dapat memanfaatkan model prediktif untuk menetapkan premi yang lebih adil dan personal?

### Tujuan

Tujuan proyek ini adalah sebagai berikut:
1. Mengidentifikasi dan menganalisis faktor-faktor yang memiliki pengaruh signifikan terhadap biaya asuransi kesehatan.
2. Membangun model machine learning yang dapat memprediksi biaya asuransi kesehatan dengan tingkat akurasi yang dapat diandalkan (R² minimal 0.75).
3. Memberikan rekomendasi berbasis data untuk perusahaan asuransi dalam menetapkan strategi penetapan harga yang lebih personal dan adil.

### Solution Statements
Untuk mencapai tujuan di atas, berikut adalah solusi yang diusulkan:
1. Melakukan analisis eksplorasi data (EDA) untuk memahami pola dan hubungan antara berbagai fitur dengan biaya asuransi kesehatan.
2. Mengimplementasikan dan membandingkan beberapa algoritma regresi:
   - Linear Regression sebagai baseline model
   - Ridge Regression untuk mengatasi potensi multikolinearitas
   - Random Forest Regression untuk menangkap pola non-linear dan interaksi kompleks
   - Gradient Boosting Regression untuk meningkatkan performa prediksi
3. Melakukan feature engineering, termasuk:
   - Transformasi data untuk fitur yang tidak berdistribusi normal
   - Pembuatan fitur interaksi untuk meningkatkan kemampuan prediksi model
   - One-hot encoding untuk variabel kategorikal
4. Evaluasi model menggunakan metrik:
   - R² Score untuk mengukur proporsi variasi yang dapat dijelaskan oleh model
   - Root Mean Squared Error (RMSE) untuk mengukur rata-rata kesalahan prediksi
   - Mean Absolute Error (MAE) untuk mengukur rata-rata absolut kesalahan prediksi

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah "Medical Cost Personal Datasets" yang berisi informasi biaya asuransi kesehatan pribadi berdasarkan karakteristik individu. Dataset ini terdiri dari 1338 sampel dengan 7 atribut. 
**Kondisi Data**
Dataset ini telah diperiksa untuk mengetahui kualitas dan kelengkapan datanya:
- Missing Values**:
Tidak ditemukan missing values pada semua kolom. Hasil dari df.isnull().sum() menunjukkan nilai 0 pada setiap kolom.
- Duplikat:
Tidak ditemukan data duplikat setelah menjalankan df.duplicated().sum().
- Outlier:
Ditemukan outlier pada variabel target charges berdasarkan metode IQR (Interquartile Range).
Jumlah outlier: ≈139 sampel, terutama pada kelompok perokok dengan BMI tinggi.
- Distribusi:
Variabel target (charges) memiliki distribusi positively skewed, artinya sebagian besar biaya rendah dan hanya sebagian kecil yang sangat tinggi. Ini penting untuk diperhatikan dalam modeling karena bisa mempengaruhi akurasi.

**Sumber Dataset**
Dataset diambil dari Kaggle:
🔗 https://www.kaggle.com/datasets/mirichoi0218/insurance

Variabel-variabel pada dataset asuransi kesehatan adalah sebagai berikut:
- age: Usia pemegang polis (numerik)
- sex: Jenis kelamin pemegang polis (kategori: female/male)
- bmi: Body Mass Index, indikator massa tubuh relatif terhadap tinggi dan berat badan (numerik)
- children: Jumlah anak/tanggungan yang tercakup dalam asuransi (numerik)
- smoker: Status merokok (kategori: yes/no)
- region: Wilayah tempat tinggal di AS (kategori: northeast, northwest, southeast, southwest)
- charges: Biaya medis yang dibebankan oleh asuransi kesehatan (numerik) - variabel target

### Exploratory Data Analysis (EDA)
Untuk memahami dataset dengan lebih baik, beberapa analisis eksplorasi data telah dilakukan:
1. Statistik Deskriptif
Statistik dasar untuk setiap kolom numerik menunjukkan:
   - Age: Rata-rata 39.2 tahun, dengan rentang 18-64 tahun
   - BMI: Rata-rata 30.66, dengan rentang 15.96-53.13
   - Children: Rata-rata 1.09 anak, dengan rentang 0-5 anak
   - Charges: Rata-rata $13,270.42, dengan rentang $1,121.87-$63,770.43

2. Distribusi Variabel Target (Charges)
Distribusi biaya asuransi terlihat sangat miring ke kanan (positively skewed), menunjukkan bahwa sebagian besar pemegang polis memiliki biaya yang relatif rendah, sementara sebagian kecil memiliki biaya yang sangat tinggi. Ini mengindikasikan ketidaknormalan distribusi yang perlu diperhatikan dalam pemodelan.

3. Korelasi Antar Variabel Numerik
Matriks korelasi menunjukkan bahwa:
  - Age memiliki korelasi positif moderat dengan charges (r ≈ 0.30)
  - BMI memiliki korelasi positif lemah dengan charges (r ≈ 0.20)
  - Children memiliki korelasi positif sangat lemah dengan charges (r ≈ 0.07)

4. Analisis Variabel Kategorikal
Beberapa insight penting dari analisis variabel kategorikal:
   - Smoker: Perbedaan biaya yang sangat signifikan antara perokok dan non-perokok. Rata-rata 
     biaya untuk perokok sekitar $32,050, sedangkan non-perokok sekitar $8,440.
   - Sex: Tidak ada perbedaan signifikan dalam biaya antara laki-laki dan perempuan.
   - Region: Perbedaan biaya antar wilayah relatif kecil, dengan northeast sedikit lebih tinggi 
     dibandingkan wilayah lain.

5. Interaksi Antar Variabel
Analisis menunjukkan interaksi yang kuat antara status merokok dan BMI. Pengaruh BMI terhadap biaya asuransi jauh lebih kuat pada kelompok perokok dibandingkan non-perokok.


## Data Preparation

Pada tahap ini, dilakukan proses pembersihan dan persiapan data agar siap digunakan dalam pemodelan machine learning. Berikut tahapan lengkapnya:

1. Menghapus Kolom Tidak Relevan

- Kolom id dihapus karena tidak memiliki pengaruh terhadap target dan hanya bersifat identifier.

2. Mengubah Format Tipe Data

- Kolom bmi dan charges dipastikan memiliki tipe data float.

- Kolom age dan children memiliki tipe data int.

3. Mendeteksi dan Menangani Duplikasi

- Data dicek untuk duplikasi, dan ditemukan 1 baris duplikat yang kemudian dihapus untuk mencegah bias pada model.

4. Penanganan Data Kategori (Encoding)

- Fitur kategorikal sex, smoker, dan region diubah menjadi numerik menggunakan One-Hot Encoding.

- Hasilnya adalah penambahan beberapa kolom dummy seperti sex_male, smoker_yes, dll.

5. Penskalaan Fitur (Feature Scaling)

- Untuk meningkatkan performa model regresi, dilakukan standardisasi pada kolom numerik menggunakan StandardScaler dari sklearn.


### 4. Feature Engineering
Beberapa fitur baru dibuat untuk meningkatkan performa model:

python
# Interaksi antara BMI dan status merokok
df['bmi_smoker'] = df['bmi'] * (df['smoker'] == 'yes').astype(int)
# Interaksi antara usia dan status merokok
df['age_smoker'] = df['age'] * (df['smoker'] == 'yes').astype(int)

Fitur interaksi ini dibuat berdasarkan insight dari EDA yang menunjukkan hubungan non-linear antara BMI, status merokok, dan biaya asuransi. Tujuannya adalah untuk membantu model menangkap pola kompleks yang tidak dapat ditangkap oleh fitur tunggal.

### 5. Transformasi Variabel Target
Karena distribusi variabel target (charges) yang sangat skewed, transformasi logaritmik dicoba untuk menghasilkan distribusi yang lebih normal.

python
df['log_charges'] = np.log(df['charges'])

Transformasi ini membantu:
   - Menstabilkan varians
   - Membuat distribusi lebih normal
   - Meningkatkan performa model regresi, terutama model linear

### 6. Train-Test Split
Data dibagi menjadi set pelatihan (80%) dan pengujian (20%) dengan strategi random sampling.

python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Split data ini penting untuk:
- Mengevaluasi performa model pada data yang tidak pernah dilihat sebelumnya
- Mencegah overfitting
- Memastikan generalisasi model

### Modeling
Pada tahap ini, beberapa algoritma machine learning diterapkan untuk memprediksi biaya asuransi kesehatan. Setiap algoritma memiliki karakteristik, kelebihan, dan kekurangan yang berbeda.
## 1. Linear Regression
Linear Regression adalah model dasar yang digunakan sebagai baseline dalam proyek ini. Model ini mencoba menemukan hubungan linear antara fitur input dan variabel target.
Parameter:
  - Tidak ada parameter spesifik yang diatur, menggunakan konfigurasi default scikit-learn

Kelebihan:
  - Interpretabilitas tinggi, memungkinkan pemahaman langsung tentang pengaruh setiap fitur
  - Komputasi ringan dan cepat
  - Baik untuk memahami hubungan dasar antar variabel
Kekurangan:
 -  Tidak dapat menangkap hubungan non-linear kompleks
 -  Sensitif terhadap outlier
 -  Asumsi independensi error dan homoskedastisitas

## 2. Ridge Regression
Ridge Regression adalah ekstensi dari Linear Regression yang menambahkan regularisasi L2 untuk mengatasi potensi masalah multikolinearitas.

Parameter:
- alpha: Parameter regularisasi yang dioptimalkan melalui cross-validation
  
python
ridge_params = {'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}

Kelebihan:
 -  Mengurangi overfitting dengan menambahkan penalti pada koefisien besar
 -  Lebih stabil daripada Linear Regression ketika terdapat korelasi tinggi antar fitur
 -  Tetap mempertahankan interpretabilitas model

Kekurangan:
  - Masih terbatas pada pola linear
  - Performa dapat lebih rendah untuk hubungan yang sangat non-linear

## 3. Random Forest Regression
Random Forest adalah model ensemble berbasis decision tree yang dapat menangkap pola non-linear dan interaksi kompleks antar fitur.
Parameter:
  - n_estimators: Jumlah pohon dalam forest [50, 100]
  - max_depth: Kedalaman maksimum setiap pohon [None, 10, 20]
  - min_samples_split: Jumlah minimum sampel untuk split [2, 5]

Kelebihan:
 -  Dapat menangkap hubungan non-linear dan interaksi kompleks
 -  Robust terhadap outlier dan noise
 -  Menyediakan feature importance untuk interpretabilitas
 -  Tidak memerlukan asumsi distribusi data

Kekurangan:
 -  Komputasi lebih berat dibandingkan model linear
 -  Dapat mengalami overfitting jika parameter tidak diatur dengan tepat
 -  Kurang interpretabel dibandingkan model linear

## 4. Gradient Boosting Regression
Gradient Boosting adalah teknik ensemble yang membangun model secara sekuensial, dengan setiap model baru mencoba memperbaiki kesalahan model sebelumnya.
Parameter:
 -  n_estimators: Jumlah estimator [50, 100]
 -  learning_rate: Kecepatan pembelajaran [0.01, 0.1]
 -  max_depth: Kedalaman maksimum [3, 5]

Kelebihan:
 -  Performa tinggi, sering menghasilkan prediksi yang sangat akurat
 -  Dapat menangkap hubungan kompleks dalam data
 -  Menyediakan feature importance
 -  Robust terhadap berbagai jenis data

Kekurangan:
 -  Komputasi paling berat di antara model yang digunakan
 -  Lebih rentan terhadap overfitting
 -  Memerlukan tuning parameter yang lebih hati-hati

## Model Development

1. Pemisahan Data
- Dataset dibagi menjadi fitur (X) dan target (charges).
- Selanjutnya, dibagi menjadi data latih dan data uji dengan rasio 80:20 menggunakan train_test_split.

2. Training Model
- Digunakan model Linear Regression dari sklearn.linear_model.
- Model dilatih menggunakan data latih (X_train, y_train).

3. Evaluasi Model
- Dilakukan evaluasi pada data uji menggunakan metrik:
  - MAE (Mean Absolute Error)
  -  MSE (Mean Squared Error)
  -  RMSE (Root Mean Squared Error)
  -  R² Score


## Hyperparameter Tuning
Untuk meningkatkan performa model, hyperparameter tuning dilakukan menggunakan GridSearchCV dengan cross-validation. Ini membantu menemukan kombinasi parameter optimal untuk setiap algoritma.

python
# Contoh untuk Random Forest
rf_params = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5]
}

rf_cv = GridSearchCV(rf_pipeline, rf_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
### Pemilihan Model Terbaik
Setelah melatih dan mengevaluasi keempat model, perbandingan dilakukan berdasarkan metrik evaluasi R², RMSE, dan MAE. Model Gradient Boosting Regression menunjukkan performa terbaik dengan R² score tertinggi dan RMSE terendah.
Model ini dipilih sebagai model final karena:
  - Memiliki kemampuan prediksi terbaik (R² > 0.85)
  - Dapat menangkap pola non-linear dan interaksi kompleks antara fitur
  - Memberikan feature importance yang bermanfaat untuk interpretasi bisnis

### Evaluation
Untuk mengevaluasi performa model dalam memprediksi biaya asuransi kesehatan, beberapa metrik evaluasi yang relevan dengan konteks regresi digunakan.
## Metrik Evaluasi
1. R² Score (Coefficient of Determination)
R² mengukur proporsi variasi variabel target yang dapat dijelaskan oleh model. Nilainya berkisar antara 0 hingga 1, dengan nilai yang lebih tinggi menunjukkan model yang lebih baik.
Formula:
R² = 1 - (Sum of Squared Residuals / Total Sum of Squares)

Di mana:
- Sum of Squared Residuals = Σ(y_actual - y_predicted)²
- Total Sum of Squares = Σ(y_actual - y_mean)²

R² sangat berguna untuk memahami seberapa baik model dapat menjelaskan variasi dalam data. Nilai R² = 0.85 berarti model dapat menjelaskan 85% variasi dalam biaya asuransi.

2. Root Mean Squared Error (RMSE)
RMSE mengukur akar rata-rata dari kesalahan kuadrat antara nilai prediksi dan nilai aktual. Metrik ini memberikan bobot yang lebih tinggi pada kesalahan besar.

Formula:

RMSE = √(Σ(y_actual - y_predicted)² / n)

RMSE memiliki unit yang sama dengan variabel target (dollar dalam kasus ini), sehingga mudah diinterpretasikan. Semakin rendah RMSE, semakin baik model.

3. Mean Absolute Error (MAE)
MAE mengukur rata-rata nilai absolut dari kesalahan antara nilai prediksi dan nilai aktual.

Formula:
MAE = Σ|y_actual - y_predicted| / n

MAE juga memiliki unit yang sama dengan variabel target dan lebih robust terhadap outlier dibandingkan RMSE.

### Hasil Evaluasi
Berikut adalah hasil evaluasi dari semua model yang diuji:
| Model | R² | RMSE | MAE |
|-------|------|------|-----|
| Linear Regression | 0.783593 | $5796.284659 | $4181.1944741 |
| Ridge Regression | 0.783281 | $5800.464938 | $4193.195353 |
| Random Forest | 0.868229 | $4522.971696 | $2524.397531 |
| Gradient Boosting | 0.879117 | $4332.083812 | $2464.898853 |

Berdasarkan evaluasi, **Gradient Boosting Regression** memberikan performa terbaik dengan R² sekitar 0.87 dan RMSE sekitar $4332.083812.

## Analisis Feature Importance

Berdasarkan model terbaik, faktor-faktor yang paling berpengaruh terhadap biaya asuransi:
1. Status merokok (kontribusi sekitar 60%)
2. Interaksi BMI dan status merokok
3. Usia
4. BMI
5. Interaksi usia dan status merokok

## Analisis Residual
Analisis residual (selisih antara nilai aktual dan prediksi) menunjukkan:
  - Distribusi residual mendekati normal dengan mean mendekati nol
  - Tidak ada pola yang jelas pada plot residual vs. nilai prediksi, menunjukkan asumsi 
    homoskedastisitas terpenuhi
  - Beberapa outlier tetap ada, terutama untuk nilai prediksi tinggi

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
