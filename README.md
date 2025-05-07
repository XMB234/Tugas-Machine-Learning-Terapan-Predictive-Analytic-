# Ekspolaris Berbagai Model Machine Learning dalam Memprediksi Penyakit Diabetes
## Latar Belakang

Diabetes mellitus (DM) adalah penyakit degeneratif kronis yang terjadi akibat produksi insulin yang tidak mencukupi di pankreas atau karena tubuh tidak dapat menggunakan insulin secara efektif, yang mengakibatkan hiperglikemi (peningkatan kadar glukosa darah) sebagai indikator utama. Menurut _World Health Organization_ (WHO), diperkirakan sebanyak 300 juta orang di seluruh dunia akan terkena DM pada tahun 2025. Selain itu, DM juga tercatat sebagai salah satu penyebab kematian utama, berdasarkan data dari International Diabetes Federation yang dirilis pada 2021, dengan Indonesia menduduki peringkat keenam, mencatatkan angka kematian sebanyak 236.711 jiwa. Karena gejalanya yang mirip dengan penyakit umum lainnya, banyak orang yang tidak menyadari bahwa mereka menderita diabetes, bahkan ketika penyakit ini sudah berkembang menjadi komplikasi. Untuk memastikan apakah seseorang mengidap diabetes, diperlukan diagnosis dari dokter melalui pemeriksaan darah. Metode yang biasa digunakan untuk deteksi diabetes adalah tes laboratorium, seperti pengukuran glukosa darah dan uji toleransi glukosa oral. Namun, hasil tes ini sering kali dipengaruhi oleh kesalahan manusia atau bias dalam pengujian, terutama pada saat analisis manual atau interpretasi data dilakukan.Oleh karena itu, diperlukan suatu metode berbasis data medis yang mempertimbangkan faktor-faktor yang mempengaruhi penyakit diabetes untuk melakukan diagnosis, serta pendekatan yang dapat mengurangi kesalahan manusia dan bias dalam pengujian prediksi penyakit diabetes.

Dalam kegiatan prediksi diagnostik, data mining dan text mining telah terbukti sebagai metode yang efektif. Metode ini terdiri dari serangkaian alat dan teknik yang dapat mengeksplorasi kumpulan data serta membantu dalam penemuan pengetahuan. Machine learning dianggap sebagai salah satu komponen utama kecerdasan buatan yang mendukung pengembangan sistem komputer yang mampu memperoleh pengetahuan dari pengalaman sebelumnya tanpa memerlukan pemrograman untuk setiap kasus. Model ini dapat dilatih menggunakan dataset besar yang mencakup berbagai variabel untuk menghasilkan prediksi yang lebih akurat. Dengan penggunaan algoritma seperti regresi logistik, pohon keputusan, atau deep learning, sistem ini dapat memberikan prediksi yang lebih cepat dan tepat. Sehingga, metode ini dapat digunakan untuk mendeteksi penyakit tanpa bergantung pada pengujian manual yang rentan terhadap kesalahan.

Berdasarkan latar belakang, pada proyek ini akan dibuat sebuah model machine learning yang dapat memprediksi penyakit diabetes berdasarkan factor factor penyakit diabetes.

## Bussiness Understanding
Selain menimbulkan masalah kesehatan langsung seperti gangguan pada kadar gula darah, penyakit diabetes juga dapat menyebabkan berbagai komplikasi serius yang dapat memengaruhi kualitas hidup penderitanyaDari pernyataan tersebut, dapat ditarik kesimpulan bahwa permasalahan utama dapat dinyatakan dengan sebuah pertanyaan berikut:

* Bagaimana memanfaatkan machine learning untuk menghasilkan sistem prediksi penyakit diabetes yang akurat, cepat, serta minim kesalahan dan bias berdasarak faktor-faktor penyakit diabetes

Dalam menyelesaikan permasalahan tersebut, berikut beberapa solusi yang akan dilakukan pada proyek ini:

* Menemukan model machine learning dengan performa terbaik dalam memprediksi penyakit diabetes berdasarkan sejumlah fitur yang tersedia, menggunakan dataset PIMA Diabetes yang terdiri dari 786 data. Dalam proses ini, akan digunakan pustaka lazypredict untuk mengevaluasi dan membandingkan berbagai model guna menentukan yang paling efektif.
* Membangun dan mengembangakan model machine learning untuk memprediksi penyakit diabetes berdasarkan hasil eksplorasi awal model. Model dengan performa terbaik yang diperoleh dari eksplorasi menggunakan lazypredict akan dipilih untuk dilakukan hyperparameter tuning. Selanjutnya, akurasi masing-masing model akan dievaluasi menggunakan metode mean squared error (MSE), dan model dengan nilai error terendah akan dipilih sebagai model utama

## Data Understanding

Dataset yang digunakan diambil dari situs Kaggle dengan nama [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). Dataset ini berasal dari National Institute of Diabetes and Digestive and Kidney Diseases. Tujuan utama dari dataset ini adalah untuk memprediksi secara diagnostik apakah seorang pasien menderita diabetes atau tidak, berdasarkan beberapa pengukuran diagnostik yang terdapat dalam dataset. Secara khusus, seluruh pasien dalam dataset ini adalah wanita berusia minimal 21 tahun yang berasal dari suku Indian Pima.Datase terdiri dari :
1. Variabel Prediktor
   1. Pregnancies – Jumlah kehamilan yang pernah dialami. 
   2. Glucose – Konsentrasi glukosa dalam plasma setelah puasa (tes gula darah puasa). 
   3. BloodPressure – Tekanan darah diastolik (mm Hg).
   4. SkinThickness – Ketebalan lipatan kulit trisep (mm).
   5. Insulin – Kadar insulin serum 2 jam setelah makan (mu U/ml).
   6.BMI - Indeks massa tubuh (berat badan dalam kg dibagi tinggi badan kuadrat dalam meter).
   7. DiabetesPedigreeFunction – Fungsi silsilah diabetes (menggambarkan riwayat diabetes dalam keluarga).
   8. Age – Usia (dalam tahun)
2. Variabel Target
    1. •	Outcome – Status diabetes (1 = diabetes, 0 = tidak diabetes).


Tabel informasi struktur data
| No | Kolom                    | Non-Null Count | Tipe Data |
|----|--------------------------|----------------|-----------|
| 0  | Pregnancies              | 768            | int64     |
| 1  | Glucose                  | 768            | int64     |
| 2  | BloodPressure            | 768            | int64     |
| 3  | SkinThickness            | 768            | int64     |
| 4  | Insulin                  | 768            | int64     |
| 5  | BMI                      | 768            | float64   |
| 6  | DiabetesPedigreeFunction| 768            | float64   |
| 7  | Age                      | 768            | int64     |
| 8  | Outcome                  | 768            | int64     |

Dari tebel diatas dapat dilihat bahwa dataset memiliki 8 kolom dengan tipe number baik int maupun float.
Selanjutnya pemeriksaan fitur numerik, berikut histogram dari masing-masing fitur pada dataset diabetes.

![Age](https://raw.githubusercontent.com/XMB234/Tugas-Machine-Learning-Terapan-Predictive-Analytic-/72a49402ca3cbf475d6b025eb6b2dff23f102d15/Age.jpg) 

Pada fitur Age, dapat dilihat bahwa beberapa data terpusat diumur 20 tahunan. Sebagin kecil data ada juga yang tersebar ke nilai 60 keatas.

![BMI](https://raw.githubusercontent.com/XMB234/Tugas-Machine-Learning-Terapan-Predictive-Analytic-/72a49402ca3cbf475d6b025eb6b2dff23f102d15/BMI.jpg)

Pada fitur BMI, sebagian data terpusat diantarra rentang 25 -40. Sebagian kecil data lainnya tersebat diluar rentang nilai tersebut.

![SkinThickness](https://raw.githubusercontent.com/XMB234/Tugas-Machine-Learning-Terapan-Predictive-Analytic-/72a49402ca3cbf475d6b025eb6b2dff23f102d15/SkinThickness.jpg)

Pada fitur SkinThickness, banyak data memiliki nilai dibawah 5, dan yang lainnya kebanyakan tersebar diantara nilai 20- 40.

![BloodPressure](https://raw.githubusercontent.com/XMB234/Tugas-Machine-Learning-Terapan-Predictive-Analytic-/72a49402ca3cbf475d6b025eb6b2dff23f102d15/BloodPressure.jpg)

Pada fitur BloodPressure. pada fitur ini, nilai data didominasi antara 60 -90, namun dapat diamati bahwa terdapat data yang jumlahnya cukup banyak pada rentang nilai 5 kebawah.

![Insulin](https://raw.githubusercontent.com/XMB234/Tugas-Machine-Learning-Terapan-Predictive-Analytic-/72a49402ca3cbf475d6b025eb6b2dff23f102d15/Insulin.jpg)

Untuk fiur Insulin, kebanyak data tersebar bearada pada rentang nilai kurang dari 200 Terutama pada rentang nilai yang mendekati 0.

![Glucose](https://raw.githubusercontent.com/XMB234/Tugas-Machine-Learning-Terapan-Predictive-Analytic-/72a49402ca3cbf475d6b025eb6b2dff23f102d15/Glucose.jpg)

Untuk fitur Glucose, data tersebar diantara nilai 75-200. Namun paling banyak terpusat diantara nilai 100 -125. Terdapat sebagian kecil data yang berada di bawah nilai 75.

![DiabetesPedigreeFunction](https://raw.githubusercontent.com/XMB234/Tugas-Machine-Learning-Terapan-Predictive-Analytic-/72a49402ca3cbf475d6b025eb6b2dff23f102d15/DiabetesPredigreeFunction.jpg)

Untuk fiur DiabetesPedigreeFunction, sebagain data tersebar pada nilai 0 - 1. Sebagian kecil lainnya berada diatas nilai 1.

![Pregnancies](https://github.com/XMB234/Tugas-Machine-Learning-Terapan-Predictive-Analytic-/blob/main/Pregnancies.jpg?raw=true)

Untuk fiur Pregnancies, kebanyakan data tersebar pada rentang nilai dibawah 5. Sebagian kecil data ada yang tersebahr diatas nilai 12. 

Salah satu syarat penting agar dataset dapat digunakan dalam pembuatan model machine learning adalah keseimbangannya. Salah satu cara untuk memeriksa apakah dataset kita seimbang atau tidak adalah dengan melakukan visualisasi. Berikut adalah visualisasi dari dataset susu yang akan digunakan dalam pembuatan model machine learning untuk proyek ini.

distribusi kelas pada kolom Outcome:
| Outcome | Count |
|---------|-------|
| 0       | 500   |
| 1       | 268   |

Pada tabel di atas, terlihat bahwa dataset memiliki distribusi data yang tidak seimbang pada setiap nilai target. Kondisi ini dapat menyebabkan model lebih banyak belajar pada data yang dominan, yang pada akhirnya dapat mempengaruhi performa model. Untuk mengatasi masalah ini, perlu dilakukan penyeimbangan data dengan menggunakan metode SMOTE (Synthetic Minority Over-sampling Technique).

Selain itu, kita juga perlu melakukan pemeriksaan terhadap missing value dalam dataset. Untuk memeriksa missing value, kita dapat menggunakan fungsi `isnull()`. Berikut adalah tabel yang menunjukkan jumlah missing value pada setiap fitur dalam dataset.

| Kolom                    | Missing Values |
|--------------------------|----------------|
| Pregnancies              | 0              |
| Glucose                  | 0              |
| BloodPressure            | 0              |
| SkinThickness            | 0              |
| Insulin                  | 0              |
| BMI                      | 0              |
| DiabetesPedigreeFunction | 0              |
| Age                      | 0              |
| Outcome                  | 0              |

Dari tabel diatas dapat dilihat bahwa tidak terdapat missing value pada dataset.Namun setelah dilakukan pemahaman lebih mendalam dari data terhadap nilai nilai yang ada pada data, ditemukan bahwa terdapat nilai yang janggal pada beberapa fitur. Hal ini dapat dilihat pada tabel statistik deskriptif pada dataset dibawah

| Feature                       | count | mean  | std   | min  | 25%  | 50%  | 75%  | max  |
|-------------------------------|-------|-------|-------|------|------|------|------|------|
| Pregnancies                   | 768   | 3.85  | 3.37  | 0.00 | 1.00 | 3.00 | 6.00 | 17.00|
| Glucose                        | 768   | 120.89| 31.97 | 0.00 | 99.00| 117.00| 140.25| 199.00|
| BloodPressure                  | 768   | 69.11 | 19.36 | 0.00 | 62.00 | 72.00 | 80.00 | 122.00|
| SkinThickness                  | 768   | 20.54 | 15.95 | 0.00 | 0.00 | 23.00 | 32.00 | 99.00|
| Insulin                        | 768   | 79.80 | 115.24| 0.00 | 0.00  | 30.50 | 127.25| 846.00|
| BMI                            | 768   | 31.99 | 7.88  | 0.00 | 27.30 | 32.00 | 36.60 | 67.10|
| DiabetesPedigreeFunction       | 768   | 0.47  | 0.33  | 0.08 | 0.24  | 0.37  | 0.63  | 2.42 |
| Age                            | 768   | 33.24 | 11.76 | 21.00| 24.00 | 29.00 | 41.00 | 81.00|
| Outcome                        | 768   | 0.35  | 0.48  | 0.00 | 0.00  | 0.00  | 1.00  | 1.00 |


Hal ini dapat dilihat pada tabel diatas bahwa nilai minimal untuk variable Glucose, BloodPressure, Insulin, SkinThickness, Indeks massa tubuh adalah 0. Tentunya ini bukan merupakan nilai yang logis dikarenakan manusia hidup tidak akan memiliki nilai 0 pada variable variable tersebut. Ini merupakan indikasi yang menandakan adanya missing value pada variable variable tersebut. Untuk mengatasi hal tersebut, kita akan mengganti nilai 0 pada variable variable tersebut ke NaN dan dilakukkan pengecekan ulang berapa missing value yang ada pada setiap variable pada proses data Preparation. Berikut tabel berisi jumlah missing value setelah perubahan nilai 0 ke NaN

| No | Fitur                    | Jumlah Missing Value |
|----|--------------------------|----------------------|
| 0  | Pregnancies              | 0                    |
| 1  | Glucose                  | 5                    |
| 2  | BloodPressure            | 35                   |
| 3  | SkinThickness            | 227                  |
| 4  | Insulin                  | 374                  |
| 5  | BMI                      | 11                   |
| 6  | DiabetesPedigreeFunction| 0                    |
| 7  | Age                      | 0                    |
| 8  | Outcome                  | 0                    |

Berdasarkan tabel diatas terdapat missing value dengan jumlah yang tinggi pada beberapa variable. Untuk mengatasi hal tersebut, kita akan mengganti nilai missing value ini dengan nilai median masing masing variable yang missing value.

Selanjutnya, kita perlu melakukan pemeriksaan terhadap outlier dalam dataset. Outlier dapat memengaruhi hasil pelatihan model machine learning, sehingga penting untuk mengidentifikasinya secara akurat. Salah satu metode yang dapat digunakan untuk mendeteksi dan menangani outlier adalah Local Outlier Factor (LOF).

LOF adalah teknik dalam machine learning yang digunakan untuk mendeteksi anomali atau data yang berbeda secara signifikan dari mayoritas. Metode ini mengukur tingkat "keanehan" suatu data berdasarkan kepadatan lingkungannya. Jika sebuah data berada di area dengan kepadatan yang jauh lebih rendah dibandingkan tetangga-tetangganya, maka data tersebut dianggap sebagai outlier. LOF menghitung skor untuk setiap titik data—semakin tinggi skornya, semakin besar kemungkinan data tersebut adalah outlier.

LOF sangat cocok digunakan pada dataset multivariat karena mampu mempertimbangkan hubungan kompleks antar variabel. Oleh karena itu, metode ini relevan untuk diterapkan pada dataset PIMA Diabetes, yang terdiri dari berbagai variabel yang saling berinteraksi, seperti kadar glukosa, tekanan darah, dan berat badan. Sebagai contoh, kelebihan berat badan dapat memicu resistensi insulin, yang selanjutnya meningkatkan kadar glukosa darah dan berpotensi berhubungan dengan tekanan darah tinggi. Kombinasi interaksi ini membuat LOF menjadi metode yang efektif dalam mendeteksi data yang tidak wajar pada dataset ini. Berikut tabel data yang terdeteksi sebagi outlier

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DiabetesPedigreeFunction | Age | Outcome | LOF_Score | LOF_Label |
|--------------|---------|---------------|---------------|---------|------|--------------------------|-----|---------|-----------|-----------|
| 1            | 189.00  | 60.00         | 23.00         | 846.00  | 30.10| 0.40                     | 59  | 1       | -3.30     | -1        |
| 1            | 103.00  | 30.00         | 38.00         | 83.00   | 43.30| 0.18                     | 33  | 0       | -1.73     | -1        |
| 0            | 100.00  | 88.00         | 60.00         | 110.00  | 46.80| 0.96                     | 31  | 0       | -1.62     | -1        |
| 5            | 44.00   | 62.00         | 27.00         | 102.50  | 25.00| 0.59                     | 36  | 0       | -1.64     | -1        |
| 1            | 96.00   | 122.00        | 27.00         | 102.50  | 22.40| 0.21                     | 27  | 0       | -2.10     | -1        |
| 0            | 162.00  | 76.00         | 56.00         | 100.00  | 53.20| 0.76                     | 25  | 1       | -1.74     | -1        |
| 1            | 88.00   | 30.00         | 42.00         | 99.00   | 55.00| 0.50                     | 26  | 1       | -2.37     | -1        |
| 0            | 129.00  | 110.00        | 46.00         | 130.00  | 67.10| 0.32                     | 26  | 1       | -1.86     | -1        |
| 7            | 142.00  | 60.00         | 33.00         | 190.00  | 28.80| 0.69                     | 61  | 0       | -1.60     | -1        |
| 4            | 197.00  | 70.00         | 39.00         | 744.00  | 36.70| 2.33                     | 31  | 0       | -2.49     | -1        |
| 0            | 165.00  | 90.00         | 33.00         | 680.00  | 52.30| 0.43                     | 23  | 0       | -2.01     | -1        |
| 9            | 106.00  | 52.00         | 27.00         | 102.50  | 31.20| 0.38                     | 42  | 0       | -1.67     | -1        |
| 13           | 152.00  | 90.00         | 33.00         | 29.00   | 26.80| 0.73                     | 43  | 1       | -1.50     | -1        |
| 1            | 139.00  | 46.00         | 19.00         | 83.00   | 28.70| 0.65                     | 22  | 0       | -1.70     | -1        |
| 5            | 103.00  | 108.00        | 37.00         | 102.50  | 39.20| 0.30                     | 65  | 0       | -1.60     | -1        |
| 4            | 116.00  | 72.00         | 12.00         | 87.00   | 22.10| 0.46                     | 37  | 0       | -1.56     | -1        |
| 3            | 96.00   | 56.00         | 34.00         | 115.00  | 24.70| 0.94                     | 39  | 0       | -1.56     | -1        |
| 0            | 180.00  | 78.00         | 63.00         | 14.00   | 59.40| 2.42                     | 25  | 1       | -2.15     | -1        |
| 9            | 134.00  | 74.00         | 33.00         | 60.00   | 25.90| 0.46                     | 81  | 0       | -2.08     | -1        |
| 9            | 120.00  | 72.00         | 22.00         | 56.00   | 20.80| 0.73                     | 48  | 0       | -1.51     | -1        |
| 0            | 84.00   | 82.00         | 31.00         | 125.00  | 38.20| 0.23                     | 23  | 0       | -1.64     | -1        |
| 0            | 57.00   | 60.00         | 27.00         | 102.50  | 21.70| 0.73                     | 67  | 0       | -1.72     | -1        |
| 1            | 164.00  | 82.00         | 43.00         | 67.00   | 32.80| 0.34                     | 50  | 0       | -1.57     | -1        |
| 6            | 92.00   | 62.00         | 32.00         | 126.00  | 32.00| 0.09                     | 46  | 0       | -1.52     | -1        |
| 2            | 197.00  | 70.00         | 99.00         | 169.50  | 34.70| 0.57                     | 62  | 1       | -3.06     | -1        |
| 1            | 89.00   | 24.00         | 19.00         | 25.00   | 27.80| 0.56                     | 21  | 0       | -1.95     | -1        |
| 6            | 98.00   | 58.00         | 33.00         | 190.00  | 34.00| 0.43                     | 43  | 0       | -1.50     | -1        |
| 10           | 68.00   | 106.00        | 23.00         | 49.00   | 35.50| 0.28                     | 47  | 0       | -1.61     | -1        |
| 5            | 126.00  | 78.00         | 27.00         | 22.00   | 29.60| 0.44                     | 40  | 0       | -1.60     | -1        |

Dari tabel diatas, dapat diketahui bahwa terdapat beberapa data pada dataset yang terdeteksi sebagai outlie dengen metode LOF. Untuk itu, kita akan menghapus beberapa outlier dan menngguanakan metode standarisasi yang tahan terhadapa outliners pada saat proses standarisasi data.

## Data Preparation
Agar dataset lebih mudah dipahami oleh model, dataset harus dipersiapkan dengan cara tertentu. Beberapa metode dapat diterapkan dalam tahap persiapan data, dan metode yang akan digunakan dalam proyek ini yaitu:
* Mengganti missing value dengan median pada setiap variabel bertujuan untuk mencegah distorsi yang mungkin terjadi jika menggunakan rata-rata, yang sangat dipengaruhi oleh outlier. Median lebih stabil terhadap nilai ekstrem, serta membantu menjaga keseimbangan data dan mengurangi potensi overfitting atau bias yang dapat muncul jika missing value diganti dengan nilai yang tidak representatif. Oleh karena itu, mengganti missing value dengan median adalah metode yang lebih efektif dan dapat diandalkan dalam mempersiapkan data untuk model machine learning.
* Mengelompokan data numerik ke babebarap kategori. Proses ini melibatkan pengelompokan nilai numerik pada variabel Glucose, BMI, dan Insulin ke dalam beberapa kategori. Langkah ini dilakukan karena pada beberapa algoritma machine learning, khususnya yang berbasis klasifikasi seperti decision tree atau model berbasis aturan, mengonversi data numerik menjadi kategori dapat meningkatkan stabilitas model dan membantu dalam menghasilkan prediksi yang lebih akurat.
* Menerapkan teknik encoding pada data kategori yang dibuat sebelumnya. Penerapan teknik encoding pada data kategorikal bertujuan mengubah nilai kategori atau label menjadi format numerik agar dapat diproses oleh algoritma machine learning, yang umumnya hanya mengenali data numerik. Encoding membantu model memahami hubungan antar fitur. Pada tahap ini, teknik yang digunakan adalah One-Hot Encoding dan diterapkan pada data kategorikal yang telah dibuat sebelumnya.

* Melakukan standarisasi pada data numerik. Standarisasi adalah proses mengubah nilai fitur dalam dataset ke skala tertentu agar memiliki rentang yang seragam. Ini penting untuk algoritma seperti regresi linier, KNN, dan SVM yang sensitif terhadap perbedaan skala antar fitur. Tanpa standarisasi, fitur dengan nilai besar bisa mendominasi model. Dalam proyek ini, fitur yang akan distandarisasi meliputi Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, dan Age. Fitur lain tidak distandarisasi karena memiliki skala yang sudah cukup besar. Proses ini akan menggunakan fungsi *RobustScaler* dari library scikit-learn.
* Membagi dataset unutk train dan test. Pembagian dataset dilakukan menggunakan fungsi *train\_test\_split* dari library scikit-learn, yang berfungsi membagi data menjadi data latih dan data uji. Pembagian ini penting untuk mengukur akurasi model secara objektif. Dalam proses ini, data dibagi dengan proporsi 80% untuk pelatihan dan 20% untuk pengujian.
* Meyeimbangkan data. Penyeimbangan data dilakukan dengan memperbanyak data pada kelas minoritas agar setara dengan kelas mayoritas, sehingga model tidak bias. Tujuannya adalah agar model dapat mengenali kedua kelas secara adil dan menghasilkan prediksi yang lebih akurat. Metode yang digunakan adalah SMOTE, yang hanya diterapkan pada data latih. Penerapan SMOTE pada data uji dihindari karena dapat menciptakan data sintetis yang tidak merepresentasikan kondisi nyata, sehingga berisiko menyebabkan evaluasi model menjadi tidak realistis dan terlalu optimis.


## Modeling
Proyek ini berfokus pada masalah prediksi apakah seseorang akan terkena diabetes atau tidak, yang dalam konteks machine learning termasuk dalam kategori masalah klasifikasi. Untuk mencari model dengan performa terbaik, digunakan pustaka machine learning bernama LazyPredict. LazyPredict adalah pustaka Python yang menyediakan alat otomatisasi untuk menguji dan mengevaluasi berbagai model machine learning. Library ini memungkinkan pengguna untuk dengan mudah mengevaluasi berbagai model machine learning tanpa perlu menulis kode yang rumit dan memakan waktu. LazyPredict secara otomatis melatih berbagai model machine learning populer dengan pengaturan dan hyperparameter yang berbeda, lalu menghasilkan ringkasan performa untuk setiap model. Hasilnya kemudian diberi peringkat berdasarkan kinerja terbaik, sehingga pengguna dapat dengan mudah memilih model yang paling sesuai untuk tugas yang dihadapi. Berikut tabel peforma model dari hasil menggunakan LazyPredict

| Model                          | Accuracy | Balanced Accuracy | ROC AUC | F1 Score | Time Taken |
|--------------------------------|----------|--------------------|---------|----------|------------|
| LGBMClassifier                 | 0.90     | 0.89               | 0.89    | 0.90     | 0.12       |
| RandomForestClassifier         | 0.89     | 0.89               | 0.89    | 0.89     | 0.20       |
| AdaBoostClassifier             | 0.89     | 0.88               | 0.88    | 0.89     | 0.15       |
| BaggingClassifier              | 0.89     | 0.88               | 0.88    | 0.89     | 0.06       |
| XGBClassifier                  | 0.89     | 0.88               | 0.88    | 0.89     | 0.32       |
| ExtraTreesClassifier           | 0.89     | 0.87               | 0.87    | 0.89     | 0.17       |
| DecisionTreeClassifier         | 0.88     | 0.86               | 0.86    | 0.88     | 0.02       |
| SVC                            | 0.84     | 0.84               | 0.84    | 0.84     | 0.03       |
| NuSVC                          | 0.83     | 0.83               | 0.83    | 0.84     | 0.06       |
| BernoulliNB                    | 0.82     | 0.82               | 0.82    | 0.82     | 0.01       |
| KNeighborsClassifier           | 0.79     | 0.79               | 0.79    | 0.80     | 0.02       |
| LogisticRegression             | 0.79     | 0.78               | 0.78    | 0.80     | 0.02       |
| CalibratedClassifierCV         | 0.78     | 0.77               | 0.77    | 0.78     | 0.06       |
| LinearSVC                      | 0.78     | 0.77               | 0.77    | 0.78     | 0.02       |
| LabelPropagation               | 0.78     | 0.77               | 0.77    | 0.78     | 0.03       |
| LabelSpreading                 | 0.78     | 0.77               | 0.77    | 0.78     | 0.04       |
| GaussianNB                     | 0.77     | 0.76               | 0.76    | 0.77     | 0.02       |
| PassiveAggressiveClassifier    | 0.78     | 0.76               | 0.76    | 0.78     | 0.02       |
| RidgeClassifier                | 0.76     | 0.74               | 0.74    | 0.76     | 0.02       |
| LinearDiscriminantAnalysis     | 0.76     | 0.74               | 0.74    | 0.76     | 0.04       |
| ExtraTreeClassifier            | 0.76     | 0.74               | 0.74    | 0.76     | 0.02       |
| RidgeClassifierCV              | 0.75     | 0.74               | 0.74    | 0.76     | 0.01       |
| NearestCentroid                | 0.73     | 0.73               | 0.73    | 0.73     | 0.02       |
| SGDClassifier                  | 0.74     | 0.73               | 0.73    | 0.74     | 0.02       |
| QuadraticDiscriminantAnalysis  | 0.74     | 0.72               | 0.72    | 0.74     | 0.01       |
| Perceptron                     | 0.69     | 0.70               | 0.70    | 0.70     | 0.02       |
| DummyClassifier                | 0.66     | 0.50               | 0.50    | 0.53     | 0.02       |

Kita akan menggunakan lima algoritma terbaik berdasarkan hasil dari LazyPredict. Berdasarkan tabel, LazyPredict, algoritma XGBoost, Random Forest, AdaBoost, Bagging, dan LightGBM menunjukkan performa terbaik. Oleh karena itu, kita akan membangun kelima model ini dan berfokus untuk mengoptimalkan performanya lebih lanjut melalui teknik hyperparameter tuning pada setiap model yang akan diuji. Selain itu, kita juga akan membuat model stacking yang menggabungkan model model tersebut menjadi satu model.

### XGBoost
XGBoost adalah teknik optimisasi berbasis pohon keputusan yang dibangun menggunakan metode penurunan gradien. Cara kerjanya adalah dengan menggabungkan hasil prediksi dari beberapa pohon keputusan. Pohon pertama menghasilkan prediksi awal, kemudian pohon berikutnya akan membuat prediksi berdasarkan kesalahan (residual) dari prediksi yang dibuat oleh pohon sebelumnya. Proses ini berlanjut hingga model mencapai prediksi yang lebih akurat. Teknik ini dikenal sangat efektif dalam meningkatkan akurasi model dengan cara meminimalkan kesalahan prediksi secara bertahap.
#### Kelebihan
* **Akurasi Tinggi**: XGBoost sering menghasilkan performa terbaik di berbagai kompetisi data science dan aplikasi nyata, seperti prediksi harga, deteksi risiko, dan analisis perilaku
* **Skalabilitas**: Dapat menangani dataset besar dan fitur yang banyak tanpa mengorbankan kecepatan
* **Fleksibilitas**: Mendukung berbagai jenis tugas (klasifikasi, regresi, ranking) dan dapat dioptimalkan dengan berbagai parameter
#### Kekurangan 
* **Keterbatasan Interpretasi**
XGBoost sulit dipahami, sehingga kurang cocok untuk aplikasi yang membutuhkan transparansi tinggi, seperti di kesehatan dan keuangan 
* **Sensitivitas terhadap Data Tidak Seimbang**: XGBoost cenderung bias terhadap kelas mayoritas pada data tidak seimbang, memerlukan teknik tambahan untuk mengatasi hal ini.
* **Kompleksitas dan Tuning**: Tuning XGBoost rumit dan memakan waktu karena banyak hyperparameter yang perlu disesuaikan.
* **Konsumsi Sumber Daya**: Pelatihan XGBoost pada dataset besar membutuhkan sumber daya komputasi yang besar, meskipun penggunaan GPU dapat mempercepat proses.
#### Parameter
* n_estimator = Menentukan jumlah total pohon (trees) yang akan dibangun dalam model. Diantara (50, 100, 200) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 100.
* learning_rate = Menentukan laju pembelajaran atau seberapa besar kontribusi setiap pohon terhadap prediksi akhir. Diantara (0.01, 0.1, 0.3) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 0,3.
* max_depth = Menentukan kedalaman maksimum pohon keputusan. Diantara (3, 5, 7) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 7.
* min_child_weight = Menentukan jumlah minimum bobot sampel yang diperlukan di dalam sebuah node untuk membagi pohon. Diantara (1, 3, 5) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 1.
* subsample = Menentukan proporsi data pelatihan yang akan digunakan untuk membangun setiap pohon. Diantara (0.8, 0.9, 1.0) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 0,9.
* colsample_bytree = Menentukan proporsi fitur yang digunakan untuk membangun setiap pohon. Diantara (0.8, 0.9, 1.0) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 0,8.
* gamma = Menentukan pengurangan loss yang diperlukan untuk membagi node lebih lanjut. Diantara (0, 0.1, 0.2) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 0,1.
### Random Forest 
Random Forest Classifier adalah metode ensemble learning dan algoritma supervised learning yang digunakan untuk tugas klasifikasi maupun regresi. Algoritma ini bekerja dengan membangun sejumlah pohon keputusan (decision tree) dan menggabungkannya untuk menghasilkan prediksi yang lebih stabil dan akurat.
#### Kelebihan
* **Mengatasi Overfitting**: Random Forest mengurangi overfitting dengan bagging dan pemilihan fitur acak, meningkatkan generalisasi model.
* **Stabilitas**: Random Forest lebih stabil dibandingkan pohon keputusan tunggal, menghasilkan prediksi yang lebih konsisten.
* **Pengukuran Kepentingan Fitur**: Random Forest menilai kontribusi fitur, memberikan wawasan tentang fitur yang paling berpengaruh.
* **Toleransi terhadap Data Tidak Seimbang**: Random Forest efektif menangani data tidak seimbang dengan pengambilan sampel acak dan pembobotan kelas, meningkatkan akurasi untuk kelas minoritas.
#### Kekurangan
* **Interpretabilitas Terbatas**: Pengaruh fitur sulit diinterpretasikan karena melibatkan banyak pohon keputusan.
* **Parameter yang Perlu Diatur**: Memerlukan eksperimen dan penyesuaian parameter, seperti jumlah pohon dan pemilihan fitur acak, untuk hasil yang optimal.
* **Kesulitan pada Data Dimensionalitas Tinggi**: Random Forest kesulitan menangani data dengan banyak fitur dibandingkan jumlah sampel, yang dapat menurunkan akurasi prediksi.
#### Parameter
* n_estimator = Menentukan jumlah total pohon (trees) yang akan dibangun dalam model. Diantara (50, 100, 200) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 100.
* max_depth = Menentukan kedalaman maksimum pohon keputusan. Diantara (None, 10, 20, 30) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah None.
* min_samples_split = Menentukan jumlah sampel minimum yang diperlukan untuk membagi sebuah node. Diantara (2, 5, 10) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 5.
* min_samples_leaf = Menentukan jumlah sampel minimum yang harus ada di daun pohon. Diantara (1, 2, 4) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 2.
* max_features = Menentukan jumlah maksimum fitur yang akan dipertimbangkan untuk pemisahan setiap node. Diantara ('auto', 'sqrt', 'log2') dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah sqrt.
* bootstrap = Menentukan apakah sampel bootstrap (sampling dengan pengembalian) digunakan untuk membangun pohon. Diantara (True, False) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah False.
### AdaBoost
Adaptive Boosting (AdaBoost) adalah teknik boosting yang digunakan sebagai metode ensemble dalam machine learning. Algoritma AdaBoost bekerja dengan cara melatih weak learners, seperti decision tree atau model linear, secara iteratif pada dataset. Setiap instance dalam dataset diberikan bobot berdasarkan kesalahan klasifikasinya, sehingga model lebih fokus pada instance yang sulit untuk diprediksi.
#### Kelebihan
* **Meningkatkan Performa Prediksi**: AdaBoost meningkatkan akurasi dengan menggabungkan weak learners menjadi strong learner, mengurangi kesalahan klasifikasi.
* **Penanganan Data Kompleks**: Efektif dalam menangani data dengan interaksi fitur rumit dan pola yang kompleks.
* **Mencegah Overfitting**: Memberikan bobot pada contoh yang salah klasifikasi untuk mengurangi overfitting dan meningkatkan generalisasi model.
#### Kekurangan
* **Sensitif terhadap Outlier**: Outlier dapat memengaruhi bobot sampel dan menghasilkan model yang kurang optimal, sehingga preprocessing data yang hati-hati diperlukan.
* **Ketergantungan pada Kualitas Data Latih**: Kualitas dataset pelatihan sangat mempengaruhi kinerja algoritma; data yang tidak representatif atau bias dapat mengurangi kemampuan model dalam menggeneralisasi.
* **Waktu Pelatihan yang Relatif Lambat**: Algoritma ini memerlukan iterasi berulang dalam membangun model, yang dapat memakan waktu lama, terutama untuk dataset besar.
#### Parameter
* n_estimator = Menentukan jumlah total pohon (trees) yang akan dibangun dalam model. Diantara (50, 100, 200) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 200.
* learning_rate = Menentukan laju pembelajaran atau seberapa besar kontribusi setiap pohon terhadap prediksi akhir. Diantara (0.01, 0.1, 1.0) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 1,0.
* estimator_max_depth = mengatur kedalaman maksimum dari pohon keputusan yang digunakan sebagai estimator dasar (base estimator). Diantara (1, 3, 5) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 5.
* estimator__min_samples_split = Menentukan jumlah minimum sampel yang diperlukan untuk membagi sebuah node dalam pohon keputusan dasar. Diantara (2, 5) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 2.

### Bagging
Bagging adalah metode pembelajaran ansambel yang digunakan untuk mengurangi varians pada dataset yang memiliki banyak derau. Algoritma ini bekerja dengan membuat beberapa subset dari dataset asli melalui proses bootstrapping. Setiap subset kemudian digunakan untuk melatih model yang berbeda, dan hasil dari semua model tersebut digabungkan (agregasi) untuk menghasilkan prediksi akhir.
#### Kelebihan
* **Mengurangi Risiko Overfitting**: Dengan menggabungkan berbagai model, bagging membantu mengurangi risiko overfitting.
* **Meningkatkan Ketahanan**: Model yang dihasilkan menjadi lebih stabil dan kurang terpengaruh oleh perubahan data.
* **Meningkatkan Presisi**: Bagging dapat meningkatkan akurasi model dengan mengurangi variansi.
#### Kekurangan
* **Waktu Pelatihan Lama**: Melatih banyak model dapat memerlukan waktu komputasi yang lebih lama.
* **Tidak Menurunkan Bias**: Bagging tidak selalu berhasil mengurangi bias, terutama jika model dasar memiliki bias yang tinggi.
#### Parameter
* n_estimator = Menentukan jumlah estimator (atau "weak learners") yang digunakan dalam ensemble. Diantara (10, 50, 100) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 50.
* max_samples = Menentukan proporsi data pelatihan yang digunakan untuk membangun setiap estimator dalam ensemble. Diantara (0.5, 0.8, 1.0) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 1,0.
* max_features = Menentukan proporsi fitur yang digunakan untuk membangun setiap estimator dalam ensemble. Diantara (0.5, 0.8, 1.0) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 0,5.
* bootstrap = Menentukan apakah bootstrap sampling (sampling dengan pengembalian) digunakan saat membangun estimator. Diantara (True, False) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah False.
* estimator__max_depth = Parameter ini digunakan untuk mengatur kedalaman maksimum pohon keputusan yang digunakan sebagai estimator dasar. Diantara (None, 5, 10) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah None.
* estimator__min_samples_split = Menentukan jumlah sampel minimum yang diperlukan untuk membagi sebuah node dalam pohon keputusan dasar. Diantara (2, 5) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 5.

### LightGBM
LightGBM adalah algoritma machine learning yang menggunakan teknik gradient boosting untuk melakukan prediksi. Algoritma ini menggabungkan beberapa pohon keputusan sederhana untuk membentuk model prediktif yang lebih kuat. LightGBM mengadopsi pendekatan berbasis histogram dalam proses pembagian data, yang meningkatkan efisiensi dan performa model.
#### Kelebihan
* **Kecepatan Tinggi**: LGBM lebih cepat dalam pelatihan dibandingkan algoritma boosting tradisional berkat penggunaan histogram-based learning.
* **Efisien untuk Dataset Besar**: Sangat efisien dalam menangani dataset besar.
* **Dukungan Fitur Kategorikal**: Secara native mendukung fitur kategorikal tanpa memerlukan one-hot encoding.
* **Akurasi Tinggi**: Memberikan performa yang kompetitif dan sering kali unggul dalam berbagai kompetisi machine learning.
* **Mengurangi Overfitting**: Dengan fitur regularisasi dan teknik boosting, LGBM membantu mengontrol overfitting.
#### Kekurangan
* **Peka terhadap Data yang Tidak Bersih**: LGBM sensitif terhadap outlier dan missing value jika tidak ditangani dengan baik.
* **Parameter Tuning Rumit**: Membutuhkan penyesuaian parameter yang cermat untuk mencapai performa optimal.
* **Kurang Optimal untuk Data Kecil**: Pada dataset kecil, LGBM tidak selalu lebih unggul dibandingkan model lain seperti XGBoost atau Random Forest.
* **Sulit Diinterpretasikan**: Model yang kompleks membuat interpretasi hasil lebih sulit dibandingkan model linear.
#### Parameter
* n_estimator = Menentukan jumlah total pohon (trees) yang akan dibangun dalam model. Diantara (50, 100, 200) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 50.
* learning_rate = Menentukan laju pembelajaran atau seberapa besar kontribusi setiap pohon terhadap prediksi akhir. Diantara (0.01, 0.1, 0.2) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 0,2.
* num_leaves = Menentukan jumlah maksimum daun (leaf nodes) dalam setiap pohon. Diantara (31, 50, 100) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 50.
* max_depth = Menentukan kedalaman maksimum pohon. Diantara (-1, 10, 20) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah -1.
* min_child_samples = Menentukan jumlah minimum sampel yang harus ada pada setiap daun pohon. Diantara (20, 50, 100) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 20
* subsample = Menentukan proporsi sampel yang digunakan untuk membangun setiap pohon. Diantara (0.6, 0.8, 1.0) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 0,6.
* colsample_bytree = Menentukan proporsi fitur yang digunakan untuk setiap pohon. Diantara (0.6, 0.8, 1.0) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 1,0.
* boosting_type = Menentukan jenis boosting yang digunakan dalam model. Diantara ('gbdt', 'dart') dengan menggunakan GridSearch didapat jenis yang terbaik untuk parameter adalah gbdt.
### Stacking Model
Stacking adalah teknik ensemble yang menggabungkan beberapa model machine learning. Cara kerjanya adalah dengan melatih beberapa model berbeda, kemudian menggunakan model yang lebih tinggi (meta-model) untuk menggabungkan hasil prediksi dari model-model tersebut dan menghasilkan prediksi akhir yang lebih akurat. meta-model yang digunakan pada project ini adalah model LogisticRegression.
#### Kelebihan
* **Peningkatan Akurasi**: Stacking seringkali meningkatkan akurasi prediksi dibandingkan dengan menggunakan model dasar secara tunggal.
* **Peningkatan Generalisasi**: Stacking dapat meningkatkan kemampuan model untuk membuat prediksi yang lebih baik pada data baru.
* **Fleksibilitas**: Stacking dapat menggabungkan berbagai jenis model dasar, memanfaatkan kelebihan masing-masing model untuk hasil yang lebih optimal.
#### Kekurangan
* **Kompleksitas Tinggi**: Stacking menggabungkan banyak model, sehingga struktur model menjadi kompleks dan sulit dipahami.
* **Waktu dan Sumber Daya Lebih Besar**: Melatih beberapa model sekaligus memakan waktu dan membutuhkan komputasi yang lebih tinggi.
* **Risiko Overfitting**: Tanpa validasi silang yang tepat, stacking bisa menyebabkan overfitting pada data pelatihan.
* **Interpretasi Sulit**: Prediksi berasal dari gabungan banyak model, sehingga sulit untuk dijelaskan secara intuitif.
* **Implementasi Lebih Rumit**: Dibandingkan dengan model tunggal, stacking memerlukan pipeline pelatihan dan prediksi yang lebih kompleks.
#### Parameter
* final_estimator__C = Menentukan kekuatan regularisasi dalam model. Diantara (0.01, 0.1, 1, 10) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 1. 
* final_estimator__C = Menentukan jenis regularisasi yang digunakan dalam model. Diantara ('l2', 'l1') dengan menggunakan GridSearch didapat jenis yang terbaik untuk parameter adalah l2.
## Evaluation
Pada tahap evaluasi, akan digunakan Mean Squared Error (MSE) untuk mengukur kesalahan prediksi model dalam memprediksi penyakit diabetes, dibandingkan dengan data aktual. MSE adalah metrik evaluasi yang umum digunakan dalam machine learning untuk menilai seberapa baik model memprediksi nilai target. MSE dihitung dengan rata-rata kuadrat selisih antara nilai prediksi model dan nilai aktual, yang memberikan gambaran mengenai besarnya kesalahan prediksi. MSE sering digunakan untuk membandingkan performa berbagai model machine learning, seperti Random Forest, Linear Regression, Support Vector Regression, dan lainnya. Model dengan nilai MSE terendah dianggap yang paling akurat, karena nilai MSE yang lebih kecil menunjukkan prediksi model yang lebih mendekati data aktual. Rumus untuk menghitung MSE adalah sebagai berikut:

\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

Keterangan:
- n = jumlah data (sample).
- y_i \) = nilai observasi yang sebenarnya (actual value).
- \( \hat{y}_i \) = nilai prediksi (predicted value).

Berikut nilai  MSE untuk masing masing model dalam memprediks penyakit diabetes

| Model                    | Mean Squared Error |
|--------------------------|--------------------|
| XGB                       | 0.1126             |
| Random Forest             | 0.1126             |
| Bagging Classifier        | 0.1126             |
| LightGBM                  | 0.1126             |
| AdaBoost                  | 0.0993             |
| Stacking                  | 0.0993             |

Dari diagram tersebut dapat dilihat bahwa model stacking memiliki nilai MSE paling rendah pada test set. Maka stacking akan digunkan pada proyek ini dalam memprediksi penyakit diabetes berdasarkan fitur Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, DiabetesPedigreeFunction, dan Age.

## Kesimpulan 
Dengan menerapkan beberapa metode, dari keenam model yang telah diuji, model stacking merupakan model yang memiliki error paling rendah dan mampu memprediksi penyakit diabetes dengan tepat. Tujuan dari proyek dalam mengatasi permasalahan yang telah disebutkan telah dapat tercapai.
## Referensi
1. Ahmadi, T., Wulandari, A., & Suhatman, H. (2019). Sistem Customer Churn Prediction Menggunakan Machine Learning pada Perusahaan ISP. Jetri: Jurnal Ilmiah Teknik Elektro, 17.
2. Apriliah, W., Kurniawan, I., Baydhowi, M., & Haryati, T. (2021). Prediksi Kemungkinan Diabetes pada Tahap Awal Menggunakan Algoritma Klasifikasi Random Forest. SISTEMASI, 10(1), 163. https://doi.org/10.32520/stmsi.v10i1.1129
3. Fernandes, W., Komati, K. S., & Assis de Souza Gazolli, K. (2024). Anomaly detection in oil-producing wells: a comparative study of one-class classifiers in a multivariate time series dataset. Journal of Petroleum Exploration and Production Technology, 14(1), 343–363. https://doi.org/10.1007/s13202-023-01710-6
4. Öngelen, G., & İnkaya, T. (2023). A novel LOF-based ensemble regression tree methodology. Neural Computing and Applications, 35(26), 19453–19463. https://doi.org/10.1007/s00521-023-08773-w
5. Septiana Rizky, P., Haiban Hirzi, R., & Hidayaturrohman, U. (2022). Perbandingan Metode LightGBM dan XGBoost dalam Menangani Data dengan Kelas Tidak Seimbang. J Statistika: Jurnal Ilmiah Teori Dan Aplikasi Statistika, 15(2), 228–236. https://doi.org/10.36456/jstat.vol15.no2.a5548
6. Silalahi, A. P., & Simanullang, H. G. (2023). SUPERVISED LEARNING METODE K-NEAREST NEIGHBOR UNTUK PREDIKSI DIABETES PADA WANITA. METHOMIKA Jurnal Manajemen Informatika Dan Komputerisasi Akuntansi, 7(1), 144–149. https://doi.org/10.46880/jmika.Vol7No1.pp144-149
 

 

