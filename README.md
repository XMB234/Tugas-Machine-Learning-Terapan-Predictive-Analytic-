# Ekspolaris Berbagai Model Machine Learning dalam Memprediksi Penyakit Diabetes - Al Hadi Busra
## Domain Proyek

Diabetes mellitus (DM) adalah penyakit degeneratif kronis yang terjadi akibat produksi insulin yang tidak mencukupi di pankreas atau karena tubuh tidak dapat menggunakan insulin secara efektif, yang mengakibatkan hiperglikemi (peningkatan kadar glukosa darah) sebagai indikator utama. Menurut _World Health Organization_ (WHO), diperkirakan sebanyak 300 juta orang di seluruh dunia akan terkena DM pada tahun 2025. Selain itu, DM juga tercatat sebagai salah satu penyebab kematian utama, berdasarkan data dari International Diabetes Federation yang dirilis pada 2021, dengan Indonesia menduduki peringkat keenam, mencatatkan angka kematian sebanyak 236.711 jiwa. Karena gejalanya yang mirip dengan penyakit umum lainnya, banyak orang yang tidak menyadari bahwa mereka menderita diabetes, bahkan ketika penyakit ini sudah berkembang menjadi komplikasi. Untuk memastikan apakah seseorang mengidap diabetes, diperlukan diagnosis dari dokter melalui pemeriksaan darah. Metode yang biasa digunakan untuk deteksi diabetes adalah tes laboratorium, seperti pengukuran glukosa darah dan uji toleransi glukosa oral. Namun, hasil tes ini sering kali dipengaruhi oleh kesalahan manusia atau bias dalam pengujian, terutama pada saat analisis manual atau interpretasi data dilakukan.Oleh karena itu, diperlukan suatu metode berbasis data medis yang mempertimbangkan faktor-faktor yang mempengaruhi penyakit diabetes untuk melakukan diagnosis, serta pendekatan yang dapat mengurangi kesalahan manusia dan bias dalam pengujian prediksi penyakit diabetes.

Dalam kegiatan prediksi diagnostik, data mining dan text mining telah terbukti sebagai metode yang efektif. Metode ini terdiri dari serangkaian alat dan teknik yang dapat mengeksplorasi kumpulan data serta membantu dalam penemuan pengetahuan. Machine learning dianggap sebagai salah satu komponen utama kecerdasan buatan yang mendukung pengembangan sistem komputer yang mampu memperoleh pengetahuan dari pengalaman sebelumnya tanpa memerlukan pemrograman untuk setiap kasus. Model ini dapat dilatih menggunakan dataset besar yang mencakup berbagai variabel untuk menghasilkan prediksi yang lebih akurat. Dengan penggunaan algoritma seperti regresi logistik, pohon keputusan, atau deep learning, sistem ini dapat memberikan prediksi yang lebih cepat dan tepat. Sehingga, metode ini dapat digunakan untuk mendeteksi penyakit tanpa bergantung pada pengujian manual yang rentan terhadap kesalahan.

Berdasarkan latar belakang, pada proyek ini akan dibuat sebuah model machine learning yang dapat memprediksi penyakit diabetes berdasarkan factor factor penyakit diabetes.

## Bussiness Understanding
### Problem Statements
Selain menimbulkan masalah kesehatan langsung seperti gangguan pada kadar gula darah, penyakit diabetes juga dapat menyebabkan berbagai komplikasi serius yang dapat memengaruhi kualitas hidup penderitanya. Dari pernyataan tersebut, dapat ditarik kesimpulan bahwa permasalahan utama dapat dinyatakan dengan sebuah pertanyaan berikut:
* Bagaimana memanfaatkan machine learning untuk menghasilkan sistem prediksi penyakit diabetes yang akurat, serta minim kesalahan dan bias berdasarak faktor-faktor penyakit diabetes

### Goals
Dalam menyelesaikan permasalahan tersebut, berikut beberapa solusi yang akan dilakukan pada proyek ini:
* Menemukan model machine learning dengan performa terbaik dalam memprediksi penyakit diabetes berdasarkan sejumlah fitur yang tersedia.
* Membangun dan mengembangakan model machine learning untuk memprediksi penyakit diabetes berdasarkan hasil eksplorasi awal model.

### Solution statements
Untuk mencapai tujuan tersebut, maka :
* Dilakukan pencarian beberapa model yang memiliki peforma bagsu dengan menggunakan Pustaka lazypredict untuk mengevaluasi dan membandingkan berbagai model guna menentukan yang paling efektif
* Melakukan hyperparameter tuning pada beberapa model terbaik hasil eksplorasi menggunakan LazyPredict. Evaluasi kinerja dilakukan menggunakan metrik klasifikasi F1-Score, Recall, dan Precision, yang dipilih karena mampu menilai keseimbangan antara ketepatan dan keberhasilan model dalam mendeteksi kasus positif. Model dengan F1-Score tertinggi dipilih sebagai model utama dalam sistem prediksi.
  
## Data Understanding

Dataset yang digunakan diambil dari situs Kaggle dengan nama [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). Dataset ini berasal dari National Institute of Diabetes and Digestive and Kidney Diseases. Dataset ini terdiri dari 768 data. Tujuan utama dari dataset ini adalah untuk memprediksi secara diagnostik apakah seorang pasien menderita diabetes atau tidak, berdasarkan beberapa pengukuran diagnostik yang terdapat dalam dataset. Secara khusus, seluruh pasien dalam dataset ini adalah wanita berusia minimal 21 tahun yang berasal dari suku Indian Pima.Datase terdiri dari :
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
* Mengapus Beberapa outlier yang didapatkan dari pemeriksaan outliers dengan metode LOF. Proses ini bertujuan untuk meningkatkan kualitas data dengan menghilangkan nilai-nilai yang dapat mengganggu hasil analisis. Dengan menghapus outlier, kita memastikan bahwa model yang dibangun tidak terpengaruh oleh nilai-nilai ekstrem yang tidak relevan dan bisa mengarah pada kesalahan dalam prediksi atau analisis.
* Mengelompokan data numerik ke babebarap kategori. Proses ini melibatkan pengelompokan nilai numerik pada variabel Glucose, BMI, dan Insulin ke dalam beberapa kategori. Langkah ini dilakukan karena pada beberapa algoritma machine learning, khususnya yang berbasis klasifikasi seperti decision tree atau model berbasis aturan, mengonversi data numerik menjadi kategori dapat meningkatkan stabilitas model dan membantu dalam menghasilkan prediksi yang lebih akurat.
* Menerapkan teknik encoding pada data kategori yang dibuat sebelumnya. Penerapan teknik encoding pada data kategorikal bertujuan mengubah nilai kategori atau label menjadi format numerik agar dapat diproses oleh algoritma machine learning, yang umumnya hanya mengenali data numerik. Encoding membantu model memahami hubungan antar fitur. Pada tahap ini, teknik yang digunakan adalah One-Hot Encoding dan diterapkan pada data kategorikal yang telah dibuat sebelumnya.
* Memisahkan data-data kategorik dan label dari dataset. Pemisahan fitur kategorik dan label bertujuan untuk memastikan bahwa hanya fitur numerik yang dikenakan standarisasi. Dengan memisahkan keduanya, kita dapat menghindari penerapan standarisasi pada data kategorik atau label yang tidak memerlukan perubahan. Hal ini membantu proses model menjadi lebih efisien dan akurat, karena standarisasi hanya diterapkan pada data numerik tanpa mengubah informasi penting pada data kategorik dan label.
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

Kita akan menggunakan lima algoritma terbaik berdasarkan hasil dari LazyPredict. Berdasarkan tabel, LazyPredict, algoritma XGBoost, Random Forest, AdaBoost, Bagging, dan LightGBM menunjukkan performa terbaik. Oleh karena itu, kita akan membangun kelima model ini dan berfokus untuk mengoptimalkan performanya lebih lanjut melalui teknik hyperparameter tuning pada setiap model yang akan diuji.

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
#### Parameter yang Digunakan untuk Hyperparameter Tuning Model
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
#### Parameter yang Digunakan untuk Hyperparameter Tuning Model
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
#### Parameter yang Digunakan untuk Hyperparameter Tuning Model
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
#### Parameter yang Digunakan untuk Hyperparameter Tuning Model
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
#### Parameter yang Digunakan untuk Hyperparameter Tuning Model
* n_estimator = Menentukan jumlah total pohon (trees) yang akan dibangun dalam model. Diantara (50, 100, 200) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 50.
* learning_rate = Menentukan laju pembelajaran atau seberapa besar kontribusi setiap pohon terhadap prediksi akhir. Diantara (0.01, 0.1, 0.2) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 0,2.
* num_leaves = Menentukan jumlah maksimum daun (leaf nodes) dalam setiap pohon. Diantara (31, 50, 100) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 50.
* max_depth = Menentukan kedalaman maksimum pohon. Diantara (-1, 10, 20) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah -1.
* min_child_samples = Menentukan jumlah minimum sampel yang harus ada pada setiap daun pohon. Diantara (20, 50, 100) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 20
* subsample = Menentukan proporsi sampel yang digunakan untuk membangun setiap pohon. Diantara (0.6, 0.8, 1.0) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 0,6.
* colsample_bytree = Menentukan proporsi fitur yang digunakan untuk setiap pohon. Diantara (0.6, 0.8, 1.0) dengan menggunakan GridSearch didapat nilai yang terbaik untuk parameter adalah 1,0.
* boosting_type = Menentukan jenis boosting yang digunakan dalam model. Diantara ('gbdt', 'dart') dengan menggunakan GridSearch didapat jenis yang terbaik untuk parameter adalah gbdt.

## Evaluation
Pada tahap evaluasi, metrik evaluasi yang digunakan adalah F1-Score, Recall, dan Precision. Penggunaan metrik f1-score, recall, dan precision sangat penting untuk mengevaluasi kinerja model secara menyeluruh karena tujuan utama dari model prediksi diabetes adalah untuk memastikan diagnosis yang akurat, yaitu untuk mengidentifikasi pasien yang benar-benar menderita diabetes tanpa terlalu banyak kesalahan prediksi. 

Recall adalah metrik evaluasi yang mengukur seberapa banyak prediksi positif yang benar dibandingkan dengan jumlah data yang sebenarnya positif. Jadi, recall menunjukkan seberapa baik model dalam menangkap semua contoh positif yang ada. Recall sangat penting dalam konteks ini karena kita ingin meminimalkan jumlah false negatives, yaitu pasien yang benar-benar menderita diabetes tetapi tidak terdeteksi oleh model. Dalam dunia medis, false negatives bisa sangat berisiko karena dapat mengarah pada penundaan diagnosis dan penanganan yang terlambat, yang berpotensi menyebabkan komplikasi serius. Nilai recall yang tinggi berarti model jarang melewatkan contoh positif (False Negative rendah). Hal ini menunjukkan model berhasil mendeteksi banyak pasien yang menderita diabetes (yaitu, pasien yang benar-benar positif). Nilai recall yang tinggi sangat penting dalam konteks medis, karena kesalahan dalam bentuk false negative dapat menyebabkan pasien yang sakit tidak terdeteksi dan tidak mendapatkan perawatan yang dibutuhkan. Berikut rumus perhitungan recall:

![Recall](https://raw.githubusercontent.com/XMB234/Tugas-Machine-Learning-Terapan-Predictive-Analytic-/c6cd3f3afcf2a6a13fa5e05c1d0da81d964165cf/Recall.jpg)

Ket :
* TP = True Positives (prediksi benar positif)
* FN =  False Negatives (prediksi salah negatif)
  
Sementara itu, precision adalah metrik evaluasi yang mengukur seberapa banyak prediksi positif yang benar dari seluruh prediksi positif yang dilakukan oleh model. Dengan kata lain, presisi menunjukkan ketepatan model dalam mengklasifikasikan objek yang sebenarnya positif. Precision digunakan untuk memastikan bahwa ketika model memprediksi seseorang sebagai penderita diabetes, prediksi tersebut benar-benar akurat. Precision membantu menghindari false positives, yaitu pasien sehat yang salah didiagnosis menderita diabetes.  Nilai precision yang tinggi berarti model jarang mengklasifikasikan contoh negatif sebagai positif (false positive rendah). Hal ini mendakan bahwa ketika model memprediksi seseorang sebagai penderita diabetes, prediksi tersebut benar-benar akurat. Precision penting untuk menghindari pemberian diagnosis yang salah pada pasien yang sebenarnya sehat, yang bisa memicu kecemasan, pengobatan yang tidak perlu, atau intervensi medis yang tidak relevan. Berikut rumus perhitungan precision:

![Presisi](https://raw.githubusercontent.com/XMB234/Tugas-Machine-Learning-Terapan-Predictive-Analytic-/462cefbb692f9a01d79e92cad7fe30968de2a9e8/Presisi1.jpg) 

Ket: 
* TP = True Positives (prediksi benar positif)
* FP = False Positives (prediksi salah positif)

F-Score adalah rata-rata harmonis antara presisi dan recall, yang memberikan keseimbangan antara keduanya. F1-Score sangat berguna ketika kita membutuhkan keseimbangan antara presisi dan recall dan tidak bisa memilih satu di antara keduanya. Ini sangat berguna karena menggabungkan dua metrik tersebut dalam satu angka yang menggambarkan keseimbangan antara kemampuan model untuk menangkap kasus positif (Recall) dan keakuratan dalam memprediksi kasus tersebut (Precision). Nilai F1-Score yang tinggi menunjukkan bahwa model dapat menangkap banyak kasus positif (recall tinggi) dan juga menghindari prediksi yang salah (precision tinggi). F1-Score sangat bermanfaat dalam kasus seperti ini karena memberikan penekanan pada model yang tidak hanya menangkap banyak kasus diabetes, tetapi juga meminimalkan kesalahan prediksi. Berikut rumus perhitungan f1-score:

![F-1 Score](https://raw.githubusercontent.com/XMB234/Tugas-Machine-Learning-Terapan-Predictive-Analytic-/refs/heads/main/F-1%20Score.jpg) 

Berikut tabel nilai-nilai recall, precision, dan f-1 score untuk masing masing model dalam memprediks penyakit diabetes

| Model              | F1-Score | Recall  | Precision |
|--------------------|----------|---------|-----------|
| XGB                | 0.8350   | 0.8431  | 0.8269    |
| Random Forest      | 0.8317   | 0.8235  | 0.8400    |
| Bagging Classifier | 0.8350   | 0.8431  | 0.8269    |
| LightGBM           | 0.8350   | 0.8431  | 0.8269    |
| AdaBoost           | 0.8515   | 0.8431  | 0.8600    |

Berdasarkan hasil evaluasi model yang ditunjukkan pada tabel, dapat dilihat bahwa model AdaBoost memberikan performa terbaik dibandingkan model-model lainnya dalam hal F1-Score, Recall, dan Precision. AdaBoost memperoleh nilai F1-Score sebesar 0.8515, dengan Recall sebesar 0.8431 dan Precision sebesar 0.8600. Nilai ini menunjukkan bahwa model tidak hanya mampu mendeteksi sebagian besar kasus diabetes secara benar (recall tinggi), tetapi juga membuat prediksi positif yang akurat (precision tinggi), sehingga menghasilkan keseimbangan kinerja yang sangat baik secara keseluruhan (F1-score tinggi).

Model lain seperti XGB, Bagging Classifier, dan LightGBM menunjukkan performa yang identik, masing-masing dengan F1-Score sebesar 0.8350, Recall 0.8431, dan Precision 0.8269. Meskipun nilai recall mereka cukup tinggi, nilai precision sedikit lebih rendah dibandingkan dengan AdaBoost, yang berarti prediksi positifnya sedikit kurang tepat. Random Forest juga menunjukkan performa yang kompetitif dengan F1-Score sebesar 0.8317, recall sedikit lebih rendah yaitu 0.8235, namun precision tertinggi kedua yaitu 0.8400.

Secara keseluruhan, model AdaBoost menempati posisi teratas dalam hal keseimbangan antara kemampuan mendeteksi kasus positif dan ketepatan prediksi, menjadikannya kandidat terbaik untuk digunakan dalam sistem prediksi penyakit diabetes berdasarkan metrik evaluasi yang digunakan. Hal ini secara langsung menjawab problem statement dalam proyek, yaitu bagaimana memanfaatkan machine learning untuk menghasilkan sistem prediksi penyakit diabetes yang akurat dan minim kesalahan. Dengan nilai recall yang tinggi, model mampu mendeteksi sebagian besar pasien yang benar-benar menderita diabetes, sehingga membantu mencegah keterlambatan diagnosis. Precision yang tinggi menunjukkan bahwa prediksi positif yang dihasilkan oleh model juga dapat diandalkan, mengurangi potensi kesalahan diagnosis yang dapat menimbulkan kecemasan atau intervensi medis yang tidak perlu.

Dari segi pencapaian goals proyek, proses eksplorasi awal yang dilakukan menggunakan berbagai algoritma machine learning telah berhasil mengidentifikasi model-model potensial. Model-model tersebut kemudian dibangun ulang, dioptimalkan, dan dievaluasi menggunakan metrik yang tepat untuk klasifikasi, yaitu Recall, Precision, dan F1-Score. Metrik ini dipilih karena mampu menggambarkan keseimbangan antara kemampuan model dalam menangkap kasus positif dan ketepatan prediksi yang dihasilkan. Dua aspek yang sangat penting dalam konteks diagnosis penyakit. Oleh karena itu, model terbaik yang teridentifikasi tidak hanya memberikan akurasi teknis yang tinggi, tetapi juga memiliki relevansi kuat terhadap kebutuhan medis dan klinis.

Dari perspektif konteks bisnis dan kesehatan, penerapan model dengan performa tinggi seperti AdaBoost berpotensi memberikan dampak signifikan, terutama dalam mendukung deteksi dini diabetes. Sistem prediktif yang akurat dapat membantu fasilitas layanan kesehatan untuk mengklasifikasikan pasien berisiko lebih cepat, mengoptimalkan alokasi sumber daya medis, dan merancang strategi pencegahan serta edukasi yang lebih tepat sasaran. Selain itu, model ini juga bisa diintegrasikan ke dalam sistem penunjang keputusan medis, memberikan rekomendasi kepada tenaga medis berdasarkan analisis data secara otomatis. Hal ini tentu berkontribusi pada peningkatan efisiensi operasional, pengurangan biaya pemeriksaan, dan peningkatan kualitas hidup pasien melalui penanganan yang lebih dini dan tepat.

Berdasarkan hasil evaluasi dan analisis performa model, AdaBoost dipilih sebagai model utama dalam proyek ini untuk memprediksi penyakit diabetes berdasarkan fitur Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, DiabetesPedigreeFunction, dan Age.

## Kesimpulan 
Dengan menerapkan berbagai metode dan evaluasi terhadap enam model machine learning, AdaBoost terbukti menjadi model dengan performa terbaik dalam memprediksi penyakit diabetes secara akurat. Model ini menunjukkan keseimbangan yang unggul antara kemampuan mendeteksi kasus positif dan ketepatan prediksi, menjadikannya pilihan yang paling andal. Berdasarkan hasil tersebut, tujuan utama proyek berhasil dicapai, yaitu mengembangkan sistem prediksi penyakit diabetes yang  dapat memprediksi penyakit diabetes berdasarkan factor factor penyakit diabetes.

## Referensi
1. Ahmadi, T., Wulandari, A., & Suhatman, H. (2019). Sistem Customer Churn Prediction Menggunakan Machine Learning pada Perusahaan ISP. Jetri: Jurnal Ilmiah Teknik Elektro, 17.
2. Apriliah, W., Kurniawan, I., Baydhowi, M., & Haryati, T. (2021). Prediksi Kemungkinan Diabetes pada Tahap Awal Menggunakan Algoritma Klasifikasi Random Forest. SISTEMASI, 10(1), 163. https://doi.org/10.32520/stmsi.v10i1.1129
3. Fernandes, W., Komati, K. S., & Assis de Souza Gazolli, K. (2024). Anomaly detection in oil-producing wells: a comparative study of one-class classifiers in a multivariate time series dataset. Journal of Petroleum Exploration and Production Technology, 14(1), 343–363. https://doi.org/10.1007/s13202-023-01710-6
4. Öngelen, G., & İnkaya, T. (2023). A novel LOF-based ensemble regression tree methodology. Neural Computing and Applications, 35(26), 19453–19463. https://doi.org/10.1007/s00521-023-08773-w
5. Septiana Rizky, P., Haiban Hirzi, R., & Hidayaturrohman, U. (2022). Perbandingan Metode LightGBM dan XGBoost dalam Menangani Data dengan Kelas Tidak Seimbang. J Statistika: Jurnal Ilmiah Teori Dan Aplikasi Statistika, 15(2), 228–236. https://doi.org/10.36456/jstat.vol15.no2.a5548
6. Silalahi, A. P., & Simanullang, H. G. (2023). SUPERVISED LEARNING METODE K-NEAREST NEIGHBOR UNTUK PREDIKSI DIABETES PADA WANITA. METHOMIKA Jurnal Manajemen Informatika Dan Komputerisasi Akuntansi, 7(1), 144–149. https://doi.org/10.46880/jmika.Vol7No1.pp144-149


 

