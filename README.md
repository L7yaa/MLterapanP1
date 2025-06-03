Laporan Proyek Machine Learning – 

Luissandro Hermawan

Domain Proyek

Industri wine merupakan salah satu sektor dengan pertumbuhan yang stabil dan kompetitif di pasar global. Kualitas wine dipengaruhi oleh berbagai faktor kimia yang kompleks, yang biasanya dinilai oleh para ahli wine (sommelier) secara subjektif. Namun, dengan kemajuan teknologi dan data science, kini kualitas wine juga dapat diprediksi secara objektif menggunakan data kuantitatif dari komposisi kimianya.

Dataset Wine Quality ini merupakan kumpulan data yang berisi hasil pengukuran berbagai karakteristik kimia dari sampel wine merah. Variabel-variabel yang diukur antara lain adalah kadar asam tetap (fixed acidity), kadar asam volatil (volatile acidity), kandungan asam sitrat (citric acid), kadar gula sisa (residual sugar), kadar klorida (chlorides), jumlah sulfur dioksida bebas (free sulfur dioxide), jumlah sulfur dioksida total (total sulfur dioxide), kepadatan (density), pH, kadar sulfat (sulphates), dan kadar alkohol (alcohol). Kualitas wine kemudian dinilai dengan menggunakan skor numerik (quality), yang memberikan gambaran obyektif mengenai kualitas dari setiap sampel wine.

Data ini penting untuk:
- Menganalisis faktor-faktor utama yang mempengaruhi kualitas wine, seperti komposisi kimia yang ada dalam sampel.
- Membangun model prediktif untuk memperkirakan skor kualitas wine berdasarkan data kimia yang ada, sehingga memungkinkan prediksi yang lebih objektif dan akurat.
- Membantu produsen wine dalam proses kontrol kualitas dan optimasi produksi dengan memanfaatkan analisis data untuk meningkatkan hasil produksi yang konsisten dan berkualitas tinggi.

Masalah penilaian kualitas wine yang subjektif harus diselesaikan karena dapat menimbulkan ketidakpastian dan kurangnya konsistensi dalam produksi. Sravan et al. (2024) menjelaskan bahwa "Pendekatan berbasis machine learning memungkinkan analisis kualitas wine secara objektif dengan memanfaatkan data kimiawi seperti kadar alkohol, pH, dan komponen lainnya, untuk membangun model prediktif yang konsisten dalam memperkirakan skor kualitas wine."Penilaian yang bergantung pada indera manusia dapat berbeda-beda dari orang ke orang, yang berdampak pada kesulitan produsen untuk memastikan kualitas yang konsisten. Untuk menyelesaikan masalah ini, kita dapat mengidentifikasi elemen-elemen yang memengaruhi kualitas dengan memanfaatkan data kimiawi anggur, seperti yang ada dalam dataset Kualitas Anggur. Dengan membuat model prediktif menggunakan teknik statistik atau machine learning, kualitas anggur dapat diprediksi secara objektif, memberikan hasil yang lebih konsisten dan dapat diandalkan. Zaza, S., Atemkeng, M., & Hamlomo, S. (2023) menyatakan bahwa “Dengan menggunakan algoritma pembelajaran mesin, pentingnya fitur dalam prediksi kualitas wine dapat dianalisis untuk mengidentifikasi faktor kimia yang paling berpengaruh terhadap kualitas.” Faktor-faktor ini kemudian dapat digunakan untuk meningkatkan ketepatan model prediktif..Faktor-faktor ini kemudian dapat digunakan untuk meningkatkan ketepatan model prediktif."Ketika model ini digunakan dalam proses produksi, mereka dapat meningkatkan kontrol kualitas, efisiensi, dan kemampuan produsen untuk membuat keputusan yang lebih baik. Akibatnya, kualitas wine dapat dipastikan dengan lebih adil dan konsisten, dengan keuntungan bagi produsen dan konsumen.

Business Understanding

Dalam industri minuman keras, menjaga kualitas produk yang konsisten sangat sulit, terutama ketika anggur, bahan baku, berasal dari berbagai panen, dan proses produksi mengalami variasi. Produsen wine biasanya bergantung pada uji organoleptik manusia (seperti mencicipi), yang mahal dan memakan waktu serta subjektif dan sulit direplikasi.

Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Mengapa penting untuk menjaga kualitas wine yang konsisten meskipun bahan baku dan proses produksi berbeda?
- Mengatasi masalah tersebut secara sistematis dan berbasis data, apa yang ingin dicapai?
- Bagaimana bisnis dapat memanfaatkan solusi berbasis data?

Goals

Menjelaskan tujuan dari pernyataan masalah:
- Meskipun persepsi konsumen sangat memengaruhi kualitas wine, metode saat ini seperti uji organoleptik mahal, tidak efektif, dan subjektif. Menjaga konsistensi mutu menjadi lebih sulit karena berbagai bahan baku dan proses produksi. Diperlukan pendekatan yang lebih terstandarisasi dan objektif.
- Tujuannya adalah untuk secara otomatis memprediksi kualitas wine berdasarkan komposisi kimia. Ini memungkinkan identifikasi faktor kimia utama, standarisasi kualitas produk, dan pengambilan keputusan berbasis data untuk meningkatkan efisiensi dan mutu produksi.
- Solusinya adalah dengan membuat model pembelajaran mesin menggunakan data kimia dari buah anggur. Proses ini mencakup eksplorasi data, analisis fitur penting, pelatihan model prediktif, dan evaluasi kinerja model. Hasilnya akan membantu produsen mengurangi ketergantungan pada uji rasa manusia dan memastikan kualitas yang konsisten.

Solution statements

- Menggunakan fitur-fitur kimia seperti keasaman, kadar alkohol, gula residual, dan pH untuk memprediksi skor kualitas wine.

Data Understanding

Dataset WineQT merupakan kumpulan data kualitas wine merah (red wine) yang banyak digunakan untuk analisis prediksi kualitas wine berdasarkan komposisi kimianya. Dataset ini berasal dari riset yang dilakukan oleh Instituto Superior de Engenharia do Porto, Portugal.
- Jumlah observasi: 1.143 baris data
- Jumlah variabel (fitur): 13 kolom
- Jenis data: Numerik (semua fitur adalah bilangan kontinu)
- Link dataset: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

Pada tahap EDA saya melakukan penghapusan Fitur Id dikarenakan tidak berkontribusi langsung dalam model prediksi/hanya sebagai pengenal unik dan tidak memberikan informasi analitis yang relevan. Pada saat mengecek missing value tidak terdapat data yang hilang dan pada saat mengecek duplikasi terdapat 125 data yang terduplikasi dan berhasil dibersihkan. Variabel-variabel pada WineQT dataset adalah sebagai berikut:
Nama Variabel Penjelasan
fixed acidity Keasaman tetap, terutama dari asam tartarat, yang tidak menguap selama fermentasi. Mempengaruhi rasa dan kestabilan.
volatile acidity Keasaman yang mudah menguap (misalnya asam asetat). Terlalu tinggi akan memberikan aroma cuka.
citric acid Asam alami dari anggur yang menambah kesegaran dan struktur rasa wine.
residual sugar Gula yang tersisa setelah fermentasi. Wine manis memiliki nilai lebih tinggi.
chlorides Kandungan garam, terutama natrium klorida, yang memengaruhi rasa wine.
free sulfur dioxide SO₂ bebas yang bertindak sebagai antiseptik dan antioksidan untuk menjaga kesegaran.
total sulfur dioxide Total SO₂, baik yang bebas maupun terikat. Terlalu tinggi bisa mengganggu rasa dan aroma.
density Massa jenis wine. Biasanya meningkat dengan kadar gula atau alkohol.
pH Tingkat keasaman atau kebasaan wine. Skala 0–14, dengan nilai rendah menunjukkan tingkat keasaman tinggi.
sulphates Aditif yang berfungsi sebagai pengawet dan meningkatkan rasa pahit/kering.
alcohol Persentase kandungan alkohol dalam wine. Umumnya, semakin tinggi alkohol, semakin tinggi persepsi kualitas.
quality Skor kualitas wine (label target), dinilai secara sensorik oleh panel (skala 0–10). Biasanya berkisar antara 3–8.


Data Preparation
- Penghapusan Kolom "Id"
dikarenakan tidak berkontribusi langsung dalam model prediksi/hanya sebagai pengenal unik dan tidak memberikan informasi analitis yang relevan

- Mengecek dan Menghapus Missing Value
Pada Dataset "WineQT" tidak ditemukan missing value pada setiap kolomnya.

- Mengecek Duplikat data
Melakukan pengecekkan duplikat data dan mendapatkan 125 data yang duplikasi, Hal ini langsung di bersihkan dengan menghapus duplikat tersebut.

- Penanganan outliers 
dalam dataset ini dilakukan dengan menggunakan metode Interquartile Range (IQR).Pertama, kuartil pertama (Q1) dan ketiga (Q3) dihitung untuk setiap kolom numerik. Selanjutnya, IQR, yang merupakan jarak antara Q3 dan Q1), digunakan untuk menemukan outliers.  Data di luar rentang [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR] dianggap sebagai outliers dan dihapus. Ini mengurangi jumlah data dari 1143 menjadi 834 baris.  Visualisasi boxplot menunjukkan perubahan distribusi data sebelum dan setelah penghapusan outliers. Setelah outliers dihapus, distribusi data menjadi lebih teratur.  Langkah ini meningkatkan akurasi data, membuatnya lebih representatif, dan membuatnya siap untuk analisis lebih lanjut, menghasilkan model prediksi yang lebih akurat dan stabil.

- Menentukan Fitur Numerik dan Target
menentukan fitur numerik dataset.  Baris kode pertama dapat memilih kolom dengan tipe data float (float64) atau integer (int64) yang digunakan sebagai fitur numerik.Jumlah kolom disimpan dalam variabel fitur numerik, dan nama kolom target—dalam hal ini kualitas—disimpan dalam variabel fitur target. HIstogram untuk masing-masing fitur numerik, seperti stabilitas asam, volatilitas asam, dan asam citrik, digunakan. Di sini, metode Kernel Density Estimation (KDE) digunakan untuk menggambarkan distribusi data secara rinci. Menunjukkan bagaimana sebaran data untuk setiap fitur numerik, apakah distribusinya simetris, miring ke kiri (positif skew), atau miring ke kanan (negatif skew). Dalam boxplot, median ditunjukkan oleh setiap garis di tengah kotak, dan kuartil ketiga (Q3) dan kuartil pertama (Q1) ditunjukkan di bagian atas dan bawah kotak. Ini membantu melihat rentang antar kuartil (IQR) dan menemukan outlier dalam fitur numerik.  Garis vertikal, atau whiskers, menunjukkan rentang data yang tidak dianggap sebagai outlier. Sebuah titik di luar whiskers dianggap sebagai outlier. membantu dalam menentukan apakah terdapat data yang sangat tinggi atau rendah yang dianggap sebagai outlier. Misalnya, kita dapat melihat beberapa titik di luar whiskers dalam fitur residu gula, yang menunjukkan adanya outlier dalam data. Visualisasi Distribusi Skor Kualitas: Pada tahap ini, kita menggunakan histogram atau countplot untuk menunjukkan distribusi skor kualitas wine. Grafik ini membantu kita memahami sebaran data pada variabel target, kualitas, yang memiliki rentang nilai 0–10. Menampilkan Jumlah Sampel untuk Setiap Kategori Skor Kualitas yang Ada pada Variabel Kualitas: Untuk memastikan distribusi data yang lebih rinci, kita dapat menampilkan jumlah sampel untuk setiap kategori skor kualitas yang ada pada variabel kualitas.   Hasil menunjukkan bahwa sampel dalam kategori kualitas 5 memiliki jumlah sampel terbesar (483 sampel), diikuti oleh sampel dalam kategori kualitas 6 (462 sampel), dan sampel dalam kategori 3 dan 8 hanya memiliki 6 dan 16 sampel, masing-masing. Dengan demikian, sebagian besar sampel dalam dataset ini memiliki kualitas wine yang relatif sedang hingga baik, dengan skor kualitas 5 dan 6.

- Heatmap Korelasi Antar Fitur Numerik
Kita melihat bagaimana fitur numerik dalam dataset berhubungan satu sama lain. Sebuah korelasi menunjukkan seberapa kuat hubungan linier antara dua variabel. Korelasi yang kuat menunjukkan bahwa kedua variabel memiliki hubungan yang saling mempengaruhi, sementara korelasi yang rendah menunjukkan bahwa tidak ada hubungan yang jelas antara kedua variabel.  Menunjukkan bagaimana fitur numerik dataset berhubungan satu sama lain.  Kekuatan hubungan antara dua fitur ditunjukkan oleh angka-angka dalam heatmap, yang memiliki nilai mulai dari nol hingga satu. Contohnya: Fixed acidity dan density memiliki korelasi positif yang cukup kuat (0.68), yang berarti peningkatan asam tetap dalam wine cenderung diikuti dengan peningkatan kepadatan. Fitur-fitur seperti kadar alkohol dan ketebalan menunjukkan kecenderungan yang lebih konsisten seiring peningkatan kualitas wine, sedangkan fitur seperti residu gula dan sulfur dioksida menunjukkan variasi yang lebih besar, yang menunjukkan adanya perbedaan di beberapa kategori kualitas wine.

- Visualisasi Pairplot Beberapa Fitur vs Quality
Visualisasi Hubungan Antar Fitur dan Kualitas: Pada langkah ini, pairplot digunakan untuk melihat hubungan antara beberapa fitur kimiawi dan kualitas anggur. Pairplot memberikan gambaran visual tentang distribusi data dan hubungan antar variabel, serta menunjukkan apakah ada korelasi yang kuat antara fitur tertentu dan kualitas anggur.  Scatter plot antara dua fitur serta distribusi masing-masing fitur pada diagonal ditampilkan di setiap panel. Misalnya, dalam grafik antara densitas dan kualitas, data kualitas wine dengan skor tinggi lebih sering ditemukan di sisi kiri, yang memiliki densitas yang lebih tinggi, sementara data kualitas wine dengan skor rendah lebih sering ditemukan di sisi kanan.

- Boxplot Setiap Fitur terhadap Target 'Quality'
Visualisasi Hubungan Antara Fitur dan Kualitas: Pada langkah ini, kita menggunakan boxplot untuk menunjukkan hubungan antara setiap fitur numerik dan variabel kualitas target. Dengan menggunakan boxplot untuk setiap fitur dibandingkan dengan variabel kualitas target, kita dapat melihat distribusi dan variabilitas setiap fitur dalam kaitannya dengan kualitas wine, dan ini membantu kita menentukan apakah beberapa fitur memainkan peran yang lebih kuat dalam menentukan kualitas wine.

- Encoding fitur kategori
Tidak ada data kategori yang dimasukkan dalam dataset. Kolom yang memiliki tipe data kategori atau objek tidak ada, yang berarti tidak ada fitur yang dapat diproses untuk encoding lebih lanjut. Kode di atas digunakan untuk melakukan encoding fitur kategori dengan cara mengganti setiap nilai kategori pada kolom tertentu dengan rata-rata nilai dari kolom quality yang terhubung dengan kategori tersebut. Banyak algoritma pembelajaran mesin, terutama yang berbasis matematis seperti regresi atau SVM, tidak dapat menangani data kategorikal langsung seperti teks atau kategori seperti warna, jenis, dll. Oleh karena itu, agar model dapat memproses fitur kategori, mereka harus diubah menjadi format numerik. Penggunaan metode encoding seperti rata-rata atau metode lainnya memungkinkan transformasi data kategori menjadi angka yang menunjukkan informasi yang ada dalam kategori tersebut.Setelah memilih kolom kategori, langkah selanjutnya adalah mengencoding fitur kategori tersebut. Ini mengubah kolom kategori menjadi representasi numerik yang lebih mudah diproses oleh model prediksi. Ini dilakukan dengan menghitung kualitas rata-rata untuk setiap kategori dalam fitur, kemudian memetakan nilai kualitas rata-rata ini kembali ke kolom kategori yang sesuai.Langkah encoding ini membantu mengubah fitur kategorikal (misalnya warna atau kategori lainnya) menjadi bentuk numerik yang dapat digunakan oleh model machine learning. Teknik ini sangat berguna ketika fitur kategorikal memiliki banyak kategori unik, sehingga encoding rata-rata quality memberikan representasi numerik yang lebih efisien daripada menggunakan teknik one-hot encoding.

- Reduksi Dimensi dengan PCA
Menghapus Kolom Tujuan dan Kategori Langkah pertama adalah menghapus kolom kategori dan kolom target dari dataset. Ini menghapus kolom kategori yang berupa string data atau kategori, sehingga dataset hanya berisi fitur numerik yang relevan. Karena kolom kualitas adalah target yang ingin diprediksi, kolom kualitas juga dihapus. Dengan demikian, PCA hanya diterapkan pada fitur numerik.melakukan standardisasi pada fitur numerik menggunakan StandardScaler. Standardisasi ini penting untuk memastikan bahwa setiap fitur memiliki skala yang sama (rata-rata 0 dan deviasi standar 1). Tanpa standardisasi, fitur dengan rentang nilai yang lebih besar akan mendominasi dalam proses analisis, yang dapat memengaruhi hasil PCA.Menggunakan Principal Component Analysis (PCA) untuk mengurangi dimensi data menjadi dua komponen utama (n_components=2). PCA akan mengubah data fitur numerik yang telah distandardisasi menjadi dua komponen utama (PC1 dan PC2), yang menyimpan sebanyak mungkin informasi dari data awal.Untuk memudahkan analisis lebih lanjut berdasarkan kualitas wine, kami menambahkan dua komponen utama (PC1 dan PC2) ke dalam DataFrame setelah memperoleh hasil PCA. Kami juga menambahkan kolom kualitas.Proses reduksi dimensi dengan PCA (Principal Component Analysis) yang terlihat di gambar bertujuan untuk menyederhanakan dataset dengan mengurangi jumlah fitur, namun tetap mempertahankan informasi penting yang ada. PCA sangat berguna untuk mengatasi masalah overfitting, kelebihan dimensi (curse of dimensionality), serta kesulitan dalam visualisasi yang sering terjadi pada data dengan banyak fitur. Dengan menggabungkan variabel yang berkorelasi menjadi komponen utama, PCA mengurangi jumlah fitur tetapi tetap mempertahankan variansi yang signifikan dalam data. Proses ini meningkatkan efisiensi model, mengurangi kompleksitas, dan memudahkan pemahaman serta visualisasi data.Jika PCA hanya digunakan untuk eksplorasi atau visualisasi data, maka tujuannya adalah untuk memberikan gambaran umum yang lebih jelas tanpa mengubah model secara signifikan. Dalam hal ini, fitur yang digunakan dalam model harus dijelaskan dengan tepat untuk memastikan bahwa interpretasi dan hasil analisis tetap akurat. Dengan mempertahankan variabilitas utama dalam kumpulan data, analisis komponen utama (PCA) berhasil mengurangi dimensi data. Ini menghasilkan dua komponen utama yang dapat menggambarkan kualitas buah-buahan berdasarkan distribusinya dalam ruang dimensi baru.Visualisasi scatter plot menunjukkan bagaimana kualitas wine tersebar setelah menurunkan dimensi, dengan titik yang lebih terang menunjukkan kualitas yang lebih tinggi. Pairplot menunjukkan hubungan yang jelas antara dua komponen utama kualitas wine, dengan kualitas tinggi terkonsentrasi di wilayah tertentu. Ini menunjukkan bahwa PCA dapat digunakan sebagai sarana untuk klasifikasi kualitas wine. Dengan membedakan kualitas wine berdasarkan distribusi dalam ruang dimensi yang lebih renda, proses ini menunjukkan bahwa PCA dapat digunakan untuk mempermudah analisis eksplorasi data.

-  Train-Test-Split
Proses Train-Test-Split digunakan untuk membagi dataset menjadi dua bagian: data pelatihan (train) dan data pengujian (test). ami menggunakan 80% data untuk pelatihan dan 20% untuk pengujian.Pembagian dataset menjadi dua bagian (train dan test) adalah langkah penting untuk mengevaluasi kemampuan model dalam menggeneralisasi data baru yang belum pernah dilihat sebelumnya. Dengan menggunakan data pelatihan untuk melatih model dan data pengujian untuk mengujinya, kita dapat memeriksa apakah model overfit pada data pelatihan atau tidak. Pembagian yang tepat memastikan bahwa model tidak hanya "menghafal" data pelatihan, melainkan dapat bekerja baik pada data yang belum pernah dilihat.

- Standarisasi Data
menstandarisasi data pelatihan dan pengujian, terutama untuk algoritma yang sensitif terhadap skala fitur. Dalam hal ini, digunakan StandardScaler untuk melakukan standardisasi, yang mengubah data sehingga setiap fitur memiliki mean = 0 dan standar deviasi = 1.Standardisasi memastikan bahwa semua fitur memiliki skala yang setara, yang sangat penting untuk model berbasis jarak seperti SVM dan KNN. Data yang digunakan untuk pemodelan telah melalui langkah-langkah persiapan yang lengkap, termasuk pemisahan fitur dan target, pembagian data menjadi train-test, standardisasi, dan reduksi dimensi dengan PCA. Semua langkah ini bertujuan untuk memastikan bahwa model yang dibangun akan bekerja secara optimal pada dataset yang bersih dan terstandarisasi.

Modeling

1. Naive Bayes (GaussianNB) : Naive Bayes adalah algoritma klasifikasi yang menggunakan Teorema Bayes dengan asumsi bahwa setiap fitur bersifat independen satu sama lain (asumsi "naive"). Untuk data kontinu, model ini mengasumsikan bahwa fitur-fitur mengikuti distribusi Gaussian (normal), sehingga digunakan varian khusus yang disebut Gaussian Naive Bayes (GaussianNB). Model ini menghitung kemungkinan suatu data termasuk ke dalam kelas tertentu berdasarkan peluang masing-masing fitur sesuai distribusi Gaussian.

Parameter (semua default)
priors=None: Model menghitung probabilitas kelas dari data pelatihan secara otomatis.
var_smoothing=1e-9: Untuk menghindari pembagian nol saat menghitung probabilitas, nilai kecil ditambahkan ke varians

Analisis:
 Precision: 58.51%
 Precision dan Recall menunjukkan hasil rendah untuk beberapa kelas, terutama untuk kelas kualitas 3 dan 8, yang memiliki jumlah sampel yang sedikit.

Keuntungan:
- Simple dan cepat.

- Sangat cocok untuk data dengan distribusi probabilitas sederhana.

Kerusakan:

- tidak efektif dalam kasus di mana data memiliki hubungan yang kompleks antar fitur.

- data dengan distribusi yang tidak normal

2. Random Forest : Random Forest adalah algoritma pembelajaran ensemble yang membangun banyak pohon keputusan (decision trees) secara acak menggunakan subset data dan fitur yang berbeda. Setiap pohon keputusan menghasilkan prediksi, lalu hasil-hasil tersebut digabungkan melalui voting mayoritas untuk menentukan kelas akhir.

 Parameter tambahan yang menjadi default:
 Jumlah pohon yang dibangun dihitung dengan n_estimators=100.
 Kriteria "gini" digunakan untuk mengevaluasi kualitas split.
 Pohon tumbuh hingga semua daun bersih.
 Minimum sampel yang dibutuhkan untuk membagi node internal adalah 2.
 Minimum sampel per daun adalah 1.

Analisis:
-  Precision: 71.61%
-  Precision dan recall model ini cukup baik (kualitas 5 dan 

Keuntungan:
- Model sangat kuat, terutama untuk dataset yang tidak linier dan banyak fitur.
- Untuk menghindari overfitting, gunakan banyak pohon keputusan.

Kerusakan:
- Dibandingkan dengan model yang lebih sederhana, membutuhkan waktu komputasi yang lebih lama.
- Karena banyaknya pohon keputusan, tidak transparan dalam hal interpretasi model.

3. Decision Tree : Decision Tree adalah model klasifikasi yang bekerja dengan membagi data ke dalam kelompok-kelompok berdasarkan fitur yang paling berpengaruh terhadap target. Proses ini dilakukan secara rekursif hingga mencapai kondisi di mana data dalam kelompok tersebut cukup homogen (purity tinggi), dan dapat diputuskan label akhirnya.
Random_State=42: Untuk memastikan bahwa hasilnya sama.

Parameter tambahan yang menjadi default:
Kriteria gini digunakan untuk mengukur kualitas split.
Maksimal kedalaman pohon tidak ada.
Minimum ukuran node diatur oleh min_samples_split=2 dan min_samples_leaf=1.

Analisis:
- Precision: 62.88%
- Pada kelas-kelas yang lebih umum, seperti kualitas 5 dan 6, model ini menunjukkan hasil yang cukup baik; namun, pada kelas dengan sampel yang lebih sedikit, seperti kualitas 3 dan 8, keakuratan dan recallnya lebih rendah.

Keuntungan:
- Mudah dipahami dan dipahami.
- Ada kemampuan untuk menangani data yang tidak linier.

Kerusakan:
- Sangat rentan terhadap overfitting, terutama untuk data dengan banyak fitur.
- Pada dataset yang kompleks, model ini sering membuat keputusan yang terlalu sederhana.

4. Logistic Regression : Logistic Regression digunakan untuk memodelkan probabilitas suatu kejadian (dalam hal ini, kualitas wine) dengan menggunakan fungsi logistik (sigmoid). Model ini termasuk dalam keluarga regresi linier, tetapi cocok untuk klasifikasi karena output-nya berupa nilai probabilitas antara 0 dan 1.

Max_iter=1000: Jumlah maksimum iterasi yang dapat dilakukan oleh penyelesai. Ini seharusnya 100, tetapi ditingkatkan untuk menjamin konvergensi.
 Random_state=42: Untuk memastikan bahwa hasilnya konsisten.
 Parameter tambahan yang menjadi default:
 Regularisasi Ridge digunakan untuk menghindari overfitting.
 Solver='lbfgs': solusi default untuk dataset ukuran sederhana hingga menengah.

Analisis:
- Precision: 65.06%
- Precision dan recall model ini kurang, terutama untuk kelas kecil.

Keuntungan:
-  Simple dan cepat.
-  mudah dipahami dan menawarkan penilaian kemungkinan.

Kerusakan:
- tidak efektif untuk masalah yang melibatkan hubungan antar fitur yang kompleks.
-  Dalam kasus data yang sangat kompleks, mungkin underfitting.

Random Forest memberikan hasil yang paling akurat dengan 71.61%, diikuti oleh Decision Tree dengan 62.88% dan Logistic Regression dengan 65.06.%.Naive Bayes memiliki akurasi yang lebih rendah (58.51%) dan kesulitan memprediksi kelas dengan sampel kecil. Meskipun Random Forest memberikan hasil terbaik, model seperti Decision Tree dan Logistic Regression memiliki hasil yang layak untuk penelitian lebih lanjut.

Evaluation

• Akurasi
evaluasi Accuracy, Precision, Recall, dan F1-Score (weighted average) dari 4 algoritma:
- Random Forest menghasilkan performa terbaik pada seluruh metrik:
Akurasi tertinggi (0.7162)
F1-Score tertinggi (0.6939)
Cocok untuk data yang memiliki banyak fitur atau relasi kompleks antar fitur.

- Logistic Regression berada di urutan kedua, cukup stabil:
Akurasi dan F1-Score cukup baik (masing-masing > 0.64)
Cocok untuk data yang memiliki hubungan linear antara fitur dan label.

- Decision Tree memiliki performa moderat:
Mudah ditafsirkan (interpretable), tapi cenderung overfitting jika tidak di-prune atau diatur parameter lebih lanjut.

- Naive Bayes menunjukkan performa paling rendah:
Akurasi dan F1-Score rendah karena asumsi independensi antar fitur tidak terpenuhi di data Anda.


Menurut semua metrik evaluasi, Random Forest adalah model terbaik untuk menangani data kompleks karena:
 kombinasi berbagai pohon keputusan
 Sangat tahan terhadap overfitting jika dibandingkan dengan Decision Tree tunggal
 Performa yang konsisten dan konsisten

• Confusion Matrix

Untuk mengevaluasi kinerja model klasifikasi, visualisasi confusion matrix menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas. Empat confusion matrix digambarkan di atas, masing-masing menunjukkan hasil klasifikasi model:
- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes
Setiap baris memiliki label sebenarnya (true label) dan label hasil prediksi (predicted label).  Warna biru yang lebih gelap menunjukkan lebih banyak prediksi, sedangkan warna biru yang lebih terang menunjukkan lebih sedikit prediksi.
Diagonal Matrix, yang dapat dilihat dari kiri atas ke kanan bawah, menunjukkan jumlah perkiraan yang benar (True Positive) untuk masing-masing kelas.  Semakin tinggi nilai, semakin baik model mengenali kelas tersebut.
Nilai di luar diagonal menunjukkan jumlah prediksi yang salah; ini terjadi ketika model memprediksi kelas yang berbeda dari label aslinya.


Logistic Regression:	Memiliki akurasi cukup tinggi pada kelas 2 dan 3, namun masih terjadi kekeliruan antar keduanya.
Decision Tree:	        Cenderung lebih menyebar prediksinya, menyebabkan overfitting dan kesalahan klasifikasi yang lebih tinggi.
Random Forest:	        Menunjukkan performa terbaik, terutama pada kelas 2 dan 3 dengan jumlah prediksi benar tertinggi.
Naive Bayes:	        Performa rendah, terutama karena prediksi tersebar ke banyak kelas yang salah; kemungkinan tidak cocok dengan distribusi data.


Daftar Pustaka:
Zaza, S., Atemkeng, M., & Hamlomo, S. (2023). Wine feature importance and quality prediction: A comparative study of machine learning algorithms with unbalanced data. arXiv preprint arXiv:2310.01584.

Sravan, K., Rao, L. G., Ramineni, K., Rachapalli, A., & Mohmmad, S. (2024). Analyze the quality of wine based on machine learning approach. Data Science and Applications: Proceedings of ICDSA 2023, Volume 3, 351–360.
