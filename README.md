# Laporan Proyek Machine Learning - Rival Moh. Wahyudi
 
## Domain Proyek
Pasar properti adalah salah satu sektor ekonomi yang sangat dinamis dan berpengaruh besar terhadap kesejahteraan masyarakat dalam penelitian yang dilakukan (Rawool dkk, 2021) dinyatakan bahwa prediksi manual yang dilakukan dalam memprediksi harga rumah memiliki sebanyak 25% error yang dimana ini dapat merugikan banyak pihak. prediksi manual juga semakin sulit untuk dilakukan dikarenakan faktor yang mempengaruhi harga rumah menjadi semakin banyak (Ravindra dkk, 2020).Oleh karena itu memahami dan memprediksi harga rumah menjadi hal penting bagi berbagai pihak, termasuk pembeli, penjual, agen properti, pengembang, hingga lembaga keuangan. Dalam hal ini prediksi harga rumah yang akurat dapat membantu:

1. Calon pembeli dalam menentukan anggaran dan lokasi yang sesuai dengan kebutuhan mereka.
2. Penjual dan agen properti dalam menentukan strategi penjualan yang tepat berdasarkan tren pasar.
3. Pengembang properti untuk membuat keputusan investasi yang lebih cerdas.
4. Lembaga keuangan dalam menilai risiko pemberian pinjaman terkait properti.
 
Namun, prediksi harga rumah bukanlah tugas yang mudah karena dipengaruhi oleh banyak faktor kompleks seperti lokasi, luas tanah dan bangunan, fasilitas sekitar, tren pasar, kondisi ekonomi, serta perubahan permintaan dan penawaran. Ketergantungan pada metode manual atau pendekatan tradisional sering kali menghasilkan estimasi yang kurang akurat dan sulit beradaptasi dengan perubahan data secara cepat.
 
Machine Learning (ML) muncul sebagai solusi potensial untuk mengatasi tantangan ini. Dengan kemampuan untuk menganalisis pola dalam data yang kompleks dan besar, ML dapat memberikan prediksi harga rumah yang lebih akurat dibandingkan metode tradisional. Penggunaan model ML tidak hanya memanfaatkan data historis, tetapi juga dapat memperhitungkan hubungan non-linear dan interaksi antar fitur, seperti pengaruh lokasi terhadap harga rumah hal ini juga sudah dibuktikan dalam beberapa penelitian (Ravindra dkk, 2020) dan (Kandasamy dkk, 2023) yang menggunakan metode machine learning dalam memprediksi harga rumah dengan berbagai feature. penelitian tersebut menggunakan berbagai jenis algoritma seperti linear regression, lasso regression, ridge regression, gradient boosting dan banyak lagi.
 
Dalam proyek ini kami mengembangkan model machine learning untuk memprediksi harga rumah menggunakan data yang mencakup berbagai faktor penting seperti karakteristik properti, dan lokasi. Model ini bertujuan untuk menyediakan alat prediksi yang andal dan dapat digunakan oleh berbagai pemangku kepentingan dalam membuat keputusan berbasis data.
 
Referensi:
- [House Price Prediction Using Machine Learning](https://www.irejournals.com/formatedpaper/1702692.pdf)
- [HOUSE PRICE PREDICTION USING ADVANCED REGRESSION TECHNIQUES](https://jespublication.com/upload/2020-1106157.pdf)
- [Prediction and Analysis of House Price Through Machine Learning Approach](https://www.ijfmr.com/papers/2023/4/5255.pdf)
 
## Business Understanding
 
Bayangkan jika rumah impian sedang dicari. Anggaran telah ditentukan, harapan mengenai lokasi dipertimbangkan, dan gambaran rumah yang sempurna telah dibayangkan. Namun, saat pencarian dimulai, satu tantangan besar dihadapi: berapa harga yang sebenarnya wajar untuk rumah tersebut?


Dilema ini tidak hanya dihadapi oleh pembeli. Penjual seringkali dihadapkan pada pertanyaan, "Apakah harga yang terlalu murah telah ditetapkan?" atau "Apakah rumah ini dihargai terlalu mahal sehingga tidak akan terjual?" Ketidakpastian yang sama dirasakan oleh para pengembang properti dan investor, yang menghadapi risiko besar jika perhitungan nilai pasar suatu properti dilakukan secara keliru. Bahkan bank yang menawarkan pinjaman pun diharuskan untuk menilai properti dengan tepat agar kerugian dapat dihindari.


Di balik setiap keputusan jual-beli rumah, ketidakpastian besar selalu dipengaruhi oleh banyak faktor: lokasi, luas tanah, jumlah kamar, fasilitas umum di sekitar, hingga tren pasar yang terus berubah. Sering kali, metode tradisional, seperti penilaian berdasarkan intuisi atau pengalaman masa lalu, dianggap tidak cukup memadai. Akibatnya, harga terlalu mahal sering kali dibayarkan oleh pembeli, keuntungan hilang dari penjual, dan peluang terbaik terlewatkan begitu saja.


Di sinilah teknologi digunakan untuk mengubah permainan. Bayangkan jika sebuah alat tersedia untuk membaca pola di balik data yang kompleks. Sebuah alat yang mampu memberikan angka akurat dalam hitungan detik.
Bagian laporan ini mencakup:
 
### Problem Statements
Berdasarkan latar belakang bisnis sebelumnya, dalam Project ini akan mengembangkan sebuah sistem prediksi harga rumah menggunakan teknologi machine learning. Berikut adalah beberapa pernyataan masalah yang dapat diuraikan:
- Feature mana saja yang mempengaruhi naik dan turunnya harga rumah?
- Berapa harga rumah dari setiap rumah dengan fitur-fitur tertentu?
 
### Goals
 
Untuk menjawab pernyataan masalah di atas, ada beberapa tujuan yang ingin dicapai, yaitu:
- Mengetahui fitur apa saja yang mempengaruhi naik dan turunnya harga rumah
- Membuat model machine learning yang dapat memprediksi harga rumah dengan akurasi tinggi
 
### Solution Statement
Untuk mencapai tujuan di atas, untuk mencapai tujuan tersebut dipakailah beberapa algoritma machine learning, yaitu:
- Linear Regression
- Random Forest
- Gradient Boosting
 
Dengan menggunakan algoritma-algoritma tersebut akan dicoba membangun model machine learning yang dapat memprediksi harga rumah dengan akurasi tinggi. dengan beberapa metrik evaluasi yang digunakan untuk memastikan keakuratan model, yaitu:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared
 
## Data Understanding
Data yang digunakan dalam proyek ini adalah data dari dataset yang saya unduh dari kaggle dengan judul "Housing Price Prediction".Dataset ini berisi informasi tentang properti seperti luas tanah, jumlah kamar, harga, dan lain sebagainya. yang dimana data ini dapat digunakan untuk memprediksi harga rumah berdasarkan fitur-fitur tersebut.


berikut adalah link dataset yang saya gunakan:
[Housing Price Dataset](https://www.kaggle.com/datasets/sukhmandeepsinghbrar/housing-price-dataset/data)




### Informasi Dataset
Dataset Housing Price ini terdiri dari 21613 baris dan 21 kolom data. Data tersebut sudah dilakukan pemeriksaan tentang data-data yang kemungkinan hilang, kosong, dan duplikat. pemeriksaan yang dilakukan menunjukkan bahwa dataset tidak memiliki data yang hilang, kosong, ataupun duplikat yang dimana ini tidak memerlukan treatment khusus untuk memperbaiki data tersebut.


Jumlah Baris : 21613
Jumlah Kolom : 21
Data Hilang : 0
Data Kosong : 0
Data Duplikat : 0


### Variabel-variabel pada Housing Price Dataset adalah sebagai berikut:
- id : merupakan identifikasi unik untuk setiap properti.
- date : merupakan tanggal saat properti terjual.
- price : merupakan harga yang diberikan oleh pemilik properti.
- bedrooms : merupakan jumlah kamar tidur dalam properti.
- bathrooms : merupakan jumlah kamar mandi dalam properti.
- sqft_living : merupakan luas tanah dalam properti.
- sqft_lot : merupakan luas lahan dalam properti.
- floors : merupakan jumlah lantai dalam properti.
- waterfront : merupakan informasi tentang properti yang terletak di tepi laut atau tidak.
- view : merupakan informasi tentang keindahan properti yang dilihat dari pemandangan laut.
- condition : merupakan kondisi properti yang mempengaruhi harga properti.
- grade : merupakan kualitas properti yang mempengaruhi harga properti.
- sqft_above : merupakan luas tanah atas dalam properti.
- sqft_basement : merupakan luas tanah dasar dalam properti.
- yr_built : merupakan tahun properti dibangun.
- yr_renovated : merupakan tahun properti diperbaiki.
- sqft_living15 : merupakan luas tanah dalam properti yang terdekat dengan 15 properti terdekat.
- sqft_lot15 : merupakan luas lahan dalam properti yang terdekat dengan 15 properti terdekat.


### Exploratory Data Analysis
Pada tahap ini saya melakukan tahapan exploratory data analysis yang meliputi:
- Univariate analysis adalah melakukan analisis pada satu variabel terhadap data tersebut
![output1](https://github.com/user-attachments/assets/64af8899-14af-41bd-be31-9f19446ad25c)
dapat dilihat dari hasil diagram setiap kolom terdapat beberapa kolom yang kebanyakkan hanya memiliki 1 nilai saja seperti kolom sqft_lot, view, sqft_lot15, yr_renovated, dan waterfront. kolom kolom ini jika terus digunakan dalam pelatihan model nantinya akan mempengaruhi hasil akurasi model yang akan dihasilkan nantinya, oleh karena itu di tahap selanjutnya kolom-kolom tersebut akan dihapus.


- Multivariate analysis adalah melakukan analisis pada beberapa variabel terhadap data tersebut
![output2](https://github.com/user-attachments/assets/fac9fcd1-fcc4-40b4-8696-16a8f51613c2)
untuk hasil diagram dalam multivariate analysis saya menggunakan heatmap untuk melihat besar korelasi pada setiap kolom terhadap kolom yang lain, tetapi fokus utama yang diinginkan adalah korelasi dengan fitur price. dapat dilihat ada beberapa kolom yang memiliki korelasi yang cukup besar dengan kolom price seperti kolom bathrooms, sqft_living, grade, sqft_above, dan sqft_living15. akan tetapi juga terdapat beberapa kolom yang memiliki korelasi yang cukup kecil dengan kolom price seperti kolom yr_built, condition, sqft_lot, yr_renovated, dan sqft_lot15. dari sini kolom yang memiliki korelasi yang kecil tidak akan terlalu berguna dalam pelatihan model machine learning nantinya jika tetap dipakai akan mempengaruhi hasil akurasi model yang akan dihasilkan nantinya, oleh karena itu di tahap selanjutnya kolom-kolom tersebut akan dihapus.


## Data Preparation
Pada bagian ini saya melakukan tahapan data preparation yang meliputi:
- Drop column/fitur yang tidak terlalu berguna seperti lat, long, zipcode, id, dan date
proses ini dilakukan karena fitur-fitur tersebut tidak terlalu berguna dalam pelatihan model machine learning nantinya dan jika tidak dihapus akan mempengaruhi hasil akurasi model yang akan dihasilkan nantinya, maka tindakan untuk dapat mengatasi hal tersebut ini adalah dengan menghapus fitur-fitur tersebut.


- Drop fitur yang tidak terlalu berkorelasi dengan harga seperti condition, sqft_lot, sqft_lot15, yr_built, dan yr_renovated
penghapusan fitur-fitur tersebut dilakukan karena fitur-fitur tersebut memiliki korelasi yang sangat kecil dengan harga rumah dimana bisa dibilang tidak berguna dalam pelatihan model machine learning nantinya dan dapat membuat bias pelatihan model.


- Drop colomn dengan value yang sebagian besar adalah value yang sama seperti view dan waterfront
seperti yang sudah dijelaskan pada tahap exploratory data analysis, kolom view dan waterfront memiliki value yang sebagian besar adalah value yang sama atau hanya memiliki 1 value sehingga dapat dihapus, penghapusan ini dilakukan dengan alasan agar model dapat memprediksi harga rumah dengan akurasi yang lebih baik.


- Feature Engineering adalah tahapan menambahkan kolom dari kolom yang sebelumnya sudah ada
penambahan kolom tersebut dilakukan untuk dapat menambahkan informasi tambahan yang relevan untuk model yang diharapkan dapat membantu model dalam memprediksi harga rumah dengan akurasi yang lebih baik. penambahan fitur-fitur tersebut adalah seperti berikut:
1. kolom bedrooms_per_sqft merupakan hasil pembagian antara jumlah kamar tidur dengan luas tanah yang menampilkan jumlah kamar tidur per meter persegi.
2. kolom price_per_sqft merupakan hasil pembagian antara harga dengan luas tanah yang menampilkan harga per meter persegi. disini kolom menampilkan harga per meter persegi dari setiap rumah.
3. kolom bathrooms_per_sqft merupakan hasil pembagian antara jumlah kamar mandi dengan luas tanah yang menampilkan jumlah kamar mandi per meter persegi.


- feature selection adalah tahapan dimana saya memilih menggunakan fitur mana saja yang berkorelasi dengan harga yang sebelumnya sudah ditampilkan melalui heatmap. hal ini diperlukan agar model dapat memprediksi harga rumah dengan akurasi yang tinggi dengan menggunakan fitur-fitur yang berkorelasi dengan harga.
- Train test split adalah tahapan dimana saya membagi data menjadi data train dan data test disini saya membagi data menjadi 80% untuk data train dan 20% untuk data test. data train digunakan untuk membangun model dan data test digunakan untuk mengevaluasi model yang telah dibangun nantinya.
- Scaling adalah tahapan dimana saya melakukan scaling pada data yang ada untuk memastikan bahwa semua fitur memiliki skala yang sama dan tidak terdapat nilai yang sangat besar atau sangat kecil dalam proses scaling ini saya menggunakan MinMaxScaler dari scikit-learn untuk mengubah skala data. proses scaling ini saya gunakan agar model dapat lebih stabil dan memiliki kemampuan untuk memprediksi harga rumah dengan akurasi yang lebih baik.
 
## Modeling
 
Tahap modeling saya menggunakan beberapa algoritma machine learning seperti Linear Regression, Random Forest, dan Gradient Boosting untuk memprediksi harga rumah. disini saya menggunakan ke 3 algoritma tersebut untuk dapat membandingkan algoritma mana yang paling cocok untuk memprediksi harga rumah. tetapi dalam hal ini setiap algoritma memiliki kelebihan dan kekurangan masing-masing.
- Linear Regression : algoritma ini memiliki kelebihan seperti sederhana dan mudah untuk diimplementasikan dan juga tidak membutuhkan parameter yang banyak, namun disisi lain juga memiliki kekurangan seperti tidak dapat memprediksi nilai yang sangat besar dan tidak dapat memprediksi nilai yang sangat kecil atau bisa dibilang sensitif outlier.
- Random Forest : algoritma ini memiliki kelebihan yaitu dapat menangkap pola yang kompleks, fleksibel dan tidak sensitif terhadap outlier, kekurangan dari algoritma ini sendiri yaitu kurang efisien untuk memproses data besar terutama jika jumlah pohon estimator tinggi.
- Gradient Boosting : algoritma ini mampu menangkap pola yang kompleks, baik dalam menangani hubungan antar fitur, dan akurasi yang tinggi, namun memiliki beberapa kekurangan seperti memerlukan komputasi yang lebih besar dan waktu yang lebih banyak dalam pelatihan, memerlukan tuning yang teliti dan waktu pelatihan lebih lama.


Cara kerja dari setiap algoritma tersebut adalah sebagai berikut:
1. Linear Regression melakukan prediksi dengan cara mengasumsikan antara variabel input dengan variable output seperti garis lurus rumusnya adalah y = m * x + b dimana m adalah slope(mengukur pengaruh setiap fitur terhadap variable output) dan b adalah intercept (menggeser garis ke atas/bawah sesuai dengan nilai intercept)
algoritma meminimalkan error dengan menemukan m dan b yang terbaik menggunakan metode seperti Ordinary Least Squares (OLS)


2. Random Forest algoritma dengan cara kerja membuat banyak pohon keputusan (decision tree) dari data pelatihan dengan proses bagging, selanjutnya setiap pohon dilatih pada subset data yang berbeda dengan fitur yang dipilih secara acak. dalam regresi hasil prediksi yang dihasilkan adalah rata-rata dari hasil prediksi dari setiap pohon. pengacakan data da fitur secara acak dapat mengurangi bias dan menghindari overfitting.


3. Gradient Boosting bekerja dega cara memulai dengan model yang sederhana seperti decision tree yang kecil, setelah pelatihan dilakukan maka model baru dilatih untuk memperbaiki error dari model, jadi setiap iterasi akan membuat model baru yang akan memperbaiki error dari model sebelumnya. hasil prediksi yang dihasilkan adalah hasil akhir dari semua model.


dalam tahapan pembangunan model ini saya melakukan dengan beberapa tahapan sebagai berikut:
- membuat variabel dictionary yang menampung algoritma-algoritma yang digunakan untuk memprediksi harga rumah.
- membuat variabel dictionary yang menampung hasil prediksi dari setiap algoritma.
- saya melakukan loop untuk membangun model dan melakukan prediksi harga rumah untuk setiap algoritma yang digunakan.
- terakhir disini saya menampilkan hasil prediksi harga rumah untuk setiap algoritma yang digunakan dari hasil ini lah dapat ditentukan model terbaik yang dapat digunakan untuk memprediksi harga rumah.
 
dalam hasil pelatihan yang dilakukan gradient boosting memiliki akurasi yang paling tinggi yang ditunjukkan oleh nilai R-squared sebesar 0.9963 lebih tinggi dibandingkan random forest dengan nilai R-squared sebesar 0.9945 dan nilai rmse sebesar 23667.04 lebih kecil daripada random forest dengan nilai rmse sebesar 28958.65 Hal ini menunjukkan bahwa model gradient boosting dapat memprediksi harga rumah dengan akurasi yang sangat baik. Dari hasil yang didapatkan tersebut
 
## Evaluation
 
metrik evaluasi yang saya gunakan dalam penyusunan proyek in adalah sebagai berikut:
- Mean Squared Error (MSE) : matrik ini bekerja dengan mengukur rata-rata selisih antara nilai prediksi dan nilai aktual. memberikan indikasi seberapa besar error yang terjadi dalam prediksi.
rumus :
![MSE Formula](https://cdn-media-1.freecodecamp.org/images/hmZydSW9YegiMVPWq2JBpOpai3CejzQpGkNG)
 
pada pelatihan ini nilai MSE dari setiap model yang sudah dilatih :
- Linear Regression : 17621966333.93
- Random Forest : 838603633.62
- Gradient Boosting : 560128675.15
 
dari hasil tersebut model linear regression menghasilkan prediksi yang memiliki error yang cukup tinggi dibandingkan random forest dan gradient boosting. dan gradient boosting memiliki error yang lebih kecil dibandingkan linear regression dan random forest.
 
- Root Mean Squared Error (RMSE) : matrik ini mengukur dengan cara mengakar kuadratkan hasil dari MSE memberikan interpretasi error dalam satuan yang sama dengan nilai aktual jadi kita dapat dengan mudah membandingkan error dengan satuan yang sama dengan data target.
rumus :
![RMSE Formula](https://media.geeksforgeeks.org/wp-content/uploads/20200622171741/RMSE1.jpg)
 
pada pelatihan ini nilai RMSE dari setiap model yang sudah dilatih :
- Linear Regression : 132747.75
- Random Forest : 28958.65
- Gradient Boosting : 23667.04
 
dari hasil tersebut dapat dilihat bahwa linear regression memiliki error yang tinggi dibandingkan yang lain yang cukup signifikan untuk konteks prediksi harga rumah. dalam model random forest penurunan yang menyimpulkan bahwa model ini lebih mampu memberikan estimasi harga rumah yang mendekati harga sebenarnya. pada model gradient boosting memiliki nilai yang lebih kecil dibandingkan random forest yang arti nya model ini lebih mampu memberikan estimasi harga rumah yang lebih akurat.
 
- R-squared : matrik ini digunakan mengukur seberapa baik model memprediksi data dibandingkan dengan rata-rata nilai sebenarnya. memberikan indikasi seberapa baik model ini dapat memprediksi data.
rumus :
![R-squared Formula](https://miro.medium.com/v2/resize:fit:1200/1*_mVvAFVEGinHlijmmeWwzg.png)
 
pada pelatihan ini nilai R-squared dari setiap model yang sudah dilatih :
- Linear Regression : 0.8834
- Random Forest : 0.9945
- Gradient Boosting : 0.9963
 
nilai dari linear regression berarti model dapat menjelaskan 88,34% varians dari data aktual. diikuti random forest dan gradient boosting memiliki nilai yang lebih tinggi dibandingkan linear regression yang menandakan bahwa sebanyak 99,45% dan 99,63% varians dari data aktual dapat dijelaskan oleh model tersebut. dan menunjukkan bahwa model gradient boosting dapat memprediksi harga rumah dengan akurasi yang sangat baik. Dari hasil tersebut menjawab beberapa problem statement yaitu fitur-fitur yang memengaruhi harga rumah dan juga dapat memprediksi harga rumah dengan akurasi yang baik adalah fitur-fitur seperti bedrooms, bathrooms, sqft_living, floors, grade, sqft_above, sqft_basement, sqft_living15, price_per_sqft, bedrooms_per_sqft, dan bathrooms_per_sqft. hasil ini juga menjawab berapa harga rumah dari setiap rumah dengan fitur-fitur tertentu yang ditunjukkan dari hasil prediksi harga rumah dari setiap algoritma yang digunakan. hal ini juga juga menjawab goals-goals yang ditetapkan yaitu dapat memprediksi harga rumah dengan akurasi yang baik dan juga mengetahui fitur fitur apa saja yang mempengaruhi harga rumah. hal ini juga menunjukkan bahwa solution statement yang telah ditentukan diawal dapat menjawab problem statement dan mencapai goals-goals yang telah ditetapkan.

## Kesimpulan
Dari hasil yang telah didapat diatas dapat dilihat bahwa model yang memiliki hasil evaluasi terbaik adalah model Gradient Boosting dengan hasil evaluasi MSE, RMSE, dan R2 Score yang paling tinggi. disini meskipun model menggunakan settingan default namun hasilnya cukup baik untuk dapat dilakukan prediksi harga rumah ini sudah sesuai dengan goals yang ditetapkan yaitu dapat memprediksi harga rumah dengan akurasi yang baik dan juga fitur fitur apa saja yang mempengaruhi harga rumah. yaitu
