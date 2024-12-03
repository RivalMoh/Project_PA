# %% [markdown]
# # Project Machine Learning Predictive Analysis - Rival Moh. Wahyudi
# Nama : Rival Moh. Wahyudi
# Asal : Semarang

# %% [markdown]
# # Import Library

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore') 

# %% [markdown]
# # Load Data
# Data adalah data yang digunakan berasal dari kaggle, dengan judul "Housing Price Prediction".
# link dataset: https://www.kaggle.com/datasets/sukhmandeepsinghbrar/housing-price-dataset

# %%
# data yang telah diunduh dari link kaggle akan di load di lokal
# load data yang sudah diunduh dengan pandas 
data = pd.read_csv('Housing.csv')
data.head()

# %% [markdown]
# # Data Understanding
# link dataset: https://www.kaggle.com/datasets/sukhmandeepsinghbrar/housing-price-dataset
# Jumlah Baris : 21613
# Jumlah Kolom : 21

# %%
# disini menampilkan informasi tentang data
# menampilkan column dan tipe datanya dengan fungsi info()
data.info()

# %% [markdown]
# pada dataset ini berjumlah sebanyak 21613 baris dengan kolom berjumlah sebanyak 21
# Jumlah Baris : 21613
# Jumlah Kolom : 21

# %% [markdown]
# Penjelasan Fitur yang ada pada data tersebut adalah sebagai berikut:
# - price: harga rumah
# - bedrooms: jumlah kamar tidur
# - bathrooms: jumlah kamar mandi
# - sqft_living: luas tanah
# - sqft_lot: luas tanah
# - floors: jumlah lantai
# - waterfront: lokasi di pantai
# - view: tampilan rumah
# - condition: kondisi rumah
# - grade: kualitas rumah
# - sqft_above: luas tanah di atas lantai
# - sqft_basement: luas tanah di bawah lantai
# - yr_built: tahun pembangunan
# - yr_renovated: tahun renovasi
# - sqft_living15: luas tanah dalam 15 blok
# - sqft_lot15: luas tanah di lot 15 blok
# - lat: koordinat latitude
# - long: koordinat longitude
# - zipcode: kode pos
# - id: id rumah
# - date: tanggal pembangunan rumah

# %% [markdown]
# terdapat fitur fitur yang tidak diperlukan untuk pembangunan model machine learning seperti lat, long, zipcode, id, dan date, pada tahap data preparation akan dihapus fitur tersebut

# %%
data.describe().round(3)

# %%
# Memeriksa Missing Value dengan fungsi isnull() dan isna()
# memeriksa duplikat dengan fungsi duplicated()
print (data.isnull().sum())
print (data.isna().sum())
print (data.duplicated().sum())

# %% [markdown]
# Hasil yang di dapat pada proses pemeriksaan data hilang, kosong, dan duplikat adalah sebagai berikut:
# Data Hilang : 0
# Data Kosong : 0
# Data Dublikat : 0

# %% [markdown]
# # Exploratory Data Analysis

# %%
feature = data.select_dtypes(include=['float64', 'int64', 'int32']).columns
print (feature)

# %%
#univariate analysis adalah melakukan analisis pada satu variabel terhadap data tersebut
# Menampilkan persebaran data pada setiap fitur numerik karena kebetulan seluruh fitur bertipe numerik
data[feature].hist(bins = 50, figsize=(20, 20))
plt.show()

# %% [markdown]
# pada hasil visualisasi diatas terdapat beberapa fitur yang kebanyakan hanya memiliki satu nilai seperti sqft_lot, view, sqft_lot15, yr_renovated, dan waterfront, jika tetap digunakan akan memengaruhi model machine learning yang akan dibangun, maka pada tahap data preparation akan dihapus fitur tersebut

# %% [markdown]
# # Multivariate Analysis
# melakukan analisis keterhubungan pada beberapa variabel
# disini saya akan melihat corelasi antar setiap fitur terhadap fitur yang lain menggunakan perintah corr()

# %%
# menampung seluruh fitur yang bersifat numerik
numerical_features = data.select_dtypes(include=['int64', 'float64', 'int32', 'object']).drop(['id', 'date', 'lat', 'long', 'zipcode'], axis=1).columns
print (numerical_features)

# %%
# Menampilkan hasil correlation matrix atau keterhubungan antar fitur dengan fitur lainnya pada data dengan heatmap
plt.figure (figsize=(20, 16))
correlation_matrix = data[numerical_features].corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix setiap Feature", size=10)
plt.show()

# %% [markdown]
# terlihat dari hasil heatmap diatas terdapat beberapa fitur yang memiliki korelasi yang cukup tinggi pada fitur harga rumah seperti bathrooms, sqft_living, grade, sqft_above, dan sqft_living15. dan juga terdapat beberapa fitur yang memiliki korelasi yang cukup rendah seperti yr_built, condition, sqft_lot, yr_renovated, dan sqft_lot15. data data yang berkorelasi rendah tersebut akan dihapus pada tahap data preparation.

# %% [markdown]
# # Data Preparation
# 1. Drop column/fitur yang tidak terlalu berguna seperti lat, long, zipcode, id, dan date
# 2. Drop fitur yang tidak terlalu berkorelasi dengan harga seperti condition, sqft_lot, sqft_lot15, yr_built, dan yr_renovated
# 3. Drop colomn dengan value yang sebagian besar adalah value yang sama seperti view dan waterfront
# 3. Feature Engineering
# 4. feature selection
# 5. train test split
# 6. scaling

# %%
# Mendrop column/fitur yang tidak terlalu berguna seperti lat, long, zipcode, id, dan date
data.drop(['lat', 'long', 'zipcode', 'id', 'date'], axis=1, inplace=True)
data.head()

# %%
# Melakukan pembersihan data yang tidak terlalu berkorelasi dengan harga seperti condition, sqft_lot, sqft_lot15, yr_built, dan yr_renovated
data.drop(['condition', 'sqft_lot', 'sqft_lot15', 'yr_built', 'yr_renovated'], axis=1, inplace=True)
data.head()

# %%
# Melakukan pembersihan pada colum dnegan sebagian besar data dengan nilai yang sama
data.drop(['view', 'waterfront'], axis=1, inplace=True)
data.head()

# %% [markdown]
# # Feature Engineering
# menambahkan beberapa feature baru yang tidak terdapat pada data yang asli seperti price_per_sqft dan bedrooms_per_sqft
# 
# - bedrooms_per_sqft = bedrooms / sqft_living
# - price_per_sqft = price / sqft_living
# - bathrooms_per_sqft = bathrooms / sqft_living

# %%
data['price_per_sqft'] = data['price'] / data['sqft_living']
data['bedrooms_per_sqft'] = data['bedrooms'] / data['sqft_living']
data['bathrooms_per_sqft'] = data['bathrooms'] / data['sqft_living']

# %%
data.info()

# %%
# Feature selection -> hanya menggunakan fitur yang berkorelasi dengan harga
feature = data.select_dtypes(include=['float64', 'int64', 'int32']).drop(['price'], axis=1).columns
print (feature)

# %%
# Data splitting -> 80:20
X = data[feature]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.head()

# %%
# Standarisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print (X_train_scaled)
print (X_test_scaled)

# %% [markdown]
# # Training dan Evaluation
# disini saya akan melakukan training dan evaluation terhadap beberapa model
# 1. Linear Regression
# 2. Random Forest
# 3. Gradient Boosting

# %%
# Model : penampungan model pada dictionary
# setiap model menggunakan settingan default
models = {
    'LinearRegression' : LinearRegression(),
    'RandomForestRegressor' : RandomForestRegressor(random_state=42),
    'GradientBoostingRegressor' : GradientBoostingRegressor(random_state=42)
}

# %% [markdown]
# model model diatas memiliki cara kerja nya sendiri seperti berikut:
# 1. Linear Regression melakuka prediksi dengan cara mengasumsikan antara variable input dengan variable output seperti garis lurus rumusnya adalah y = m * x + b dimana m adalah slope(mengukur pengaruh setiap fitur terhadap variable output) dan b adalah intercept (menggeser garis ke atas/bawah sesuai dengan nilai intercept)
# algoritma meminimilakan error dengan menemukan m dan b yang terbaik menggunakan motode seperti Ordinary Least Squares (OLS)
# 
# 2. Random Forest algoritma dengan cara kerja membuat banyak pohon keputusan (decision tree) dari data pelatihan dengan proses bagging, selanjutnya setiap pohon dilatih pada subset data yang berbeda dengan fitur yang dipilih secara acak. dalam regresi hasil prediksi yang dihasilkan adalah rata-rata dari hasil prediksi dari setiap pohon. pengacakan data da fitur secara acak dapat mengurangi bias dan menghindari overfitting.
# 
# 3. Gradient Boosting bekerja dega cara memulai degan model yang sederhana seperti decision tree yang kecil, setelah pelatihan dilakukan maka modle baru dilatih unutk memperbaiki error dari model, jadi setiap iterasi akan membuat model baru yang akan memperbaiki error dari model sebelumnya. hasil prediksi yang dihasilkan adalah hasil akhir dari semua model.

# %%
# digunakan untuk menyimpan hasil evaluasi model yang akan dilakukan
results = {}

# %%
# menggunakan perulangan untuk melatih model dan melakukan evaluasi prediksi model yang selanjutnya hasilnya kaan disimpan pada variabel results
for name, model in models.items():
    # melatih model
    model.fit(X_train_scaled, y_train)
    #melakukan prediksi
    y_pred = model.predict(X_test_scaled)
    
    #melakukan evaluasi
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    # menyimpan hasil
    results[name] = {'MSE': mse, 'RMSE': rmse, 'R2': r2}
    
    # menampilkan hasil
    print(f"\n{name} Results:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

# %% [markdown]
# dari hasil yang telah didapat diatas dapat dilihat bahwa model yang memiliki hasil evaluasi terbaik adalah model Gradient Boosting dengan hasil evaluasi MSE, RMSE, dan R2 Score yang paling tinggi. disini meskipun model menggunakan settingan default namun hasilnya cukup baik untuk dapat dilakukan prediksi harga rumah ini sudah sesua dengan goals yang ditetapkan yaitu dapat memprediksi harga rumah dengan akurasi yang baik dan juga fitur fitur apa saja yang mempengaruhi harga rumah. yaitu


