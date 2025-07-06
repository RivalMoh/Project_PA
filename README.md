# House Price Prediction Project - Rival Moh. Wahyudi

Welcome to the House Price Prediction project! In this project, I developed a machine learning model to predict house prices using a rich dataset containing various property characteristics and location factors. The goal is to provide a reliable prediction tool that can help different stakeholders make informed, data-driven decisions.

## Project Domain

The property market is a dynamic sector with a significant impact on society's well-being. Manual house price predictions can have up to 25% error (Rawool et al., 2021), which can be costly for all parties. As the factors influencing house prices become more complex (Ravindra et al., 2020), accurate predictions are crucial for buyers, sellers, agents, developers, and financial institutions. Accurate house price predictions help:

1. Buyers set realistic budgets and choose suitable locations.
2. Sellers and agents develop effective sales strategies based on market trends.
3. Developers make smarter investment decisions.
4. Financial institutions assess loan risks more accurately.

However, predicting house prices is challenging due to many complex factors—location, land/building area, nearby facilities, market trends, economic conditions, and supply-demand changes. Relying on manual or traditional methods often leads to inaccurate estimates and slow adaptation to new data.

Machine Learning (ML) offers a powerful solution. ML can analyze complex, large datasets and provide more accurate predictions than traditional methods. ML models leverage historical data and can capture non-linear relationships and feature interactions, as shown in research (Ravindra et al., 2020; Kandasamy et al., 2023) using various algorithms like linear regression, lasso, ridge, and gradient boosting.

In this project, I developed a machine learning model to predict house prices using data covering key property and location factors. The goal is to provide a reliable prediction tool for all stakeholders to make data-driven decisions.

## References

- [House Price Prediction Using Machine Learning](https://www.irejournals.com/formatedpaper/1702692.pdf)
- [HOUSE PRICE PREDICTION USING ADVANCED REGRESSION TECHNIQUES](https://jespublication.com/upload/2020-1106157.pdf)
- [Prediction and Analysis of House Price Through Machine Learning Approach](https://www.ijfmr.com/papers/2023/4/5255.pdf)

## Business Understanding

Imagine searching for your dream home. You have a set budget, a preferred location, and a vision of the perfect house. But as you begin your search, a big question arises: what is the fair price for the house?

This dilemma is not only faced by buyers. Sellers often wonder, "Is my price too low?" or "Is my house overpriced and unlikely to sell?" Property developers and investors face similar uncertainties, risking losses if market values are miscalculated. Even banks need accurate property valuations to avoid financial risk when offering loans.

Behind every real estate transaction, uncertainty is influenced by many factors: location, land area, number of rooms, nearby facilities, and ever-changing market trends. Traditional methods—like relying on intuition or past experience—are often not enough. As a result, buyers may overpay, sellers may lose profit, and the best opportunities can be missed.

This is where technology changes the game. Imagine a tool that can read patterns in complex data and deliver accurate numbers in seconds. This project aims to build such a tool, using machine learning to bring clarity and confidence to property pricing decisions.

### Problem Statements

Based on the business background, this project aims to develop a house price prediction system using machine learning. The main questions addressed are:
- Which features most influence house prices?
- What is the predicted price for a house with specific features?

### Goals

To answer these questions, the project sets out to:
- Identify the key features that drive house prices up or down
- Build a machine learning model that can predict house prices with high accuracy

### Solution Approach

To achieve these goals, I experimented with several machine learning algorithms:
- Linear Regression
- Random Forest
- Gradient Boosting

These models are evaluated using metrics such as:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared

## Data Understanding
The dataset used in this project was downloaded from Kaggle: [Housing Price Dataset](https://www.kaggle.com/datasets/sukhmandeepsinghbrar/housing-price-dataset/data). It contains detailed information about properties, such as land area, number of rooms, price, and more—making it ideal for predicting house prices based on these features.

**Dataset Overview:**
- Rows: 21,613
- Columns: 21
- Missing Values: 0
- Empty Values: 0
- Duplicates: 0

All data has been checked for missing, empty, or duplicate values, so no special data cleaning was required.

### Dataset Variables

The Housing Price dataset includes the following variables:
- **id**: Unique identifier for each property
- **date**: Date the property was sold
- **price**: Sale price of the property
- **bedrooms**: Number of bedrooms
- **bathrooms**: Number of bathrooms
- **sqft_living**: Living area in square feet
- **sqft_lot**: Lot area in square feet
- **floors**: Number of floors
- **waterfront**: Whether the property is on the waterfront
- **view**: Quality of the view
- **condition**: Condition of the property
- **grade**: Construction and design grade
- **sqft_above**: Square footage above ground
- **sqft_basement**: Square footage of the basement
- **yr_built**: Year the property was built
- **yr_renovated**: Year the property was renovated
- **sqft_living15**: Living area of the 15 nearest neighbors
- **sqft_lot15**: Lot area of the 15 nearest neighbors

## Exploratory Data Analysis (EDA)

In this stage, I performed exploratory data analysis to better understand the dataset and its features:

- **Univariate Analysis:**
  - Examined each variable individually to understand its distribution and characteristics.
  - ![output1](https://github.com/user-attachments/assets/64af8899-14af-41bd-be31-9f19446ad25c)
  - Some columns, such as `sqft_lot`, `view`, `sqft_lot15`, `yr_renovated`, and `waterfront`, mostly contain a single value. Keeping these columns in the model could negatively impact accuracy, so they are removed in later steps.

- **Multivariate Analysis:**
  - Analyzed relationships between multiple variables, especially their correlation with `price`.
  - ![output2](https://github.com/user-attachments/assets/fac9fcd1-fcc4-40b4-8696-16a8f51613c2)
  - The heatmap shows that features like `bathrooms`, `sqft_living`, `grade`, `sqft_above`, and `sqft_living15` have strong correlations with price, while others like `yr_built`, `condition`, `sqft_lot`, `yr_renovated`, and `sqft_lot15` have weak correlations. Features with low correlation are also removed to improve model performance.

## Data Preparation

To ensure optimal model performance, I implemented a comprehensive data preparation pipeline:

### Feature Cleaning and Selection

- **Removed Non-Essential Features**: Dropped irrelevant columns like `lat`, `long`, `zipcode`, `id`, and `date` that don't contribute to price prediction and could introduce noise to the model.

- **Eliminated Weak Correlations**: Removed features with low correlation to price such as `condition`, `sqft_lot`, `sqft_lot15`, `yr_built`, and `yr_renovated` to reduce model bias and improve accuracy.

- **Dropped Low-Variance Features**: Eliminated columns like `view` and `waterfront` that contain mostly identical values, as they provide minimal predictive power.

### Feature Engineering

To enhance the model's predictive capabilities, I created new meaningful features:

1. **`bedrooms_per_sqft`**: Bedroom density (bedrooms ÷ living area) - indicates space efficiency
2. **`price_per_sqft`**: Price per square foot - a key real estate metric for value assessment  
3. **`bathrooms_per_sqft`**: Bathroom density (bathrooms ÷ living area) - reflects property convenience

### Data Preprocessing Steps

- **Feature Selection**: Selected only highly correlated features identified through correlation analysis
- **Train-Test Split**: Divided data into 80% training and 20% testing sets for robust model evaluation
- **Data Scaling**: Applied MinMaxScaler to normalize all features to the same scale (0-1), ensuring model stability and improved convergence


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

I tested three machine learning algorithms to find the best approach for house price prediction. Each algorithm has unique strengths and characteristics:

### Algorithm Comparison

**🔵 Linear Regression**
- ✅ **Strengths**: Simple, fast, interpretable, requires minimal parameters
- ❌ **Weaknesses**: Assumes linear relationships, sensitive to outliers, limited for complex patterns

**🌳 Random Forest**  
- ✅ **Strengths**: Handles complex patterns, robust to outliers, provides feature importance
- ❌ **Weaknesses**: Less efficient with large datasets, can overfit with many trees

**🚀 Gradient Boosting**
- ✅ **Strengths**: Excellent accuracy, captures complex relationships, handles feature interactions well
- ❌ **Weaknesses**: Computationally intensive, longer training time, requires careful tuning

### How Each Algorithm Works

1. **Linear Regression**: Creates a straight-line relationship between features and price using the formula `y = mx + b`, where it finds the best slope (m) and intercept (b) using Ordinary Least Squares.

2. **Random Forest**: Builds multiple decision trees using different data subsets and random feature selection, then averages their predictions to reduce overfitting and improve accuracy.

3. **Gradient Boosting**: Starts with a simple model, then iteratively adds new models that correct the errors of previous ones, creating a powerful ensemble.

### Model Development Process

- Created a dictionary of algorithms for systematic comparison
- Implemented cross-validation for robust performance assessment  
- Trained each model on the prepared dataset
- Evaluated predictions using multiple metrics
- Selected the best-performing model based on evaluation results

### Results Summary

**Gradient Boosting emerged as the winner** with:
- **R-squared: 0.9963** (vs Random Forest: 0.9945)
- **RMSE: $23,667** (vs Random Forest: $28,959)

This demonstrates that Gradient Boosting can predict house prices with exceptional accuracy.

## Model Evaluation

To ensure our models perform reliably, I used three comprehensive evaluation metrics:

### 📊 Evaluation Metrics Explained

**1. Mean Squared Error (MSE)**
- Measures the average squared difference between predicted and actual prices
- Lower values indicate better performance
- Formula: MSE = (1/n) × Σ(actual - predicted)²

![MSE Formula](https://cdn-media-1.freecodecamp.org/images/hmZydSW9YegiMVPWq2JBpOpai3CejzQpGkNG)

**MSE Results:**
- 🔴 Linear Regression: 17,621,966,333.93
- 🟡 Random Forest: 838,603,633.62  
- 🟢 **Gradient Boosting: 560,128,675.15** ✨

**2. Root Mean Squared Error (RMSE)**
- Square root of MSE, providing error in the same units as house prices (dollars)
- More interpretable than MSE for understanding prediction accuracy
- Formula: RMSE = √MSE

![RMSE Formula](https://media.geeksforgeeks.org/wp-content/uploads/20200622171741/RMSE1.jpg)

**RMSE Results:**
- 🔴 Linear Regression: $132,747.75
- 🟡 Random Forest: $28,958.65
- 🟢 **Gradient Boosting: $23,667.04** ✨

**3. R-squared (Coefficient of Determination)**
- Measures how well the model explains the variance in house prices
- Values closer to 1.0 indicate better model performance
- Formula: R² = 1 - (SS_res / SS_tot)

![R-squared Formula](https://miro.medium.com/v2/resize:fit:1200/1*_mVvAFVEGinHlijmmeWwzg.png)

**R-squared Results:**
- 🔴 Linear Regression: 0.8834 (88.34% variance explained)
- 🟡 Random Forest: 0.9945 (99.45% variance explained)
- 🟢 **Gradient Boosting: 0.9963 (99.63% variance explained)** ✨

### 🎯 Key Insights

**Linear Regression** shows significant limitations with high error rates, indicating that house prices have complex, non-linear relationships that simple linear models cannot capture effectively.

**Random Forest** demonstrates substantial improvement, reducing prediction errors dramatically and explaining 99.45% of price variance.

**Gradient Boosting** achieves the best performance across all metrics, with:
- The lowest prediction error ($23,667 RMSE)
- The highest explanatory power (99.63% R-squared)
- Superior ability to capture complex feature interactions

### ✅ Answering Our Research Questions

This evaluation successfully addresses our initial problem statements:

1. **Which features influence house prices?** 
   - Key drivers: `bedrooms`, `bathrooms`, `sqft_living`, `floors`, `grade`, `sqft_above`, `sqft_basement`, `sqft_living15`, and our engineered features (`price_per_sqft`, `bedrooms_per_sqft`, `bathrooms_per_sqft`)

2. **Can we predict house prices accurately?**
   - Yes! Our Gradient Boosting model achieves 99.63% accuracy, making it highly reliable for real-world price predictions.

## Conclusion

This project successfully developed a high-performance house price prediction system using machine learning. **Gradient Boosting emerged as the optimal algorithm**, achieving exceptional accuracy with minimal prediction error.

### 🏆 Key Achievements

- **99.63% prediction accuracy** (R-squared)
- **$23,667 average prediction error** (RMSE) 
- **Robust feature engineering** that improved model performance
- **Comprehensive model comparison** ensuring the best solution

### 🚀 Real-World Impact

This model provides valuable insights for:
- **Buyers**: Make informed decisions with accurate price estimates
- **Sellers**: Set competitive, market-driven prices
- **Real Estate Professionals**: Enhance client advisory services with data-driven valuations
- **Financial Institutions**: Improve loan risk assessment accuracy

The exceptional performance demonstrates that machine learning can effectively solve complex real estate valuation challenges, providing stakeholders with reliable, data-driven price predictions.
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
