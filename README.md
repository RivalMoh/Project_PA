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
