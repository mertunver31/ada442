# Bank Marketing Prediction App

This application deploys a machine learning model to predict whether a customer will subscribe to a term deposit based on marketing campaign data from a Portuguese banking institution.

## Features

- **Prediction Interface**: Enter customer and campaign information to get subscription probability predictions.
- **Business Recommendations**: Get actionable recommendations on whether to contact a customer based on expected profit.
- **Model Insights**: Explore feature importance, model performance metrics, and business optimization details.
- **Project Overview**: Learn about the data analysis methodology and key findings.

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd <repository-folder>
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Run the Streamlit app:
```
streamlit run app.py
```

## Using the App

The app has three main sections:

### 1. Predict

- Enter customer demographic information (age, job, marital status, etc.)
- Input campaign-related details (contact type, call duration, etc.)
- Add economic indicators (employment rate, consumer indices, etc.)
- Click "Predict Subscription Likelihood" to get results
- View the probability, expected profit, and ROI calculations
- Review the recommendation on whether to contact the customer

### 2. Model Insights

- **Feature Importance**: See which features most strongly predict subscription
- **Model Performance**: Examine confusion matrix, ROC curve, and performance metrics
- **Business Optimization**: Understand how probability thresholds affect profit

### 3. Project Overview

- Learn about the project objectives and methodology
- Review key findings from the data analysis
- Understand how the model can be applied to improve marketing campaigns

## Model Details

The deployed model is a Stacking Classifier that combines multiple machine learning algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- SVM
- Gradient Boosting

The model was optimized for business metrics, using a lower probability threshold (10%) to maximize expected profit rather than traditional accuracy metrics.

## Data Source

The data comes from the UCI Machine Learning Repository's Bank Marketing dataset, which contains information about direct marketing campaigns of a Portuguese banking institution.

## Business Value

By using this predictive model, marketing teams can:
- Prioritize which customers to contact
- Increase campaign efficiency
- Improve conversion rates
- Maximize ROI from marketing spending

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib 