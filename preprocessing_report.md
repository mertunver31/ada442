# Data Preprocessing Report
## Approach 1: Basic Preprocessing
This approach includes handling missing values, encoding categorical variables, and scaling features.
### 1. Handling Missing Values
#### Numerical Features
For numerical features with missing values, we replaced them with the median of each column.
Missing values before imputation:
- emp.var.rate: 2384 values
- euribor3m: 422 values

#### Categorical Features
For categorical features with 'unknown' values, we treated 'unknown' as a separate category.
'Unknown' values found in categorical columns:
- job: 39 values (0.95%)
- marital: 11 values (0.27%)
- education: 167 values (4.05%)
- default: 803 values (19.50%)
- housing: 105 values (2.55%)
- loan: 105 values (2.55%)

### 2. Encoding Categorical Variables
We applied one-hot encoding to all categorical variables, creating binary features for each category.
After one-hot encoding, the number of features increased from 20 to 53.

### 3. Feature Scaling
We applied StandardScaler to normalize numerical features.
Numerical features were scaled to have mean=0 and standard deviation=1.

## Approach 2: Advanced Preprocessing with Feature Selection
This approach builds on the basic preprocessing and adds feature selection and handling class imbalance.
### 4. Feature Selection
#### PCA for Dimensionality Reduction
We applied PCA to reduce dimensionality while preserving most of the variance.
PCA reduced the number of features from 53 to 25 while preserving 95% of variance.
Explained variance ratios: [0.21843262 0.10279618 0.09086654 0.07850759 0.07168626 0.06576787
 0.05295201 0.03289926 0.02821403 0.02559703 0.01963687 0.01782314
 0.01729396 0.01592458 0.01514888 0.01441224 0.0141041  0.01370992
 0.01079608 0.00947447 0.00942266 0.00752464 0.00720605 0.00613869
 0.00569847]

#### SelectKBest for Feature Selection
SelectKBest identified the top 10 most important features:
1. duration
2. pdays
3. previous
4. emp.var.rate
5. euribor3m
6. nr.employed
7. contact_telephone
8. month_mar
9. poutcome_nonexistent
10. poutcome_success

### 5. Handling Class Imbalance
We applied SMOTE (Synthetic Minority Over-sampling Technique) to address the class imbalance problem.
Class distribution before SMOTE:
- no: 3668 samples (89.05%)
- yes: 451 samples (10.95%)

Class distribution after SMOTE (training set only):
- no: 2936 samples (50.00%)
- yes: 2936 samples (50.00%)

### Final Preprocessed Datasets
We created several preprocessed datasets for modeling:
1. **Basic Preprocessed**: Missing values handled, categorical variables encoded, features scaled
2. **PCA Transformed**: Dimensionality reduction while preserving 95% of variance
3. **Feature Selected**: Top 10 most important features selected
4. **SMOTE Resampled**: Class imbalance addressed using SMOTE
