import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import os
from sklearn.model_selection import train_test_split
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Create a directory for preprocessed data if it doesn't exist
if not os.path.exists('preprocessed_data'):
    os.makedirs('preprocessed_data')

# Load the dataset
print("Loading the dataset...")
try:
    print("Trying to read as tab-delimited file...")
    df = pd.read_csv('bank-additional.xls', sep='\t')
    print("Successfully read as tab-delimited file.")
except Exception as e:
    print(f"Failed to read as tab-delimited file: {str(e)}")
    print("All attempts to read the file failed.")
    exit(1)

# Fix specific columns with numeric data
numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                   'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                   'euribor3m', 'nr.employed']

# Convert numeric columns from string to proper numeric types
for col in df.columns:
    if col in numeric_columns:
        try:
            if df[col].dtype == 'object':
                # Replace commas with periods for decimal values
                df[col] = df[col].str.replace(',', '.') if isinstance(df[col].iloc[0], str) else df[col]
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Converted {col} to numeric type.")
        except Exception as e:
            print(f"Could not convert {col} to numeric: {e}")

# Separate features and target variable
X = df.drop('y', axis=1)
y = df['y']

# Separate numerical and categorical variables
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical columns: {numerical_cols}")
print(f"Categorical columns: {categorical_cols}")

# Make copies of the original data for different preprocessing approaches
df_original = df.copy()

# Create a preprocessing report to document steps and results
preprocessing_report = []
preprocessing_report.append("# Data Preprocessing Report\n")

# ---------- Approach 1: Basic Preprocessing ----------
preprocessing_report.append("## Approach 1: Basic Preprocessing\n")
preprocessing_report.append("This approach includes handling missing values, encoding categorical variables, and scaling features.\n")

# 1. Handle missing values
preprocessing_report.append("### 1. Handling Missing Values\n")

# For numerical features, replace missing values with the median
preprocessing_report.append("#### Numerical Features\n")
preprocessing_report.append("For numerical features with missing values, we replaced them with the median of each column.\n")

df_preprocessed = df.copy()
numerical_imputer = SimpleImputer(strategy='median')
df_preprocessed[numerical_cols] = numerical_imputer.fit_transform(df_preprocessed[numerical_cols])

missing_before = df[numerical_cols].isnull().sum()
missing_columns = missing_before[missing_before > 0].index.tolist()

if missing_columns:
    preprocessing_report.append("Missing values before imputation:\n")
    for col in missing_columns:
        preprocessing_report.append(f"- {col}: {missing_before[col]} values\n")
else:
    preprocessing_report.append("No missing values found in numerical columns.\n")

# For categorical features, treat 'unknown' as a separate category
preprocessing_report.append("\n#### Categorical Features\n")
preprocessing_report.append("For categorical features with 'unknown' values, we treated 'unknown' as a separate category.\n")

unknown_before = {}
for col in categorical_cols:
    unknown_count = (df_preprocessed[col] == 'unknown').sum()
    if unknown_count > 0:
        unknown_before[col] = unknown_count

if unknown_before:
    preprocessing_report.append("'Unknown' values found in categorical columns:\n")
    for col, count in unknown_before.items():
        preprocessing_report.append(f"- {col}: {count} values ({count/len(df)*100:.2f}%)\n")
else:
    preprocessing_report.append("No 'unknown' values found in categorical columns.\n")

# 2. Encode categorical variables
preprocessing_report.append("\n### 2. Encoding Categorical Variables\n")

# One-hot encoding for categorical variables
preprocessing_report.append("We applied one-hot encoding to all categorical variables, creating binary features for each category.\n")

# Create a one-hot encoder
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoded_cats = encoder.fit_transform(df_preprocessed[categorical_cols])

# Get the feature names from one-hot encoding
encoded_feature_names = []
for i, col in enumerate(categorical_cols):
    cat_values = encoder.categories_[i][1:]  # Drop first category
    for cat in cat_values:
        encoded_feature_names.append(f"{col}_{cat}")

# Create a DataFrame with the encoded values
encoded_df = pd.DataFrame(encoded_cats, columns=encoded_feature_names)

# Combine with numerical features
df_preprocessed = pd.concat([df_preprocessed[numerical_cols].reset_index(drop=True), 
                             encoded_df.reset_index(drop=True)], axis=1)

preprocessing_report.append(f"After one-hot encoding, the number of features increased from {len(numerical_cols) + len(categorical_cols)} to {df_preprocessed.shape[1]}.\n")

# 3. Feature scaling
preprocessing_report.append("\n### 3. Feature Scaling\n")
preprocessing_report.append("We applied StandardScaler to normalize numerical features.\n")

# Apply standard scaling to numerical features
scaler = StandardScaler()
df_preprocessed[numerical_cols] = scaler.fit_transform(df_preprocessed[numerical_cols])

preprocessing_report.append("Numerical features were scaled to have mean=0 and standard deviation=1.\n")

# Save the preprocessed dataframe
df_preprocessed.to_csv('preprocessed_data/preprocessed_basic.csv', index=False)

# ---------- Approach 2: Advanced Preprocessing with Feature Selection ----------
preprocessing_report.append("\n## Approach 2: Advanced Preprocessing with Feature Selection\n")
preprocessing_report.append("This approach builds on the basic preprocessing and adds feature selection and handling class imbalance.\n")

# Start with the basic preprocessed data (before adding the target)
df_advanced = df_preprocessed.copy()

# 4. Feature selection
preprocessing_report.append("### 4. Feature Selection\n")

# 4.1 PCA for dimensionality reduction
preprocessing_report.append("#### PCA for Dimensionality Reduction\n")
preprocessing_report.append("We applied PCA to reduce dimensionality while preserving most of the variance.\n")

pca = PCA(n_components=0.95)  # Retain 95% of variance
pca_result = pca.fit_transform(df_advanced)

preprocessing_report.append(f"PCA reduced the number of features from {df_advanced.shape[1]} to {pca_result.shape[1]} while preserving 95% of variance.\n")
preprocessing_report.append(f"Explained variance ratios: {pca.explained_variance_ratio_}\n")

# Create DataFrame with PCA results
pca_cols = [f'PC{i+1}' for i in range(pca_result.shape[1])]
df_pca = pd.DataFrame(pca_result, columns=pca_cols)

# Save the PCA components
pd.DataFrame(pca.components_, columns=df_advanced.columns).to_csv('preprocessed_data/pca_components.csv', index=False)

# Save preprocessed data with PCA
df_pca.to_csv('preprocessed_data/preprocessed_pca.csv', index=False)

# 4.2 SelectKBest for feature selection
preprocessing_report.append("\n#### SelectKBest for Feature Selection\n")

# Combine the preprocessed features with the target for SelectKBest
X_for_select = df_advanced
y_label_encoded = LabelEncoder().fit_transform(y)

# Apply SelectKBest to find the top features
selector = SelectKBest(f_classif, k=10)  # Select top 10 features
X_selected = selector.fit_transform(X_for_select, y_label_encoded)

# Get selected feature names
selected_indices = selector.get_support(indices=True)
selected_feature_names = X_for_select.columns[selected_indices].tolist()

preprocessing_report.append(f"SelectKBest identified the top 10 most important features:\n")
for i, feature in enumerate(selected_feature_names):
    preprocessing_report.append(f"{i+1}. {feature}\n")

# Create DataFrame with selected features
df_selected = X_for_select.iloc[:, selected_indices]

# Save selected features dataset
df_selected.to_csv('preprocessed_data/preprocessed_selected_features.csv', index=False)

# 5. Handle class imbalance
preprocessing_report.append("\n### 5. Handling Class Imbalance\n")
preprocessing_report.append("We applied SMOTE (Synthetic Minority Over-sampling Technique) to address the class imbalance problem.\n")

# Calculate class distribution before SMOTE
class_distribution_before = y.value_counts()
preprocessing_report.append("Class distribution before SMOTE:\n")
for label, count in class_distribution_before.items():
    preprocessing_report.append(f"- {label}: {count} samples ({count/len(y)*100:.2f}%)\n")

# Create a dataset for SMOTE application
X_train, X_test, y_train, y_test = train_test_split(df_selected, y_label_encoded, test_size=0.2, random_state=42)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Calculate class distribution after SMOTE
y_train_smote_series = pd.Series(y_train_smote).map({0: 'no', 1: 'yes'})
class_distribution_after = y_train_smote_series.value_counts()
preprocessing_report.append("\nClass distribution after SMOTE (training set only):\n")
for label, count in class_distribution_after.items():
    preprocessing_report.append(f"- {label}: {count} samples ({count/len(y_train_smote)*100:.2f}%)\n")

# Save SMOTE resampled data
smote_df = pd.DataFrame(X_train_smote, columns=df_selected.columns)
smote_df['y'] = y_train_smote_series.values
smote_df.to_csv('preprocessed_data/smote_resampled.csv', index=False)

# Create final datasets for modeling
preprocessing_report.append("\n### Final Preprocessed Datasets\n")
preprocessing_report.append("We created several preprocessed datasets for modeling:\n")
preprocessing_report.append("1. **Basic Preprocessed**: Missing values handled, categorical variables encoded, features scaled\n")
preprocessing_report.append("2. **PCA Transformed**: Dimensionality reduction while preserving 95% of variance\n")
preprocessing_report.append("3. **Feature Selected**: Top 10 most important features selected\n")
preprocessing_report.append("4. **SMOTE Resampled**: Class imbalance addressed using SMOTE\n")

# Save preprocessing objects for later use
preprocessing_objects = {
    'numerical_imputer': numerical_imputer,
    'encoder': encoder,
    'scaler': scaler,
    'pca': pca,
    'selector': selector,
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'selected_feature_names': selected_feature_names
}
joblib.dump(preprocessing_objects, 'preprocessed_data/preprocessing_objects.pkl')

# Save the preprocessing report
with open('preprocessing_report.md', 'w') as f:
    for line in preprocessing_report:
        f.write(line)

# Update the analysis report with preprocessing findings
print("Updating analysis report with preprocessing results...")

with open('analysis_report.md', 'a') as f:
    f.write("\n## 10. Data Preprocessing\n\n")
    
    f.write("### 10.1 Handling Missing Values\n")
    f.write("Two types of missing values were addressed:\n")
    f.write("1. **Traditional missing values**: Found in numerical columns like 'emp.var.rate' and 'euribor3m'. These were imputed using the median value of each column.\n")
    f.write("2. **'Unknown' values**: Found in categorical columns like 'job', 'education', and 'default'. These were treated as separate categories during encoding.\n\n")
    
    f.write("### 10.2 Encoding Categorical Variables\n")
    f.write("All categorical variables were encoded using one-hot encoding, which created binary features for each category (except the first category of each variable to avoid multicollinearity).\n")
    f.write(f"This increased the number of features from {len(numerical_cols) + len(categorical_cols)} to {df_preprocessed.shape[1]}.\n\n")
    
    f.write("### 10.3 Feature Scaling\n")
    f.write("Numerical features were standardized using StandardScaler to have a mean of 0 and a standard deviation of 1. This ensures that all features contribute equally to the models.\n\n")
    
    f.write("### 10.4 Feature Selection and Dimensionality Reduction\n")
    f.write("Two approaches were implemented:\n")
    f.write("1. **PCA (Principal Component Analysis)**: Reduced the dimensions while preserving 95% of the variance, resulting in fewer features.\n")
    f.write("2. **SelectKBest**: Identified the 10 most important features for predicting the target variable.\n\n")
    
    f.write("The top 10 most important features according to SelectKBest were:\n")
    for i, feature in enumerate(selected_feature_names):
        f.write(f"{i+1}. {feature}\n")
    f.write("\n")
    
    f.write("### 10.5 Handling Class Imbalance\n")
    f.write("The dataset showed significant class imbalance with approximately 89% 'no' and 11% 'yes' responses. SMOTE (Synthetic Minority Over-sampling Technique) was applied to create a balanced training dataset by generating synthetic examples of the minority class.\n\n")
    
    f.write("Class distribution before SMOTE:\n")
    for label, count in class_distribution_before.items():
        f.write(f"- {label}: {count} samples ({count/len(y)*100:.2f}%)\n")
    
    f.write("\nClass distribution after SMOTE (training set only):\n")
    for label, count in class_distribution_after.items():
        f.write(f"- {label}: {count} samples ({count/len(y_train_smote)*100:.2f}%)\n")
    
    f.write("\n### 10.6 Preprocessing Results\n")
    f.write("Multiple preprocessed datasets were created for model development:\n")
    f.write("1. **Basic Preprocessed**: Complete dataset with encoding and scaling\n")
    f.write("2. **PCA Transformed**: Dataset with reduced dimensions\n")
    f.write("3. **Feature Selected**: Dataset with only the most important features\n")
    f.write("4. **SMOTE Resampled**: Balanced dataset for addressing class imbalance\n\n")
    
    f.write("These preprocessed datasets will be used in the model development phase to evaluate which preprocessing approach yields the best predictive performance.\n")

print("Data preprocessing completed and results documented.") 