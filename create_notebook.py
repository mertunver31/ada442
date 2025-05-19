import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Bank Marketing Prediction Pipeline\n",
                "\n",
                "This notebook demonstrates the process of building a machine learning model to predict whether a customer will subscribe to a term deposit based on marketing campaign data from a Portuguese banking institution."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Initial Setup\n",
                "\n",
                "First, we import all the necessary libraries and packages for our pipeline."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import essential libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder\n",
                "from sklearn.impute import SimpleImputer\n",
                "from sklearn.decomposition import PCA\n",
                "from sklearn.feature_selection import SelectKBest, f_classif\n",
                "from imblearn.over_sampling import SMOTE\n",
                "from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.tree import DecisionTreeClassifier\n",
                "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier\n",
                "from sklearn.svm import SVC\n",
                "from xgboost import XGBClassifier\n",
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\n",
                "from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve\n",
                "import os\n",
                "import joblib\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Set random seed for reproducibility\n",
                "np.random.seed(42)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Data Loading\n",
                "\n",
                "We load the Bank Marketing dataset which contains information about direct marketing campaigns of a Portuguese banking institution."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the dataset\n",
                "print(\"Loading the dataset...\")\n",
                "try:\n",
                "    df = pd.read_csv('bank-additional.xls', sep='\\t')\n",
                "    print(\"Successfully loaded the dataset.\")\n",
                "except Exception as e:\n",
                "    print(f\"Failed to read the file: {str(e)}\")\n",
                "\n",
                "# Display basic information about the dataset\n",
                "print(f\"Dataset shape: {df.shape}\")\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Data Exploration\n",
                "\n",
                "Let's explore the dataset to understand its structure, feature distributions, and relationships."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check data types and missing values\n",
                "print(\"Data types:\")\n",
                "print(df.dtypes)\n",
                "\n",
                "print(\"\\nMissing values:\")\n",
                "print(df.isnull().sum())\n",
                "\n",
                "# Check for 'unknown' values in categorical columns\n",
                "categorical_cols = df.select_dtypes(include=['object']).columns\n",
                "print(\"\\nUnknown values in categorical columns:\")\n",
                "for col in categorical_cols:\n",
                "    unknown_count = (df[col] == 'unknown').sum()\n",
                "    if unknown_count > 0:\n",
                "        print(f\"{col}: {unknown_count} ({unknown_count/len(df)*100:.2f}%)\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Analyze target variable distribution\n",
                "target_counts = df['y'].value_counts()\n",
                "print(\"Target variable distribution:\")\n",
                "print(target_counts)\n",
                "print(f\"Positive class percentage: {target_counts['yes']/len(df)*100:.2f}%\")\n",
                "\n",
                "# Visualize target distribution\n",
                "plt.figure(figsize=(8, 6))\n",
                "sns.countplot(x='y', data=df)\n",
                "plt.title('Target Variable Distribution')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Data Preprocessing\n",
                "\n",
                "Now we'll preprocess the data to prepare it for model training. This includes handling missing values, encoding categorical variables, and scaling features."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Fix specific columns with numeric data\n",
                "numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', \n",
                "                   'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', \n",
                "                   'euribor3m', 'nr.employed']\n",
                "\n",
                "# Convert numeric columns from string to proper numeric types\n",
                "for col in df.columns:\n",
                "    if col in numeric_columns:\n",
                "        try:\n",
                "            if df[col].dtype == 'object':\n",
                "                # Replace commas with periods for decimal values\n",
                "                df[col] = df[col].str.replace(',', '.') if isinstance(df[col].iloc[0], str) else df[col]\n",
                "                # Convert to numeric\n",
                "                df[col] = pd.to_numeric(df[col], errors='coerce')\n",
                "        except Exception as e:\n",
                "            print(f\"Could not convert {col} to numeric: {e}\")\n",
                "\n",
                "# Separate features and target variable\n",
                "X = df.drop('y', axis=1)\n",
                "y = df['y']\n",
                "\n",
                "# Separate numerical and categorical variables\n",
                "numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()\n",
                "categorical_cols = X.select_dtypes(include=['object']).columns.tolist()\n",
                "\n",
                "print(f\"Numerical columns: {numerical_cols}\")\n",
                "print(f\"Categorical columns: {categorical_cols}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Handle missing values\n",
                "df_preprocessed = df.copy()\n",
                "\n",
                "# For numerical features, replace missing values with the median\n",
                "numerical_imputer = SimpleImputer(strategy='median')\n",
                "df_preprocessed[numerical_cols] = numerical_imputer.fit_transform(df_preprocessed[numerical_cols])\n",
                "\n",
                "# Encode categorical variables\n",
                "encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')\n",
                "encoded_cats = encoder.fit_transform(df_preprocessed[categorical_cols])\n",
                "\n",
                "# Get the feature names from one-hot encoding\n",
                "encoded_feature_names = []\n",
                "for i, col in enumerate(categorical_cols):\n",
                "    cat_values = encoder.categories_[i][1:]  # Drop first category\n",
                "    for cat in cat_values:\n",
                "        encoded_feature_names.append(f\"{col}_{cat}\")\n",
                "\n",
                "# Create a DataFrame with the encoded values\n",
                "encoded_df = pd.DataFrame(encoded_cats, columns=encoded_feature_names)\n",
                "\n",
                "# Combine with numerical features\n",
                "df_preprocessed = pd.concat([df_preprocessed[numerical_cols].reset_index(drop=True), \n",
                "                         encoded_df.reset_index(drop=True)], axis=1)\n",
                "\n",
                "print(f\"After one-hot encoding, the number of features increased from {len(numerical_cols) + len(categorical_cols)} to {df_preprocessed.shape[1]}.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Feature scaling\n",
                "scaler = StandardScaler()\n",
                "df_preprocessed[numerical_cols] = scaler.fit_transform(df_preprocessed[numerical_cols])\n",
                "\n",
                "print(\"Numerical features were scaled to have mean=0 and standard deviation=1.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Feature selection - SelectKBest\n",
                "X_for_select = df_preprocessed\n",
                "y_label_encoded = LabelEncoder().fit_transform(y)\n",
                "\n",
                "# Apply SelectKBest to find the top features\n",
                "selector = SelectKBest(f_classif, k=10)  # Select top 10 features\n",
                "X_selected = selector.fit_transform(X_for_select, y_label_encoded)\n",
                "\n",
                "# Get selected feature names\n",
                "selected_indices = selector.get_support(indices=True)\n",
                "selected_feature_names = X_for_select.columns[selected_indices].tolist()\n",
                "\n",
                "print(f\"SelectKBest identified the top 10 most important features:\")\n",
                "for i, feature in enumerate(selected_feature_names):\n",
                "    print(f\"{i+1}. {feature}\")\n",
                "\n",
                "# Create DataFrame with selected features\n",
                "df_selected = X_for_select.iloc[:, selected_indices]"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Handle class imbalance with SMOTE\n",
                "smote = SMOTE(random_state=42)\n",
                "X_smote, y_smote = smote.fit_resample(df_selected, y_label_encoded)\n",
                "\n",
                "print(f\"SMOTE resampling: Original shape {df_selected.shape[0]} -> Resampled shape {X_smote.shape[0]}\")\n",
                "print(f\"Class distribution after SMOTE: 0={sum(y_smote==0)}, 1={sum(y_smote==1)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Model Development\n",
                "\n",
                "Now we develop and train multiple machine learning models to predict whether a customer will subscribe to a term deposit."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Split the data into training and testing sets\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    df_selected, y_label_encoded, test_size=0.3, random_state=42, stratify=y_label_encoded)\n",
                "\n",
                "print(f\"Training set size: {len(X_train)} samples\")\n",
                "print(f\"Testing set size: {len(X_test)} samples\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define models and their parameter grids for hyperparameter optimization\n",
                "models = {\n",
                "    'Logistic Regression': {\n",
                "        'model': LogisticRegression(max_iter=1000, random_state=42),\n",
                "        'params': {\n",
                "            'C': [0.01, 0.1, 1, 10],\n",
                "            'solver': ['liblinear', 'saga'],\n",
                "            'class_weight': [None, 'balanced']\n",
                "        }\n",
                "    },\n",
                "    'Random Forest': {\n",
                "        'model': RandomForestClassifier(random_state=42),\n",
                "        'params': {\n",
                "            'n_estimators': [50, 100, 200],\n",
                "            'max_depth': [5, 10, None],\n",
                "            'min_samples_split': [2, 5],\n",
                "            'min_samples_leaf': [1, 2],\n",
                "            'class_weight': [None, 'balanced', 'balanced_subsample']\n",
                "        }\n",
                "    },\n",
                "    'XGBoost': {\n",
                "        'model': XGBClassifier(random_state=42),\n",
                "        'params': {\n",
                "            'n_estimators': [50, 100, 200],\n",
                "            'max_depth': [3, 5, 7],\n",
                "            'learning_rate': [0.01, 0.1, 0.2],\n",
                "            'subsample': [0.8, 1.0],\n",
                "            'colsample_bytree': [0.8, 1.0],\n",
                "            'scale_pos_weight': [1, 3, 8]  # to handle class imbalance\n",
                "        }\n",
                "    }\n",
                "}"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train a Logistic Regression model\n",
                "print(\"Training Logistic Regression...\")\n",
                "lr_model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42, solver='liblinear')\n",
                "lr_model.fit(X_smote, y_smote)\n",
                "\n",
                "# Make predictions\n",
                "y_pred = lr_model.predict(X_test)\n",
                "y_pred_proba = lr_model.predict_proba(X_test)[:, 1]\n",
                "\n",
                "# Calculate metrics\n",
                "accuracy = accuracy_score(y_test, y_pred)\n",
                "precision = precision_score(y_test, y_pred)\n",
                "recall = recall_score(y_test, y_pred)\n",
                "f1 = f1_score(y_test, y_pred)\n",
                "roc_auc = roc_auc_score(y_test, y_pred_proba)\n",
                "\n",
                "print(f\"Logistic Regression Results:\")\n",
                "print(f\"Accuracy: {accuracy:.4f}\")\n",
                "print(f\"Precision: {precision:.4f}\")\n",
                "print(f\"Recall: {recall:.4f}\")\n",
                "print(f\"F1 Score: {f1:.4f}\")\n",
                "print(f\"ROC AUC: {roc_auc:.4f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train a Random Forest model\n",
                "print(\"Training Random Forest...\")\n",
                "rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, \n",
                "                                 min_samples_leaf=1, class_weight='balanced', random_state=42)\n",
                "rf_model.fit(X_smote, y_smote)\n",
                "\n",
                "# Make predictions\n",
                "y_pred = rf_model.predict(X_test)\n",
                "y_pred_proba = rf_model.predict_proba(X_test)[:, 1]\n",
                "\n",
                "# Calculate metrics\n",
                "accuracy = accuracy_score(y_test, y_pred)\n",
                "precision = precision_score(y_test, y_pred)\n",
                "recall = recall_score(y_test, y_pred)\n",
                "f1 = f1_score(y_test, y_pred)\n",
                "roc_auc = roc_auc_score(y_test, y_pred_proba)\n",
                "\n",
                "print(f\"Random Forest Results:\")\n",
                "print(f\"Accuracy: {accuracy:.4f}\")\n",
                "print(f\"Precision: {precision:.4f}\")\n",
                "print(f\"Recall: {recall:.4f}\")\n",
                "print(f\"F1 Score: {f1:.4f}\")\n",
                "print(f\"ROC AUC: {roc_auc:.4f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train an XGBoost model\n",
                "print(\"Training XGBoost...\")\n",
                "xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, \n",
                "                         subsample=0.8, colsample_bytree=0.8, scale_pos_weight=3, random_state=42)\n",
                "xgb_model.fit(X_smote, y_smote)\n",
                "\n",
                "# Make predictions\n",
                "y_pred = xgb_model.predict(X_test)\n",
                "y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]\n",
                "\n",
                "# Calculate metrics\n",
                "accuracy = accuracy_score(y_test, y_pred)\n",
                "precision = precision_score(y_test, y_pred)\n",
                "recall = recall_score(y_test, y_pred)\n",
                "f1 = f1_score(y_test, y_pred)\n",
                "roc_auc = roc_auc_score(y_test, y_pred_proba)\n",
                "\n",
                "print(f\"XGBoost Results:\")\n",
                "print(f\"Accuracy: {accuracy:.4f}\")\n",
                "print(f\"Precision: {precision:.4f}\")\n",
                "print(f\"Recall: {recall:.4f}\")\n",
                "print(f\"F1 Score: {f1:.4f}\")\n",
                "print(f\"ROC AUC: {roc_auc:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Model Ensemble (Stacking)\n",
                "\n",
                "Now we'll build an ensemble model using stacking to combine the strengths of individual models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create a stacking classifier\n",
                "estimators = [\n",
                "    ('lr', lr_model),\n",
                "    ('rf', rf_model),\n",
                "    ('xgb', xgb_model)\n",
                "]\n",
                "\n",
                "stacking_model = StackingClassifier(\n",
                "    estimators=estimators,\n",
                "    final_estimator=LogisticRegression(),\n",
                "    cv=5,\n",
                "    stack_method='predict_proba',\n",
                "    n_jobs=-1\n",
                ")\n",
                "\n",
                "# Train the stacking classifier\n",
                "print(\"Training Stacking Classifier...\")\n",
                "stacking_model.fit(X_train, y_train)\n",
                "\n",
                "# Make predictions\n",
                "y_pred = stacking_model.predict(X_test)\n",
                "y_pred_proba = stacking_model.predict_proba(X_test)[:, 1]\n",
                "\n",
                "# Calculate metrics\n",
                "accuracy = accuracy_score(y_test, y_pred)\n",
                "precision = precision_score(y_test, y_pred)\n",
                "recall = recall_score(y_test, y_pred)\n",
                "f1 = f1_score(y_test, y_pred)\n",
                "roc_auc = roc_auc_score(y_test, y_pred_proba)\n",
                "\n",
                "print(f\"Stacking Classifier Results:\")\n",
                "print(f\"Accuracy: {accuracy:.4f}\")\n",
                "print(f\"Precision: {precision:.4f}\")\n",
                "print(f\"Recall: {recall:.4f}\")\n",
                "print(f\"F1 Score: {f1:.4f}\")\n",
                "print(f\"ROC AUC: {roc_auc:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Business Optimization\n",
                "\n",
                "Now we'll optimize the model for business metrics. We'll calculate expected profit and ROI for different probability thresholds."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define business parameters\n",
                "call_cost = 5  # Cost of making a call in euros\n",
                "conversion_profit = 100  # Profit from a successful conversion in euros\n",
                "\n",
                "# Function to calculate business metrics\n",
                "def calculate_business_metrics(y_true, y_pred_proba, threshold):\n",
                "    # Predict using the threshold\n",
                "    y_pred = (y_pred_proba >= threshold).astype(int)\n",
                "    \n",
                "    # Calculate confusion matrix elements\n",
                "    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()\n",
                "    \n",
                "    # Calculate costs and profits\n",
                "    total_calls = tp + fp  # Total number of predicted positives\n",
                "    call_costs = total_calls * call_cost\n",
                "    conversion_profits = tp * conversion_profit\n",
                "    net_profit = conversion_profits - call_costs\n",
                "    \n",
                "    # Calculate ROI\n",
                "    roi = net_profit / call_costs if call_costs > 0 else 0\n",
                "    \n",
                "    # Calculate other metrics\n",
                "    accuracy = (tp + tn) / (tp + tn + fp + fn)\n",
                "    precision = tp / (tp + fp) if (tp + fp) > 0 else 0\n",
                "    recall = tp / (tp + fn) if (tp + fn) > 0 else 0\n",
                "    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0\n",
                "    \n",
                "    print(f\"Business Metrics at threshold {threshold:.2f}:\")\n",
                "    print(f\"Accuracy: {accuracy:.4f}\")\n",
                "    print(f\"Precision: {precision:.4f}\")\n",
                "    print(f\"Recall: {recall:.4f}\")\n",
                "    print(f\"F1 Score: {f1:.4f}\")\n",
                "    print(f\"True Positives: {tp}\")\n",
                "    print(f\"False Positives: {fp}\")\n",
                "    print(f\"Total Calls: {total_calls}\")\n",
                "    print(f\"Call Costs: €{call_costs:.2f}\")\n",
                "    print(f\"Conversion Profits: €{conversion_profits:.2f}\")\n",
                "    print(f\"Net Profit: €{net_profit:.2f}\")\n",
                "    print(f\"ROI: {roi:.2f}\")\n",
                "    \n",
                "    return {\n",
                "        'threshold': threshold,\n",
                "        'accuracy': accuracy,\n",
                "        'precision': precision,\n",
                "        'recall': recall,\n",
                "        'f1': f1,\n",
                "        'true_positives': tp,\n",
                "        'false_positives': fp,\n",
                "        'total_calls': total_calls,\n",
                "        'call_costs': call_costs,\n",
                "        'conversion_profits': conversion_profits,\n",
                "        'net_profit': net_profit,\n",
                "        'roi': roi\n",
                "    }"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Calculate business metrics for the stacking model with different thresholds\n",
                "y_pred_proba = stacking_model.predict_proba(X_test)[:, 1]\n",
                "\n",
                "# Calculate metrics for thresholds 0.1, 0.3, and 0.5\n",
                "results_01 = calculate_business_metrics(y_test, y_pred_proba, 0.1)\n",
                "print(\"\\n\")\n",
                "results_03 = calculate_business_metrics(y_test, y_pred_proba, 0.3)\n",
                "print(\"\\n\")\n",
                "results_05 = calculate_business_metrics(y_test, y_pred_proba, 0.5)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Save the Final Model\n",
                "\n",
                "Finally, we save the optimized model for deployment."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create a directory for models if it doesn't exist\n",
                "if not os.path.exists('models'):\n",
                "    os.makedirs('models')\n",
                "\n",
                "# Save the stacking model\n",
                "joblib.dump(stacking_model, 'models/stacking_model.pkl')\n",
                "\n",
                "# Save preprocessing objects\n",
                "preprocessing_objects = {\n",
                "    'numerical_imputer': numerical_imputer,\n",
                "    'encoder': encoder,\n",
                "    'scaler': scaler,\n",
                "    'selector': selector,\n",
                "    'selected_feature_names': selected_feature_names,\n",
                "    'numerical_cols': numerical_cols,\n",
                "    'categorical_cols': categorical_cols\n",
                "}\n",
                "joblib.dump(preprocessing_objects, 'models/preprocessing_objects.pkl')\n",
                "\n",
                "print(\"Model and preprocessing objects saved successfully.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Conclusion\n",
                "\n",
                "We've successfully built a machine learning pipeline for predicting bank marketing success. The model can be used to prioritize which customers to contact in marketing campaigns, increasing efficiency and ROI.\n",
                "\n",
                "Key achievements:\n",
                "1. Comprehensive data preprocessing including handling missing values, encoding categorical variables, and scaling\n",
                "2. Feature selection to identify the most important predictors\n",
                "3. Model development with multiple algorithms\n",
                "4. Ensemble modeling with stacking to improve prediction performance\n",
                "5. Business optimization to maximize profit and ROI\n",
                "\n",
                "The final model is ready for deployment in a production environment, such as a Streamlit app for real-time predictions."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook to a file
with open('bank_marketing_prediction_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2) 