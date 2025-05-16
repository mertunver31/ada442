import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve


def load_data():
    """Load the bank marketing dataset"""
    try:
        # Try to read the dataset
        df = pd.read_csv('bank-additional.xls', sep='\t')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def preprocess_data(df, preprocessing_options):
    """Preprocess the data based on user selections"""
    # Make a copy of the dataframe
    processed_df = df.copy()
    
    # Separate features and target variable
    X = processed_df.drop('y', axis=1)
    y = processed_df['y']
    
    # Separate numerical and categorical variables
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Step 1: Handle missing values
    missing_strategy = preprocessing_options['missing_values']
    
    # Handle numerical missing values
    if missing_strategy == 'Simple Imputer (Median)':
        num_imputer = SimpleImputer(strategy='median')
        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    elif missing_strategy == 'Simple Imputer (Mean)':
        num_imputer = SimpleImputer(strategy='mean')
        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    elif missing_strategy == 'KNN Imputer':
        num_imputer = KNNImputer(n_neighbors=5)
        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    
    # For categorical features, keep 'unknown' as a separate category
    # No special handling needed as the bank dataset uses 'unknown' explicitly
    
    # Step 2: Encoding categorical variables
    encoding_strategy = preprocessing_options['categorical_encoding']
    
    if encoding_strategy == 'One-Hot Encoding':
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded_cats = encoder.fit_transform(X[categorical_cols])
        
        # Get feature names
        encoded_feature_names = []
        for i, col in enumerate(categorical_cols):
            cat_values = encoder.categories_[i][1:]  # Drop first category
            for cat in cat_values:
                encoded_feature_names.append(f"{col}_{cat}")
        
        # Create DataFrame with encoded values
        encoded_df = pd.DataFrame(encoded_cats, columns=encoded_feature_names)
        
        # Combine with numerical features
        X = pd.concat([X[numerical_cols].reset_index(drop=True), 
                       encoded_df.reset_index(drop=True)], axis=1)
    
    elif encoding_strategy == 'Label Encoding':
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    elif encoding_strategy == 'Ordinal Encoding':
        encoder = OrdinalEncoder()
        X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
    
    # Step 3: Feature scaling
    scaling_strategy = preprocessing_options['feature_scaling']
    
    if scaling_strategy == 'StandardScaler':
        scaler = StandardScaler()
        if encoding_strategy == 'Label Encoding' or encoding_strategy == 'Ordinal Encoding':
            # Scale all columns
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        else:
            # Scale only numerical columns
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    elif scaling_strategy == 'MinMaxScaler':
        scaler = MinMaxScaler()
        if encoding_strategy == 'Label Encoding' or encoding_strategy == 'Ordinal Encoding':
            # Scale all columns
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        else:
            # Scale only numerical columns
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    elif scaling_strategy == 'RobustScaler':
        scaler = RobustScaler()
        if encoding_strategy == 'Label Encoding' or encoding_strategy == 'Ordinal Encoding':
            # Scale all columns
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        else:
            # Scale only numerical columns
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Step 4: Feature selection/reduction
    feature_selection = preprocessing_options['feature_selection']
    
    if feature_selection == 'PCA':
        n_components = preprocessing_options['pca_components']
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X)
        feature_names = [f"PC{i+1}" for i in range(X_transformed.shape[1])]
        X = pd.DataFrame(X_transformed, columns=feature_names)
        # Save explained variance for later display
        explained_variance = pca.explained_variance_ratio_
    
    elif feature_selection == 'SelectKBest (ANOVA F-test)':
        k = preprocessing_options['k_best_features']
        # Convert target to numeric for feature selection
        y_numeric = LabelEncoder().fit_transform(y)
        selector = SelectKBest(f_classif, k=k)
        X_transformed = selector.fit_transform(X, y_numeric)
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        X = pd.DataFrame(X_transformed, columns=selected_features)
    
    elif feature_selection == 'SelectKBest (Mutual Information)':
        k = preprocessing_options['k_best_features']
        # Convert target to numeric for feature selection
        y_numeric = LabelEncoder().fit_transform(y)
        selector = SelectKBest(mutual_info_classif, k=k)
        X_transformed = selector.fit_transform(X, y_numeric)
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        X = pd.DataFrame(X_transformed, columns=selected_features)
    
    # Step 5: Handle class imbalance
    imbalance_strategy = preprocessing_options['imbalance_handling']
    
    # Convert target to categorical for SMOTE
    y_encoded = LabelEncoder().fit_transform(y)
    
    if imbalance_strategy == 'SMOTE':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
        X, y = X_resampled, y_resampled
    
    elif imbalance_strategy == 'Random Oversampling':
        oversampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(X, y_encoded)
        X, y = X_resampled, y_resampled
    
    elif imbalance_strategy == 'ADASYN':
        adasyn = ADASYN(random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y_encoded)
        X, y = X_resampled, y_resampled
    
    elif imbalance_strategy == 'Random Undersampling':
        undersampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X, y_encoded)
        X, y = X_resampled, y_resampled
    
    elif imbalance_strategy == 'NearMiss Undersampling':
        nearmiss = NearMiss(version=1)
        X_resampled, y_resampled = nearmiss.fit_resample(X, y_encoded)
        X, y = X_resampled, y_resampled
    
    # Return processed features and target, plus original data for reference
    return X, y, numerical_cols, categorical_cols


def train_model(X, y, model_options):
    """Train model based on user selections"""
    # Split the data
    test_size = model_options['test_size']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Select model type
    model_type = model_options['model_type']
    
    if model_type == 'Logistic Regression':
        C = model_options['log_reg_c']
        solver = model_options['log_reg_solver']
        model = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
    
    elif model_type == 'Random Forest':
        n_estimators = model_options['rf_n_estimators']
        max_depth = model_options['rf_max_depth']
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth if max_depth > 0 else None, 
            random_state=42
        )
    
    elif model_type == 'XGBoost':
        n_estimators = model_options['xgb_n_estimators']
        max_depth = model_options['xgb_max_depth']
        learning_rate = model_options['xgb_learning_rate']
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
    
    elif model_type == 'Decision Tree':
        max_depth = model_options['dt_max_depth']
        min_samples_split = model_options['dt_min_samples_split']
        model = DecisionTreeClassifier(
            max_depth=max_depth if max_depth > 0 else None,
            min_samples_split=min_samples_split,
            random_state=42
        )
    
    elif model_type == 'Support Vector Machine':
        C = model_options['svm_c']
        kernel = model_options['svm_kernel']
        model = SVC(
            C=C,
            kernel=kernel,
            probability=True,
            random_state=42
        )
    
    elif model_type == 'KNN':
        n_neighbors = model_options['knn_n_neighbors']
        weights = model_options['knn_weights']
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights
        )
    
    elif model_type == 'Gradient Boosting':
        n_estimators = model_options['gb_n_estimators']
        max_depth = model_options['gb_max_depth']
        learning_rate = model_options['gb_learning_rate']
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            learning_rate=learning_rate,
            random_state=42
        )
    
    elif model_type == 'AdaBoost':
        n_estimators = model_options['ada_n_estimators']
        learning_rate = model_options['ada_learning_rate']
        model = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save the model
    os.makedirs('models_live', exist_ok=True)
    joblib.dump(model, f'models_live/{model_type}.pkl')
    
    # Return results
    return {
        'model': model,
        'model_type': model_type,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        },
        'confusion_matrix': cm
    }


def predict_with_model(model, input_data):
    """Make a prediction with the trained model"""
    # Convert input_data to DataFrame if it's not already
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data])
    
    # Make prediction
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = model.predict(input_data)[0]
    
    return prediction, prediction_proba


def plot_training_results(results):
    """Plot training results"""
    # Create confusion matrix plot
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'Confusion Matrix - {results["model_type"]}')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Create ROC curve
    fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(fpr, tpr, label=f'ROC Curve (AUC = {results["metrics"]["roc_auc"]:.4f})')
    ax2.plot([0, 1], [0, 1], 'k--', label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'ROC Curve - {results["model_type"]}')
    ax2.legend()
    
    # Create precision-recall curve
    precision, recall, _ = precision_recall_curve(results['y_test'], results['y_pred_proba'])
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(recall, precision, label=f'Precision-Recall Curve')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title(f'Precision-Recall Curve - {results["model_type"]}')
    ax3.legend()
    
    return fig1, fig2, fig3 