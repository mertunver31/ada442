import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

print("Starting model retraining with current library versions...")

# Create directories if they don't exist
if not os.path.exists('models_current'):
    os.makedirs('models_current')

# Load the dataset
try:
    print("Loading data...")
    df_original = pd.read_csv('bank-additional.xls', sep='\t')
    
    # Load preprocessed data if available
    try:
        df_selected = pd.read_csv('preprocessed_data/preprocessed_selected_features.csv')
        print("Loaded preprocessed selected features dataset.")
    except:
        print("Could not load preprocessed selected features. Will use original data.")
        df_selected = df_original.copy()
    
    # Target variable
    y_original = df_original['y'].map({'yes': 1, 'no': 0})
    
    # If using original data, do some basic preprocessing
    if df_selected.equals(df_original):
        print("Performing basic preprocessing...")
        # Handle categorical variables with one-hot encoding
        cat_cols = [col for col in df_original.columns if df_original[col].dtype == 'object']
        df_selected = pd.get_dummies(df_original.drop('y', axis=1), columns=cat_cols)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df_selected, y_original, test_size=0.3, random_state=42, stratify=y_original)
    
    print(f"Data loaded. Training set size: {len(X_train)} samples, test set size: {len(X_test)} samples")
    
    # Define the list of selected feature names for the app
    selected_feature_names = df_selected.columns.tolist()
    
    # Save selected feature names
    feature_info = {'selected_feature_names': selected_feature_names}
    joblib.dump(feature_info, 'models_current/feature_info.pkl')
    print(f"Saved feature information with {len(selected_feature_names)} features.")
    
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Define models
print("\nDefining models...")
models = {
    'Logistic Regression': LogisticRegression(C=1.0, solver='liblinear', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    'SVM': SVC(probability=True, C=1.0, kernel='rbf', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
}

# Train and save each model
print("Training and saving models...")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Save model
    model_filename = f"models_current/{name.replace(' ', '_')}.pkl"
    joblib.dump(model, model_filename)
    print(f"Saved {name} to {model_filename}")
    
    # Simple evaluation
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"{name} - Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")

# Create and save stacking classifier
print("\nTraining Stacking Classifier...")
base_estimators = [
    ('lr', models['Logistic Regression']),
    ('dt', models['Decision Tree']),
    ('rf', models['Random Forest']),
    ('xgb', models['XGBoost']),
    ('svm', models['SVM']),
    ('gb', models['Gradient Boosting'])
]

stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(random_state=42),
    cv=5,
    stack_method='predict_proba'
)

stacking.fit(X_train, y_train)
joblib.dump(stacking, 'models_current/Stacking.pkl')

# Evaluate stacking
stacking_train_score = stacking.score(X_train, y_train)
stacking_test_score = stacking.score(X_test, y_test)
print(f"Stacking Classifier - Train accuracy: {stacking_train_score:.4f}, Test accuracy: {stacking_test_score:.4f}")

print("\nModel retraining complete. All models saved with current library versions.")
print("To use these models, update the app.py file to load models from the 'models_current' directory.") 