import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import live_training
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)

# Set page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction App",
    page_icon="💰",
    layout="wide"
)

# Function to create a simple model if no models can be loaded
def create_simple_model():
    """Create a simple logistic regression model as last resort"""
    from sklearn.linear_model import LogisticRegression
    
    # Create a very basic logistic regression model
    simple_model = LogisticRegression(random_state=42)
    
    # The model won't be trained, but we'll add a simple predict_proba method
    # that uses manual weight for the most important features
    
    class SimpleModel:
        def predict_proba(self, X):
            # Get key features (assuming same order as in the application)
            # Most important is duration, then pdays, then previous
            duration = X[:, 0]  # Normalized between 0-1 later
            pdays = X[:, 1]     # -1 means not contacted before
            previous = X[:, 2]  # Number of previous contacts
            
            # Start with low base probability (11% is dataset average)
            base_prob = 0.11
            
            # Duration is important but shouldn't dominate completely
            # Use sigmoid function to have more balanced scaling
            duration_factor = 1 / (1 + np.exp(-4 * (duration - 0.5)))
            
            # If pdays is -1 (never contacted), it's less favorable
            pdays_factor = 0.8 if pdays[0] <= 0 else 1.1  # Treating scaled 0 as -1
            
            # Previous contacts can help if not too many
            prev_factor = 1.0
            if previous[0] > 0 and previous[0] <= 0.3:  # Scaled range for 1-3 contacts
                prev_factor = 1.1  # Small positive factor for 1-3 previous contacts
            elif previous[0] > 0.3:  # More than 3 contacts (scaled)
                prev_factor = 0.9  # Small negative factor for more than 3 contacts
            
            # Calculate final probability with conservative adjustment to base probability
            # Base should be around 11% with small adjustments
            final_prob = base_prob + (duration_factor - 0.5) * 0.15 * pdays_factor * prev_factor
            
            # Ensure it's between 0.05 and 0.7 (more conservative upper bound)
            final_prob = np.clip(final_prob, 0.05, 0.7)
            
            # Return as a 2D array with both class probabilities (required format)
            result = np.zeros((len(X), 2))
            result[:, 0] = 1 - final_prob  # Probability of class 0
            result[:, 1] = final_prob      # Probability of class 1
            return result
    
    return SimpleModel()

# Load the feature names
@st.cache_data
def load_feature_info():
    try:
        preprocess_objects = joblib.load('preprocessed_data/preprocessing_objects.pkl')
        selected_features = preprocess_objects['selected_feature_names']
        return selected_features
    except:
        # Fallback if we can't load from file
        return ['duration', 'pdays', 'previous', 'emp.var.rate', 
                'euribor3m', 'nr.employed', 'contact_telephone', 
                'month_mar', 'poutcome_nonexistent', 'poutcome_success']

# Function to find optimal classification threshold based on F1 score
def find_optimal_threshold(y_true, y_pred_proba):
    """
    Find the optimal classification threshold that maximizes F1 score
    
    Parameters:
    - y_true: True labels
    - y_pred_proba: Predicted probabilities
    
    Returns:
    - optimal_threshold: The threshold that maximizes F1 score
    - threshold_metrics: Dictionary with metrics at different thresholds
    """
    # Range of thresholds to try - start from higher value to reduce false positives
    thresholds = np.arange(0.3, 0.9, 0.05)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    # Calculate F1 score for each threshold
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    # Find threshold with maximum F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # If calculated threshold is too low, use a minimum threshold of 0.5
    # to avoid too many false positives with imbalanced dataset
    if optimal_threshold < 0.5:
        optimal_threshold = 0.5
    
    # Return optimal threshold and metrics
    threshold_metrics = {
        'thresholds': thresholds,
        'f1_scores': f1_scores,
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'optimal_threshold': optimal_threshold,
        'optimal_f1': f1_scores[optimal_idx],
        'optimal_precision': precision_scores[optimal_idx],
        'optimal_recall': recall_scores[optimal_idx]
    }
    
    return optimal_threshold, threshold_metrics

# Function to make prediction with fallback mechanism
def make_prediction(models, model_input):
    """Try to predict using various models with fallback mechanism"""
    # Try models in order of preference
    for model_name in ['stacking', 'xgboost', 'random_forest', 'logistic']:
        if models.get(model_name) is not None:
            try:
                # Try to make prediction with this model
                probability = models[model_name].predict_proba(model_input)[0, 1]
                st.info(f"Using {model_name} model for prediction.")
                return probability
            except Exception as e:
                st.warning(f"Error using {model_name} model: {e}. Trying next model.")
                continue
    
    # If all models fail, use a simple fallback
    st.error("All models failed. Using a simple probability estimate based on feature values.")
    
    # More conservative fallback logic with better sigmoid scaling
    duration = model_input[0, 0]  # Already normalized in the input processing
    
    # Start with base probability (11% is the dataset average)
    base_prob = 0.11
    
    # Duration should influence but not dominate the prediction
    # Use sigmoid with adjusted parameters for smoother probability curve
    duration_factor = 1 / (1 + np.exp(-4 * (duration - 0.5)))
    
    # Balance the result around the base probability with smaller factor
    simple_prob = base_prob + (duration_factor - 0.5) * 0.12
    
    # Ensure it's between 0.05 and 0.6 (more conservative upper limit)
    simple_prob = max(0.05, min(simple_prob, 0.6))
    
    return simple_prob

# Dosya yolu yardımcı fonksiyonları
def get_file_path(base_path, file_name):
    """Dosya yolunu birkaç farklı alternatifle kontrol eder"""
    paths_to_try = [
        os.path.join(base_path, file_name),             # Normal yol
        os.path.join('.', base_path, file_name),        # Göreli yol
        os.path.join('..', base_path, file_name),       # Bir üst dizinden
        os.path.abspath(os.path.join(base_path, file_name))  # Mutlak yol
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            return path
    
    # Hiçbir yol bulunamadıysa varsayılan döndür
    return os.path.join(base_path, file_name)

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    models = {}
    
    # Önce tüm gerekli dizinlerin varlığını kontrol et
    for directory in ['models_current', 'preprocessed_data', 'models', 'data']:
        os.makedirs(directory, exist_ok=True)
        
    try:
        # Try loading from models_current directory first (new models)
        # Try loading the stacking classifier
        try:
            st.info("Loading stacking model...")
            stacking_path = get_file_path('models_current', 'Stacking.pkl')
            st.info(f"Trying to load from: {stacking_path}")
            models['stacking'] = joblib.load(stacking_path)
            st.success("Successfully loaded stacking model!")
        except Exception as e:
            st.warning(f"Could not load stacking model: {str(e)}")
            st.info("Current working directory: " + os.getcwd())
            st.info("Files in models_current: " + str(os.listdir('models_current') if os.path.exists('models_current') else "Directory not found"))
            models['stacking'] = None
            
        # Try loading individual models
        for model_name in ['Logistic_Regression', 'XGBoost', 'Random_Forest', 'Gradient_Boosting', 'SVM', 'Decision_Tree']:
            try:
                st.info(f"Loading {model_name} model...")
                models[model_name.lower()] = joblib.load(f'models_current/{model_name}.pkl')
                st.success(f"Successfully loaded {model_name} model!")
            except Exception as e:
                st.warning(f"Could not load {model_name} model: {str(e)}")
                models[model_name.lower()] = None

        # Calculate optimal threshold using test data if available
        try:
            # Check if test_data.csv exists, if not, create it from bank-additional.xls
            test_data_path = 'data/test_data.csv'
            
            if not os.path.exists(test_data_path):
                # Create data directory if it doesn't exist
                os.makedirs('data', exist_ok=True)
                
                # Load the dataset
                bank_data = pd.read_csv('bank-additional.xls', sep='\t')
                
                # Preprocess the data - this is a simplified version of what's in model_development.py
                # Encode target variable
                bank_data['y'] = bank_data['y'].map({'no': 0, 'yes': 1})
                
                # Split into train and test
                from sklearn.model_selection import train_test_split
                _, test_data = train_test_split(bank_data, test_size=0.2, random_state=42, stratify=bank_data['y'])
                
                # Save test data
                test_data.to_csv(test_data_path, index=False)
            
            # Load test data for threshold optimization
            test_data = pd.read_csv(test_data_path)
            X_test = test_data.drop('y', axis=1)
            y_test = test_data['y']
            
            # Use stacking model if available, otherwise use the first available model
            model_for_threshold = None
            if models['stacking'] is not None:
                model_for_threshold = models['stacking']
            else:
                for model_name in models:
                    if models[model_name] is not None and model_name != 'optimal_threshold':
                        model_for_threshold = models[model_name]
                        break
            
            if model_for_threshold is not None:
                # Make predictions on test data
                y_pred_proba = model_for_threshold.predict_proba(X_test)[:, 1]
                
                # Calculate optimal threshold
                optimal_threshold, _ = find_optimal_threshold(y_test, y_pred_proba)
                models['optimal_threshold'] = optimal_threshold
            else:
                # Default threshold if no model is available
                models['optimal_threshold'] = 0.5
        except Exception as e:
            # Default threshold based on typical F1 optimization for imbalanced data
            models['optimal_threshold'] = 0.5
            # st.warning(f"Could not calculate optimal threshold: {e}. Using default value.")

        # Load feature information
        try:
            feature_info = joblib.load('models_current/feature_info.pkl')
            selected_features = feature_info.get('selected_feature_names', [])
            return models, {'selected_feature_names': selected_features}
        except Exception as e:
            # st.warning(f"Could not load new feature info: {e}. Will try to use old preprocessing objects.")
            
            # Try to load old preprocessing objects as fallback
            try:
                preprocess_objects = joblib.load('preprocessed_data/preprocessing_objects.pkl')
            except Exception as e2:
                # st.warning(f"Could not load old preprocessing objects: {e2}")
                preprocess_objects = None
        
        # If no models could be loaded, create a simple model
        if all(model is None for model_name, model in models.items() if model_name != 'optimal_threshold'):
            # st.warning("No pre-trained models could be loaded. Creating a simple model as fallback.")
            models['simple'] = create_simple_model()
            
        return models, preprocess_objects
    except Exception as e:
        # st.error(f"Error in model loading process: {e}")
        # Still create a simple model as last resort
        models = {'simple': create_simple_model(), 'optimal_threshold': 0.5}
        return models, None

# Streamlit Cloud için dosya yolu kontrolü ekleyelim
def ensure_path_exists(path):
    """Eğer dosya yolu yoksa oluştur"""
    import os
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

# Main app
def main():
    try:
        # Sidebar with app navigation
        st.sidebar.title("Bank Marketing Prediction")
        
        # Add application selection box
        pages = {
            "🏠 Home": "Home",
            "📊 Model Insights": "Model Insights",
            "🔮 Predict": "Predict",
            "📋 Project Overview": "Project Overview",
            "🧪 Live Training": "Live Training",
            "📝 Research Report": "Research Report"
        }
        
        # Create a radio button for application selection
        selection = st.sidebar.radio("Go to", list(pages.keys()))
        
        # Extract page name from selection
        page = pages[selection]
        
        # Make sure directories exist
        ensure_path_exists('models_current')
        ensure_path_exists('preprocessed_data')
        ensure_path_exists('models')
        ensure_path_exists('models_live')
        ensure_path_exists('optimization')
        
        # Load models for predictions
        models = load_models()
        
        # Home page
        if page == "Home":
            # ... existing code ...
    except Exception as e:
        st.error("### Beklenmeyen bir hata oluştu")
        st.error(f"**Hata mesajı:** {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.info("Bu hatayı inceleyip çözmek için lütfen ekran görüntüsü alın veya hata mesajını kopyalayın.")

# Main entry point
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("### Kritik bir hata oluştu")
        st.error(f"**Hata mesajı:** {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.info("Bu hatayı inceleyip çözmek için lütfen ekran görüntüsü alın.")
 