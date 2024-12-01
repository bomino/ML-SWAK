from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import joblib
import os

def evaluate_model(y_true, y_pred):
    """Calculate regression metrics."""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def save_model(model, model_name, model_dir='models'):
    """Save trained model to disk."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, f'{model_name}.joblib')
    joblib.dump(model, model_path)
    return model_path

def load_model(model_name, model_dir='models'):
    """Load trained model from disk."""
    model_path = os.path.join(model_dir, f'{model_name}.joblib')
    return joblib.load(model_path)

def recommend_model(data_characteristics):
    """Recommend best model based on data characteristics."""
    n_samples = data_characteristics['n_samples']
    n_features = data_characteristics['n_features']
    
    if n_samples < 1000:
        if n_features < 10:
            return "Linear Regression"
        else:
            return "SVR"
    else:
        if n_features < 10:
            return "Gradient Boosting"
        else:
            return "Random Forest"