import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def handle_missing_values(df):
    """Handle missing values in the dataframe."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Fill numeric columns with median
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Fill categorical columns with mode
    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
    
    return df

def encode_categorical_features(df):
    """Encode categorical features."""
    categorical_columns = df.select_dtypes(include=['object']).columns
    encoders = {}
    
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        encoders[column] = le
    
    return df, encoders

def prepare_time_series(df, date_column):
    """Prepare dataframe for time series analysis."""
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)
    df = df.set_index(date_column)
    
    return df