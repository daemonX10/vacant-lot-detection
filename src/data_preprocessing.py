import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load raw data from a specified file path."""
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """Clean the data by handling missing values and duplicates."""
    data = data.drop_duplicates()
    data = data.fillna(method='ffill')  # Forward fill for missing values
    return data

def preprocess_data(file_path):
    """Load and preprocess the data."""
    raw_data = load_data(file_path)
    cleaned_data = clean_data(raw_data)
    return cleaned_data

def split_data(data, target_column, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test