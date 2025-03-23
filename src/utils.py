def load_data(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def save_model(model, file_path):
    import joblib
    joblib.dump(model, file_path)

def load_model(file_path):
    import joblib
    return joblib.load(file_path)

def preprocess_data(data):
    # Implement any necessary preprocessing steps here
    return data

def split_data(data, target_column):
    from sklearn.model_selection import train_test_split
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)