"""
Data processing module for the Customer Churn project.
Handles missing values, categorical encoding, and feature preparation.
"""

import pandas as pd


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input pandas DataFrame.
        
    Returns:
        Cleaned pandas DataFrame.
    """
    initial_shape = df.shape
    
    # Drop rows with any missing values 
    # (Safe to do here since we only have 1 missing row out of 440k+)
    df_cleaned = df.dropna().reset_index(drop=True)
    
    print(f"✅ Cleaned missing values. Shape changed from {initial_shape} to {df_cleaned.shape}")
    return df_cleaned


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical string columns into numerical format for ML models.
    
    Args:
        df: Input pandas DataFrame.
        
    Returns:
        DataFrame with encoded categorical features.
    """
    df_encoded = df.copy()
    
    # Identify categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object', 'str']).columns.tolist()
    
    # Remove target variable ('Churn') and ID ('CustomerID') from encoding if present
    cols_to_encode = [col for col in categorical_cols if col not in ['CustomerID', 'Churn']]
    
    # One-Hot Encoding for categorical variables
    df_encoded = pd.get_dummies(df_encoded, columns=cols_to_encode, drop_first=True, dtype=int)
    
    print(f"✅ Encoded categorical columns: {cols_to_encode}")
    return df_encoded


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main preprocessing pipeline. Chains all cleaning and transformation steps.
    
    Args:
        df: Raw input pandas DataFrame.
        
    Returns:
        Fully preprocessed pandas DataFrame ready for training.
    """
    print("\n--- Starting Data Preprocessing ---")
    
    # Step 1: Clean missing values
    df_processed = clean_missing_values(df)
    
    # Step 2: Encode categorical features
    df_processed = encode_categorical_features(df_processed)
    
    print("--- Data Preprocessing Complete! ---\n")
    return df_processed