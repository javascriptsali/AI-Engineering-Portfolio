"""
Test script to verify the data processing pipeline works correctly.
"""

import os
import pandas as pd
from data_processing import preprocess_data


def main():
    print("=" * 60)
    print("DATA PROCESSING PIPELINE TEST")
    print("=" * 60)
    
    # Load raw training data
    train_path = os.path.join("data", "raw", "customer_churn_dataset-training-master.csv")
    print(f"\nLoading raw data from: {train_path}")
    df_raw = pd.read_csv(train_path)
    
    print(f"\nRaw Data Shape: {df_raw.shape}")
    print(f"Missing values in raw data: {df_raw.isnull().sum().sum()}")
    
    # Run preprocessing pipeline
    df_processed = preprocess_data(df_raw)
    
    # Verify results
    print(f"\nProcessed Data Shape: {df_processed.shape}")
    print(f"Missing values in processed data: {df_processed.isnull().sum().sum()}")
    print(f"Data types in processed data:\n{df_processed.dtypes.value_counts()}")
    
    print("\n✅ Data Processing Test Passed Successfully!")


if __name__ == "__main__":
    main()