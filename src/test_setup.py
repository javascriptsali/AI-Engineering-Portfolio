"""
Test script to verify project setup and data loading.
Checks if both training and testing datasets are present and loads them.
"""

import os
import pandas as pd
from dotenv import load_dotenv


def main():
    """Main entry point of the script."""
    print("=" * 60)
    print("PROJECT SETUP & DATA LOADING VERIFICATION")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    print("\n[Step 1] Environment variables loaded.")
    
    # Define dataset paths
    train_filename = "customer_churn_dataset-training-master.csv"
    test_filename = "customer_churn_dataset-testing-master.csv"
    
    train_path = os.path.join("data", "raw", train_filename)
    test_path = os.path.join("data", "raw", test_filename)
    
    print(f"\n[Step 2] Loading datasets from 'data/raw/'...")
    
    try:
        # Check if files exist
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training dataset not found at: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Testing dataset not found at: {test_path}")
            
        # Load data into pandas DataFrames
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        print(f"\n✅ Datasets loaded successfully!")
        print(f"Training set -> Rows: {df_train.shape[0]} | Columns: {df_train.shape[1]}")
        print(f"Testing set  -> Rows: {df_test.shape[0]} | Columns: {df_test.shape[1]}")
        
        print("\n--- First 5 rows of Training Data ---")
        print(df_train.head())
        
        print("\n--- Data Types & Missing Values (Training Set) ---")
        df_train.info()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please make sure both CSV files are inside the 'data/raw/' folder.")


if __name__ == "__main__":
    main()