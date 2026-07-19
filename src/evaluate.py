"""
Evaluation script for the Customer Churn project.
Loads the trained model and evaluates it on the unseen testing dataset.
"""

import os
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import our custom data processing module
from data_processing import preprocess_data


def main():
    """Main entry point for model evaluation."""
    print("=" * 60)
    print("MODEL EVALUATION ON UNSEEN TEST DATA")
    print("=" * 60)
    
    # 1. Load the trained model
    model_path = os.path.join("models", "churn_model.pkl")
    print(f"\n[Step 1] Loading trained model from: {model_path}")
    model = joblib.load(model_path)
    
    # 2. Load raw testing data
    test_path = os.path.join("data", "raw", "customer_churn_dataset-testing-master.csv")
    print(f"\n[Step 2] Loading raw test data from: {test_path}")
    df_test_raw = pd.read_csv(test_path)
    
    # 3. Preprocess the test data
    print("\n[Step 3] Preprocessing test data...")
    df_test_processed = preprocess_data(df_test_raw)
    
    # 4. Separate Features (X) and Target (y)
    print("\n[Step 4] Separating features and target...")
    X_test = df_test_processed.drop(columns=['Churn', 'CustomerID'])
    y_test = df_test_processed['Churn']
    
    print(f"Test Features shape: {X_test.shape}")
    print(f"Test Target shape: {y_test.shape}")
    
    # 5. Make predictions
    print("\n[Step 5] Making predictions on test data...")
    y_pred = model.predict(X_test)
    
    # 6. Evaluate and print metrics
    print("\n[Step 6] Final Evaluation Metrics:")
    print("-" * 40)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn (0)', 'Churn (1)']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nInterpretation:")
    print(f"  - True Negatives (Correctly predicted No Churn): {cm[0][0]}")
    print(f"  - False Positives (Incorrectly predicted Churn): {cm[0][1]}")
    print(f"  - False Negatives (Incorrectly predicted No Churn): {cm[1][0]}")
    print(f"  - True Positives (Correctly predicted Churn): {cm[1][1]}")
    
    print("\n" + "=" * 60)
    print("EVALUATION PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()