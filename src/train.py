"""
Model training script for the Customer Churn project.
Loads data, preprocesses it, trains a Random Forest model, and saves it.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Import our custom data processing module
from data_processing import preprocess_data


def main():
    """Main entry point for model training."""
    print("=" * 60)
    print("MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Load raw training data
    train_path = os.path.join("data", "raw", "customer_churn_dataset-training-master.csv")
    print(f"\n[Step 1] Loading raw data from: {train_path}")
    df_raw = pd.read_csv(train_path)
    
    # 2. Preprocess the data
    print("\n[Step 2] Preprocessing data...")
    df_processed = preprocess_data(df_raw)
    
    # 3. Separate Features (X) and Target (y)
    print("\n[Step 3] Separating features and target...")
    # 'Churn' is our target variable
    X = df_processed.drop(columns=['Churn', 'CustomerID']) 
    y = df_processed['Churn']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # 4. Split into Training and Validation sets (80% train, 20% validation)
    print("\n[Step 4] Splitting data into Train and Validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    
    # 5. Initialize and train the Random Forest model
    print("\n[Step 5] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        #class_weight='balanced' # Handles imbalanced data automatically
    )
    model.fit(X_train, y_train)
    print("✅ Model training completed!")
    
    # 6. Evaluate on the Validation set
    print("\n[Step 6] Evaluating model on Validation set...")
    y_pred = model.predict(X_val)
    
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # 7. Save the trained model
    print("\n[Step 7] Saving model to 'models/churn_model.pkl'...")
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, "churn_model.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Model successfully saved at: {model_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()