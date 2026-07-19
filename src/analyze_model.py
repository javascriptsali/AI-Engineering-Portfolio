"""
Model analysis script to understand what the model has learned.
Checks feature importance and identifies potential issues.
"""

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    """Main entry point for model analysis."""
    print("=" * 60)
    print("MODEL ANALYSIS - FEATURE IMPORTANCE")
    print("=" * 60)
    
    # 1. Load the trained model
    model_path = os.path.join("models", "churn_model.pkl")
    print(f"\n[Step 1] Loading trained model from: {model_path}")
    model = joblib.load(model_path)
    
    # 2. Load and preprocess training data to get feature names
    train_path = os.path.join("data", "raw", "customer_churn_dataset-training-master.csv")
    df_train = pd.read_csv(train_path)
    
    # Import preprocessing function
    from data_processing import preprocess_data
    df_processed = preprocess_data(df_train)
    
    # Get feature names (exclude Churn and CustomerID)
    feature_names = df_processed.drop(columns=['Churn', 'CustomerID']).columns.tolist()
    
    # 3. Get feature importances
    print("\n[Step 2] Extracting feature importances...")
    importances = model.feature_importances_
    
    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10))
    
    # 4. Visualize feature importances
    print("\n[Step 3] Creating visualization...")
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
    plt.title('Top 15 Feature Importances in Random Forest Model')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join("docs", "feature_importance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Feature importance plot saved at: {plot_path}")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("MODEL ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()