"""
Prediction script for the Customer Churn project.
Loads the trained model and predicts churn for a single new customer.
"""

import os
import pandas as pd
import joblib


def predict_churn(customer_data: dict) -> dict:
    """
    Predicts the churn probability and class for a single customer.
    
    Args:
        customer_data: A dictionary containing the customer's raw features.
        
    Returns:
        A dictionary with the prediction result and probabilities.
    """
    # 1. Load the trained model
    model_path = os.path.join("models", "churn_model.pkl")
    model = joblib.load(model_path)
    
    # 2. Convert dictionary to DataFrame
    df = pd.DataFrame([customer_data])
    
    # 3. Preprocess the single row (Matching the training pipeline)
    # Drop CustomerID if it exists
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
        
    # One-Hot Encoding for categorical variables 
    # (Manual mapping for single row stability to avoid missing columns)
    
    # Gender encoding
    df['Gender_Male'] = 1 if df['Gender'].values[0] == 'Male' else 0
    df = df.drop(columns=['Gender'])
    
    # Subscription Type encoding
    sub_type = df['Subscription Type'].values[0]
    df['Subscription Type_Standard'] = 1 if sub_type == 'Standard' else 0
    df['Subscription Type_Premium'] = 1 if sub_type == 'Premium' else 0
    df = df.drop(columns=['Subscription Type'])
    
    # Contract Length encoding
    contract = df['Contract Length'].values[0]
    df['Contract Length_Monthly'] = 1 if contract == 'Monthly' else 0
    df['Contract Length_Quarterly'] = 1 if contract == 'Quarterly' else 0
    df = df.drop(columns=['Contract Length'])
    
    # 4. Ensure column order matches the training data exactly
    expected_columns = [
        'Age', 'Tenure', 'Usage Frequency', 'Support Calls', 
        'Payment Delay', 'Total Spend', 'Last Interaction', 
        'Gender_Male', 'Subscription Type_Premium', 'Subscription Type_Standard', 
        'Contract Length_Monthly', 'Contract Length_Quarterly'
    ]
    
    # Reorder and select only the expected columns
    df = df[expected_columns]
    
    # 5. Make prediction
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    
    return {
        "prediction": "Churn (1)" if prediction == 1 else "No Churn (0)",
        "probability_no_churn": f"{probabilities[0]*100:.2f}%",
        "probability_churn": f"{probabilities[1]*100:.2f}%"
    }


if __name__ == "__main__":
    # Test with a sample customer
    # Notice the high Support Calls and Payment Delay, which usually indicate churn
    # A "Perfect Customer" profile designed to avoid churn
    sample_customer = {
        "CustomerID": 1000,
        "Age": 50,
        "Gender": "Female",
        "Tenure": 60,             # Very long tenure (loyal customer)
        "Usage Frequency": 20,    # High usage
        "Support Calls": 0,       # Zero complaints!
        "Payment Delay": 0,       # Never late on payment
        "Subscription Type": "Premium", # Highest tier
        "Contract Length": "Annual",    # Long-term commitment (Not Monthly!)
        "Total Spend": 2500,      # Very high spend
        "Last Interaction": 5     # Recent interaction
    }
    
    print("=" * 60)
    print("SINGLE CUSTOMER PREDICTION TEST")
    print("=" * 60)
    
    result = predict_churn(sample_customer)
    
    print(f"\nCustomer Profile:")
    for k, v in sample_customer.items():
        if k != 'CustomerID':
            print(f"  - {k}: {v}")
            
    print(f"\n🔮 Model Prediction:")
    print(f"  - Result: {result['prediction']}")
    print(f"  - Probability of No Churn: {result['probability_no_churn']}")
    print(f"  - Probability of Churn: {result['probability_churn']}")
    
    print("\n" + "=" * 60)