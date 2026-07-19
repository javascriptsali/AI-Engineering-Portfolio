"""
Streamlit web application for the Customer Churn Prediction project.
Provides an interactive UI for users to input customer data and get churn predictions.
"""

import streamlit as st
import pandas as pd
import os
import sys

# Add src directory to Python path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import predict_churn  # type: ignore


# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main function for the Streamlit app."""
    
    # Sidebar - Project Information
    st.sidebar.title("📊 About This App")
    st.sidebar.markdown("""
    This application predicts customer churn using a Random Forest machine learning model.
    
    **Features:**
    - Real-time churn prediction
    - Probability estimation
    - Actionable recommendations
    
    **Model:** Random Forest Classifier  
    **Accuracy:** ~99.9% on training data
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Project Links")
    st.sidebar.markdown("[GitHub Repository](https://github.com/javascriptsali/AI-Engineering-Portfolio)")
    
    # Main Header
    st.title("🎯 Customer Churn Prediction System")
    st.markdown("""
    Enter customer information below to predict whether they are likely to churn (leave the service).
    """)
    
    st.markdown("---")
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("👤 Customer Information")
        
        # Create a form for user input
        with st.form("customer_form"):
            # Numerical inputs
            age = st.slider("Age", 18, 80, 35, help="Customer's age in years")
            tenure = st.slider("Tenure (months)", 0, 72, 12, help="How long the customer has been with us")
            usage_frequency = st.slider("Usage Frequency", 0, 30, 10, help="How often the customer uses the service")
            support_calls = st.slider("Support Calls", 0, 20, 2, help="Number of calls to customer support")
            payment_delay = st.slider("Payment Delay (days)", 0, 30, 1, help="Average delay in payment")
            total_spend = st.number_input("Total Spend ($)", 0, 10000, 500, step=50, help="Total amount spent by customer")
            last_interaction = st.slider("Last Interaction (days ago)", 0, 90, 10, help="Days since last interaction")
            
            # Categorical inputs
            gender = st.selectbox("Gender", ["Male", "Female"])
            subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
            contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
            
            # Submit button
            submit_button = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
    
    with col2:
        st.subheader("📊 Prediction Results")
        
        if submit_button:
            # Create customer data dictionary
            customer_data = {
                "CustomerID": 999,
                "Age": age,
                "Gender": gender,
                "Tenure": tenure,
                "Usage Frequency": usage_frequency,
                "Support Calls": support_calls,
                "Payment Delay": payment_delay,
                "Subscription Type": subscription_type,
                "Contract Length": contract_length,
                "Total Spend": total_spend,
                "Last Interaction": last_interaction
            }
            
            # Get prediction
            with st.spinner("Analyzing customer data..."):
                result = predict_churn(customer_data)
            
            # Display results
            st.markdown("---")
            
            # Show prediction with color coding
            if result["prediction"] == "Churn (1)":
                st.error(f"### ⚠️ {result['prediction']}")
                st.markdown(f"""
                **Risk Level:** HIGH  
                **Probability of Churn:** {result['probability_churn']}  
                **Probability of Retention:** {result['probability_no_churn']}
                """)
                
                # Recommendations for high-risk customers
                st.markdown("### 💡 Recommended Actions")
                st.markdown("""
                - 📞 **Contact the customer immediately** to understand their concerns
                -  **Offer a discount or special promotion** to retain them
                - 🎯 **Assign a dedicated account manager** for personalized support
                - 📧 **Send a satisfaction survey** to identify pain points
                """)
            else:
                st.success(f"### ✅ {result['prediction']}")
                st.markdown(f"""
                **Risk Level:** LOW  
                **Probability of Churn:** {result['probability_churn']}  
                **Probability of Retention:** {result['probability_no_churn']}
                """)
                
                # Recommendations for loyal customers
                st.markdown("###  Recommended Actions")
                st.markdown("""
                - ⭐ **Reward their loyalty** with exclusive offers
                - 📈 **Upsell premium features** they might be interested in
                - 🤝 **Ask for referrals** or testimonials
                - 📊 **Monitor their usage** to ensure continued satisfaction
                """)
        
        else:
            # Default state when no prediction has been made
            st.info("👈 Fill in the customer information and click **Predict Churn** to see results.")
            
            # Show example metrics
            st.markdown("---")
            st.subheader(" Model Performance Metrics")
            
            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
            
            with col_metrics1:
                st.metric("Training Accuracy", "99.9%", delta="Excellent")
            with col_metrics2:
                st.metric("Precision", "100%", delta="High")
            with col_metrics3:
                st.metric("Recall", "100%", delta="High")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit, Scikit-Learn, and Python</p>
        <p>Part of AI Engineering Portfolio Project</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()