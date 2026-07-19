# 🎯 Customer Churn Prediction System

A professional end-to-end machine learning project that predicts customer churn using Random Forest classifier and provides actionable business recommendations through an interactive web application.

## 🌟 Live Demo

[Add your Streamlit Cloud link here after deployment]

## 📋 Project Overview

This project demonstrates a complete ML pipeline from data preprocessing to model deployment, following industry best practices for production-ready machine learning systems.

### Key Features

- ✅ **Automated Data Preprocessing**: Handles missing values and categorical encoding
- ✅ **Machine Learning Model**: Random Forest classifier with hyperparameter tuning
- ✅ **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, and F1-score
- ✅ **Interactive Web App**: Built with Streamlit for real-time predictions
- ✅ **Business Recommendations**: AI-powered actionable insights for customer retention

## 🏗️ Project Structure

AI-Engineering-Portfolio/
├── data/
│ ├── raw/ # Original datasets
│ └── processed/ # Cleaned datasets
├── src/
│ ├── data_processing.py # Data cleaning and preprocessing
│ ├── train.py # Model training pipeline
│ ├── evaluate.py # Model evaluation on test data
│ ├── predict.py # Single prediction function
│ └── analyze_model.py # Feature importance analysis
├── app/
│ └── streamlit_app.py # Interactive web application
├── models/
│ └── churn_model.pkl # Trained model artifact
├── notebooks/
│ └── 01_eda.ipynb # Exploratory Data Analysis
├── tests/
│ └── test_data_processing.py # Unit tests
├── docs/
│ └── feature_importance.png # Model visualization
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/javascriptsali/AI-Engineering-Portfolio.git
   cd AI-Engineering-Portfolio
    2-Create and activate virtual environment:
   
    python -m venv venv

    # On Windows:
    venv\Scripts\activate

    # On Linux/Mac:
    source venv/bin/activate

    3-Install dependencies:
    pip install -r requirements.txt

    4-Run the web application:
    streamlit run app/streamlit_app.py

    5-Open your browser:
    Navigate to http://localhost:8501

📊 Model Performance
Training Metrics
Accuracy: 99.9%
Precision: 100%
Recall: 100%
F1-Score: 100%
Key Features Identified
Support Calls (28.9% importance)
Total Spend (22.1% importance)
Age (14.1% importance)
Contract Length - Monthly (13.6% importance)
Payment Delay (12.4% importance)
🔧 Technical Stack
Programming Language: Python 3.13
Machine Learning: Scikit-Learn (Random Forest)
Data Processing: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Web Framework: Streamlit
Version Control: Git & GitHub
📈 Business Impact
This system helps businesses:
🎯 Identify at-risk customers before they churn
Reduce customer acquisition costs by retaining existing customers
📊 Make data-driven decisions for customer retention strategies
⚡ Take proactive actions with AI-powered recommendations
🔮 Future Improvements
Deploy on Streamlit Cloud for public access
Add user authentication and customer database integration
Implement A/B testing for retention strategies
Add automated retraining pipeline with new data
Integrate with CRM systems (Salesforce, HubSpot)
Add email notification system for high-risk customers
📝 Lessons Learned
Challenges Faced
Covariate Shift: Discovered significant distribution differences between training and testing datasets
Data Leakage Prevention: Implemented proper train-test split and validation strategies
Feature Engineering: Learned the importance of proper categorical encoding with drop_first=True
Best Practices Applied
✅ Modular code structure following SOLID principles
✅ Type hints for better code documentation
✅ Comprehensive error handling
✅ Environment variable management with .env files
✅ Git version control with meaningful commit messages
👨‍💻 Author
Saleh Bakhtiyari
linkedin.com/in/deve-loper-4870b5376
[GitHub Profile](https://github.com/javascriptsali)
📄 License
This project is open source and available under the MIT License.
If you found this project helpful, please give it a ⭐ on GitHub!
