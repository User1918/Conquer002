import streamlit as st
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(page_title="Attrition Analysis", layout="wide")

st.title("Employee Attrition Analysis & SHAP Explanations")

@st.cache_data
def load_data():
    # Loading the 3 CSV files provided
    train = pd.read_csv('df_train.csv')
    test = pd.read_csv('df_test.csv')
    raw = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition_Raw.csv')
    return train, test, raw

@st.cache_resource
def load_shap_models():
    # Loading pre-computed SHAP values from your PKL files
    with open('lr_shap_results.pkl', 'rb') as f:
        lr_data = pickle.load(f)
    with open('rf_shap_results.pkl', 'rb') as f:
        rf_data = pickle.load(f)
    return lr_data, rf_data

# Load everything
df_train, df_test, df_raw = load_data()
lr_res, rf_res = load_shap_models()

# Prepare features (excluding target column 'Attrition')
X_test = df_test.drop(columns=['Attrition'])

# Navigation Sidebar
menu = ["Data Tables", "SHAP Explanations"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Data Tables":
    st.subheader("📋 Data Overview")
    tab_raw, tab_train, tab_test = st.tabs(["Raw Data", "Train Data (Processed)", "Test Data (Processed)"])
    
    with tab_raw:
        st.write(f"Raw dataset shape: {df_raw.shape}")
        st.dataframe(df_raw.head(100)) # Showing first 100 rows for speed
        
    with tab_train:
        st.write(f"Train set shape: {df_train.shape}")
        st.dataframe(df_train.head(100))
        
    with tab_test:
        st.write(f"Test set shape: {df_test.shape}")
        st.dataframe(df_test.head(100))

elif choice == "SHAP Explanations":
    st.subheader("🧠 Model Interpretation")
    
    model_type = st.radio("Choose Model:", ["Logistic Regression (Kernel SHAP)", "Random Forest (Tree SHAP)"])
    
    # Extract pre-computed values based on selection
    if "Logistic" in model_type:
        shap_vals = lr_res['shap_values_lr']
        st.info("Explaining Logistic Regression using Kernel SHAP results")
    else:
        shap_vals = rf_res['shap_values_rf']
        st.info("Explaining Random Forest using Tree SHAP results")

    # Handle binary classification (check if it's a list of arrays)
    # Usually index 1 represents 'Attrition = Yes'
    if isinstance(shap_vals, list):
        shap_vals_to_plot = shap_vals[1]
    else:
        shap_vals_to_plot = shap_vals

    st.write("### Global Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals_to_plot, X_test, show=False)
    st.pyplot(fig)
    
    st.divider()
    
    st.write("### Local Prediction Explanation")
    idx = st.number_input("Select Employee Index from Test Set:", min_value=0, max_value=len(X_test)-1, value=0)
    
    # Individual Bar Plot for a specific employee
    fig_ind, ax_ind = plt.subplots(figsize=(10, 4))
    exp = shap.Explanation(
        values=shap_vals_to_plot[idx],
        data=X_test.iloc[idx],
        feature_names=X_test.columns.tolist()
    )
    shap.plots.bar(exp, show=False)
    st.pyplot(fig_ind)