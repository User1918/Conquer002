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
    # Loading the 3 CSV files
    train = pd.read_csv('df_train.csv')
    test = pd.read_csv('df_test.csv')
    raw = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition_Raw.csv')
    return train, test, raw

@st.cache_resource
def load_shap_models():
    # Loading pre-computed SHAP results
    with open('lr_shap_results.pkl', 'rb') as f:
        lr_data = pickle.load(f)
    with open('rf_shap_results.pkl', 'rb') as f:
        rf_data = pickle.load(f)
    return lr_data, rf_data

# Load everything
try:
    df_train, df_test, df_raw = load_data()
    lr_res, rf_res = load_shap_models()
    # Ensure X_test only contains the feature columns
    X_test = df_test.drop(columns=['Attrition'])
except Exception as e:
    st.error(f"Error loading files: {e}. Please check if all CSV/PKL files are in the repo.")
    st.stop()

# Navigation
menu = ["Data Tables", "SHAP Explanations"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Data Tables":
    st.subheader("📋 Data Overview")
    tab_raw, tab_train, tab_test = st.tabs(["Raw Data", "Train Data", "Test Data"])
    with tab_raw: st.dataframe(df_raw.head(100))
    with tab_train: st.dataframe(df_train.head(100))
    with tab_test: st.dataframe(df_test.head(100))

elif choice == "SHAP Explanations":
    st.subheader("🧠 Model Interpretation")
    model_type = st.radio("Choose Model:", ["Logistic Regression (Kernel)", "Random Forest (Tree)"])
    
    # 1. Select the correct SHAP values and Expected Value based on model
    if "Logistic" in model_type:
        shap_vals = lr_res['shap_values_lr']
        base_val = lr_res.get('expected_value_lr', 0)
        st.info("Explaining Logistic Regression")
    else:
        shap_vals = rf_res['shap_values_rf']
        base_val = rf_res.get('expected_value_rf', 0)
        st.info("Explaining Random Forest")

    # 2. Logic to handle 3D Arrays vs Lists (Binary Classification Fix)
    # Tree SHAP often returns a list [class0, class1] or a 3D array (samples, features, outputs)
    if isinstance(shap_vals, list):
        # It's a list, take the positive class (Attrition = Yes)
        shap_vals_to_plot = shap_vals[1]
        if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
            base_val = base_val[1]
    elif len(np.shape(shap_vals)) == 3:
        # It's a 3D array (samples, features, 2), slice for class 1
        shap_vals_to_plot = shap_vals[:, :, 1]
        if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
            base_val = base_val[1]
    else:
        # It's already 2D (samples, features)
        shap_vals_to_plot = shap_vals

    # Global Plot
    st.write("### Global Feature Importance")
    fig_global, ax_global = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals_to_plot, X_test, show=False)
    st.pyplot(fig_global)
    
    st.divider()
    
    # Local Plot
    st.write("### Local Prediction Explanation")
    idx = st.number_input("Select Employee Row Index:", 0, len(X_test)-1, 0)
    
    # DATA EXTRACTION FIX:
    # Ensure row_shap and row_data are strictly 1D arrays of the same length
    row_shap = np.array(shap_vals_to_plot[idx]).flatten()
    row_data = X_test.iloc[idx].values.flatten()
    
    # Create the Explanation object correctly
    exp = shap.Explanation(
        values=row_shap,
        base_values=float(base_val) if isinstance(base_val, (int, float, np.number)) else 0.0,
        data=row_data,
        feature_names=X_test.columns.tolist()
    )

    # Use Waterfall for Tree models if you want, but Bar is also great
    fig_local, ax_local = plt.subplots(figsize=(10, 5))
    try:
        shap.plots.bar(exp, show=False)
        st.pyplot(fig_local)
    except Exception as e:
        st.error(f"Visualization error: {e}")
        st.write("Falling back to text-based importance:")
        st.write(pd.Series(row_shap, index=X_test.columns).sort_values(ascending=False))
