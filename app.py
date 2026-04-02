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
    
    # 1. Selection Logic
    if "Logistic" in model_type:
        shap_vals = lr_res['shap_values_lr']
        base_val = lr_res.get('expected_value_lr', 0)
    else:
        shap_vals = rf_res['shap_values_rf']
        base_val = rf_res.get('expected_value_rf', 0)

    # 2. Extract specific class for Binary Classification
    if isinstance(shap_vals, list):
        shap_vals_to_plot = shap_vals[1]
        actual_base_val = base_val[1] if isinstance(base_val, (list, np.ndarray)) else base_val
    elif len(np.shape(shap_vals)) == 3:
        shap_vals_to_plot = shap_vals[:, :, 1]
        actual_base_val = base_val[1] if isinstance(base_val, (list, np.ndarray)) else base_val
    else:
        shap_vals_to_plot = shap_vals
        actual_base_val = base_val

    # 3. Row Selection & Prediction Display
    idx = st.number_input("Select Employee Row Index:", 0, len(X_test)-1, 0)
    
    # Get Actual Label from df_test
    actual_label = df_test.iloc[idx]['Attrition']
    label_text = "Yes (Left)" if actual_label == 1 else "No (Stayed)"
    
    # Calculate Prediction based on SHAP sum
    # Prediction = Base Value + Sum(SHAP Values)
    row_shap = np.array(shap_vals_to_plot[idx]).flatten()
    prediction_value = actual_base_val + np.sum(row_shap)
    
    # For Logistic/RF, if prediction_value > 0 (log-odds) or > 0.5 (prob), it's a "Yes"
    # Note: Adjust threshold if your model uses a specific probability cutoff
    pred_threshold = 0.5 if "Random" in model_type else 0.0
    prediction_text = "Yes (Left)" if prediction_value > pred_threshold else "No (Stayed)"

    # Display Metrics
    col1, col2 = st.columns(2)
    col1.metric("Actual Label", label_text)
    col2.metric("Model Prediction", prediction_text, 
               delta=f"{prediction_value:.2f} (Score)", 
               delta_color="inverse" if actual_label != (prediction_value > pred_threshold) else "normal")

    st.divider()

    # 4. Plots
    tab1, tab2 = st.tabs(["Local Bar Plot", "Global Summary"])
    
    with tab1:
        st.write(f"### Explanation for Row {idx}")
        row_data = X_test.iloc[idx].values.flatten()
        exp = shap.Explanation(
            values=row_shap,
            base_values=float(actual_base_val),
            data=row_data,
            feature_names=X_test.columns.tolist()
        )
        fig_local, ax_local = plt.subplots(figsize=(10, 5))
        shap.plots.bar(exp, show=False)
        st.pyplot(fig_local)

    with tab2:
        st.write("### Global Feature Importance")
        fig_global, ax_global = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_vals_to_plot, X_test, show=False)
        st.pyplot(fig_global)
