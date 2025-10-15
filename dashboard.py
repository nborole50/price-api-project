# dashboard.py (Enhanced Version with Fairness and Drift)

import streamlit as st
import pandas as pd
import pickle
import json
import shap
import matplotlib.pyplot as plt
import numpy as np
from fairlearn.metrics import MetricFrame
from sklearn.metrics import mean_squared_error
import warnings
from datetime import datetime

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Indigo Flight Price Predictor", page_icon="✈️", layout="wide")

# --- Suppress Warnings ---
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

# --- 1. Load Artifacts (Model, Columns, and Dataset) ---
@st.cache_resource
def load_artifacts():
    try:
        with open("flight_price_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model_columns.json", "r") as f:
            model_columns = json.load(f)
        df = pd.read_csv('indigo_flights_cleaned (1).csv')
        df.dropna(subset=['Price'], inplace=True)
        return model, model_columns, df
    except FileNotFoundError as e:
        st.error(f"Error: A required file was not found: '{e.filename}'.")
        return None, None, None

model, model_columns, df = load_artifacts()

# --- 2. Caching Functions for Analyses ---
@st.cache_data
def generate_shap_plot(_model, _df_sample, _model_columns):
    X_sample = _df_sample.drop(columns=['Price'], errors='ignore')
    X_encoded = pd.get_dummies(X_sample)
    X_aligned = X_encoded.reindex(columns=_model_columns, fill_value=0)
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_aligned)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_aligned, plot_type="bar", show=False)
    plt.title("Feature Importance (based on SHAP values)")
    return fig

@st.cache_data
def generate_fairness_report(_model, _df, _model_columns):
    X = _df.drop(columns=['Price'], errors='ignore')
    y_true = _df['Price']
    X_encoded = pd.get_dummies(X)
    X_aligned = X_encoded.reindex(columns=_model_columns, fill_value=0)
    y_pred = _model.predict(X_aligned)

    sensitive_features = _df['Class']
    metrics = {'RMSE': lambda y, y_p: np.sqrt(mean_squared_error(y, y_p))}
    grouped_on_class = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)

    fig, ax = plt.subplots()
    grouped_on_class.by_group.plot.bar(y="RMSE", ax=ax, legend=False)
    plt.title("Model Error (RMSE) by Travel Class")
    plt.ylabel("RMSE (₹)")
    return grouped_on_class.by_group, fig

@st.cache_data
def check_data_drift(_df, column_name):
    _df_sorted = _df.sort_index().reset_index(drop=True)
    split_point = len(_df_sorted) // 2
    reference_data = _df_sorted.loc[:split_point-1, column_name]
    current_data = _df_sorted.loc[split_point:, column_name]

    fig, ax = plt.subplots()
    reference_data.plot(kind='hist', ax=ax, density=True, alpha=0.5, label='Reference Data (First 50%)')
    current_data.plot(kind='hist', ax=ax, density=True, alpha=0.5, label='Current Data (Last 50%)')
    plt.title(f"Data Drift Check for '{column_name}'")
    plt.legend()
    return fig

# --- 3. UI Layout and Design ---
st.title("✈️ Indigo Flight Price Prediction Dashboard")
st.write("An end-to-end project demonstrating model deployment, explainability, and responsible AI.")

if all([model, model_columns, df is not None]):
    tab1, tab2, tab3, tab4 = st.tabs(["**Get a Prediction**", "**Model Insights**", "**Fairness Audit**", "**Data Drift**"])

    # --- Prediction Tab ---
    with tab1:
        st.header("Enter Flight Details for Prediction")
        col1, col2 = st.columns(2)
        # ... (Prediction form code from previous version remains the same)
        with col1:
            day = st.selectbox("Day of the Week", options=sorted(df['Day_of_Week'].unique()))
            source = st.selectbox("Source Airport", options=sorted(df['Source'].unique()))
            destination = st.selectbox("Destination Airport", options=sorted(df['Destination'].unique()))
            flight_class = st.selectbox("Travel Class", options=sorted(df['Class'].unique()))
        with col2:
            aircraft = st.selectbox("Aircraft Type", options=sorted(df['Aircraft_Type'].unique()))
            weather = st.selectbox("Weather Conditions", options=sorted(df['Weather_Conditions'].unique()))
            meal = st.selectbox("Meal Opted", options=sorted(df['Meal_Opted'].unique()))
            booking = st.selectbox("Booking Channel", options=sorted(df['Booking_Channel'].unique()))

        seat_rate = st.slider("Seat Occupancy Rate (%)", 50.0, 100.0, 85.0)
        rating = st.slider("Passenger Rating", 1.0, 5.0, 4.0)
        delay = st.number_input("Delay in Minutes", min_value=0, max_value=500, value=0)
        current_year = datetime.now().year
        month = st.selectbox("Month", options=range(1, 13), index=datetime.now().month - 1)
        year = st.number_input("Year", min_value=current_year - 1, max_value=current_year + 2, value=current_year)

        if st.button("Predict Price", type="primary"):
            input_data = {'Day_of_Week': day, 'Source': source, 'Destination': destination, 'Aircraft_Type': aircraft, 'Class': flight_class, 'Weather_Conditions': weather, 'Meal_Opted': meal, 'Booking_Channel': booking, 'Seat_Occupancy_Rate': seat_rate, 'Passenger_Rating': rating, 'Delay_Minutes': delay, 'Month': month, 'Year': year}
            input_df = pd.DataFrame([input_data])
            input_encoded = pd.get_dummies(input_df)
            input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
            prediction = model.predict(input_aligned)
            st.success(f"**Predicted Flight Price: ₹{prediction[0]:.2f}**")

    # --- Model Insights Tab ---
    with tab2:
        st.header("Understanding the Model's Predictions (Global Explainability)")
        with st.spinner("Generating SHAP plot..."):
            shap_fig = generate_shap_plot(model, df.sample(100, random_state=42), model_columns)
            st.pyplot(shap_fig)

    # --- Fairness Audit Tab ---
    with tab3:
        st.header("Fairness Audit: Is the Model Fair Across Travel Classes?")
        st.write("Here, we check if the model's prediction error (RMSE) is significantly different for Economy vs. Premium Economy flights.")
        with st.spinner("Running fairness audit..."):
            fairness_df, fairness_fig = generate_fairness_report(model, df, model_columns)
            st.pyplot(fairness_fig)
            st.write("#### Performance by Group:")
            st.dataframe(fairness_df)
            rmse_diff = fairness_df['RMSE'].max() - fairness_df['RMSE'].min()
            st.metric(label="Disparity (Difference in RMSE)", value=f"₹{rmse_diff:.2f}")

    # --- Data Drift Tab ---
    with tab4:
        st.header("Data Drift Detection")
        st.write("This section checks if the statistical properties of the data have changed over time. We simulate this by comparing the first half of the dataset with the second half.")
        drift_column = st.selectbox("Select a feature to check for drift:", ['Seat_Occupancy_Rate', 'Passenger_Rating', 'Delay_Minutes'])
        with st.spinner(f"Checking drift for {drift_column}..."):
            drift_fig = check_data_drift(df, drift_column)
            st.pyplot(drift_fig)

else:
    st.warning("Dashboard could not be loaded because one or more essential files are missing.")
