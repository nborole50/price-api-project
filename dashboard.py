# dashboard.py (Final Enhanced Version)

import streamlit as st
import pandas as pd
import pickle
import json
import shap
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Indigo Flight Price Predictor", page_icon="✈️", layout="wide")

# --- Suppress Warnings ---
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

# --- 1. Load Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        with open("flight_price_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model_columns.json", "r") as f:
            model_columns = json.load(f)
        df = pd.read_csv('indigo_flights_cleaned (1).csv')
        df.dropna(subset=['Price'], inplace=True)
        df['Date'] = pd.to_datetime(df['Date']) # Ensure Date is datetime
        return model, model_columns, df
    except FileNotFoundError as e:
        st.error(f"Error: A required file was not found: '{e.filename}'. Please ensure all artifacts are in the GitHub repository.")
        return None, None, None

model, model_columns, df = load_artifacts()

# --- 2. Caching Function for SHAP Plot ---
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

# --- 3. UI Layout ---
st.title("✈️ Indigo Flight Price Prediction Dashboard")

if all([model, model_columns, df is not None]):
    tab1, tab2, tab3, tab4 = st.tabs(["**Project Overview & Analysis**", "**Get a Prediction**", "**Model Insights**", "**Responsible AI**"])

    # --- Project Overview & Analysis Tab ---
    with tab1:
        st.header("Project Overview")
        st.markdown("""
        **Why does this project matter?** Airline ticket pricing is highly dynamic and complex, influenced by a multitude of factors like seasonality, route demand, and aircraft type. For both customers and the airline, understanding these price variations is crucial.

        This project aims to demystify Indigo's flight pricing by building a machine learning model that can:
        - **Predict flight prices** based on various travel parameters.
        - **Provide insights** into the key drivers that determine ticket costs.
        - Serve as an end-to-end demonstration of a real-world data science project, from data cleaning and modeling to deployment and responsible AI auditing.
        """)

        # --- NEW: Dataset Snapshot ---
        st.header("Dataset Snapshot")
        st.write("A quick look at the cleaned data used for training the model.")
        st.dataframe(df.head())

        st.header("Exploratory Data Analysis")
        st.write("Visualizing the relationships within the flight dataset.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Average Price by Day of Week")
            avg_price_day = df.groupby('Day_of_Week')['Price'].mean().sort_values()
            st.bar_chart(avg_price_day)

            st.subheader("Flight Volume by Source Airport")
            flights_by_source = df['Source'].value_counts()
            st.bar_chart(flights_by_source)
        with col2:
            st.subheader("Average Price by Month")
            df['Month'] = df['Date'].dt.month
            avg_price_month = df.groupby('Month')['Price'].mean()
            st.line_chart(avg_price_month)

            st.subheader("Flight Distribution by Class")
            fig, ax = plt.subplots()
            df['Class'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)

    # --- Prediction Tab ---
    with tab2:
        st.header("Enter Flight Details for Prediction")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            day = st.selectbox("Day of the Week", options=sorted(df['Day_of_Week'].unique()), key='pred_day')
            source = st.selectbox("Source Airport", options=sorted(df['Source'].unique()), key='pred_source')
            destination = st.selectbox("Destination Airport", options=sorted(df['Destination'].unique()), key='pred_dest')
            flight_class = st.selectbox("Travel Class", options=sorted(df['Class'].unique()), key='pred_class')
        with p_col2:
            aircraft = st.selectbox("Aircraft Type", options=sorted(df['Aircraft_Type'].unique()), key='pred_aircraft')
            weather = st.selectbox("Weather Conditions", options=sorted(df['Weather_Conditions'].unique()), key='pred_meal')
            meal = st.selectbox("Meal Opted", options=sorted(df['Meal_Opted'].unique()), key='pred_meal_opt')
            booking = st.selectbox("Booking Channel", options=sorted(df['Booking_Channel'].unique()), key='pred_booking')

        seat_rate = st.slider("Seat Occupancy Rate (%)", 50.0, 100.0, 85.0, key='pred_seat')
        rating = st.slider("Passenger Rating", 1.0, 5.0, 4.0, key='pred_rating')
        delay = st.number_input("Delay in Minutes", min_value=0, max_value=500, value=0, key='pred_delay')
        current_year = datetime.now().year
        month = st.selectbox("Month", options=range(1, 13), index=datetime.now().month - 1, key='pred_month')
        year = st.number_input("Year", min_value=current_year - 1, max_value=current_year + 2, value=current_year, key='pred_year')

        if st.button("Predict Price", type="primary"):
            input_data = {'Day_of_Week': day, 'Source': source, 'Destination': destination, 'Aircraft_Type': aircraft, 'Class': flight_class, 'Weather_Conditions': weather, 'Meal_Opted': meal, 'Booking_Channel': booking, 'Seat_Occupancy_Rate': seat_rate, 'Passenger_Rating': rating, 'Delay_Minutes': delay, 'Month': month, 'Year': year}
            input_df = pd.DataFrame([input_data])
            input_encoded = pd.get_dummies(input_df)
            input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
            prediction = model.predict(input_aligned)
            st.success(f"**Predicted Flight Price: ₹{prediction[0]:.2f}**")

    # --- Model Insights Tab ---
    with tab3:
        st.header("Understanding the Model's Predictions (Global Explainability)")
        with st.spinner("Generating SHAP plot..."):
            shap_fig = generate_shap_plot(model, df.sample(100, random_state=42), model_columns)
            st.pyplot(shap_fig)

    # --- Responsible AI Tab ---
    with tab4:
        st.header("Responsible AI (RAI) Checklist")
        st.markdown("""
        This section outlines the steps taken to ensure the model was developed and evaluated responsibly.

        ---

        ### 1. Transparency and Explainability
        **Objective:** To ensure that the model's decision-making process is understandable.
        - **Global Explainability (SHAP):** We used SHAP to identify the most influential factors across all predictions, confirming the model learned logical patterns.
        - **Local Explainability (LIME):** We used LIME to explain individual predictions, building trust and allowing for debugging.

        ---

        ### 2. Fairness and Bias
        **Objective:** To identify and quantify any systematic biases in the model's performance.
        - **Sensitive Attribute:** We audited the model for performance bias across the `Class` feature (Economy vs. Premium Economy).
        - **Fairness Audit Finding:** The model's average prediction error (RMSE) is **~₹178 higher** for Economy class flights, indicating a performance bias.
        - **Proposed Mitigation:** Strategies like reweighting training data or applying post-prediction adjustments were proposed to address this.

        ---

        ### 3. Privacy and Data Governance
        - **No Personal Data:** The model was trained exclusively on anonymized flight data. No Personally Identifiable Information (PII) was used.
        - **User Input:** The dashboard does not ask for or store any user-specific data.

        ---

        ### 4. Accountability and Human Oversight
        - **Intended Use:** The model is intended as a tool for price estimation and market analysis, not as a fully autonomous system.
        - **CI/CD Pipeline:** An automated CI/CD pipeline tests the model's integrity with every code change, ensuring reliability.
        """)
        st.info("Note: The LIME and Fairness Audit analyses were performed in Experiment 5. The results are summarized here and presented on the dashboard.")

else:
    st.warning("Dashboard could not be loaded because one or more essential files are missing from the repository.")
