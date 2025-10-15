# dashboard.py

import streamlit as st
import pandas as pd
import pickle
import json
import shap
import matplotlib.pyplot as plt
import warnings

# Suppress warnings and set Streamlit options for a cleaner interface
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

# --- 1. Load Artifacts (Model, Columns, and Dataset) ---

@st.cache_resource
def load_artifacts():
    """
    Loads the model, model columns, and the dataset.
    This function is cached so artifacts are not reloaded on every interaction.
    """
    try:
        with open("flight_price_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model_columns.json", "r") as f:
            model_columns = json.load(f)
        # Use the cleaned CSV for the dashboard as it's lighter and pre-processed
        df = pd.read_csv('indigo_flights_cleaned (1).csv')
        df.dropna(subset=['Price'], inplace=True)
        return model, model_columns, df
    except FileNotFoundError as e:
        st.error(f"Error: A required file was not found. Please ensure '{e.filename}' is in the repository.")
        return None, None, None

model, model_columns, df = load_artifacts()

# --- 2. Function to Generate and Cache SHAP Plot ---

@st.cache_data
def generate_shap_plot(_model, _df_sample, _model_columns):
    """
    Generates and caches the SHAP summary plot using a sample of the data for speed.
    """
    st.write("Generating SHAP plot for the first time... This may take a moment.")
    # Prepare data for SHAP
    X_sample = _df_sample.drop(columns=['Price'], errors='ignore')
    X_encoded = pd.get_dummies(X_sample)
    X_aligned = X_encoded.reindex(columns=_model_columns, fill_value=0)
    
    # Explain model predictions using SHAP
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_aligned)
    
    # Create the plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_aligned, plot_type="bar", show=False)
    plt.title("Feature Importance (based on SHAP values)")
    return fig

# --- 3. UI Layout and Design ---

st.set_page_config(page_title="Indigo Flight Price Predictor", page_icon="✈️", layout="wide")
st.title("✈️ Indigo Flight Price Prediction Dashboard")
st.write(
    "This dashboard allows you to get real-time price predictions for Indigo flights "
    "and understand the key factors that influence the price."
)

# Create tabs for different sections
tab1, tab2 = st.tabs(["**Get a Prediction**", "**Model Insights**"])


# --- 4. Prediction Tab ---

with tab1:
    st.header("Enter Flight Details for Prediction")

    if df is not None:
        # Create columns for a cleaner layout
        col1, col2 = st.columns(2)

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

        # Sliders and number inputs for numerical features
        seat_rate = st.slider("Seat Occupancy Rate (%)", 50.0, 100.0, 85.0)
        rating = st.slider("Passenger Rating", 1.0, 5.0, 4.0)
        delay = st.number_input("Delay in Minutes", min_value=0, max_value=500, value=0)
        
        from datetime import datetime
        current_year = datetime.now().year
        month = st.selectbox("Month", options=range(1, 13), index=datetime.now().month - 1)
        year = st.number_input("Year", min_value=current_year - 1, max_value=current_year + 2, value=current_year)

        if st.button("Predict Price", type="primary"):
            if model is not None:
                input_data = {
                    'Day_of_Week': day, 'Source': source, 'Destination': destination,
                    'Aircraft_Type': aircraft, 'Class': flight_class, 'Weather_Conditions': weather,
                    'Meal_Opted': meal, 'Booking_Channel': booking, 'Seat_Occupancy_Rate': seat_rate,
                    'Passenger_Rating': rating, 'Delay_Minutes': delay, 'Month': month, 'Year': year
                }
                
                input_df = pd.DataFrame([input_data])
                input_encoded = pd.get_dummies(input_df)
                input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
                
                prediction = model.predict(input_aligned)
                
                st.success(f"**Predicted Flight Price: ₹{prediction[0]:.2f}**")
            else:
                st.error("Model is not loaded. Cannot make a prediction.")
    else:
        st.warning("Dataset could not be loaded. Prediction form is disabled.")


# --- 5. Model Insights Tab ---

with tab2:
    st.header("Understanding the Model's Predictions")
    st.write(
        "The following chart shows the most important factors the model uses to predict flight prices. "
        "Features with longer bars have a greater impact on the final price prediction."
    )
    
    if model is not None and df is not None:
        # Use a smaller sample for the public dashboard to ensure it runs quickly
        shap_fig = generate_shap_plot(model, df.sample(100, random_state=42), model_columns)
        st.pyplot(shap_fig)
    else:
        st.error("Model or data not loaded. Cannot display insights.")
