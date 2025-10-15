# dashboard.py

import streamlit as st
import pandas as pd
import pickle
import json
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings

# Suppress warnings for a cleaner interface
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

# --- 1. Load Artifacts (Model, Columns, and Dataset) ---

# Use Streamlit's caching to load artifacts only once
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
        df = pd.read_excel('indigo_flights_dataset.xlsx')
        df.dropna(subset=['Price'], inplace=True)
        return model, model_columns, df
    except FileNotFoundError:
        st.error("Error: Model or data files not found. Please ensure all required files are in the repository.")
        return None, None, None

model, model_columns, df = load_artifacts()

# --- 2. Function to Generate and Cache SHAP Plot ---

@st.cache_data
def generate_shap_plot(_model, _df, _model_columns):
    """
    Generates and caches the SHAP summary plot.
    The plot is only regenerated if the underlying model or data changes.
    """
    st.write("Generating SHAP plot for the first time... This may take a moment.")
    # Prepare data for SHAP (using a sample for speed)
    X = _df.drop(columns=['Price'], errors='ignore')
    X_encoded = pd.get_dummies(X)
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

    # Create columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        # Get unique values from the dataframe for dropdowns
        source_options = df['Source'].unique()
        dest_options = df['Destination'].unique()
        day_options = df['Day_of_Week'].unique()
        class_options = df['Class'].unique()

        day = st.selectbox("Day of the Week", options=day_options)
        source = st.selectbox("Source Airport", options=source_options)
        destination = st.selectbox("Destination Airport", options=dest_options)
        flight_class = st.selectbox("Travel Class", options=class_options)
        
    with col2:
        aircraft_options = df['Aircraft_Type'].unique()
        weather_options = df['Weather_Conditions'].unique()
        meal_options = df['Meal_Opted'].unique()
        booking_options = df['Booking_Channel'].unique()
        
        aircraft = st.selectbox("Aircraft Type", options=aircraft_options)
        weather = st.selectbox("Weather Conditions", options=weather_options)
        meal = st.selectbox("Meal Opted", options=meal_options)
        booking = st.selectbox("Booking Channel", options=booking_options)

    # Sliders and number inputs for numerical features
    seat_rate = st.slider("Seat Occupancy Rate (%)", 50.0, 100.0, 85.0)
    rating = st.slider("Passenger Rating", 1.0, 5.0, 4.0)
    delay = st.number_input("Delay in Minutes", min_value=0, max_value=500, value=0)
    
    # Use current time for default month and year
    from datetime import datetime
    current_year = datetime.now().year
    month = st.selectbox("Month", options=range(1, 13), index=datetime.now().month - 1)
    year = st.number_input("Year", min_value=current_year -1, max_value=current_year + 2, value=current_year)


    # Prediction logic when button is clicked
    if st.button("Predict Price", type="primary"):
        if model is not None:
            # Create a dictionary from the user input
            input_data = {
                'Day_of_Week': day,
                'Source': source,
                'Destination': destination,
                'Aircraft_Type': aircraft,
                'Class': flight_class,
                'Weather_Conditions': weather,
                'Meal_Opted': meal,
                'Booking_Channel': booking,
                'Seat_Occupancy_Rate': seat_rate,
                'Passenger_Rating': rating,
                'Delay_Minutes': delay,
                'Month': month,
                'Year': year
            }
            
            # Convert to DataFrame and preprocess
            input_df = pd.DataFrame([input_data])
            input_encoded = pd.get_dummies(input_df)
            input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
            
            # Make prediction
            prediction = model.predict(input_aligned)
            
            # Display the result
            st.success(f"**Predicted Flight Price: ₹{prediction[0]:.2f}**")
        else:
            st.error("Model is not loaded. Cannot make a prediction.")


# --- 5. Model Insights Tab ---

with tab2:
    st.header("Understanding the Model's Predictions")
    st.write(
        "The following chart shows the most important factors the model uses to predict flight prices. "
        "Features with longer bars have a greater impact on the final price prediction."
    )
    
    # Generate and display the SHAP plot
    if model is not None:
        shap_fig = generate_shap_plot(model, df.sample(100), model_columns) # Use a sample for speed
        st.pyplot(shap_fig)
    else:
        st.error("Model is not loaded. Cannot display insights.")
