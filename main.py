# main.py

import pickle
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# 1. INITIALIZE THE FASTAPI APP
app = FastAPI(title="Indigo Flight Price Prediction API")

# 2. LOAD MODEL ARTIFACTS
# Load the trained machine learning model
with open("flight_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the list of column names the model was trained on
with open("model_columns.json", "r") as f:
    model_columns = json.load(f)

# 3. DEFINE THE INPUT DATA MODEL
# This tells FastAPI what kind of data to expect in a request
# It's based on the columns from your original Excel file
class Flight(BaseModel):
    Day_of_Week: str
    Source: str
    Destination: str
    Aircraft_Type: str
    Class: str
    Weather_Conditions: str
    Meal_Opted: str
    Booking_Channel: str
    Seat_Occupancy_Rate: float
    Passenger_Rating: float
    Delay_Minutes: int
    Month: int
    Year: int

# Define the structure for a list of flights
class PredictionRequest(BaseModel):
    flights: List[Flight]


# 4. DEFINE THE PREDICTION ENDPOINT
@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Takes a list of flight records, preprocesses them, and returns price predictions.
    """
    # Convert the incoming list of flights into a pandas DataFrame
    input_df = pd.DataFrame([flight.dict() for flight in request.flights])

    # --- Data Preprocessing ---
    # One-hot encode the categorical features
    # This must match the encoding done during training
    try:
        input_encoded = pd.get_dummies(input_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data encoding error: {e}")

    # --- Align Columns ---
    # Reindex the input dataframe to match the exact columns the model was trained on
    # This adds missing columns (with value 0) and removes extra columns
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)

    # --- Make Predictions ---
    try:
        predictions = model.predict(input_aligned)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    # Return the predictions in a JSON response
    return {"predictions": predictions.tolist()}

# 5. DEFINE A ROOT ENDPOINT (for health checks)
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the Flight Price Prediction API!"}