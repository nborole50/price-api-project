# test_model.py

import pickle
import pandas as pd
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

def test_model_prediction():
    """
    This test loads the dataset and the trained model, preprocesses a sample,
    and checks if a valid prediction is made.
    """
    print("--- Running test: test_model_prediction ---")

    # 1. Load the dataset
    try:
        df = pd.read_excel('indigo_flights_dataset.xlsx')
        df.dropna(subset=['Price'], inplace=True)
        df.columns = df.columns.str.strip()
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        assert False, "Dataset file 'indigo_flights_dataset.xlsx' not found."

    # 2. Perform the same preprocessing as in training
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    features_to_encode = [
        'Day_of_Week', 'Source', 'Destination', 'Aircraft_Type', 'Class',
        'Weather_Conditions', 'Meal_Opted', 'Booking_Channel'
    ]
    df_encoded = pd.get_dummies(df, columns=features_to_encode, drop_first=True)

    columns_to_drop = [
        'Price', 'Date', 'Flight_ID', 'Departure_Time', 'Arrival_Time',
        'Duration', 'Delay_Status'
    ]
    existing_cols_to_drop = [col for col in columns_to_drop if col in df_encoded.columns]
    X = df_encoded.drop(columns=existing_cols_to_drop)
    print("Data preprocessing complete.")

    # 3. Load the pre-trained model
    try:
        with open("flight_price_model.pkl", "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except FileNotFoundError:
        assert False, "Model file 'flight_price_model.pkl' not found."

    # 4. Make a prediction on the first sample
    sample_data = X.head(1)
    prediction = model.predict(sample_data)
    print(f"Prediction on sample data: {prediction}")

    # 5. Assert test conditions
    assert prediction is not None, "Prediction result is None."
    assert isinstance(prediction[0], float), "Prediction is not a float."
    assert prediction[0] > 0, "Prediction is not a positive number."

    print("--- Test PASSED ---")

# Run the test function if the script is executed directly
if __name__ == "__main__":
    test_model_prediction()
