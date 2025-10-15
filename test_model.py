# test_model.py (Corrected)

import pickle
import pandas as pd
import json
import warnings

# Suppress warnings for cleaner test output...
warnings.filterwarnings("ignore")

def test_model_prediction():
    """
    This test loads the dataset, model, and required columns, preprocesses a sample,
    and checks if a valid prediction is made.
    """
    print("--- Running test: test_model_prediction ---")

    # 1. Load artifacts
    try:
        # Load the pre-trained model
        with open("flight_price_model.pkl", "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")

        # Load the list of columns the model was trained on
        with open("model_columns.json", "r") as f:
            model_columns = json.load(f)
        print("Model columns loaded successfully.")

        # Load the dataset for creating a test sample
        df = pd.read_excel('indigo_flights_dataset.xlsx')
        df.dropna(subset=['Price'], inplace=True)
        df.columns = df.columns.str.strip()
        print("Dataset loaded successfully.")

    except FileNotFoundError as e:
        assert False, f"Artifact not found: {e}"

    # 2. Create a sample and preprocess it
    sample_raw = df.head(1) # Use the first row as our test case

    sample_raw['Date'] = pd.to_datetime(sample_raw['Date'], dayfirst=True)
    sample_raw['Month'] = sample_raw['Date'].dt.month
    sample_raw['Year'] = sample_raw['Date'].dt.year

    # One-hot encode the sample
    sample_encoded = pd.get_dummies(sample_raw)
    print("Sample data preprocessed.")

    # 3. Align columns with the trained model
    # This is the crucial step to prevent errors
    sample_aligned = sample_encoded.reindex(columns=model_columns, fill_value=0)
    print("Sample columns aligned with model columns.")

    # 4. Make a prediction
    prediction = model.predict(sample_aligned)
    print(f"Prediction on sample data: {prediction}")

    # 5. Assert test conditions
    assert prediction is not None, "Prediction result is None."
    assert isinstance(prediction[0], float), "Prediction is not a float."
    assert prediction[0] > 0, "Prediction is not a positive number."

    print("--- Test PASSED ---")

# Run the test function if the script is executed directly
if __name__ == "__main__":
    test_model_prediction()
