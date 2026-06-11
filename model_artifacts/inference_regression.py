import os
import joblib
import numpy as np
import pandas as pd
from io import StringIO

# Column order must match training data
FEATURE_COLUMNS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

def model_fn(model_dir):
    """Load model and scaler from model_dir when endpoint starts."""
    model  = joblib.load(os.path.join(model_dir, "rf_regressor.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler_regression.joblib"))
    return {"model": model, "scaler": scaler}

def input_fn(request_body, content_type=None):
    """Parse incoming request into a DataFrame."""
    if content_type is None or content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body), header=None)
        df.columns = FEATURE_COLUMNS
        return df
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model_dict):
    """Scale input and run prediction."""
    scaler = model_dict["scaler"]
    model  = model_dict["model"]
    scaled = scaler.transform(input_data)
    return model.predict(scaled)

def output_fn(prediction, accept="text/csv"):
    """Format prediction as CSV string."""
    return str(round(float(prediction[0]), 4))
