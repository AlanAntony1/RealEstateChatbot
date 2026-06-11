import os
import joblib
import numpy as np
import pandas as pd
from io import StringIO

def model_fn(model_dir):
    """Load all artifacts from model_dir when endpoint starts."""
    model    = joblib.load(os.path.join(model_dir, "logistic_classifier.joblib"))
    scaler   = joblib.load(os.path.join(model_dir, "scaler_classification.joblib"))
    encoders = joblib.load(os.path.join(model_dir, "label_encoders.joblib"))
    columns  = joblib.load(os.path.join(model_dir, "feature_columns.joblib"))
    return {"model": model, "scaler": scaler,
            "encoders": encoders, "columns": columns}

def input_fn(request_body, content_type="text/csv"):
    """Parse incoming CSV request into a DataFrame."""
    if content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body), header=None)
        df.columns = [
            "age", "job", "marital", "education", "default",
            "housing", "loan", "contact", "month", "day_of_week",
            "duration", "campaign", "pdays", "previous", "poutcome",
            "emp.var.rate", "cons.price.idx", "cons.conf.idx",
            "euribor3m", "nr.employed"
        ]
        return df
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model_dict):
    """Encode, scale, and predict."""
    encoders = model_dict["encoders"]
    scaler   = model_dict["scaler"]
    model    = model_dict["model"]
    columns  = model_dict["columns"]

    df = input_data.copy()
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    df     = df[columns]
    scaled = scaler.transform(df)
    pred   = int(model.predict(scaled)[0])
    prob   = round(float(model.predict_proba(scaled)[0][1]), 4)
    return {"prediction": pred, "probability": prob,
            "label": "yes" if pred == 1 else "no"}

def output_fn(prediction, accept="text/csv"):
    """Format prediction as string."""
    return f"{prediction['label']},{prediction['probability']}"
