from fastapi import FastAPI, Request
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from datetime import datetime
import joblib
import sqlite3
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

app = FastAPI()

# Instrument the app before startup
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Load model
model = joblib.load("models/best_model.pkl")

# Ensure logs.db and table exist
def init_db():
    conn = sqlite3.connect("logs.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            client_ip TEXT,
            input_data TEXT,
            prediction REAL,
            true_value REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Schema for input data
class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Prediction endpoint
@app.post("/predict")
async def predict(data: InputData, request: Request):
    input_features = [[
        data.MedInc, data.HouseAge, data.AveRooms,
        data.AveBedrms, data.Population, data.AveOccup,
        data.Latitude, data.Longitude
    ]]
    prediction = model.predict(input_features)[0]

    # Log to SQLite
    conn = sqlite3.connect("logs.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO logs (timestamp, client_ip, input_data, prediction)
        VALUES (?, ?, ?, ?)
    ''', (
        str(datetime.now()),
        request.client.host,
        str(data.dict()),
        prediction
    ))
    conn.commit()
    conn.close()

    return {"prediction": prediction}

# Monitoring endpoint
@app.get("/metrics")
def metrics():
    conn = sqlite3.connect("logs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), AVG(prediction) FROM logs")
    count, avg_pred = cursor.fetchone()
    conn.close()
    return {
        "total_predictions": count,
        "average_prediction": avg_pred
    }

class LabelUpdate(BaseModel):
    id: int
    true_value: float

@app.post("/submit-label")
def submit_true_label(update: LabelUpdate):
    conn = sqlite3.connect("logs.db")
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE logs SET true_value = ? WHERE id = ?",
        (update.true_value, update.id)
    )
    conn.commit()
    conn.close()
    return {"status": "label updated"}

@app.post("/retrain")
def retrain_model():
    conn = sqlite3.connect("logs.db")
    df = pd.read_sql_query("SELECT * FROM logs WHERE true_value IS NOT NULL", conn)
    conn.close()

    if df.empty:
        return {"error": "No labeled data to retrain"}

    df["input_data"] = df["input_data"].apply(eval)
    X = pd.DataFrame(df["input_data"].tolist())
    y = df["true_value"]

    model = LinearRegression()
    model.fit(X, y)

    version = len([f for f in os.listdir("models") if f.startswith("model_v")])
    model_path = f"models/model_v{version+1}.pkl"
    joblib.dump(model, model_path)

    return {
        "status": "model retrained",
        "model_path": model_path,
        "samples_used": len(X)
    }