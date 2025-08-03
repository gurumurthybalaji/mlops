from fastapi import FastAPI, Request
from pydantic import BaseModel
from datetime import datetime
import joblib
import sqlite3
import os

app = FastAPI()

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
            prediction REAL
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
async def predict(data: InputData,request: Request):
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
