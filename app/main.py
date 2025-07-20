# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the trained pipeline
model = joblib.load("model/model.pkl")

# Create FastAPI app
app = FastAPI()

# Input schema
class Request(BaseModel):
    text: str

# Predict route
@app.post("/predict")
def predict_sentiment(req: Request):
    prediction = model.predict([req.text])[0]
    return {"sentiment": prediction}
