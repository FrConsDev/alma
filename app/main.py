from .schemas import SentimentRequest, SentimentPrediction
import numpy as np
from fastapi import FastAPI, HTTPException
import os
from .bert_model import load_model, predict

app = FastAPI()
model_path = "model/distilbert_model.keras"

model, preprocessor = load_model(model_path)

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get('/health')
def health_check():
    return {"status": "ok", "message": "API is running smoothly.", "config": "uv+pyproject.toml"}

@app.post("/predict_sentiment", response_model=SentimentPrediction)
async def predict_sentiment(request: SentimentRequest):
    try:
        encoded = preprocessor([request.text])
        prediction = model.predict(encoded)
        proba = prediction[0]

        sentiment_idx = int(np.argmax(proba))
        confidence = float(proba[sentiment_idx])

        sentiment_label = "positive" if sentiment_idx == 1 else "negative"

        proba_dict = {
            "negative": float(proba[0]),
            "positive": float(proba[1])
        }

        return {
            "Proba": proba_dict,
            "Sentiment": sentiment_label,
            "Confidence": confidence
        }

    except Exception as e:
        print("ERREUR :", e)
        raise HTTPException(status_code=500, detail=str(e))
