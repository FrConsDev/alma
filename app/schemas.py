from pydantic import BaseModel
from typing import Dict

class SentimentRequest(BaseModel):
    text: str

class SentimentPrediction(BaseModel):
    Proba: Dict[str, float]
    Sentiment: str
    Confidence: float
