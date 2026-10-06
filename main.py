"""
FastAPI app serving a sentiment analysis model (positive / negative).

Run locally:
    uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(
    title="Sentiment Analysis API",
    description="Classifies a review as positive or negative.",
    version="1.0.0",
)

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["This movie was absolutely fantastic!"])


class ReviewResponse(BaseModel):
    sentiment: str
    confidence: float | None = None


LABELS = {0: "negative", 1: "positive"}


@app.get("/")
def root():
    return {"message": "Sentiment API is running. See /docs for usage."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=ReviewResponse)
def predict(request: ReviewRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Did you run the training notebook?")

    pred = model.predict([request.text])[0]

    # Works whether the model returns 0/1 or "positive"/"negative"
    if isinstance(pred, str):
        sentiment = pred.lower()
    else:
        sentiment = LABELS.get(int(pred), str(pred))

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = round(float(max(model.predict_proba([request.text])[0])), 3)

    return ReviewResponse(sentiment=sentiment, confidence=confidence)