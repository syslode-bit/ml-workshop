"""
FastAPI app serving a sentiment analysis model (positive / negative).

Run locally:
    uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(
    title="Sentiment Analysis API",
    description="Classifies a review as positive or negative.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your static site URL later
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["This product is really good!"])


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

from fastapi.responses import HTMLResponse


@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sentiment Tester</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #f4f5f7; display: flex;
           justify-content: center; padding: 60px 16px; margin: 0; }
    .card { background: white; max-width: 520px; width: 100%; padding: 28px;
            border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
    h1 { margin: 0 0 16px; font-size: 22px; }
    textarea { width: 100%; box-sizing: border-box; height: 100px; padding: 12px;
               font-size: 15px; border: 1px solid #ccc; border-radius: 8px; resize: vertical; }
    button { margin-top: 12px; width: 100%; padding: 12px; font-size: 15px; font-weight: 600;
             color: white; background: #2563eb; border: none; border-radius: 8px; cursor: pointer; }
    button:disabled { background: #93a7d6; cursor: wait; }
    #result { margin-top: 20px; padding: 16px; border-radius: 8px; font-size: 16px; display: none; }
    .positive { background: #e7f6ec; color: #166534; }
    .negative { background: #fdecec; color: #991b1b; }
    .error    { background: #fff4e5; color: #92400e; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Sentiment Tester</h1>
    <textarea id="text" placeholder="e.g. This product is really good"></textarea>
    <button id="run" onclick="runModel()">Run model</button>
    <div id="result"></div>
  </div>
  <script>
    async function runModel() {
      const text = document.getElementById("text").value.trim();
      const button = document.getElementById("run");
      const result = document.getElementById("result");
      result.style.display = "block";

      if (!text) {
        result.className = "error";
        result.textContent = "Please enter a sentence first.";
        return;
      }

      button.disabled = true;
      button.textContent = "Running...";
      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text })
        });
        if (!res.ok) throw new Error("Server returned " + res.status);
        const data = await res.json();
        const pct = data.confidence !== null ? " (" + Math.round(data.confidence * 100) + "% confident)" : "";
        result.className = data.sentiment;
        result.textContent = "Sentiment: " + data.sentiment.toUpperCase() + pct;
      } catch (err) {
        result.className = "error";
        result.textContent = "Something went wrong: " + err.message;
      }
      button.disabled = false;
      button.textContent = "Run model";
    }
  </script>
</body>
</html>
"""