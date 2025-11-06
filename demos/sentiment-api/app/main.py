# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model_loader import load_or_train_model, retrain_model_if_requested
from typing import Dict, Any
import logging
import os
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment-api")

app = FastAPI(title="Sentiment Analysis API - Workshop Demo")


origins = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:8501",   # Streamlit (if used)
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8501",
    "http://127.0.0.1:3000",
    # add any host your frontend will run on
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,    # OR ["*"] for workshop/dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure model is loaded at startup (train if required)
model_bundle = load_or_train_model()  # returns dict with vectorizer, model, classes

class MessageIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    text: str
    prediction: str
    proba: Dict[str, float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=PredictionOut)
def analyze(msg: MessageIn):
    if not msg.text or not msg.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    vectorizer = model_bundle["vectorizer"]
    model = model_bundle["model"]
    classes = model_bundle.get("classes")
    X = vectorizer.transform([msg.text])
    pred = model.predict(X)[0]
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(X)[0]
        proba = {str(c): float(p) for c, p in zip(model.classes_, proba_arr)}
    else:
        proba = {str(pred): 1.0}
    return {"text": msg.text, "prediction": str(pred), "proba": proba}

@app.post("/train")
def train_endpoint(force: bool = False):
    """
    Retrain the model. If force=True, retrains even if model exists.
    """
    new_bundle = retrain_model_if_requested(force=force)
    return {"status": "trained", "classes": [str(c) for c in new_bundle["model"].classes_]}
