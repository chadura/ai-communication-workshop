# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model_loader import KerasModelManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment-api-keras")

app = FastAPI(title="Sentiment API (Keras) - Workshop Demo")

mgr = KerasModelManager()

class MessageIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    text: str
    prediction: str
    proba: dict

@app.on_event("startup")
def startup_event():
    # Attempt to load model; will train lazily if not present
    mgr.load_or_train()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=PredictionOut)
def analyze(msg: MessageIn):
    if not msg.text or not msg.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    pred_label, proba = mgr.predict_text(msg.text)
    return {"text": msg.text, "prediction": pred_label, "proba": proba}

@app.post("/train")
def train_endpoint(force: bool = False):
    mgr.train(force=force)
    return {"status": "trained", "classes": mgr.class_names}
