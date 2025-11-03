from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app = FastAPI(title="Sentiment Analysis API")
vectorizer, model = joblib.load('model.joblib')


class Message(BaseModel):
    text: str


@app.post('/analyze')
def analyze(message: Message):
    X = vectorizer.transform([message.text])
    prediction = model.predict(X)[0]
    return {"text": message.text, "prediction": prediction}