# Sentiment API (Workshop Demo)

## Overview
This FastAPI service serves a small sentiment classifier. If no model exists, the service will train one from `data/sentiment_sample.csv`. A default sample CSV will be created automatically if missing.

## Endpoints
- `GET /health` — basic health check
- `POST /analyze` — body: `{ "text": "..." }` → returns `{"text","prediction","proba"}`
- `POST /train?force=true` — retrain the model (force overwrite)

## Run locally
```bash
# from demos/sentiment-api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# generate data (optional)
python data/generate_sample_data.py
# start server
uvicorn app.main:app --reload --port 8000
```


```bash
# from repository root (where Dockerfile is located)
docker build -t sentiment-api:local .
docker run -p 8000:8000 sentiment-api:local
```


## Test health:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```


## Test analyze:
```bash
curl -X POST "http://127.0.0.1:8000/analyze" -H "Content-Type: application/json" -d '{"text":"I love this product, excellent service!"}'
# -> returns JSON with prediction and class probabilities
```

## Force retrain:

```bash
curl -X POST "http://127.0.0.1:8000/train?force=true"

```