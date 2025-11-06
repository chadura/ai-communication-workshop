# Sentiment API (Keras) - Workshop Demo

This service uses TensorFlow / Keras to train a small text classifier.

## Endpoints
- `GET /health`
- `POST /analyze` — JSON: `{ "text": "..." }` → returns `{ "text","prediction","proba" }`
- `POST /train?force=true` — retrain model

## Run locally
1. Create venv & install deps:
```bash
cd demos/sentiment-api-keras
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2.(optional) generate sample data (if missing):

```bash
python data/generate_sample_data.py
```

3.Start server:

```bash
uvicorn app.main:app --reload --port 8000
```

**Note on Docker**

TensorFlow container images are large. Local venv development is recommended for workshops where possible.

**How to run (local dev recommended)**

1.Create & activate venv:
```bash
cd demos/sentiment-api-keras
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2.(Optional) generate data:

```bash
python data/generate_sample_data.py
```

3.Start the API:
```bash
uvicorn app.main:app --reload --port 8001
```

4.Test
```bash
curl -X POST "http://127.0.0.1:8001/analyze" -H "Content-Type: application/json" -d '{"text":"I love this service!"}'
```

5.Retrain:
```bash
curl -X POST "http://127.0.0.1:8001/train?force=true"
```