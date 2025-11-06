# Streamlit Dashboard - AI in Communication Workshop

## Overview
Simple Streamlit dashboard to test sentiment predictions from the FastAPI service.

## Quickstart (local)
1. Create venv and install deps:
```bash
cd demos/streamlit-dashboard
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```


**Ensure the Sentiment API is running at http://localhost:8000 (or set SENTIMENT_API_URL in Streamlit secrets).**

## Run:
```bash

Start Streamlit:

```bash
cd demos/streamlit-dashboard
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

`Open http://localhost:8501`


**Quickstart (docker-compose)**
```bash
docker compose up --build
```



### Docker (if using Dockerfile):

```bash
cd demos/streamlit-dashboard
docker build -t streamlit-dashboard:local .
docker run -p 8501:8501 --link sentiment-api streamlit-dashboard:local
```





