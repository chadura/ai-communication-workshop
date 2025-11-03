#!/bin/bash
set -e
python3 -m venv venv
source venv/bin/activate
pip install -r demos/spam-detector/requirements.txt
pip install -r demos/sentiment-api/requirements.txt
pip install -r demos/streamlit-dashboard/requirements.txt
echo "Setup complete. Run 'docker compose up' or 'streamlit run demos/streamlit-dashboard/app.py'"