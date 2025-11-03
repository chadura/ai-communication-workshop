# ai-communication-workshop


## Requirements

* Python ≥ 3.10
* Docker ≥ 24
* VS Code recommended


```bash
ai-communication-workshop/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ workshop-schedule.md
├─ participants-checklist.md
├─ demos/
│  ├─ spam-detector/
│  │  ├─ data/sms_spam_collection.csv
│  │  ├─ notebooks/train_model.ipynb
│  │  ├─ train.py
│  │  ├─ model.joblib (gitignored; fetch via script or storage)
│  │  └─ requirements.txt
│  ├─ sentiment-api/
│  │  ├─ app/
│  │  │  └─ main.py
│  │  ├─ Dockerfile
│  │  ├─ requirements.txt
│  │  └─ README.md
│  ├─ streamlit-dashboard/
│  │  ├─ app.py
│  │  ├─ requirements.txt
│  │  └─ README.md
│  └─ react-dashboard/         # optional frontend
│     ├─ package.json
│     ├─ src/
│     │  └─ App.jsx
│     └─ README.md
├─ infra/
│  └─ docker-compose.yml
├─ scripts/
│  ├─ setup_local.sh
│  └─ run_all.sh
└─ docs/
   ├─ ai-for-communication-handbook.pdf
   └─ code-snippets.md
```


## Key repo files explained

`README.md` — Quick start, goals, workshop flow and timing

`workshop-schedule.md` — Timed agenda for the two days with break times and lab durations

`demos/spam-detector/` — Full pipeline: dataset, training script, saved model, and a FastAPI wrapper (Python-only)

`demos/sentiment-api/` — FastAPI service that wraps a simple sentiment model and exposes /analyze endpoint (Python)

`demos/streamlit-dashboard/` — Pure-Python dashboard (Streamlit) that queries sentiment API and visualizes message-level analytics

`demos/react-dashboard/` — Optional React app for teams comfortable with JS that queries the FastAPI endpoints and shows charts

`infra/docker-compose.yml` — Bring up sentiment API + streamlit dashboard + mock services in one command (all Python containers)

`scripts/setup_local.sh` — One-click helper to prepare local Python venvs and download small model artifacts


### Quick Start (Python)

**1. Clone repo:**

```bash
git clone https://github.com/chadura/ai-communication-workshop.git
cd ai-communication-workshop

bash scripts/setup_local.sh
```

**2. Start services with Docker Compose (recommended)::**

```bash
docker compose up --build
```

This will build and run the sentiment API and the Streamlit dashboard (or React dashboard if configured) on ports documented in `infra/docker-compose.yml`

**3.Run the spam detector training locally (optional):**
```bash
cd demos/spam-detector
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

**4.Run the Streamlit dashboard (optional pure-Python dashboard):**

```bash
cd demos/streamlit-dashboard
pip install -r requirements.txt
streamlit run app.py
```

**5.(Optional) If using React dashboard:**

```bash
cd demos/react-dashboard
npm install
npm run dev
```

## CI / Branching / Versioning recommendations

- Branch model: `main` (stable), `dev` (workshop artifacts), `lab-N` (each hands-on lab). Instructor merges lab branches into`dev` post-workshop.

- Tag releases for official workshop artifacts (e.g., `v2025-11-03-workshop`).

- Avoid committing large model files; use Git LFS or store models in a storage bucket and provide `scripts/download_artifacts.sh`.

## Datasets and licensing

- Demo datasets included are small, publicly available datasets (e.g., SMS Spam Collection Dataset). Include citation and license in each demo's README.

- Do **not** include proprietary customer logs. Participants who want to test with real logs must anonymize data and follow compliance rules.

## Security & compliance notes for participants

- Never upload PII or production logs to the public repository.

- For cloud demos with sensitive data, use masked/anonymized samples only.

- Consult security/compliance (HIPAA, GDPR, TCPA) before running any workshop using real production data.


## Trainer notes (docs/trainer-notes.md)

Environment variables needed for demos (store in .env.example):

`SENTIMENT_API_PORT=8000`

`MODEL_PATH=./model.joblib`

Suggested runtime for each demo and fallback paths if Docker is unavailable.

Troubleshooting section: common Docker issues, port conflicts, Windows firewall steps, Python venv pitfalls.
