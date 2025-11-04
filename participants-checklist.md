# Participant Requirements & Pre-Workshop Checklist  
**Workshop:** AI in Communication — Demystifying AI/ML for SMS and Messaging Engineers  
**Edition:** Python-Only (FastAPI + Streamlit + scikit-learn)

---

## 1. System Requirements
To ensure smooth participation, please prepare your environment before Day 1.

### Minimum Hardware
- 8 GB RAM (16 GB recommended)
- 20 GB free disk space
- 64-bit operating system (Windows 10+, macOS 12+, or Ubuntu 24.04+)

### Network
- Reliable internet connection to pull Docker images and Git repo  
- Ability to access `github.com` and `docker.com` (corporate firewalls should allow outbound HTTPS)

---

## 2. Software to Install (Before Day 1)

| Software | Version | Purpose | Download |
|-----------|----------|----------|-----------|
| **Git** | Latest stable | Clone repository | https://git-scm.com/ |
| **Docker Desktop / Engine** | 24+ | Container runtime | https://www.docker.com/ |
| **Python** | 3.10 or 3.11 | Primary coding language | https://www.python.org/ |
| **VS Code** | Latest | Code editor (recommended) | https://code.visualstudio.com/ |
| **Node.js** *(optional)* | 22+ | For React dashboard demo only | https://nodejs.org/ |

---

## 3. Verify Installations

Run the following commands in your terminal:

```bash
git --version
docker --version
python --version
```

## 4. Clone the Repository

```bash
git clone https://github.com/chadura/ai-communication-workshop.git
cd ai-communication-workshop
```

Confirm that the directory structure matches:

```bash
demos/
  spam-detector/
  sentiment-api/
  streamlit-dashboard/
infra/
scripts/
docs/

```


## 5. Python Environment Setup (Local Run Option)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r demos/spam-detector/requirements.txt
pip install -r demos/sentiment-api/requirements.txt
pip install -r demos/streamlit-dashboard/requirements.txt

```


To verify:

```bash
python -c "import fastapi, streamlit, sklearn; print('Environment OK')"
```

## 6. Docker Setup (Recommended)

```bash
docker run hello-world
```

If successful, you’re ready for the hands-on lab.

To start workshop containers:

```bash
docker compose up --build
```


This launches:


`sentiment-api` at http://localhost:8000
`streamlit-dashboard` at http://localhost:8501

| Step                                  | Status |
| ------------------------------------- | ------ |
| Git, Docker, Python installed         | ☐      |
| Repo cloned successfully              | ☐      |
| Docker hello-world test passes        | ☐      |
| Virtual environment ready             | ☐      |
| Ports 8000 & 8501 free                | ☐      |
| Streamlit or React dashboard launches | ☐      |
| Slack/Teams access confirmed          | ☐      |