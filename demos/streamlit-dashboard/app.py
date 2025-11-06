"""
Streamlit Dashboard for AI in Communication workshop
- Single message analysis
- Batch analysis (sample messages / uploaded CSV)
- Visual summary of predictions
- Fallback local rule-based classifier if API is unavailable

Expect the Sentiment API at: http://sentiment-api:8000/analyze (docker-compose)
or http://localhost:8000/analyze for local dev.
"""

import streamlit as st
import pandas as pd
import requests
from pathlib import Path
from typing import List, Dict
import time

# Config
API_URL_ENV = st.secrets.get("SENTIMENT_API_URL") if "SENTIMENT_API_URL" in st.secrets else None
DEFAULT_API_URL = API_URL_ENV or "http://localhost:8000/analyze"
SAMPLES_PATH = Path(__file__).resolve().parent / "assets" / "sample_messages.json"

st.set_page_config(page_title="AI Communication Dashboard", layout="wide")

st.title("AI Communication — Sentiment Dashboard")
st.markdown(
    "Analyze SMS / message text with the workshop sentiment API. "
    "If the API is not reachable, a local fallback classifier will provide a basic prediction."
)

# -------------------------
# Helpers
# -------------------------
def call_api_single(text: str, api_url: str = DEFAULT_API_URL, timeout: float = 3.0) -> Dict:
    payload = {"text": text}
    try:
        r = requests.post(api_url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        # Return a dict with error; caller will use fallback
        return {"error": str(e)}

def call_api_batch(texts: List[str], api_url: str = DEFAULT_API_URL, timeout: float = 3.0) -> List[Dict]:
    results = []
    for t in texts:
        r = call_api_single(t, api_url=api_url, timeout=timeout)
        results.append(r)
    return results

# very simple fallback classifier (rule-based) for offline demo
POS_KEYWORDS = {"thank", "thanks", "success", "congrat", "great", "awesome", "good", "earned", "reward", "won", "successfully", "thanks"}
NEG_KEYWORDS = {"fail", "failed", "unable", "sorry", "delay", "declin", "error", "unsuccessful", "cancel", "suspend", "blocked", "complaint", "issue"}

def fallback_classify(text: str) -> Dict:
    txt = text.lower()
    pos = sum(1 for k in POS_KEYWORDS if k in txt)
    neg = sum(1 for k in NEG_KEYWORDS if k in txt)
    if pos > neg:
        pred = "positive"
    elif neg > pos:
        pred = "negative"
    else:
        pred = "neutral"
    # crude probability proxy
    proba = {"positive": float(pos / (pos + neg + 1)), "negative": float(neg / (pos + neg + 1)), "neutral": float(1.0 - ((pos + neg) / (pos + neg + 1)))}
    return {"text": text, "prediction": pred, "proba": proba, "fallback": True}

# load sample messages
def load_samples() -> pd.DataFrame:
    if not SAMPLES_PATH.exists():
        return pd.DataFrame({"text": []})
    df = pd.read_json(SAMPLES_PATH)
    if "text" not in df.columns:
        df = pd.DataFrame({"text": df})
    return df

# run batch analysis with progress
def run_batch_analysis(texts: List[str], api_url: str = DEFAULT_API_URL) -> pd.DataFrame:
    results = []
    progress = st.progress(0)
    total = len(texts)
    for i, t in enumerate(texts):
        r = call_api_single(t, api_url=api_url)
        if r.get("error"):
            # fallback when API not reachable
            fb = fallback_classify(t)
            results.append({"text": t, "prediction": fb["prediction"], "proba": fb["proba"], "fallback": True, "api_error": r.get("error")})
        else:
            results.append({"text": t, "prediction": r.get("prediction"), "proba": r.get("proba", {}), "fallback": False, "api_error": None})
        time.sleep(0.05)  # tiny throttle to show progress
        progress.progress(int((i + 1) / total * 100))
    progress.empty()
    return pd.DataFrame(results)

# -------------------------
# Layout
# -------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Single Message Analysis")
    sample_df = load_samples()
    sample_list = sample_df["text"].tolist() if not sample_df.empty else []
    select_sample = st.selectbox("Choose a sample message", options=["(type your own)"] + sample_list, index=0)
    input_text = st.text_area("Text to analyze", value=(select_sample if select_sample != "(type your own)" else ""), height=120)
    st.write("Model endpoint:", DEFAULT_API_URL)
    if st.button("Analyze"):
        if not input_text or not input_text.strip():
            st.warning("Please enter text to analyze.")
        else:
            resp = call_api_single(input_text)
            if resp.get("error"):
                st.error(f"API Error: {resp.get('error')}. Using fallback classifier.")
                fb = fallback_classify(input_text)
                st.write("Prediction (fallback):", fb["prediction"])
                st.json(fb["proba"])
            else:
                st.success(f"Prediction: {resp.get('prediction')}")
                st.json(resp.get("proba"))

    st.markdown("---")
    st.subheader("Batch Analysis")
    st.markdown("Choose a source of messages for batch analysis.")

    batch_source = st.radio("Source", options=["Sample messages file", "Upload CSV (column: text)"], index=0)
    texts_for_batch = []
    if batch_source == "Sample messages file":
        st.write(f"{len(sample_list)} sample messages loaded.")
        sample_selector = st.multiselect("Pick sample messages (or leave empty to run all)", options=sample_list, default=sample_list[:10])
        texts_for_batch = sample_selector if sample_selector else sample_list
    else:
        uploaded = st.file_uploader("Upload CSV with `text` column", type=["csv"])
        if uploaded is not None:
            df_up = pd.read_csv(uploaded)
            if "text" not in df_up.columns:
                st.error("CSV must contain a `text` column.")
            else:
                st.write(f"Loaded {len(df_up)} rows.")
                texts_for_batch = df_up["text"].astype(str).tolist()

    if texts_for_batch:
        if st.button("Run batch analysis"):
            with st.spinner("Analyzing messages..."):
                result_df = run_batch_analysis(texts_for_batch, api_url=DEFAULT_API_URL)
            st.success("Batch analysis complete.")
            st.dataframe(result_df[["text", "prediction", "fallback"]].rename(columns={"text":"message"}), height=300)
            # summarize
            counts = result_df["prediction"].value_counts().reset_index()
            counts.columns = ["prediction", "count"]
            st.subheader("Summary")
            st.table(counts)
            st.bar_chart(counts.set_index("prediction"))
            # show probability breakdown for first few
            st.subheader("Sample probabilities (first 10)")
            def normalize_proba(p):
                # input may be dict of strings->float or missing
                if not isinstance(p, dict):
                    return {}
                return {k: float(v) for k,v in p.items()}
            proba_df = pd.DataFrame([normalize_proba(x) for x in result_df["proba"].tolist()]).fillna(0)
            proba_df["message"] = result_df["text"].astype(str)
            st.dataframe(proba_df.head(10).set_index("message"))

with col2:
    st.subheader("Utilities & Settings")
    st.markdown(
        "You can change the API endpoint below. If running via Docker Compose the hostname is often `sentiment-api`."
    )
    api_input = st.text_input("Sentiment API URL", value=DEFAULT_API_URL)
    if api_input and api_input != DEFAULT_API_URL:
        # update runtime default (note: won't persist across reloads)
        DEFAULT_API_URL = api_input
        st.success(f"API endpoint set to: {api_input}")

    st.markdown("---")
    st.write("Quick actions")
    if st.button("Health check API"):
        try:
            r = requests.get(DEFAULT_API_URL.replace("/analyze", "/health"), timeout=2.0)
            r.raise_for_status()
            st.success(f"API healthy: {r.json()}")
        except Exception as e:
            st.error(f"API health check failed: {e}")

    if st.button("Run test batch (10 samples)"):
        samples = sample_list[:10] if sample_list else []
        if not samples:
            st.warning("No sample messages available.")
        else:
            with st.spinner("Running test batch..."):
                df_res = run_batch_analysis(samples, api_url=DEFAULT_API_URL)
            st.success("Done")
            st.table(df_res["prediction"].value_counts().rename_axis("prediction").reset_index(name="count"))

    st.markdown("---")
    st.info(
        "If the API is down the app will fallback to a rule-based classifier (basic demo only). "
        "This dashboard is intended for workshop use and demonstration rather than production."
    )

st.markdown("---")
st.caption("AI in Communication — Streamlit Dashboard (workshop demo).")
