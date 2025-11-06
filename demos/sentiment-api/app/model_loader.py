# app/model_loader.py
import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

logger = logging.getLogger("model_loader")

BASE_DIR = Path(__file__).resolve().parent.parent  # demos/sentiment-api
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "model" / "sentiment_model.joblib"

def ensure_sample_data():
    """
    If sample CSV doesn't exist, create a default one (60 rows).
    """
    csv_path = DATA_DIR / "sentiment_sample.csv"
    if csv_path.exists():
        logger.info("Found existing sentiment_sample.csv")
        return csv_path
    logger.info("No sample CSV found. Creating default sentiment_sample.csv")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        # positive (20)
        ("Your order has been successfully delivered!", "positive"),
        ("Thank you for choosing our service!", "positive"),
        ("We appreciate your feedback and look forward to serving you again.", "positive"),
        ("Congratulations! You've been selected for a reward.", "positive"),
        ("Your payment was received successfully.", "positive"),
        ("Our representative will call you shortly.", "positive"),
        ("Your subscription is active and valid until 2026.", "positive"),
        ("Awesome! You just earned 50 loyalty points.", "positive"),
        ("Your return request has been approved.", "positive"),
        ("Your appointment is confirmed for tomorrow at 3 PM.", "positive"),
        ("Your feedback made our day. Thank you!", "positive"),
        ("Discount applied successfully. Enjoy your shopping!", "positive"),
        ("Your recharge was completed successfully.", "positive"),
        ("We value you as our premium customer.", "positive"),
        ("Your gift voucher has been activated.", "positive"),
        ("Limited-time offer: 40% off for all premium users!", "positive"),
        ("Exciting news! New features are now available in your app.", "positive"),
        ("Refer your friends and earn extra rewards!", "positive"),
        ("Congratulations! You’re now a Gold Member.", "positive"),
        ("Enjoy ad-free access for the next 30 days.", "positive"),
        # neutral (20)
        ("Package shipped. Expected delivery by 6 PM today.", "neutral"),
        ("Your OTP for verification is 285639.", "neutral"),
        ("Meeting reminder: 11:30 AM at HQ Room 5.", "neutral"),
        ("System update scheduled at midnight.", "neutral"),
        ("Your subscription will renew automatically next month.", "neutral"),
        ("Invoice generated and sent to your registered email.", "neutral"),
        ("Ticket ID 15423 is under process.", "neutral"),
        ("Your account balance is ₹524.78.", "neutral"),
        ("Payment of ₹1200 scheduled for 10th Nov.", "neutral"),
        ("We are verifying your request. Please wait.", "neutral"),
        ("Update: Our office timings will change from next week.", "neutral"),
        ("Please verify your email to continue.", "neutral"),
        ("You’re registered for webinar 'AI in Communication'.", "neutral"),
        ("New version available. Tap to update now.", "neutral"),
        ("Your support ticket has been assigned to an agent.", "neutral"),
        ("Your account statement is ready for download.", "neutral"),
        ("Session timeout after 10 minutes of inactivity.", "neutral"),
        ("Reminder: Feedback survey closes tomorrow.", "neutral"),
        ("Your profile information has been updated.", "neutral"),
        ("Tracking ID created. Await pickup confirmation.", "neutral"),
        # negative (20)
        ("Unable to process your payment. Try again later.", "negative"),
        ("Your session has expired. Please log in again.", "negative"),
        ("Payment failed due to insufficient balance.", "negative"),
        ("Service temporarily unavailable. We apologize for the inconvenience.", "negative"),
        ("We couldn’t locate your delivery address.", "negative"),
        ("Invalid credentials. Access denied.", "negative"),
        ("We are experiencing delays due to server issues.", "negative"),
        ("Sorry, your request could not be processed at this time.", "negative"),
        ("Delivery failed. Recipient unavailable.", "negative"),
        ("Subscription expired. Please renew to continue.", "negative"),
        ("We regret the delay in resolving your complaint.", "negative"),
        ("Your refund request was declined.", "negative"),
        ("Account suspended due to multiple failed login attempts.", "negative"),
        ("Offer expired. Please check new deals on our website.", "negative"),
        ("Unexpected error occurred. Please contact support.", "negative"),
        ("Recharge unsuccessful. Please retry.", "negative"),
        ("We are sorry for the poor experience you had.", "negative"),
        ("Complaint received. Investigation ongoing.", "negative"),
        ("Technical issue detected. Our team is on it.", "negative"),
        ("Login attempt blocked due to suspicious activity.", "negative"),
    ]
    df = pd.DataFrame(rows, columns=["text", "sentiment"])
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"Created sample CSV at {csv_path}")
    return csv_path

def train_model(csv_path):
    """
    Train a simple CountVectorizer + MultinomialNB classifier and save it.
    Returns a pipeline object and class labels.
    """
    df = pd.read_csv(csv_path)
    X = df["text"].astype(str)
    y = df["sentiment"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    pipeline = Pipeline([
        ("vec", CountVectorizer(ngram_range=(1,2), max_features=5000)),
        ("clf", MultinomialNB())
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    logger.info("Training complete. Evaluation:\n" + classification_report(y_test, preds))
    # Ensure model dir exists
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info(f"Saved model to {MODEL_PATH}")
    return pipeline

def load_or_train_model():
    """
    Load model if available; otherwise create data and train a model and return bundle.
    """
    csv_path = ensure_sample_data()
    if MODEL_PATH.exists():
        logger.info(f"Loading model from {MODEL_PATH}")
        pipeline = joblib.load(MODEL_PATH)
    else:
        logger.info("No saved model found. Training a new model from sample CSV.")
        pipeline = train_model(csv_path)
    bundle = {
        "vectorizer": pipeline.named_steps["vec"],
        "model": pipeline.named_steps["clf"] if "clf" in pipeline.named_steps else pipeline,
    }
    # store pipeline in bundle so we can persist/predict if needed
    bundle["pipeline"] = pipeline
    return bundle

def retrain_model_if_requested(force=False):
    """
    Retrain the model; if force=True retrain regardless of existing model.
    """
    csv_path = ensure_sample_data()
    if not force and MODEL_PATH.exists():
        logger.info("Model exists and force=False. Loading existing model.")
        return {"pipeline": joblib.load(MODEL_PATH), "vectorizer": None, "model": None}
    pipeline = train_model(csv_path)
    logger.info("Retraining complete.")
    return {"pipeline": pipeline, "vectorizer": pipeline.named_steps["vec"], "model": pipeline.named_steps["clf"]}
