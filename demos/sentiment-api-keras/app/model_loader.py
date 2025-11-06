# app/model_loader.py
import os
from pathlib import Path
import json
import logging
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

logger = logging.getLogger("model_loader")

BASE_DIR = Path(__file__).resolve().parent.parent  # demos/sentiment-api-keras
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "keras_sentiment_model"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"
META_PATH = MODEL_DIR / "meta.json"

DEFAULT_MAX_WORDS = 8000
DEFAULT_MAXLEN = 40
DEFAULT_EMBED_DIM = 64
DEFAULT_BATCH = 16
DEFAULT_EPOCHS = 10

class KerasModelManager:
    def __init__(self,
                 max_words=DEFAULT_MAX_WORDS,
                 maxlen=DEFAULT_MAXLEN,
                 embed_dim=DEFAULT_EMBED_DIM):
        self.max_words = max_words
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.tokenizer = None
        self.model = None
        self.class_names = None

    def ensure_data(self):
        csv_path = DATA_DIR / "sentiment_sample.csv"
        if csv_path.exists():
            logger.info(f"Found dataset at {csv_path}")
            return csv_path
        logger.info("No sample CSV found. Creating default sentiment_sample.csv")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        rows = [
            # keep same 60 rows as previous generator (20 per class)
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

    def build_tokenizer_and_sequences(self, texts):
        tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.maxlen, padding="post", truncating="post")
        return tokenizer, padded

    def build_model(self, vocab_size, embed_dim=None):
        embed_dim = embed_dim or self.embed_dim
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=self.maxlen))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.3))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(len(self.class_names), activation="softmax"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def train(self, force=False):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = self.ensure_data()
        df = pd.read_csv(csv_path)
        texts = df["text"].astype(str).tolist()
        labels = df["sentiment"].astype(str).tolist()
        # Map classes to ints
        classes = sorted(list(set(labels)))
        self.class_names = classes
        label_to_idx = {c: i for i, c in enumerate(classes)}
        y = np.array([label_to_idx[l] for l in labels], dtype=np.int32)

        # tokenizer + sequences
        tokenizer, padded = self.build_tokenizer_and_sequences(texts)
        vocab_size = min(self.max_words, len(tokenizer.word_index) + 1)

        # split
        idxs = np.arange(len(padded))
        # small dataset so do simple split
        split = int(len(padded) * 0.85)
        train_x, test_x = padded[:split], padded[split:]
        train_y, test_y = y[:split], y[split:]

        model = self.build_model(vocab_size=vocab_size)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
        ]
        model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH, callbacks=callbacks, verbose=1)

        # Save model and tokenizer and meta
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_PATH.as_posix(), include_optimizer=False)
        joblib.dump(tokenizer, TOKENIZER_PATH)
        meta = {"class_names": self.class_names, "max_words": self.max_words, "maxlen": self.maxlen, "embed_dim": self.embed_dim}
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        # load into instance
        self.tokenizer = tokenizer
        self.model = model
        logger.info(f"Trained and saved model at {MODEL_PATH}")
        return True

    def ensure_data(self):
        # compatibility: existing generate_sample_data.py path may differ
        csv_path = DATA_DIR / "sentiment_sample.csv"
        if not csv_path.exists():
            return self.ensure_data()  # this will create via ensure_data's logic above
        return csv_path

    def load_or_train(self):
        # Attempt to load model and tokenizer; if missing, train
        if MODEL_PATH.exists():
            try:
                self.model = load_model(MODEL_PATH.as_posix(), compile=False)
                if TOKENIZER_PATH.exists():
                    self.tokenizer = joblib.load(TOKENIZER_PATH)
                if META_PATH.exists():
                    with open(META_PATH, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        self.class_names = meta.get("class_names", ["negative", "neutral", "positive"])
                logger.info("Loaded existing Keras model and tokenizer.")
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}. Will retrain.")
        # Create sample data and train
        csv_path = self.ensure_data()
        return self.train(force=True)

    def predict_text(self, text: str):
        if self.model is None or self.tokenizer is None:
            logger.info("Model/tokenizer missing. Triggering training.")
            self.load_or_train()
        seq = self.tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=self.maxlen, padding="post", truncating="post")
        preds = self.model.predict(pad, verbose=0)[0]  # softmax vector
        idx = int(np.argmax(preds))
        label = self.class_names[idx]
        proba = {c: float(preds[i]) for i, c in enumerate(self.class_names)}
        return label, proba
