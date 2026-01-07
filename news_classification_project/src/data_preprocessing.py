import pandas as pd
import re
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.config import *

def clean_text(text):
    # Convert to string (safety)
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove numbers and special characters
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]

    # Join words back
    text = " ".join(words)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_data():
    # 1️⃣ Load dataset
    df = pd.read_csv(TRAIN_FILE)

    # 2️⃣ Handle missing values
    df = df.dropna(subset=["Title", "Description", "Class Index"])

    # 3️⃣ Combine Title and Description
    df["text"] = df["Title"] + " " + df["Description"]

    # 4️⃣ Clean text
    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["Class Index"]

    # 5️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6️⃣ Save processed data
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    pickle.dump(X_train, open(os.path.join(PROCESSED_DATA_DIR, "X_train.pkl"), "wb"))
    pickle.dump(X_test, open(os.path.join(PROCESSED_DATA_DIR, "X_test.pkl"), "wb"))
    pickle.dump(y_train, open(os.path.join(PROCESSED_DATA_DIR, "y_train.pkl"), "wb"))
    pickle.dump(y_test, open(os.path.join(PROCESSED_DATA_DIR, "y_test.pkl"), "wb"))

    print("✅ Data preprocessing completed successfully")
