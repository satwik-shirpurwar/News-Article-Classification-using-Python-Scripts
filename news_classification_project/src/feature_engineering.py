import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import *

def vectorize_text():
    # Load preprocessed text
    X_train = pickle.load(open(os.path.join(PROCESSED_DATA_DIR, "X_train.pkl"), "rb"))
    X_test = pickle.load(open(os.path.join(PROCESSED_DATA_DIR, "X_test.pkl"), "rb"))

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Save features and vectorizer
    pickle.dump(X_train_tfidf, open(os.path.join(PROCESSED_DATA_DIR, "X_train_tfidf.pkl"), "wb"))
    pickle.dump(X_test_tfidf, open(os.path.join(PROCESSED_DATA_DIR, "X_test_tfidf.pkl"), "wb"))
    pickle.dump(vectorizer, open(os.path.join(PROCESSED_DATA_DIR, "tfidf.pkl"), "wb"))

    print("✅ Feature engineering using TF-IDF completed")
