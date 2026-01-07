import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from src.config import *

def train_models():
    X_train = pickle.load(open(PROCESSED_DATA_DIR + "/X_train_tfidf.pkl", "rb"))
    y_train = pickle.load(open(PROCESSED_DATA_DIR + "/y_train.pkl", "rb"))

    os.makedirs(MODEL_DIR, exist_ok=True)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "naive_bayes": MultinomialNB(),
        "linear_svm": LinearSVC()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        pickle.dump(model, open(os.path.join(MODEL_DIR, f"{name}.pkl"), "wb"))
        print(f"✅ {name} trained and saved")

