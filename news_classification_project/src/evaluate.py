import pickle
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from src.config import *

def evaluate_models():
    X_test = pickle.load(open(os.path.join(PROCESSED_DATA_DIR, "X_test_tfidf.pkl"), "rb"))
    y_test = pickle.load(open(os.path.join(PROCESSED_DATA_DIR, "y_test.pkl"), "rb"))

    os.makedirs(RESULTS_DIR, exist_ok=True)

    models = [
        "logistic_regression.pkl",
        "naive_bayes.pkl",
        "linear_svm.pkl"
    ]

    with open(METRICS_PATH, "w") as f:
        for model_file in models:
            model_path = os.path.join(MODEL_DIR, model_file)
            model = pickle.load(open(model_path, "rb"))

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            accuracy_percent = accuracy * 100
            cm = confusion_matrix(y_test, predictions)

            model_name = model_file.replace(".pkl", "")

            f.write(f"\nModel: {model_name}\n")
            f.write(f"Accuracy: {accuracy_percent:.2f}%\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")

            print(f"\nModel: {model_name}")
            print(f"Accuracy: {accuracy_percent:.2f}%")
            print("Confusion Matrix:")
            print(cm)
