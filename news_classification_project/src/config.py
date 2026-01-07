import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TRAIN_FILE = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_FILE = os.path.join(RAW_DATA_DIR, "test.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "news_classifier.pkl")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.txt")
