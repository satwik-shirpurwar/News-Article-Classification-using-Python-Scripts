from src.data_preprocessing import preprocess_data
from src.feature_engineering import vectorize_text
from src.train import train_models
from src.evaluate import evaluate_models

def main():
    preprocess_data()
    vectorize_text()
    train_models()
    evaluate_models()

if __name__ == "__main__":
    main()
