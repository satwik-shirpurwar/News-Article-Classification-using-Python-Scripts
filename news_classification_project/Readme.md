
📰 News Article Classification using Machine Learning

📌 Project Overview

This project implements a complete machine learning pipeline to classify news articles into predefined categories using Python scripts only. It covers data preprocessing, feature engineering, model training, and evaluation, and is fully runnable from the terminal without using Jupyter notebooks.


📊 Dataset Source

AG News Classification Dataset (Public Dataset)
Source: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset/data

The dataset contains news articles categorized into four classes:

- World
- Sports
- Business
- Science/Technology

Each record consists of:

- Class Index
- Title
- Description


📁 Folder Structure Explanation

news_classification_project/
│
├── data/                              # Contains all dataset-related files
│   ├── processed/                    # Preprocessed and feature-engineered data
│   │   ├── tfidf.pkl                 # Saved TF-IDF vectorizer
│   │   ├── X_test_tfidf.pkl          # TF-IDF features for test data
│   │   ├── X_test.pkl                # Cleaned test text
│   │   ├── X_train_tfidf.pkl         # TF-IDF features for training data
│   │   ├── X_train.pkl               # Cleaned training text
│   │   ├── y_test.pkl                # Test labels
│   │   └── y_train.pkl               # Training labels
│   │
│   └── raw/                          # Original dataset (unchanged)
│       ├── test.csv                  # Raw test dataset
│       └── train.csv                 # Raw training dataset
│
├── models/                           # Trained machine learning models
│   ├── linear_svm.pkl                # Linear Support Vector Machine model
│   ├── logistic_regression.pkl       # Logistic Regression model
│   └── naive_bayes.pkl               # Multinomial Naive Bayes model
│
├── results/                          # Evaluation outputs
│   └── metrics.txt                   # Accuracy and confusion matrices
│
├── src/                              # Source code files│
│   ├── config.py                     # Project paths and configuration settings
│   ├── data_preprocessing.py         # Data loading, cleaning, preprocessing
│   ├── evaluate.py                   # Model evaluation logic
│   ├── feature_engineering.py        # TF-IDF feature extraction
│   └── train.py                      # Model training scripts
│
├── main.py                           # Entry point to run full ML pipeline
├── Readme.md                         # Project documentation
└── requirements.txt                  # Required Python libraries




▶️ Steps to Run the Project

1. Navigate to the project root directory: cd news_classification_project
2. Install required dependencies: pip install -r requirements.txt
3. Run the complete machine learning pipeline: python main.py

This will execute:

✅Data preprocessing
✅Feature engineering
✅Model training
✅Model evaluation



🤖 Models Used

The following machine learning models were implemented and compared:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (Linear SVM)
- All models were trained using TF-IDF features.


📈 Final Result Summary

Model	             Accuracy
Logistic Regression   90.92%
Naive Bayes	          89.53%
Linear SVM	          91.05%


Model: logistic_regression
Accuracy: 90.92%
Confusion Matrix:
[[5401  190  243  166]
 [  84 5840   40   36]
 [ 200   45 5287  468]
 [ 204   77  427 5292]]

Model: naive_bayes
Accuracy: 89.53%
Confusion Matrix:
[[5346  223  258  173]
 [  98 5826   37   39]
 [ 242   62 5128  568]
 [ 256  112  444 5188]]

Model: linear_svm
Accuracy: 91.05%
Confusion Matrix:
[[5395  188  259  158]
 [  65 5861   37   37]
 [ 190   50 5287  473]
 [ 200   77  413 5310]]


🏆 Conclusion

- Linear SVM achieved the highest balanced performance among all models.
- The project demonstrates a complete, modular, and industry-standard ML workflow.
- All evaluation metrics and confusion matrices are saved in results/metrics.txt.
- The classification results show strong performance across all four news categories.