# 📰 News Article Classification using Machine Learning

## 📌 Project Overview
This project implements a complete end-to-end machine learning pipeline to classify news articles into predefined categories using Python scripts only. The pipeline includes data preprocessing, feature engineering, model training, and evaluation, and the project is fully runnable from the terminal without using Jupyter notebooks.

---

## 📊 Dataset Source
**AG News Classification Dataset (Public Dataset)**  
Source: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset  

The dataset contains news articles categorized into four classes:
- World  
- Sports  
- Business  
- Science/Technology  

Each record consists of:
- Class Index  
- Title  
- Description  

---

## 📁 Folder Structure Explanation

<img width="843" height="820" alt="image" src="https://github.com/user-attachments/assets/e22c457f-c7d5-4a93-84b0-002131c80f10" />


## ▶️ Steps to Run the Project

1. Navigate to the project root directory:
   ```bash
   cd news_classification_project


This will execute:
✅Data preprocessing
✅Feature engineering
✅Model training
✅Model evaluation


## 🤖 Models Used
The following machine learning models were implemented and compared:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (Linear SVM)

- All models were trained using TF-IDF feature representation.


## 📈 Final Result Summary

<img width="554" height="700" alt="image" src="https://github.com/user-attachments/assets/599a03f9-3786-4308-9843-0fdcb231a6da" />


## 🏆 Conclusion

- Linear SVM achieved the highest balanced performance among all models.
- The project demonstrates a complete, modular, and industry-standard ML workflow.
- All evaluation metrics and confusion matrices are saved in results/metrics.txt.
- The classification results show strong performance across all four news categories.
