# Heart-disease-prediction

## 📌 Overview
This project is a **Machine Learning application** that predicts the likelihood of heart disease using clinical data.  
It was developed as a **graduation project** for the *AI and Machine Learning program at Sprints*.  
The project implements a **complete ML pipeline**:
- Data preprocessing & cleaning  
- Dimensionality reduction (PCA)  
- Feature selection (Random Forest, Chi-Square, RFE)  
- Model training & evaluation (Logistic Regression, SVM, Random Forest, KNN, Decision Tree)  
- Hyperparameter tuning  
- Unsupervised learning (K-Means, Hierarchical Clustering)


## 🗂️ Project Structure

│── data/

│ └── heart_disease.csv # Original dataset

│── notebooks/

│ ├── 01_data_preprocessing.ipynb # Data cleaning & preprocessing

│ ├── 02_pca_analysis.ipynb # Principal Component Analysis

│ ├── 03_feature_selection.ipynb # Feature importance & selection

│ ├── 04_supervised_learning.ipynb # Classification models

│ ├── 05_unsupervised_learning.ipynb # Clustering approaches

│ ├── 06_hyperparameter_tuning.ipynb # GridSearch, RandomizedSearch

│── models/

│ └── final_model.pkl # Trained best model

│ └── ngrok_setup.txt # Guide for public deployment

│── results/

│ └── evaluation_metrics.txt # Accuracy, precision, recall, etc.

│── README.md

│── requirements.txt

│── .gitignore


## 🖥️ Technologies Used

- **Python 3.9+**
- **Jupyter Notebook** 
- **Pandas** – data manipulation  
- **NumPy** – numerical operations  
- **Matplotlib & Seaborn** – data visualization  
- **Scikit-learn** – preprocessing, PCA, feature selection, ML models



## 📊 Results

Models evaluated: Logistic Regression, Random Forest, SVM, KNN, Decision Tree

Best model achieved 91% accuracy .

Full evaluation metrics are saved in: results/evaluation_metrics.txt.


## 🔮 Future Improvements

Add deep learning models (TensorFlow/Keras) for comparison.

Collect and use a larger dataset.

Deploy on cloud platforms (Heroku, AWS, or Azure) for persistent hosting.


## 👩‍💻 Author

Youssef Abdeltawab 

GitHub: Youssef20015

LinkedIn: www.linkedin.com/in/youssefabdeltawab
