
## App Demo
![Demo](static/video/churn_demo.gif)

# 🧠 Customer Churn Prediction – End-to-End MLOps Project

This is an end-to-end **Machine Learning Operations (MLOps)** project focused on predicting **customer churn** using a real-world dataset. The goal is to develop a robust ML pipeline—from data ingestion to deployment—that can be monitored, versioned, and maintained in production environments.

---

## 📌 Problem Statement

Customer churn is a critical problem in industries like telecom, banking, and SaaS, where retaining existing customers is more cost-effective than acquiring new ones. This project aims to predict which customers are at risk of leaving, enabling proactive retention strategies.

We use data from **ABC Multistate Bank**, which includes demographic and account-related information for each customer. The objective is to build a predictive model that accurately determines whether a customer will churn (leave the bank) based on their profile.

---

## 📂 Project Highlights

- ✅ Modular ML codebase (components for ingestion, preprocessing, training, and evaluation)
- 🧪 Experiment tracking using **MLflow**
- 📦 Data versioning with **DVC**
- 🧑‍💻 Code versioning & collaboration using **Git + GitHub**
- 🐳 Containerized using **Docker**
- ☁️ Deployment-ready with **Azure Blob Storage**, **CI/CD workflows**, and cloud infrastructure
- 📈 Model & data drift monitoring with **Grafana**
- 📋 Logging and exception handling implemented for reliability

---

## 📊 Data Dictionary

The dataset contains the following columns:

| Column Name        | Description                                                                 | Data Type     | Notes                     |
|--------------------|-----------------------------------------------------------------------------|---------------|---------------------------|
| `customer_id`      | Unique identifier for each customer                                          | String        | Unused in modeling        |
| `credit_score`     | Customer’s credit score                                                      | Integer       | Used as input             |
| `country`          | Country of residence (e.g., France, Germany, Spain)                          | Categorical   | Used as input             |
| `gender`           | Customer gender (Male/Female)                                                | Categorical   | Used as input             |
| `age`              | Customer age in years                                                        | Integer       | Used as input             |
| `tenure`           | Number of years the customer has been with the bank                          | Integer       | Used as input             |
| `balance`          | Account balance in dollars                                                   | Float         | Used as input             |
| `products_number`  | Number of banking products used by the customer                              | Integer       | Used as input             |
| `credit_card`      | Whether the customer has a credit card (1 = Yes, 0 = No)                     | Binary        | Used as input             |
| `active_member`    | Whether the customer actively uses their account (1 = Yes, 0 = No)           | Binary        | Used as input             |
| `estimated_salary` | Estimated annual salary of the customer                                      | Float         | Used as input             |
| `churn`            | Target variable (1 = Customer churned, 0 = Stayed with the bank)             | Binary        | **Target** for prediction |

## ⚙️ Project Architecture



```
ml_project/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── models/
│   └── model.pkl
│
├── notebooks/                      
│   ├── 01_eda.ipynb
│   ├── 02_model_experiments.ipynb
│   └── 
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── model_training.py
│   │   ├── evaluation.py
│   │   ├── inference.py
│   │   └── model_serving.py
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── inference_pipeline.py
│   │
│   └── utils/
│       ├── logger.py
│       └── exception.py
│
├── mlruns/
│
├── dvc.yaml
├── Dockerfile
├── requirements.txt
├── .gitignore
├── .dvcignore
├── README.md
└── .github/
    └── workflows/
        └── ci-cd.yml

```
## ✅ Model Training & Evaluation

After completing preprocessing and feature engineering, multiple models were trained and compared:

- 🎯 **Random Forest**
- 🎯 **XGBoostClassifier**
- 🎯 **CatBoostClassifier**
- ✅ **LightGBMClassifier** *(Selected)*

The **LightGBMClassifier** was selected as the final model due to its superior **recall and precision** for the **positive class** (i.e., customers who churned).

---

### 📊 Evaluation Metrics (Test Set)

| Metric            | Score    |
|-------------------|----------|
| Accuracy          | 0.7870   |
| Precision         | 0.4746   |
| Recall            | 0.7837   |
| F1 Score          | 0.5912   |
| ROC AUC Score     | 0.8604   |

---

### 📋 Classification Report

          precision    recall  f1-score   support

       0       0.94      0.79      0.86      1607
       1       0.47      0.78      0.59       393

accuracy                           0.79      2000


✅ The model correctly identified **308 out of 393** churning customers.

---

### 📦 MLflow Tracking

- ✅ All models and metrics logged to **MLflow**
- ✅ Best-performing model (**LightGBMClassifier**) **registered**
- ✅ Tracked metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC






---

## 🏗️ Technologies Used

| Category        | Tools/Tech |
|----------------|------------|
| Language        | Python |
| Data Source     | Kaggle Dataset ([Link](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)) |
| Data Handling   | Pandas, Scikit-learn |
| Experiment Tracking | MLflow |
| Data Versioning | DVC + Azure Blob |
| Source Control  | Git + GitHub |
| Containerization | Docker |
| Cloud Deployment | Azure VMs + ACR |
| CI/CD Automation | GitHub Actions |
| Monitoring & Logging | Custom Logger + Exception Handling |

---


## 🚧 Current Progress

- ✅ Data ingestion module implemented
- ✅ Custom logger and exception handling added
- 🔄 Next: preprocessing → feature engineering → model training

---

## 📈 Future Plans

- [ ] Add data preprocessing pipeline  
- [ ] Implement feature engineering  
- [ ] Train multiple models & log with MLflow  
- [ ] Setup DVC for data versioning  
- [ ] Containerize with Docker  
- [ ] Setup CI/CD with GitHub Actions  
- [ ] Deploy on AWS (EC2 + ECR)  
- [ ] Monitor drift and model health  

---

## 📂 Getting Started

Clone the repository:

```bash
git clone https://github.com/your-username/ml-project-churn.git
cd ml-project-churn
