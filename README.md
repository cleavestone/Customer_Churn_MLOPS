# 🧠 Customer Churn Prediction – End-to-End MLOps Project

This is an end-to-end **Machine Learning Operations (MLOps)** project focused on predicting **customer churn** using a real-world dataset. The goal is to develop a robust ML pipeline—from data ingestion to deployment—that can be monitored, versioned, and maintained in production environments.

---

## 📌 Problem Statement

Customer churn is a critical problem in industries like telecom, banking, and SaaS, where retaining existing customers is more cost-effective than acquiring new ones. This project aims to predict which customers are at risk of leaving, enabling proactive retention strategies.

---

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

## ✅ Key Features

- 📥 **Data Ingestion**: Download and store raw data from Kaggle using `kagglehub`
- 🧹 **Data Preprocessing**: Clean, transform, and handle missing values (to be implemented)
- ⚙️ **Feature Engineering**: Create meaningful features for better model performance
- 🧪 **Model Training**: Train and validate using different ML algorithms
- 📊 **Evaluation**: Log metrics using **MLflow**
- 🧠 **Model Serving**: Serve the trained model via API (FastAPI/Flask)
- 🗃️ **Data/Model Versioning**: Track datasets and model changes using **DVC**
- 🔁 **Automation**: Enable CI/CD pipelines for test + deployment
- ☁️ **Deployment**: Deploy via Docker containers on **AWS EC2**
- 🧾 **Logging & Monitoring**: Custom logging and error handling integrated

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
