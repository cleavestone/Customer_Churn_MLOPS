from src.components.data_preprocessing import run_preprocessing
from src.components.data_ingestion import data_ingestion
from src.components.model_trainer import train_model_with_mlflow
from src.components.evaluation import evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import pandas as pd
import os

from utils.helper import configs
from utils.logging import logger
from utils.exceptions import CustomException

def train_pipeline():
    try:
        # load data
        data_ingestion()
        # handling class imbalance and preprocessing
        logger.info("Running preprocessing pipeline...")
        run_preprocessing()
        # Training model
        logger.info("Training model ....")
        train_model_with_mlflow()
        # model evaluation
        logger.info("Evaluating the model")
        evaluate_model()


    except Exception as e:
        raise CustomException("Training pipeline failed", e)


if __name__ == "__main__":
    train_pipeline()
