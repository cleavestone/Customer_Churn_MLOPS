from src.components.data_preprocessing import run_preprocessing
from src.components.data_ingestion import data_ingestion
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
        logger.info("Running training pipeline...")
        run_preprocessing()

    except Exception as e:
        raise CustomException("Training pipeline failed", e)


if __name__ == "__main__":
    train_pipeline()
