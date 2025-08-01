import mlflow
import mlflow.sklearn
import pandas as pd
from utils.logging import logger
from utils.exceptions import CustomException
from utils.helper import configs
import inspect
import sys
from sklearn.ensemble import RandomForestClassifier


def train_model_with_mlflow(X_path: str = configs['processed_X_path_csv'],
                             y_path: str = configs['processed_y_path_csv']):
    """
    Trains the best model as specified in config using MLflow,
    logs all hyperparameters and model artifact.
    """
    try:
        logger.info("📥 Loading training data...")
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path).values.ravel()  # Flatten y

        logger.info("🚀 Starting MLflow run...")
        with mlflow.start_run():

            # Get model class and hyperparams from config
            model_config = configs['best_model']
            model_name = model_config['name']
            hyperparams = model_config['hyperparameters']

            # Instantiate model
            model_class = eval(model_name)
            model = model_class(**hyperparams)

            # Train
            model.fit(X, y)

            # Log model name and all hyperparameters
            mlflow.log_param("model_type", model_name)
            for param, value in hyperparams.items():
                mlflow.log_param(param, value)

            # Save the model artifact
            mlflow.sklearn.log_model(model, "model")

            logger.info(f"✅ {model_name} trained and logged to MLflow.")

            return model

    except Exception as e:
        raise CustomException("Error training model with MLflow", e)
