import mlflow
import mlflow.sklearn
import pandas as pd
from utils.logging import logger
from utils.exceptions import CustomException
from utils.helper import configs


def train_model_with_mlflow(X_path, y_path):
    """
    Trains a RandomForestClassifier using data loaded from CSV files,
    logs training info with MLflow, and saves the model.
    """
    try:
        logger.info("Loading data for training...")
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path).values.ravel()  # flatten to 1d array
        
        logger.info("Starting MLflow run for training...")
        with mlflow.start_run():
            # Instantiate model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Train model
            model.fit(X, y)
            
            # Log parameters
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)
            
            # (Optional) Calculate & log metrics here if you have a validation set
            
            # Log model artifact
            mlflow.sklearn.log_model(model, "model")
            
            logger.info("Model trained and logged with MLflow successfully.")
            
            return model
    
    except Exception as e:
        raise CustomException("Error training model with MLflow", e)

