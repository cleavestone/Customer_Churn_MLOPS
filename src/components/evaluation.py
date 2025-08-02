import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from utils.helper import configs
from utils.logging import logger
from utils.exceptions import CustomException
import mlflow
from mlflow.sklearn import load_model

def evaluate_model():
    try:
        logger.info("Loading validation data...")
        val_df = pd.read_csv(configs['val_path'])
        y_val = val_df[configs['target_column']]
        X_val = val_df.drop(columns=[configs['target_column'], 'customer_id'])

        logger.info("Loading preprocessor and transforming validation data...")
        preprocessor = joblib.load(configs['preprocessor_obj'])
        X_val_transformed = preprocessor.transform(X_val)
        cat_cols = configs['categorical_columns']
        
        X_val_df = pd.DataFrame(X_val_transformed, columns=configs['all_columns'])


        logger.info("📥 Loading best model from MLflow...")
        logged_model = configs['logged_model']

        # Load model as a PyFuncModel.
        model = mlflow.pyfunc.load_model(logged_model)

        #logged_model_uri = f"runs:/{configs['best_run_id']}/model"
        #model = mlflow.sklearn.load_model(logged_model_uri)
        
        print(X_val_df)

        logger.info("Making predictions...")
        y_pred = model.predict(X_val_df)

        logger.info("Computing evaluation metrics...")
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)

        logger.info("Logging metrics to MLflow...")
        with mlflow.start_run(run_name="model_evaluation"):
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)

            cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
            cm_df.to_csv("confusion_matrix.csv", index=True)
            mlflow.log_artifact("confusion_matrix.csv")

            report_df = pd.DataFrame(report).transpose()
            #report_df.to_csv("classification_report.csv")
            mlflow.log_artifact("classification_report.csv")

        logger.info("Evaluation complete.")
        print("Evaluation Complete!")
        print(f"Accuracy     : {acc:.4f}")
        print(f"F1 Score     : {f1:.4f}")
        print(f"Precision    : {precision:.4f}")
        print(f"Recall       : {recall:.4f}")
        print("\n🧾 Confusion Matrix:")
        print(cm)

    except Exception as e:
        logger.error("❌ Error occurred during evaluation.", exc_info=True)
        raise CustomException("Error occurred during model evaluation", e)


if __name__ == "__main__":
    evaluate_model()
