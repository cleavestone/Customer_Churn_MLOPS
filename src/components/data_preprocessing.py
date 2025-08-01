from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from utils.logging import logger
from utils.exceptions import CustomException
from utils.helper import configs
import sys
import os
import joblib



def handle_imbalance(path: str = configs['train_path']):
    """
    Loads training data and handles class imbalance using RandomOverSampler.
    Returns resampled X and y.
    """
    try:
        logger.info("Reading training data for imbalance handling...")
        df = pd.read_csv(path)
        X_train = df.drop(columns=configs['target_column'])
        y_train = df[configs['target_column']]
        
        logger.info("Handling imbalance using RandomOverSampler...")
        ros = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        
        return X_resampled, y_resampled
    
    except Exception as e:
        raise CustomException('Error while handling imbalance', e)


def get_preprocessor(save_path: str = configs['preprocessor_obj'],save:bool=True):
    """
    Builds a preprocessor pipeline with imputation and one-hot encoding.
    Saves the fitted preprocessor if save=True.
    """
    try:
        logger.info("Building preprocessor pipeline...")
        
        cat_cols = configs['categorical_columns']
        
        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', cat_pipeline, cat_cols)
            ],
            remainder='passthrough'  # Keep numerical features
        )

        if save:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(preprocessor, save_path)
            logger.info(f"Preprocessor saved at {save_path}")
        
        return preprocessor

    except Exception as e:
        raise CustomException('Error building/saving preprocessor', e)


def run_preprocessing():
    """
    Preprocesses training data:
    - Handles imbalance using RandomOverSampler
    - Fits and transforms with preprocessor
    - Saves the preprocessor as .pkl
    - Saves processed X and y as CSVs
    """
    try:
        
        logger.info("Starting preprocessing pipeline...")

        # Step 1: Handle class imbalance
        X_resampled, y_resampled = handle_imbalance()
        
        # drop customer_id column
        X_resampled=X_resampled.drop(columns=configs['columns_to_drop'])

        # Step 2: Build preprocessor and save it
        preprocessor = get_preprocessor(save=True)

        # Step 3: Fit and transform X
        logger.info("Fitting and transforming training data...")
        X_processed = preprocessor.fit_transform(X_resampled)

        # Step 4: Convert to DataFrame with correct column names
        cat_cols = configs['categorical_columns']
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        ohe_feature_names = ohe.get_feature_names_out(cat_cols)

        num_cols = [col for col in X_resampled.columns if col not in cat_cols]
        all_columns = list(ohe_feature_names) + num_cols

        X_df = pd.DataFrame(X_processed, columns=all_columns)
        y_df = pd.DataFrame(y_resampled, columns=[configs['target_column']])

        # Step 5: Save to CSV
        X_csv_path = configs['processed_X_path_csv']
        y_csv_path = configs['processed_y_path_csv']

        os.makedirs(os.path.dirname(X_csv_path), exist_ok=True)

        X_df.to_csv(X_csv_path, index=False)
        y_df.to_csv(y_csv_path, index=False)

        logger.info(f"Saved preprocessed X to {X_csv_path}")
        logger.info(f"Saved target y to {y_csv_path}")

        return X_df, y_df

    except Exception as e:
        raise CustomException("Error in preprocessing pipeline", e)

