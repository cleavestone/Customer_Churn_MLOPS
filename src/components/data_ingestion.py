import pandas as pd
from utils.logging import logger
from utils.exceptions import CustomException
from utils.helper import configs
import os
import sys
import kagglehub
from sklearn.model_selection import train_test_split

def data_ingestion(path:str='gauravtopre/bank-customer-churn-dataset'):
    try:
        logger.info("downloading data from kaggle...")
        path = kagglehub.dataset_download(path)
        csv_path=os.path.join(path,'Bank Customer Churn Prediction.csv')
        df=pd.read_csv(csv_path)
        logger.info("saving raw data...")
        df.to_csv(configs['raw_data_path'],index=False)
        
        # spliting into train, test, val
        logger.info("splitting into train, test and validation")
        temp,test=train_test_split(df,test_size=0.2,random_state=42)
        train,val=train_test_split(temp,test_size=0.2,random_state=42)
        logger.info("saving train, test and validation")


        # save train
        train.to_csv(configs['train_path'],index=False)
        # save test
        test.to_csv(configs['test_path'],index=False)
        # save validation
        val.to_csv(configs['val_path'],index=False)

        
    except Exception as e:
        raise CustomException("Failed to download dataset",sys)
    
if __name__ == "__main__":
    data_ingestion()