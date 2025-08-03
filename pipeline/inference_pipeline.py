import pandas as pd
import numpy as np
from utils.helper import configs
from utils.logging import logger
from utils.exceptions import CustomException

def make_batch_prediction(path:str=configs['val_path']):
    
