from src.logger.logger import logger
from src.exceptions.custom_exception import CustomException
import pandas as pd
from pathlib import Path
import sys
from src.utils.constant import EXPECTED_COLUMNS


class DataLoader:
    def __init__(self, train_path:Path, test_path:Path, rul_path:Path):
        self.train_path = train_path
        self.test_path = test_path
        self.rul_path = rul_path


    def _load_file(self, file_path:Path, columns:list[str])->pd.DataFrame:
        try:
            logger.info(f"Reading {file_path.name}")
            df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
            df.columns = columns
            logger.info(f"{file_path.name} Loaded Successfully")
            return df


        except Exception as e:
            raise CustomException(e, sys)

    def load_train(self):
        return self._load_file(self.train_path, EXPECTED_COLUMNS)        
    
    def load_test(self):
        return self._load_file(self.test_path, EXPECTED_COLUMNS)

    def load_rul(self):
        return self._load_file(self.rul_path, ["RUL"])  
        


