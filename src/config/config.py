from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

## Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"



## Dataset Paths
TRAIN_DATA_PATH = RAW_DATA_DIR / "train_FD004.txt"
TEST_DATA_PATH = RAW_DATA_DIR / "test_FD004.txt"
RUL_DATA_PATH = RAW_DATA_DIR / "RUL_FD004.txt"