"""One-off helper: dump a small CSV that Pipeline/predict.py can read,
built from the project's own DataLoader (guaranteed correct schema) —
avoids hand-crafting a CSV from the raw whitespace-delimited txt file."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.config import TEST_DATA_PATH, TRAIN_DATA_PATH, RUL_DATA_PATH
from src.data.loader import DataLoader

loader = DataLoader(train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH, rul_path=RUL_DATA_PATH)
test_raw = loader.load_test()
test_raw.to_csv("predict_input_sample.csv", index=False)
print(f"Wrote predict_input_sample.csv with {test_raw['unit_number'].nunique()} engines, {len(test_raw)} rows.")
