from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.config.config import TRAIN_DATA_PATH, TEST_DATA_PATH, RUL_DATA_PATH
from src.preprocessing.rul_generator import RULGenerator


loader = DataLoader(
    train_path=TRAIN_DATA_PATH,
    test_path=TEST_DATA_PATH,
    rul_path=RUL_DATA_PATH,
)

train_df = loader.load_train()
test_df = loader.load_test()
rul_df = loader.load_rul()

validator = DataValidator(
    train_df,
    test_df,
    rul_df,
)

report = validator.validate_all()

print(report)


generator = RULGenerator(train_df)

raw_df = generator.generate()

print(raw_df.head())