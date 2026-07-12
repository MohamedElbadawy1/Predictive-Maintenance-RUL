from pathlib import Path

# ============================================================
# Project Name
# ============================================================

PROJECT_NAME = "NASA-RUL-Prediction"

# ============================================================
# Files & Folders
# ============================================================

structure = [

    # Root
    ".gitignore",
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "main.py",

    # Configs
    "configs/__init__.py",
    "configs/config.yaml",

    # Data
    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/.gitkeep",
    "data/external/.gitkeep",

    # Notebooks
    "notebooks/.gitkeep",

    # Source
    "src/__init__.py",

    "src/data/__init__.py",
    "src/data/loader.py",
    "src/data/validator.py",

    "src/preprocessing/__init__.py",
    "src/preprocessing/preprocessor.py",

    "src/features/__init__.py",
    "src/features/feature_engineer.py",

    "src/models/__init__.py",
    "src/models/base_trainer.py",
    "src/models/xgboost_trainer.py",
    "src/models/lstm_trainer.py",
    "src/models/predictor.py",
    "src/models/factory.py",

    "src/evaluation/__init__.py",
    "src/evaluation/evaluator.py",

    "src/explainability/__init__.py",
    "src/explainability/shap_explainer.py",

    "src/visualization/__init__.py",
    "src/visualization/plots.py",

    "src/pipelines/__init__.py",
    "src/pipelines/training_pipeline.py",
    "src/pipelines/inference_pipeline.py",

    "src/logger/__init__.py",
    "src/logger/logger.py",

    "src/exceptions/__init__.py",
    "src/exceptions/custom_exception.py",

    "src/utils/__init__.py",
    "src/utils/common.py",

    # App
    "app/__init__.py",
    "app/app.py",

    # Tests
    "tests/__init__.py",

    # Other folders
    "models/.gitkeep",
    "reports/.gitkeep",
    "artifacts/.gitkeep",

    # GitHub Actions
    ".github/workflows/.gitkeep",
]


def create_structure():
    root = Path(PROJECT_NAME)

    for item in structure:

        path = root / item

        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)

            if not path.exists():
                path.touch()

        else:
            path.mkdir(parents=True, exist_ok=True)

    print(f"\nProject '{PROJECT_NAME}' created successfully!\n")


if __name__ == "__main__":
    create_structure()