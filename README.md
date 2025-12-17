🫀 Heart Disease Classification — Production ML Package
This repository contains a production-ready machine learning package for heart disease classification, built with a strong focus on:
* clean project structure
* reproducible training
* automated testing
* static code checks
* packaging and distribution best practices
The project demonstrates how an experimental notebook-based ML solution can be transformed into a maintainable, testable and distributable Python package.

📌 Project Goals
* Build an end-to-end ML pipeline for medical classification
* Ensure reproducibility and deterministic training
* Emphasize model interpretability (important for healthcare)
* Apply software engineering practices to ML (tests, linting, packaging)
* Provide a clean example of ML → package → build workflow

📊 Dataset
* Domain: Medical / Healthcare
* Task: Binary classification
* Target: Presence of heart disease
* Source: UCI-style heart disease dataset
The dataset is intentionally included in the repository because:
* it is small (< 1 MB)
* it is public
* it allows full reproducibility of training and tests

🧠 Modeling Approach
The project prioritizes interpretability and reliability over raw performance.
Models and techniques include:
* feature engineering
* structured preprocessing pipeline
* deterministic training
* validation and evaluation logic
* stored trained artifacts for inference
The package is designed so that training, prediction and validation are clearly separated.

📁 Project Structure
```
heart_disease_classification/
│
├── __init__.py
├── heart_classification_model
│   ├── __init__.py
│   │
│   ├── config
│   │   ├── __init__.py
│   │   └── core.py
│   ├── config.yml
│   ├── datasets
│   │   ├── __init__.py
│   │   └── heartDisease_dataset_TP.csv
│   ├── pipeline.py
│   ├── predict.py
│   ├── processing
│   │   ├── __init__.py
│   │   ├── data_manager.py
│   │   ├── features.py
│   │   └── validation.py
│   ├── train_pipeline.py
│   ├── trained_models
│   │   └── __init__.py
│   └── VERSION
├── MANIFEST.in
├── mypy.ini
├── pyproject.toml
├── README.md
├── requirements
│   ├── requirements.txt
│   ├── test_requirements.txt
│   └── typing_requirements.txt
├── setup.py
├── tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_features.py
│   └── test_prediction.py
└── tox.ini
```


✅ Automated Workflow (tox)

This project uses tox to standardize all workflows.

Run full test suite and code checks
```
tox
```
This executes:
* model training
* unit tests (pytest)
* code quality checks (flake8, isort, black, mypy)


Train model only
```
tox -e train
```


Run package tests only
```
tox -e test_package
```


🧪 Testing
* Unit tests validate:
    * feature engineering logic
    * prediction pipeline behavior
    * consistency of outputs
* Tests are deterministic and reproducible
* Training is executed as part of the test environment


📦 Build Package

After tests pass, build distributable artifacts:
```
python3 -m build
```


This creates:
* .whl (wheel)
* .tar.gz (source distribution)
Artifacts are placed in the dist/ directory.


🧠 Why This Structure Matters
This repository demonstrates:
* separation of concerns (data / features / training / inference)
* testable ML code
* reproducible experiments
* readiness for CI/CD integration
* transition from research code to production-ready package

📌 Intended Audience
* ML Engineers
* Data Scientists moving toward production ML
* Teams interested in MLOps foundations
* Recruiters reviewing real-world ML engineering work

🚧 Project Status
The package is stable and fully functional. Future extensions may include:
* experiment tracking
* model versioning
* deployment examples (API / batch inference)