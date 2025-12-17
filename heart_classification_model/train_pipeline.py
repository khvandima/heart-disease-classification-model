import sys

from sklearn.model_selection import train_test_split

from heart_classification_model.config.core import config
from heart_classification_model.pipeline import heart_pipe
from heart_classification_model.processing.data_manager import (
    load_dataset,
    save_pipeline,
)

sys.path.append(
    "/Users/khvandima/Documents/Programming/KS_AI_JLR_18/ml_practice/Heart_Disease_classification/"
)


def run_training() -> None:
    data = load_dataset(file_name=config.app_config.raw_data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    heart_pipe.fit(X_train, y_train)

    save_pipeline(pipeline_to_persist=heart_pipe)


if __name__ == "__main__":
    run_training()
