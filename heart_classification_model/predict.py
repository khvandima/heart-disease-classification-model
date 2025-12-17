import sys
import typing as t

import pandas as pd

from heart_classification_model import __version__ as _version
from heart_classification_model.config.core import config
from heart_classification_model.processing.data_manager import load_pipeline
from heart_classification_model.processing.validation import validate_inputs

sys.path.append(
    "/Users/khvandima/Documents/Programming/KS_AI_JLR_18/ml_practice/Heart_Disease_classification/"
)

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"

_heart_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict]) -> dict:
    data = pd.DataFrame(input_data)

    validated_data, errors = validate_inputs(input_data=data)

    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _heart_pipe.predict(
            X=validated_data[config.model_config.features]
        )

        results = {"predictions": predictions, "version": _version, "errors": errors}

    return results
