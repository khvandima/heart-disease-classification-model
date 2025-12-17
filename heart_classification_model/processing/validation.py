import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from heart_classification_model.config.core import config

sys.path.append(
    "/Users/khvandima/Documents/Programming/KS_AI_JLR_18/ml_practice/Heart_Disease_classification/"
)


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    validated_data = input_data[config.model_config.features].copy()

    errors = None

    try:
        MultipleHeartDiseaseDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records"),
        )

    except ValidationError as error:
        errors = error.errors()

    return validated_data, errors


class HeartDiseaseDataInputSchema(BaseModel):
    age: Optional[float]
    sex: Optional[float]
    cp: Optional[float]
    trestbps: Optional[float]
    chol: Optional[float]
    fbs: Optional[float]
    restecg: Optional[float]
    thalach: Optional[float]
    exang: Optional[float]
    oldpeak: Optional[float]
    slope: Optional[float]
    ca: Optional[float]
    thal: Optional[float]


class MultipleHeartDiseaseDataInputs(BaseModel):
    inputs: List[HeartDiseaseDataInputSchema]
