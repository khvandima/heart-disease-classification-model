import sys

import numpy as np
from sklearn.metrics import accuracy_score

from heart_classification_model.predict import make_prediction

sys.path.append(
    "/Users/khvandima/Documents/Programming/KS_AI_JLR_18/ml_practice/Heart_Disease_classification/"
)


def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = 76

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")

    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.int64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    _predictions = list(predictions)
    y_true = sample_input_data["target"]
    accuracy = accuracy_score(y_true, _predictions)

    assert accuracy > 0.83
