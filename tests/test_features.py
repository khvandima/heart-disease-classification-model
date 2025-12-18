from heart_classification_model.config.core import config
from heart_classification_model.processing.features import (
    AgeBinningTransformer,
    CholesterolBinningTransformer,
    OldpeakBinningTransformer,
    ThalachBinningTransformer,
    TrestbpsBinningTransformer,
)


def test_age_binning_transformer(sample_input_data):
    # Given
    transformer = AgeBinningTransformer(variables=config.model_config.age_vars)
    assert sample_input_data["age"].iat[0] == 70

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["age"].iat[0] == "59 years more"


def test_trestbps_binning_transformer(sample_input_data):
    # Given
    transformer = TrestbpsBinningTransformer(
        variables=config.model_config.trestbps_vars
    )
    assert sample_input_data["trestbps"].iat[0] == 145

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["trestbps"].iat[0] == "hypertension"


def test_cholesterol_binning_transformer(sample_input_data):
    # Given
    transformer = CholesterolBinningTransformer(variables=config.model_config.chol_vars)
    assert sample_input_data["chol"].iat[0] == 174

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["chol"].iat[0] == "high"


def test_thalach_binning_transformer(sample_input_data):
    # Given
    transformer = ThalachBinningTransformer(variables=config.model_config.thalach_vars)
    assert sample_input_data["thalach"].iat[0] == 125

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["thalach"].iat[0] == "moderate"


def test_oldpeak_binning_transformer(sample_input_data):
    # Given
    transformer = OldpeakBinningTransformer(variables=config.model_config.oldpeak_vars)
    assert sample_input_data["oldpeak"].iat[0] == 2.6

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["oldpeak"].iat[0] == "normal"
