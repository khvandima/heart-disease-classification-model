import sys

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from heart_classification_model.config.core import config
from heart_classification_model.processing.features import (
    AgeBinningTransformer,
    CholesterolBinningTransformer,
    CustomOneHotEncoder,
    IterativeImputerTransformer,
    KNNImputerTransformer,
    OldpeakBinningTransformer,
    SimpleImputerTransformer,
    ThalachBinningTransformer,
    TrestbpsBinningTransformer,
)

sys.path.append(
    "/Users/khvandima/Documents/Programming/KS_AI_JLR_18/ml_practice/Heart_Disease_classification/"
)

heart_pipe = Pipeline(
    steps=[
        (
            "iterative_imputer",
            IterativeImputerTransformer(
                variables=config.model_config.iterative_imputer_vars
            ),
        ),
        (
            "simple_imputer_median",
            SimpleImputerTransformer(
                strategy="median",
                variables=config.model_config.simple_imputer_median_vars,
            ),
        ),
        (
            "simple_imputer_most_frequent",
            SimpleImputerTransformer(
                strategy="most_frequent",
                variables=config.model_config.simple_imputer_most_frequent_vars,
            ),
        ),
        (
            "knn_imputer",
            KNNImputerTransformer(variables=config.model_config.knn_imputer_vars),
        ),
        ("age_binning", AgeBinningTransformer(variables=config.model_config.age_vars)),
        (
            "treshtbps_binning",
            TrestbpsBinningTransformer(variables=config.model_config.trestbps_vars),
        ),
        (
            "cholesterol_binning",
            CholesterolBinningTransformer(variables=config.model_config.chol_vars),
        ),
        (
            "thalach_binning",
            ThalachBinningTransformer(variables=config.model_config.thalach_vars),
        ),
        (
            "old_peak_binning",
            OldpeakBinningTransformer(variables=config.model_config.oldpeak_vars),
        ),
        ("encoder", CustomOneHotEncoder(variables=config.model_config.features)),
        (
            "LogisticRegression",
            LogisticRegression(C=1, penalty="l2", solver="liblinear"),
        ),
    ]
)
