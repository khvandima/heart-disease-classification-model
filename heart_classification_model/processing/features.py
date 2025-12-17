from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class AgeBinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError("Variables should be a list")
        self.variable = variables[0]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.variable] = pd.cut(
            X[self.variable],
            bins=[0, 50, 59, 100],
            labels=["up to 50 years", "50-59 years", "59 years more"],
            right=False,
        )
        return X


class TrestbpsBinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError("Variables should be a list")
        self.variable = variables[0]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.variable] = pd.cut(
            X[self.variable],
            bins=[-np.inf, 120, 139, np.inf],
            labels=["normal", "prehypertension", "hypertension"],
            right=False,
        )
        return X


class CholesterolBinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError("Variables should be a list")
        self.variable = variables[0]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.variable] = pd.cut(
            X[self.variable],
            bins=[-np.inf, 120, 139, np.inf],
            labels=["low", "moderate", "high"],
            right=False,
        )
        return X


class ThalachBinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError("Variables should be a list")
        self.variable = variables[0]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.variable] = pd.cut(
            X[self.variable],
            bins=[-np.inf, 100, 140, np.inf],
            labels=["low", "moderate", "high"],
            right=False,
        )
        return X


class OldpeakBinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError("Variables should be a list")
        self.variable = variables[0]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.variable] = pd.cut(
            X[self.variable],
            bins=[-np.inf, 100, 140, np.inf],
            labels=["normal", "moderate", "severe"],
            right=False,
        )
        return X


class IterativeImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError("variables must be a list")

        self.variables = variables
        self.imputer = IterativeImputer(random_state=0)

    def fit(self, X, y=None):
        self.imputer.fit(X[self.variables])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.variables] = self.imputer.transform(X[self.variables])
        return X


class SimpleImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, *, variables: List[str], strategy: str = "median"):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list of column names")
        self.variables = variables
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X: pd.DataFrame, y=None):
        # учим импьютер только на нужных колонках
        self.imputer.fit(X[self.variables])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = self.imputer.transform(X[self.variables])
        return X


class KNNImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        variables: List[str],
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "nan_euclidean",
        add_indicator=False,
    ):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list of column names")

        self.variables = variables
        self.imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            add_indicator=add_indicator,
        )

    def fit(self, X: pd.DataFrame, y=None):
        self.imputer.fit(X[self.variables])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = self.imputer.transform(X[self.variables])
        return X


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            raise ValueError("variables must be a list")

        self.variables = variables
        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,  # Важно: возвращать плотный массив
        )
        self.feature_names_out = None

    def fit(self, X, y=None):
        self.encoder.fit(X[self.variables])
        self.feature_names_out = list(
            self.encoder.get_feature_names_out(self.variables)
        )
        return self

    def transform(self, X):
        X = X.copy()

        # Преобразуем в плотный массив
        encoded = self.encoder.transform(X[self.variables])

        encoded_df = pd.DataFrame(
            encoded, columns=self.feature_names_out, index=X.index
        )

        X.drop(columns=self.variables, inplace=True)
        X = pd.concat([X, encoded_df], axis=1)

        return X
