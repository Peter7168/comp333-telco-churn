"""Step 4: data transformation. Normalization, scaling and encoding.

build_preprocessor() returns one scikit-learn ColumnTransformer, so every
choice is fit on training rows only and applied the same way to new rows.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MaxAbsScaler, MinMaxScaler,
                                   OneHotEncoder, OrdinalEncoder, PowerTransformer,
                                   QuantileTransformer, RobustScaler, StandardScaler,
                                   TargetEncoder)

SCALERS = ["none", "standard", "minmax", "robust", "maxabs", "decimal"]
NORMALIZERS = ["none", "log", "yeo_johnson", "quantile"]
ENCODERS = ["drop", "ordinal", "onehot", "frequency", "target"]


class DecimalScaler(BaseEstimator, TransformerMixin):
    """Decimal scaling from the course: divide by 10^j so every |value| is below 1."""

    def fit(self, X, y=None):
        biggest = np.abs(np.asarray(X, dtype=float)).max(axis=0)
        # I pick the smallest j with max / 10^j < 1. A column of zeros keeps divisor 1.
        j = np.where(biggest > 0, np.floor(np.log10(np.where(biggest > 0, biggest, 1))) + 1, 0)
        self.divisor_ = 10.0 ** j
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) / self.divisor_

    def get_feature_names_out(self, names=None):
        return np.asarray(names, dtype=object)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with how often it appears in the training data."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.freqs_ = [X[c].value_counts(normalize=True).to_dict() for c in X.columns]
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        # I give unseen categories a frequency of 0.
        out = [X[c].map(f).fillna(0.0).to_numpy() for c, f in zip(self.columns_, self.freqs_)]
        return np.column_stack(out)

    def get_feature_names_out(self, names=None):
        return np.asarray([f"{c}_freq" for c in self.columns_], dtype=object)


def make_scaler(name):
    return {
        "none": "passthrough",
        "standard": StandardScaler(),   # z-score: mean 0, std 1
        "minmax": MinMaxScaler(),       # min-max normalization to [0, 1]
        "robust": RobustScaler(),       # median and IQR, less sensitive to outliers
        "maxabs": MaxAbsScaler(),
        "decimal": DecimalScaler(),
    }[name]


def make_normalizer(name):
    return {
        "none": "passthrough",
        "log": FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        "yeo_johnson": PowerTransformer(method="yeo-johnson", standardize=False),
        "quantile": QuantileTransformer(output_distribution="normal", n_quantiles=200),
    }[name]


def make_encoder(name):
    return {
        "drop": "drop",
        "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        "onehot": OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                sparse_output=False),
        "frequency": FrequencyEncoder(),
        "target": TargetEncoder(),
    }[name]


def build_preprocessor(binary, numeric, categorical, skewed,
                       scaler="standard", normalizer="none", encoder="onehot"):
    skewed = [c for c in numeric if c in skewed]
    regular = [c for c in numeric if c not in skewed]

    parts = [("binary", "passthrough", binary)]
    if regular:
        parts.append(("numeric", make_scaler(scaler), regular))
    if skewed:
        # Skewed columns get normalized first, then scaled like the rest.
        steps = [("normalize", make_normalizer(normalizer)), ("scale", make_scaler(scaler))]
        parts.append(("skewed", Pipeline(steps), skewed))
    if categorical:
        parts.append(("categorical", make_encoder(encoder), categorical))

    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)
