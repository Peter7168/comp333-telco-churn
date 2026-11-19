"""Normalization, scaling and encoding in one ColumnTransformer."""
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
    """Decimal scaling: divide by 10^j so every |value| < 1."""

    def fit(self, X, y=None):
        biggest = np.abs(np.asarray(X, dtype=float)).max(axis=0)
        # Smallest j with max / 10^j < 1 (all-zero columns keep divisor 1).
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
        # Unseen categories get frequency 0.
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


def rare_onehot():
    """Feature-engine groups categories under 0.5% of customers into "Rare", then one-hot."""
    from feature_engine.encoding import RareLabelEncoder
    return Pipeline([
        ("rare", RareLabelEncoder(tol=0.005, n_categories=1, replace_with="Rare",
                                  ignore_format=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
    ])


def make_encoder(name):
    if name == "rare_onehot":
        return rare_onehot()
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
        # Skewed columns: normalize, then scale.
        steps = [("normalize", make_normalizer(normalizer)), ("scale", make_scaler(scaler))]
        parts.append(("skewed", Pipeline(steps), skewed))
    if categorical:
        parts.append(("categorical", make_encoder(encoder), categorical))

    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)
