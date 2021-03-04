import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


def _cast_nan_to_none(val):
    return val if not np.isnan(val) else None


def check_missing_columns(df, wanted_columns):
    assert set(df.columns).issuperset(
        set(wanted_columns)
    ), f"{set(wanted_columns) - set(df.columns)} is missing"


def convert_to_timestamp(datetime_array: np.ndarray) -> np.ndarray:
    """
    Convert a datetime array to timestamp in milliseconds.

    Nans are converted to negatives values

    :param datetime_array: An array of datetimes
    :return: array of timestamps
    :rtype: np.ndarray
    """

    return datetime_array.astype(int) // 10 ** 6


class IdentityTransformer(TransformerMixin, BaseEstimator):
    """The identity transformer returns exactly what the input is"""

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X
