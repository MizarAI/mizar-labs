from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import pandas as pd


class LocalMaxToCloseRatio(BaseEstimator, TransformerMixin):
    def __init__(self, window):
        self.window = window

    def transform(self, X: pd.DataFrame) -> pd.Series:
        return (X.high.rolling(window=self.window).max() / X.close).values.reshape(
            -1, 1
        )
