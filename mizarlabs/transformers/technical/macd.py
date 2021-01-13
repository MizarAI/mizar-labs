from mizarlabs.transformers.technical.factory import TAFactory
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class MACDHistogramCrossOver(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column_name: str = "close",
    ):

        if fast_period > slow_period:
            raise ValueError("Fast should be smaller than slow")
        self.fast_period_ = fast_period
        self.slow_period_ = slow_period
        self.signal_period_ = signal_period
        self.column_name = column_name
        self.transformer_ = TAFactory().create_transformer(
            "MACD",
            self.column_name,
            kw_args={
                "fastperiod": fast_period,
                "slowperiod": slow_period,
                "signalperiod": signal_period,
            },
        )

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame):

        macd_df = pd.DataFrame(self.transformer_.transform(X))

        return np.sign(macd_df[2]).diff().values.reshape(-1, 1) / 2


class MACDHistogramCrossOverPredictor(MACDHistogramCrossOver):
    classes_ = [-1.0, 0.0, 1.0]
    n_classes_ = 3

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.transform(X).flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        predictions = self.predict(X)
        probabilities = np.zeros(shape=(len(predictions), len(self.classes_)))
        for i, class_value in enumerate(self.classes_):
            probabilities[:, i] = predictions == class_value
        return probabilities