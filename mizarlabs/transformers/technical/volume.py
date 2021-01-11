import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class BuySellImbalance(BaseEstimator, TransformerMixin):
    allowed_volume_types = ["quote", "base"]

    def __init__(self, fast: int, slow: int, volume_type: str):
        self.fast = fast
        self.slow = slow
        if volume_type not in self.allowed_volume_types:
            raise ValueError("The allowed volume types are 'quote' and 'base'")
        self.volume_type = volume_type

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:

        fast_buy_sell_diff = (
            X[f"{self.volume_type}_asset_buy_volume"].ewm(span=self.fast).mean()
            - X[f"{self.volume_type}_asset_sell_volume"].ewm(span=self.fast).mean()
        )
        slow_buy_sell_diff = (
            X[f"{self.volume_type}_asset_buy_volume"].ewm(span=self.slow).mean()
            - X[f"{self.volume_type}_asset_sell_volume"].ewm(span=self.slow).mean()
        )

        fast_sell_crossover_normalised = (fast_buy_sell_diff - slow_buy_sell_diff) / (
            X[f"{self.volume_type}_asset_volume"] + 1
        )

        return fast_sell_crossover_normalised.values.reshape(-1, 1)
