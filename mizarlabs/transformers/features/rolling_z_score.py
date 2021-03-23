import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from mizarlabs import static


class RollingZScoreTransformer(BaseEstimator, TransformerMixin):
    """Transformer which calculates rolling z score based on a transformation
    of a dataframe into a series.

    :param window: lookbackwindow of rolling z score
    :type window: int
    :param to_series_func: function to apply on dataframe to transform into a series
    :type to_series_func: callable
    """
    def __init__(self, window: int, to_series_func: callable):
        self.window = window
        self.to_series_func = to_series_func

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """Transform dataframe into rolling z score.

        :param X: DataFrame to transform
        :type X: pd.DataFrame
        :return: Series with rolling z scores
        :rtype: pd.Series
        """
        series = self.to_series_func(X)
        rolling_z_score = self.calc_z_score(series)
        return rolling_z_score.values.reshape(-1, 1)

    def calc_z_score(self, X: pd.Series) -> pd.Series:
        """calculated the rolling z score

        :param X: series to compute z score of
        :type X: pd.Series
        :return: series with rolling z scores
        :rtype: pd.Series
        """
        X_rolling_mean = X.rolling(window=self.window).mean()
        X_rolling_std = X.rolling(window=self.window).std()
        rolling_z_score = (X - X_rolling_mean) / X_rolling_std
        return rolling_z_score


def bar_arrival_time(bars_df: pd.DataFrame) -> pd.Series:
    return bars_df.index.to_series().astype(int).diff()

def buy_sell_diff(bars_df: pd.DataFrame) -> pd.Series:
    return bars_df.quote_asset_buy_volume - bars_df.quote_asset_sell_volume

def average_buy_size(bars_df: pd.DataFrame) -> pd.Series:
    return bars_df.quote_asset_buy_volume / bars_df.num_buy_ticks

def high_to_low_ratio(bars_df: pd.DataFrame) -> pd.Series:
    return (bars_df.high / bars_df.low)

class RollingZScoreTransformerFactory:
    """Factory for creating predefined rolling z score transformers.
    """
    types = dict(
        bar_arrival_time=bar_arrival_time,
        buy_sell_diff=buy_sell_diff,
        average_buy_size=average_buy_size,
        high_to_low_ratio=high_to_low_ratio,
    )

    @classmethod
    def get_transformer(self, key: str, window: int) -> RollingZScoreTransformer:
        """Builds a rolling z score transformer.

        Select from the following: bar_arrival_time, buy_sell_diff, average_buy_size, high_to_low_ratio.

        :param key: name of the transformer to build
        :type key: str
        :param window: lookback window for rolling mean and standard deviation
        :type window: int
        :raises KeyError: cannot build transformers, which are not registered.
        :return: Transformer with requested specification
        :rtype: RollingZScoreTransformer
        """
        if key not in self.types:
            raise KeyError(f"{key} does not exist, please select from {', '.join(self.types.keys())}")
        return RollingZScoreTransformer(window=window, to_series_func=self.types[key])

