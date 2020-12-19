import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class CUSUMFilter(BaseEstimator, TransformerMixin):
    """
    Downsamples a timeseries to filter out non significant value
    changing samples.

    :param threshold: Sets the threshold to trigger a sample, setting
                      the threshold higher will result in less samples being selected.
    :type threshold: float
    """

    def __init__(self, threshold: float):
        self._threshold = threshold

    def transform(self, X: pd.Series) -> pd.DatetimeIndex:
        """
        Returns a pandas DatetimeIndex indicating which samples have been
        selected by the CUSUM filter.

        :param X: Time series with time indices.
        :type X: pd.Series
        :return: DatetimeIndex indicating which samples have been selected.
        :rtype: pd.DatetimeIndex
        """
        filtered_indices = []
        pos_threshold = 0
        neg_threshold = 0
        diff = X.diff()
        for index in diff.index[1:]:
            pos_threshold, neg_threshold = (
                max(0, pos_threshold + diff.loc[index]),
                min(0, neg_threshold + diff.loc[index]),
            )

            if neg_threshold < -self._threshold:
                neg_threshold = 0
                filtered_indices.append(index)

            elif pos_threshold > self._threshold:
                pos_threshold = 0
                filtered_indices.append(index)

        return pd.DatetimeIndex(filtered_indices)
