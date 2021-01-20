import numpy as np
import pandas as pd
from mizarlabs.model.bootstrapping import get_ind_matrix
from mizarlabs.static import CLOSE
from mizarlabs.transformers.sampling.average_uniqueness import AverageUniqueness
from mizarlabs.static import EVENT_END_TIME
from mizarlabs.transformers.utils import check_missing_columns
from mizarlabs.transformers.utils import convert_to_timestamp
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class SampleWeightsByReturns(BaseEstimator, TransformerMixin):
    """
    Calculate the sample weights by absolute returns.

    See page 69 of Advances in Financial Machine Learning by Marcos Lopez de
    Prado for additional information.

    :param event_end_time_column_name: The column name of the event end time
    :type event_end_time_column_name: str, optional
    :param close_column_name: The column name of the close price
    :type close_column_name: str, optional
    """

    def __init__(
        self,
        event_end_time_column_name: str = EVENT_END_TIME,
        close_column_name: str = CLOSE,
    ):

        self._event_end_time_column_name = event_end_time_column_name
        self._close_column_name = close_column_name

    def transform(self, X: pd.DataFrame) -> pd.Series:
        check_missing_columns(
            X, [self._close_column_name, self._event_end_time_column_name]
        )

        indicators_matrix, ind_mat_indices = get_ind_matrix(
            samples_info_sets=X[self._event_end_time_column_name],
            price_bars=X,
            event_end_time_column_name=self._event_end_time_column_name,
            return_indices=True,
        )

        num_concurrent_events = pd.Series(
            indicators_matrix.tocsc().sum(axis=1).A1, index=ind_mat_indices
        ).loc[X.index]

        returns = np.log(X[self._close_column_name]).diff()

        weights_array = self._calculate_weights(
            convert_to_timestamp(X.index.values),
            convert_to_timestamp(X[self._event_end_time_column_name].values),
            num_concurrent_events.values,
            returns.values,
        )

        weights = pd.Series(weights_array, index=X.index)

        return weights

    @staticmethod
    def _calculate_weights(
        bars_index: np.ndarray,
        expiration_barriers: np.ndarray,
        num_concurrent_events: np.ndarray,
        returns: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the weights by absolute returns

        It is based on numpy arrays for optimization

        :param bars_index:
        :type bars_index: np.ndarray
        :param expiration_barriers:
        :type expiration_barriers: np.ndarray
        :param num_concurrent_events:
        :type num_concurrent_events: np.ndarray
        :param returns:
        :type returns: np.ndarray
        :return: The weights by absolute returns
        :rtype: np.ndarray
        """
        # init weights array
        weights = np.zeros_like(bars_index, dtype=np.float64)
        for i in range(len(expiration_barriers)):
            # creating mask between start time and expiration barrier
            mask = np.greater_equal(bars_index, bars_index[i]) * np.less_equal(
                bars_index, expiration_barriers[i]
            )
            # calculating weights based on the returns and the concurrent events
            weights[i] = np.abs(np.nansum(returns[mask] / num_concurrent_events[mask]))

        return weights


class SampleWeightsByTimeDecay(AverageUniqueness):
    """
    Calculate the sample weights by time decay.

    See page 70 of Advances in Financial Machine Learning by Marcos Lopez de
    Prado for additional information.

    :param event_end_time_column_name: The column name of the event end time
    :type event_end_time_column_name: str, optional
    :param minimum_decay_weight: Is the minimum desired value in the decay weights
        - minimum_decay_weight = 1 means there is no time decay
        - 0 < minimum_decay_weight < 1 means that weights decay linearly over
          time, but every observation still receives a strictly positive weight, regadless of how old
        - minimum_decay_weight = 0 means that weights converge linearly to zero, as they become older
        - minimum_decay_weight < 0 means that the oldest portion of the observations
          receive zero weight (i.e they are erased from memory)
    :type minimum_decay_weight: float
    """

    def __init__(
        self,
        minimum_decay_weight: float,
        event_end_time_column_name: str = EVENT_END_TIME,
    ):
        super().__init__(event_end_time_column_name)
        self._decay_factor = minimum_decay_weight

    def transform(self, X: pd.DataFrame) -> pd.Series:
        assert self._decay_factor <= 1, "Decay factor should be less or equal to 1"
        check_missing_columns(X, [self._event_end_time_column_name])
        av_uniqueness = super().transform(X)
        time_decay_weights = av_uniqueness.sort_index().cumsum()
        if self._decay_factor >= 0:
            slope = (1 - self._decay_factor) / time_decay_weights.iloc[-1]
        else:
            slope = 1 / ((self._decay_factor + 1) * time_decay_weights.iloc[-1])
        const = 1 - slope * time_decay_weights.iloc[-1]
        time_decay_weights = const + slope * time_decay_weights
        time_decay_weights[time_decay_weights < 0] = 0  # Weights can't be negative

        return time_decay_weights
