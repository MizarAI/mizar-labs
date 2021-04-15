from abc import abstractmethod

import numpy as np
import pandas as pd
from mizarlabs.static import EVENT_END_TIME
from mizarlabs.transformers.utils import check_missing_columns
from mizarlabs.transformers.utils import convert_to_timestamp
from numba import jit
from numba import prange
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

BET_SIZE = "bet_size"
PROBABILITY = "prob"
PREDICTION = "pred"
SIDE = "side"


class BetSizingBase(BaseEstimator, TransformerMixin):
    """
    Base class for bet sizing transformers
    """

    def transform(self, X: pd.DataFrame) -> pd.Series:

        bet_sizing_signal = self._transform(X)

        assert (bet_sizing_signal >= -1).all(), "The bet size signal should be >= -1"
        assert (bet_sizing_signal <= 1).all(), "The bet size signal should be <= 1"

        return bet_sizing_signal

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.Series:
        pass


class BetSizingFromProbabilities(BetSizingBase):
    """
    Calculate the bet size using the predicted probability.

    :param num_classes: Number of labeled classes
    :type num_classes: int
    :param average_active: Whether we need to apply the average
                           active to the bet sizing signal
    :type average_active: bool, optional
    :param meta_labeling: Whether the bet sizing is calculated from a
                          metalabeling signal
    :type meta_labeling: bool, optional
    :param discretise: Whether the output needs to be discretised
    :type discretise: bool, optional
    :param step_size: The step size of the discretisation
    :type step_size: int, optional
    :param probability_column_name: The column name of the probabilities
    :type probability_column_name: str, optional
    :param prediction_column_name: The column name of the predictions
    :type prediction_column_name: str, optional
    :param side_column_name: The column name of the side of the 'simpler'
                             metalabeling model
    :type side_column_name: str, optional
    :param event_end_time_column_name: The column name of the event end time
    :rtype event_end_time_column_name: str, optional
    """

    def __init__(
        self,
        num_classes: int,
        average_active: bool = False,
        meta_labeling: bool = False,
        discretise: bool = False,
        step_size: float = None,
        probability_column_name: str = PROBABILITY,
        prediction_column_name: str = PREDICTION,
        side_column_name: str = SIDE,
        event_end_time_column_name: str = EVENT_END_TIME,
        bet_size_column_name: str = BET_SIZE,
    ):

        self._side_column_name = side_column_name
        self._metalabeling = meta_labeling
        self._average_active = average_active
        self._step_size = step_size
        self._num_classes = num_classes
        self._probability_column_name = probability_column_name
        self._prediction_column_name = prediction_column_name
        self._event_end_time_column_name = event_end_time_column_name
        self._discretise = discretise
        self._bet_size_column_name = bet_size_column_name

        if self._discretise:
            assert self._discretise and self._step_size, (
                "When discretise is activated, step size should be "
                "set with value between 0 and 1"
            )
            assert (
                0 < self._step_size < 1
            ), "The step size should be greater than zero and less than 1"

    def _transform(self, X: pd.DataFrame) -> pd.Series:
        check_missing_columns(
            X, [self._probability_column_name, self._prediction_column_name]
        )

        # generate signals from multinomial classification (one-vs-rest, OvR)
        test_statistic_z = (
            X[self._probability_column_name] - 1.0 / self._num_classes
        ) / (
            X[self._probability_column_name] * (1.0 - X[self._probability_column_name])
        ) ** 0.5

        # signal=side*size
        bet_sizing_signal = X[self._prediction_column_name] * (
            2 * norm.cdf(test_statistic_z) - 1
        )

        if self._metalabeling:
            assert set(X[self._side_column_name].unique()).issubset(
                {1, -1, 0}
            ), "The side should be 1, -1 or 0"

            check_missing_columns(X, [self._side_column_name])
            bet_sizing_signal *= X.loc[bet_sizing_signal.index, self._side_column_name]

        if self._average_active:
            bet_sizing_signal_with_barrier = bet_sizing_signal.to_frame(BET_SIZE).join(
                X[[self._event_end_time_column_name]], how="left"
            )

            bet_sizing_signal = avg_active_signals(
                bet_sizing_signal_with_barrier,
                self._event_end_time_column_name,
                self._bet_size_column_name,
            )

        if self._discretise:
            bet_sizing_signal = discretise_signal(bet_sizing_signal, self._step_size)

        return bet_sizing_signal.abs()


def avg_active_signals(
    signals: pd.DataFrame,
    event_end_time_column_name: str = EVENT_END_TIME,
    bet_size_column_name: str = BET_SIZE,
) -> pd.Series:
    """
    Average the bet sizes of all concurrently not closed positions
    (e.i. no barrier has been hit yet)

    :param signals: Signal from which the active average is calculated
    :rtype signals: pd.DataFrame
    :param event_end_time_column_name: the name of the event end time
    :type event_end_time_column_name: str, optional
    :param bet_size_column_name: the name of the bet size column
    :type bet_size_column_name: str, optional
    :return: The active average signal
    :rtype: pd.DataFrame
    """
    # compute the average bet size among those active
    # time points were bet size change (either one starts or one ends)
    active_bet_size_time_indices = set(
        signals[event_end_time_column_name].dropna().values
    )
    active_bet_size_time_indices = active_bet_size_time_indices.union(
        signals.index.values
    )
    active_bet_size_time_indices = list(active_bet_size_time_indices)
    active_bet_size_time_indices.sort()
    active_bet_size_time_indices = np.array(active_bet_size_time_indices)

    avg_active_bet_size_list = _get_avg_active_signals(
        signals.loc[:, bet_size_column_name].values,
        convert_to_timestamp(active_bet_size_time_indices),
        convert_to_timestamp(signals.index.values),
        convert_to_timestamp(signals[event_end_time_column_name].values),
    )

    avg_active_bet_size = pd.Series(
        avg_active_bet_size_list, index=active_bet_size_time_indices, dtype=float
    )

    return avg_active_bet_size


@jit(parallel=True, nopython=True)
def _get_avg_active_signals(
    bet_size_signal: np.ndarray,
    active_bet_size_time_indices: np.ndarray,
    signal_timestamp_index: np.ndarray,
    expiration_barrier_timestamp: np.ndarray,
) -> np.ndarray:
    """
    Calculate the average active bet signal from the overlapping bets

    :param bet_size_signal: The bet size signal not averaged by active signals
    :type bet_size_signal: np.ndarray
    :param active_bet_size_time_indices: The timestamps when at least one
                                         signal is active
    :type active_bet_size_time_indices: np.ndarray
    :param signal_timestamp_index: The timestamps of the signal bet signal
    :type signal_timestamp_index: np.ndarray
    :param expiration_barrier_timestamp: The timestamps of the expiration
                                         barriers
    :type expiration_barrier_timestamp: np.ndarray
    :return: The average active bet size
    :rtype: np.ndarray
    """
    # init the average active bet sizes array with zeros
    avg_active_bet_size = np.zeros_like(active_bet_size_time_indices, dtype=np.float64)

    for i in prange(len(active_bet_size_time_indices)):
        active_bet_size_time = active_bet_size_time_indices[i]
        # mask that finds where the bet signals are overlapping
        mask = np.less_equal(
            signal_timestamp_index, active_bet_size_time
        ) * np.logical_or(
            np.less(active_bet_size_time, expiration_barrier_timestamp),
            np.less(expiration_barrier_timestamp, 0),
        )
        # select the active bet sizes signals and calculates the mean
        active_bets_timestamps = signal_timestamp_index[mask]
        if len(active_bets_timestamps) > 0:
            avg_active_bet_size[i] = np.mean(bet_size_signal[mask])

    return avg_active_bet_size


def discretise_signal(signal: pd.Series, step_size: float) -> pd.Series:
    """
    Discretise the bet size signal based on the step size given.

    :param signal: Signal to discretise
    :type signal: pd.Series
    :param step_size: the step size to use for the discretisation
    :type step_size: float
    :return: Discretised signal
    :rtype: pd.Series
    """
    assert 0 < step_size < 1, "The step size should be between 0 and 1"

    discretised_signal = ((signal / step_size).round() * step_size).round(3)
    # Capping the discretised signal to 1
    discretised_signal[discretised_signal > 1] = 1
    # Flooring the discretised signal to 0
    discretised_signal[discretised_signal < -1] = -1

    return discretised_signal
