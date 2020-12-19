"""
Implementation of Trend-Scanning labels described in `Advances in Financial Machine Learning: Lecture 3/10
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>`_
"""
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from mizarlabs.static import EVENT_END_TIME
from mizarlabs.static import LABEL
from mizarlabs.static import RETURN
from mizarlabs.static import T_VALUE
from mizarlabs.structural_breaks.sdfc import get_beta_and_beta_var
from numba import jit
from numba import prange
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class TrendScannerLabeling(BaseEstimator, TransformerMixin):
    """
    `Trend scanning <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419>`_ is both a classification and
    regression labeling technique.
    That can be used in the following ways:
    1. Classification: By taking the sign of t-value for a given observation we can set {-1, 1} labels to define the
       trends as either downward or upward.
    2. Classification: By adding a minimum t-value threshold you can generate {-1, 0, 1} labels for downward, no-trend,
       upward.
    3. The t-values can be used as sample weights in classification problems.
    4. Regression: The t-values can be used in a regression setting to determine the magnitude of the trend.
    The output of this algorithm is a DataFrame with t1 (time stamp for the farthest observation), t-value, returns for
    the trend, and bin.
    :param t_events: filtered events, array/list of pd.Timestamps, defaults to None
    :type t_events: Union[np.ndarray, list, None], optional
    :param look_forward_window: maximum look forward window used to get the trend value, defaults to 20
    :type look_forward_window: int, optional
    :param min_sample_length: minimum sample length used to fit regression, defaults to 5
    :type min_sample_length: int, optional
    :param step: optimal t-value index is searched every 'step' indices, defaults to 1
    :type step: int, optional
    """

    def __init__(
        self,
        t_events: Union[np.ndarray, list, None] = None,
        look_forward_window: int = 20,
        min_sample_length: int = 5,
        step: int = 1,
    ):
        assert min_sample_length < look_forward_window, (
            f"look_forward_window is {look_forward_window} "
            f"and should be larger than min_sample_length {min_sample_length}"
        )
        self.t_events = t_events
        self.look_forward_window = look_forward_window
        self.min_sample_length = min_sample_length
        self.step = step

    def fit(self, y):
        return self

    def _check_input(self, input_series: pd.Series):
        assert isinstance(
            input_series, pd.Series
        ), "Please provide only a Pandas series with close prices with datetime indices."
        if self.t_events is not None:
            assert set(input_series.index.values).issuperset(set(self.t_events)), (
                f"The following time indices are in t_events but "
                f"not in the provided series "
                f"{set(input_series.index.values) - set(self.t_events)}"
            )

    def _check_output(self, output_df: pd.DataFrame):
        required_columns = [EVENT_END_TIME, T_VALUE, RETURN, LABEL]

        assert set(required_columns) == set(
            output_df.columns
        ), f"Missing {set(required_columns) - set(output_df.columns)} in output DataFrame."
        assert set(self.t_events) == set(
            output_df.index.values
        ), f"Missing following time indices in output DataFrame: {set(self.t_events) - set(output_df.index.values)}"

    def transform(self, y: pd.Series) -> pd.DataFrame:
        """Scans for trends in the provided series and provides
        a DataFrame with results.

        DataFrame contains the start_time, event_end_time, t_value,
        return and label.

        :param y: series used to label the data set
        :type y: pd.Series
        :return: DataFrame with as index the start time and
                 in the columns the event_end_time, t_value, return and
                 label.
        :rtype: pd.DataFrame
        """
        self._check_input(y)
        df = self._transform(y)
        self._check_output(df)
        return df

    def _transform(self, y: pd.Series) -> pd.DataFrame:
        """Scans for trends in the provided series and provides
        a DataFrame with results.

        DataFrame contains the start_time, event_end_time, t_value,
        return and label.

        :param y: series used to label the data set
        :type y: pd.Series
        :return: DataFrame with as index the start time and
                 in the columns the event_end_time, t_value, return and
                 label.
        :rtype: pd.DataFrame
        """

        # check if only subset of indices is looped over
        # else scan every index
        if self.t_events is None:
            self.t_events = y.index.values

        # get t values and end indices per index
        t_values_array, end_indices_array = _trend_scan_each_index(
            y_array=y.values.astype(np.float64),
            indices=y.index.get_indexer(self.t_events),
            min_sample_length=self.min_sample_length,
            look_forward_window=self.look_forward_window,
            step=self.step,
        )

        # convert end indices into datetime
        event_end_times_array = [
            y.index[int(i)] if not np.isnan(i) else i for i in end_indices_array
        ]

        # cast results into dataframe and compute the returns and the sign of the return
        labels = pd.DataFrame(
            {EVENT_END_TIME: event_end_times_array, T_VALUE: t_values_array},
            index=self.t_events,
        )
        non_nan_labels = labels[EVENT_END_TIME].dropna()
        labels.loc[non_nan_labels.index, RETURN] = (
            y.loc[non_nan_labels].values / y.loc[non_nan_labels.index].values - 1
        )
        labels[LABEL] = np.sign(labels.t_value)
        return labels


@jit(parallel=True, nopython=True)
def _trend_scan_each_index(
    y_array: np.ndarray,
    indices: np.ndarray,
    min_sample_length: int,
    look_forward_window: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scans for the trend in the index.

    :param look_forward_window: maximum look forward window used to get the trend value, defaults to 20
    :type look_forward_window: int, optional
    :param min_sample_length: minimum sample length used to fit regression, defaults to 5
    :type min_sample_length: int, optional
    :param step: optimal t-value index is searched every 'step' indices, defaults to 1
    :type step: int, optional

    :param y_array: series to trend scan
    :type y_array: np.ndarray
    :param indices: indices in the series to scan
    :type indices: np.ndarray
    :param min_sample_length: minimum sample length used to fit regression
    :type min_sample_length: int
    :param look_forward_window: maximum look forward window used to get the trend value
    :type look_forward_window: int
    :param step: optimal t-value index is searched every 'step' indices
    :type step: int
    :return: Tuple with the max t-values and their respective location in the series
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    # array to save the end time index values
    end_indices_array = np.empty(len(indices), dtype=np.float64)
    # array to save the t values per index
    t_values_array = np.empty(len(indices), dtype=np.float64)

    # iterate over each index in t_events
    for i in prange(len(indices)):
        index = indices[i]

        # select a subset of the series of interest based on forward looking window
        y_subset = y_array[index : (index + look_forward_window)].astype(np.float64)

        # check if sufficient data else set values to NaN
        if y_subset.shape[0] == look_forward_window:

            # compute max t value and its index location in the subset of y
            (
                t_values_array[i],
                index_max_t_value,
            ) = _get_t_val_and_event_end_time_index_at_index(
                y_subset, min_sample_length, step
            )

            # convert index location to index in whole of y
            end_indices_array[i] = index + index_max_t_value + min_sample_length - 1

        else:
            end_indices_array[i] = np.nan
            t_values_array[i] = np.nan
    return t_values_array, end_indices_array


@jit(nopython=True)
def _get_t_val_and_event_end_time_index_at_index(
    y_subset: np.ndarray,
    min_sample_length: int,
    step: int,
) -> Tuple[np.float64, np.float64]:
    """Returns the maximum t-value and its index location.

    Loop over possible look-ahead windows to get the one
    which yields maximum t values for b_1 regression coef

    :param y_subset: series being trend scanned
    :type y_subset: np.ndarray
    :param min_sample_length: minimum sample length used to fit regression
    :type min_sample_length: int
    :param step: optimal t-value index is searched every 'step' indices
    :type step: int
    :return: tuple with max t-value and its index location
    :rtype: Tuple[np.float64, np.float64]
    """

    # init array to stores t values in
    t_values_array = np.empty(y_subset.shape[0] - min_sample_length)

    # expand forward looking window, compute and store t value, keeping into account
    # the min_sample_length and the stepsize
    for forward_window in np.arange(min_sample_length, y_subset.shape[0], step):
        # y{t}:y_{t+l}
        y_subset_forward_window = np.ascontiguousarray(
            y_subset[:forward_window].reshape(-1, 1)
        )

        # Array of [1, 0], [1, 1], [1, 2], ... [1, l] # b_0, b_1 coefficients
        X_constant = np.ones_like(y_subset_forward_window, dtype=np.float64)
        X_trend = (
            np.arange(y_subset_forward_window.shape[0])
            .astype(np.float64)
            .reshape(-1, 1)
        )
        X_subset_forward_window = np.ascontiguousarray(
            np.concatenate((X_constant, X_trend), axis=1)
        )

        # get OLS estimates
        b_mean_, b_var_ = get_beta_and_beta_var(
            X_subset_forward_window, y_subset_forward_window
        )

        # compute t value for trend coefficient
        t_values_array[forward_window - min_sample_length] = b_mean_[1, 0] / np.sqrt(
            b_var_[1, 1]
        )

    # identify max abs t-value and its location
    index_max_t_value = np.argmax(np.abs(t_values_array))
    max_t_value = t_values_array[index_max_t_value]
    return max_t_value, index_max_t_value
