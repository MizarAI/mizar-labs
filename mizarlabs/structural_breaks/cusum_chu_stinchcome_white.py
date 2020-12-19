"""
Implementation of Chu-Stinchcombe-White test
"""
import numpy as np
import pandas as pd
from numba import jit
from numba import njit
from numba import prange

ONE_SIDED_POSITIVE = "one_sided_positive"
ONE_SIDED_NEGATIVE = "one_sided_negative"
TWO_SIDED = "two_sided"


@jit(parallel=True, nopython=True)
def get_test_statistics_array(
    side_test: str, t_indices: np.ndarray, time_series_array: np.ndarray
) -> np.ndarray:
    """Returns an array with the test statistics of
    the Chu-Stinchcombe-White and the critical values

    :param side_test: 'one_sided_positive' or 'one_sided_negative' or 'two_sided'
    :type side_test: str
    :param t_indices: t indices to loop over
    :type t_indices: np.ndarray
    :param time_series_array: Time series to be tested
    :type time_series_array: np.ndarray
    :return: Array with two columns, one for the
             test statistic and one for the critical values
    :rtype: np.ndarray
    """
    b_alpha_5_pct = 4.6  # 4.6 is b_a estimate derived via Monte-Carlo
    test_statistics_array = np.empty((t_indices.shape[0], 2), dtype=np.float64)
    # outer loops goes over all t
    for i in prange(len(t_indices)):

        # compute variance
        t = t_indices[i]
        array_t = time_series_array[:t]
        sigma_squared_t = np.sum(np.square(np.diff(array_t))) / (t - 1)

        # init supremum vals
        max_S_n_t_value = -np.inf
        max_S_n_t_critical_value = np.nan  # Corresponds to c_alpha[n,t]

        y_t = array_t[t - 1]
        # inner loop goes over all n between 1 and t
        for j in prange(len(array_t) - 1):
            # compute test statistic
            n = j + 1
            y_n = time_series_array[j]
            y_t_y_n_diff = get_y_t_y_n_diff(side_test, y_t, y_n)
            S_n_t = y_t_y_n_diff / np.sqrt(sigma_squared_t * (t - n))

            # check if new val is better than supremum
            # if so compute new critical value
            if S_n_t > max_S_n_t_value:
                max_S_n_t_value = S_n_t
                max_S_n_t_critical_value = np.sqrt(b_alpha_5_pct + np.log(t - n))
        # store result of iteration
        test_statistics_array[i, :] = np.array(
            [max_S_n_t_value, max_S_n_t_critical_value], dtype=np.float64
        )
    return test_statistics_array


@njit
def get_y_t_y_n_diff(side_test: str, y_t: np.float64, y_n: np.float64) -> np.float64:
    """Returns the difference between y_t and y_n given
    a test specification.

    :param side_test: 'one_sided_positive' or 'one_sided_negative' or 'two_sided'
    :type side_test: str
    :param y_t: value of the series to be tested at time t
    :type y_t: np.float64
    :param y_n: value of the series to be tested at time n, where n < t
    :type y_n: np.float64
    :return: difference between y_t and y_n given a test specification
    :rtype: np.float64
    """
    if side_test == ONE_SIDED_POSITIVE:
        values_diff = y_t - y_n
    elif side_test == ONE_SIDED_NEGATIVE:
        values_diff = y_n - y_t
    else:
        values_diff = abs(y_t - y_n)
    return values_diff


class ChuStinchcombeWhiteStatTest:
    STATISTIC = "statistic"
    CRITICAL_VALUE = "critical_value"

    def __init__(self, side_test: str):
        """
        Chu-Stinchcombe-White test CUSUM Tests on Levels as
        described on page 250-251 in Advances in Financial Machine Learning
        by Marcos Lopez de Prado.

        :param side_test: 'one_sided_positive' or 'one_sided_negative' or 'two_sided'
        :type side_test: str
        """
        expected_vals = [ONE_SIDED_NEGATIVE, ONE_SIDED_POSITIVE, TWO_SIDED]
        assert (
            side_test in expected_vals
        ), f"Expected one of {expected_vals} got {side_test}..."
        self.side_test = side_test

    def run(self, series_to_test: pd.Series) -> pd.DataFrame:
        """Runs the Chu-Stinchcombe-White stat test and returns
        a dataframe with the test statistics and corresponding critical values.

        :param series_to_test: Time series to be tested.
        :type series_to_test: pd.Series
        :return: DataFrame with results.
        :rtype: pd.DataFrame
        """
        t_indices = np.arange(2, len(series_to_test) + 1)
        S_n_t_array = get_test_statistics_array(
            self.side_test, t_indices, series_to_test.values
        )
        return pd.DataFrame(
            S_n_t_array,
            index=series_to_test.index[1:],
            columns=[self.STATISTIC, self.CRITICAL_VALUE],
        )
