"""
Explosiveness tests: Chow-Type Dickey-Fuller Test
"""
from typing import Tuple

import numpy as np
import pandas as pd
from numba import jit
from numba import njit
from numba import prange


@jit(parallel=True, nopython=True)
def get_dfc_array(
    indices: np.ndarray, y_diff: np.ndarray, y_lag: np.ndarray
) -> np.ndarray:
    """Returns the Chow-Type Dickey-Fuller t-values.

    :param indices: Indices to iterate over
    :type indices: np.ndarray
    :param y_diff: Differenced time series
    :type y_diff: np.ndarray
    :param y_lag: Lagged time series
    :type y_lag: np.ndarray
    :return: Array with t-values.
    :rtype: np.ndarray
    """
    dfc_array = np.empty_like(indices, dtype=np.float64)
    for i in prange(len(indices)):
        dummy_var = np.ones_like(y_lag)
        dummy_var[: indices[i]] = 0  # D_t* indicator: before t* D_t* = 0
        X = y_lag * dummy_var
        beta_hat, beta_var = get_beta_and_beta_var(
            X,
            y_diff,
        )
        dfc_array[i] = beta_hat[0, 0] / np.sqrt(beta_var[0, 0])
    return dfc_array


@njit
def get_beta_and_beta_var(X: np.ndarray, y: np.ndarray) -> Tuple[float]:
    """
    Returns the OLS estimates of the coefficients and the variance of the coefficients.

    :param X: Matrix with features values
    :type X: pd.DataFrame
    :param y: Series with target values.
    :type y: pd.Series
    :return: Tuple with the coefficients and the variance
                of the coefficients estimate.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    Xy = np.dot(X.T, y)
    XX = np.dot(X.T, X)
    XX_inv = np.linalg.inv(XX)
    beta_hat = np.dot(np.ascontiguousarray(XX_inv), Xy)
    err = y - np.dot(X, beta_hat)
    beta_hat_var = np.dot(err.T, err) / (X.shape[0] - X.shape[1]) * XX_inv
    return beta_hat, beta_hat_var


class SupremumDickeyFullerChowStatTest:
    def __init__(self, min_num_samples: int = 20):
        """
        Chow-Type Dickey-Fuller Test statistics as
        described on page 251-252 in Advances in Financial
        Machine Learning by Marcos Lopez de Prado.

        :param min_num_samples: min. no. of samples for the dummy variable
                                in the test specification to ensure enough
                                ones and zeros, defaults to 20
        :type min_num_samples: int, optional
        """
        self.min_num_samples = min_num_samples

    def run(self, series_to_test: pd.Series) -> pd.Series:
        indices = np.arange(
            self.min_num_samples, series_to_test.shape[0] - self.min_num_samples
        )
        series_diff = series_to_test.diff().dropna()
        series_lag = series_to_test.shift(1).dropna()
        dfc_array = get_dfc_array(
            indices,
            np.ascontiguousarray(series_diff.values.reshape(-1, 1), dtype=np.float64),
            np.ascontiguousarray(series_lag.values.reshape(-1, 1), dtype=np.float64),
        )
        return pd.Series(
            dfc_array,
            index=series_to_test.index.values[
                self.min_num_samples : series_to_test.shape[0] - self.min_num_samples
            ],
        )
