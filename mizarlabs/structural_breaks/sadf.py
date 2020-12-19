"""
Explosiveness tests: SADF
"""
import warnings
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from numba import jit
from numba import prange

from .sdfc import get_beta_and_beta_var


class SupremumAugmentedDickeyFullerStatTest:
    """
    Implementation of SADF, p. 252-261.

    SADF fits the ADF regression at each end point t with backwards expanding start points. For the estimation
    of SADF(t), the right side of the window is fixed at t. SADF recursively expands the beginning of the sample
    up to t - min_length, and returns the sup of this set.

    When doing with sub- or super-martingale test, the variance of beta of a weak long-run bubble may be smaller than
    one of a strong short-run bubble, hence biasing the method towards long-run bubbles. To correct for this bias,
    ADF statistic in samples with large lengths can be penalized with the coefficient phi in [0, 1] such that:

    ADF_penalized = ADF / (t - t0) ^ phi), where t0 is between 1 and t - min_length

    :param model: model test specification, which can be 'no_trend', 'linear',
                  'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :type model: str
    :param lags: either number of lags to use or array of specified lags
    :type lags: Union[int, list]
    :param min_length: minimum number of observations needed for estimation
    :type min_length: int
    :param add_constant: flag to add constant, NOTE: sub- and super-martingale tests always
                         have a constant.
    :type add_constant: bool
    :param phi: coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :type phi: float
    """

    def __init__(
        self,
        model: str,
        lags: Union[int, list],
        min_length: int,
        add_constant: bool,
        phi: float,
    ):
        assert 0 <= phi <= 1, f"Expected phi to be between 0 and 1, got {phi}."
        assert (
            min_length > lags
        ), f"min_length {min_length} should be higher than lags {lags}."
        self.model = model
        self.lags = lags
        self.min_length = min_length
        self.add_constant = add_constant
        self.phi = phi

    def run(self, series_to_test: pd.Series) -> pd.Series:
        """
        Runs the Supremum Augmented Dickey Fuller test and returns a
        series with SADF statistics.

        When the model specification is for sub- and super-martingale
        then a series with SMT statistics is returned. See page 260
        for additional details.

        :param series_to_test: series for which SADF statistics are generated
        :type series_to_test: pd.Series
        :return: series of SADF statistics (or SMT in case of sub- and super-martingale tests)
        :rtype: pd.Series
        """
        X, y = _get_X_y(series_to_test, self.model, self.lags, self.add_constant)
        indices = y.index[self.min_length :]

        sadf_array = _sadf_outer_loop(
            X=X.values.astype(np.float64),
            y=y.values.astype(np.float64).reshape(-1, 1),
            min_length=self.min_length,
            model=self.model,
            phi=self.phi,
        )

        return pd.Series(sadf_array, index=indices)


@jit(parallel=False, nopython=True)
def _get_sadf_at_t(
    X: np.ndarray, y: np.ndarray, min_length: int, model: str, phi: float
) -> np.float64:
    """
    Snippet 17.2, page 258. SADF's Inner Loop (get SADF value at t)

    :param X: matrix of lagged values, constants, trend coefficients
    :type X: np.ndarray
    :param y: y values (either y or y.diff() or np.log(y))
    :type y: np.ndarray
    :param min_length: minimum number of samples needed for estimation
    :type min_length: int
    :param model: either 'no_trend', 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :type model: str
    :param phi: coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :type phi: float
    :return: SADF statistic at time t
    :rtype: np.float64
    """

    adf_array = np.empty(y.shape[0] - min_length + 1)

    # inner loop starts from 1 to t - tau see page 253
    for start in prange(0, y.shape[0] - min_length + 1):
        X_ = np.ascontiguousarray(X[start:])
        y_ = np.ascontiguousarray(y[start:])

        b_mean_, b_var_ = get_beta_and_beta_var(X_, y_)

        current_adf = b_mean_[0, 0] / b_var_[0, 0] ** 0.5

        # if the test specification is a sub- or super-martingale test
        # adjust the test statistic as described on page 260.
        if model[:2] == "sm":
            t = y.shape[0]
            t0 = start + 1  # t0 index starts from 1 to t - tau (page 260)
            current_adf = np.abs(current_adf) / ((y.shape[0] - (t - t0)) ** phi)

        adf_array[start] = current_adf

    return np.max(adf_array)


def _get_X_y_standard_specification(
    series: pd.Series, lags: Union[int, list]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns the matrix with features X and y for the
    standard specifcation without constant as described on
    page 252.

    :param series: series to calculated SADF values for.
    :type series: pd.Series
    :param lags: either number of lags to use or array of specified lags
    :type lags: Union[int, list]
    :return: matrix with features X and vector y with target values
    :rtype: Tuple[pd.DataFrame, pd.Series]
    """
    series_diff = series.diff().dropna()
    X = _lag_df(series_diff.to_frame(), lags).dropna()
    X["y_lagged"] = series.shift(1).loc[X.index]  # add y_(t-1) column
    y = series_diff.loc[X.index]
    return X, y


def _get_X_y(
    series: pd.Series, model: str, lags: Union[int, list], add_const: bool
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Snippet 17.2, page 258-259. Preparing The Datasets

    :param series: to prepare for test statistics generation (for example log prices)
    :type series: pd.Series
    :param model: either 'no_trend', 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :type model: str
    :param lags: either number of lags to use or array of specified lags
    :type lags: Union[int, list]
    :param add_const: flag to add constant
    :type add_const: bool
    :raises ValueError: if unknown model specification is given.
    :return: prepared X and y for SADF calculation
    :rtype: Tuple[pd.DataFrame, pd.Series]
    """
    if not add_const and model in ["sm_poly_1", "sm_poly_2", "sm_exp", "sm_power"]:
        warnings.warn(
            f"Model specification {model} always has a constant.", category=UserWarning
        )

    if model == "no_trend":
        X, y = _get_X_y_standard_specification(series, lags)
        beta_column = "y_lagged"
        if add_const:
            X["const"] = 1
    elif model == "linear":
        X, y = _get_X_y_standard_specification(series, lags)
        X["trend"] = np.arange(
            1, X.shape[0] + 1
        )  # Add t to the model (0, 1, 2, 3, 4, 5, .... t)
        beta_column = (
            "y_lagged"  # Column which is used to estimate test beta statistics
        )
        if add_const:
            X["const"] = 1
    elif model == "quadratic":
        X, y = _get_X_y_standard_specification(series, lags)
        X["trend"] = np.arange(
            1, X.shape[0] + 1
        )  # Add t to the model (0, 1, 2, 3, 4, 5, .... t)
        X["quad_trend"] = X["trend"] ** 2  # Add t^2 to the model (0, 1, 4, 9, ....)
        beta_column = (
            "y_lagged"  # Column which is used to estimate test beta statistics
        )
        if add_const:
            X["const"] = 1
    elif model == "sm_poly_1":
        y = series.copy()
        X = pd.DataFrame(index=y.index)
        X["const"] = 1
        X["trend"] = np.arange(1, X.shape[0] + 1)
        X["quad_trend"] = X["trend"] ** 2
        beta_column = "quad_trend"
    elif model == "sm_poly_2":
        y = np.log(series.copy())
        X = pd.DataFrame(index=y.index)
        X["const"] = 1
        X["trend"] = np.arange(1, X.shape[0] + 1)
        X["quad_trend"] = X["trend"] ** 2
        beta_column = "quad_trend"
    elif model == "sm_exp":
        y = np.log(series.copy())
        X = pd.DataFrame(index=y.index)
        X["const"] = 1
        X["trend"] = np.arange(1, X.shape[0] + 1)
        beta_column = "trend"
    elif model == "sm_power":
        y = np.log(series.copy())
        X = pd.DataFrame(index=y.index)
        X["const"] = 1
        X["log_trend"] = np.log(np.arange(1, X.shape[0] + 1))
        beta_column = "log_trend"
    else:
        raise ValueError("Unknown model")

    # Move y_lagged column to the front for further extraction
    columns = list(X.columns)
    columns.insert(0, columns.pop(columns.index(beta_column)))
    X = X[columns]

    assert (
        ~X.isna().any().any() and ~y.isna().any().any()
    ), f"The constructed X and y contain NaNs based on the model specification {model}."
    return X, y


def _lag_df(df: pd.DataFrame, lags: Union[int, list]) -> pd.DataFrame:
    """
    Snipet 17.3, page 259. Apply Lags to DataFrame

    :param df: dataframe with features
    :type df: pd.DataFrame
    :param lags: either number of lags to use or array of specified lags
    :type lags: Union[int, list]
    :return: dataframe with features and lagged differenced y features
    :rtype: pd.DataFrame
    """
    df_lagged = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(1, lags + 1)
    else:
        lags = [int(lag) for lag in lags]

    for lag in lags:
        temp_df = df.shift(lag).copy(deep=True)
        temp_df.columns = [str(i) + "_diff_lag_" + str(lag) for i in temp_df.columns]
        df_lagged = df_lagged.join(temp_df, how="outer")
    return df_lagged


@jit(parallel=True, nopython=True)
def _sadf_outer_loop(
    X: np.ndarray,
    y: np.ndarray,
    min_length: np.int64,
    model: str,
    phi: np.float64,
) -> np.ndarray:
    """Runs the SADF outer loop, i.e for each t in T.

    :param X: array with features
    :type X: np.ndarray
    :param y: array with target values
    :type y: np.ndarray
    :param min_length: minimum number of observations
    :type min_length: np.int64
    :param model: either 'no_trend', 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :type model: str
    :param phi: coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :type phi: np.float64
    :return: array with SADF statistics for each point in time t for all T
    :rtype: np.ndarray
    """
    sadf_array = np.empty(y.shape[0] - min_length, dtype=np.float64)
    for index in prange(min_length, y.shape[0]):
        sadf_array[index - min_length] = _get_sadf_at_t(
            X[: (index + 1), :], y[: (index + 1), :], min_length, model, phi
        )
    return sadf_array
