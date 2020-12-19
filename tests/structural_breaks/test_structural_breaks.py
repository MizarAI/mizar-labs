import numpy as np
import pandas as pd
import pytest
from mizarlabs import static
from mizarlabs.structural_breaks.cusum_chu_stinchcome_white import (
    ChuStinchcombeWhiteStatTest,
)
from mizarlabs.structural_breaks.cusum_chu_stinchcome_white import ONE_SIDED_NEGATIVE
from mizarlabs.structural_breaks.cusum_chu_stinchcome_white import TWO_SIDED
from mizarlabs.structural_breaks.sadf import SupremumAugmentedDickeyFullerStatTest
from mizarlabs.structural_breaks.sdfc import SupremumDickeyFullerChowStatTest


@pytest.mark.parametrize(
    "model",
    ["no_trend", "linear", "quadratic", "sm_poly_1", "sm_poly_2", "sm_exp", "sm_power"],
)
@pytest.mark.parametrize("add_constant", [True, False])
@pytest.mark.parametrize("max_num_samples", np.arange(100, 400, step=100))
@pytest.mark.parametrize("min_length", [np.random.randint(20, 40)])
@pytest.mark.parametrize("lags", [np.random.randint(1, 19)])
@pytest.mark.parametrize("phi", [np.random.rand()])
def test_supremum_augmented_dickey_fuller_stat_test(
    dollar_bar_dataframe: pd.DataFrame,
    model: str,
    add_constant: bool,
    max_num_samples: int,
    min_length: int,
    lags: int,
    phi: float,
):
    """
    Checks the following:
        1) Warnings being raised correctly
        2) Expected no. of computed values
        3) No NaNs in calculated test statistics
        4) Computed test statistics are finite
        5) Sub- and super-martingale tests are positive
    """
    # set data
    close_reduced = get_samples(max_num_samples, dollar_bar_dataframe)

    # run test
    sadf_test = SupremumAugmentedDickeyFullerStatTest(
        model=model,
        lags=lags,
        min_length=min_length,
        add_constant=add_constant,
        phi=phi,
    )

    # check if warnings are raised correctly
    if "sm_" in model and not add_constant:
        sadf_results = pytest.warns(UserWarning, sadf_test.run, close_reduced)
    else:
        sadf_results = sadf_test.run(close_reduced)

    # check if no. of computed values is as expected
    if "sm_" in model:
        assert sadf_results.shape[0] == (max_num_samples - min_length)
    else:
        assert sadf_results.shape[0] == (max_num_samples - min_length - lags - 1)

    # check if NaNs in series
    assert not sadf_results.isna().any().any()

    # check if compute statistics are finite
    assert np.isfinite(sadf_results.values).all()

    # check if sub- and super-martingale test statistics are positive
    if "sm_" in model:
        assert all(sadf_results.values >= 0)


def get_test_statistic(idx: int, close_reduced: pd.Series) -> float:
    """
    Computes a single t-value in the Chow Dickey Fuller stat test.
    """
    # set data
    y_diff = close_reduced.diff().dropna()
    y_lag = close_reduced.shift().dropna()
    dummy_var = np.zeros_like(y_lag)
    dummy_var[idx:] = 1

    # ols: beta and beta_var
    X = (y_lag * dummy_var).values.reshape(-1, 1)
    y = y_diff.values.reshape(-1, 1)
    XX = np.dot(X.T, X)
    Xy = np.dot(X.T, y)
    XX_inv = np.linalg.inv(XX)
    beta_hat = np.dot(XX_inv, Xy)
    err = y - np.dot(X, beta_hat)
    beta_var = np.dot(err.T, err) / (X.shape[0] - X.shape[1]) * XX_inv

    # compute t value
    test_statistic = (beta_hat / np.sqrt(beta_var))[0, 0]
    return test_statistic


def get_samples(max_num_samples: int, dollar_bar_dataframe: pd.DataFrame) -> pd.Series:
    """
    Returns subset of the series.
    """
    close = dollar_bar_dataframe[static.CLOSE]
    return close.iloc[:max_num_samples]


@pytest.mark.parametrize("max_num_samples", [100, 200, 300])
@pytest.mark.parametrize("min_num_samples_factor", np.linspace(0.1, 0.4, num=10))
def test_supremum_dickey_fuller_chow_stat_test(
    max_num_samples: int,
    min_num_samples_factor: float,
    dollar_bar_dataframe: pd.DataFrame,
):
    """
    Checks the following:
        1) Expected no. of t-values to calculate
        2) Expected values with t-values
        3) No NaNs in calculated test statistics
        4) Compares manual calculation of first and last t-value
    """
    # set data
    min_num_samples = int(max_num_samples * min_num_samples_factor)
    close_reduced = get_samples(max_num_samples, dollar_bar_dataframe)

    # run test
    sdfc_test = SupremumDickeyFullerChowStatTest(min_num_samples=min_num_samples)
    sdfc_results = sdfc_test.run(close_reduced)

    # check if correct no. of t values have been calculated
    assert len(sdfc_results) == close_reduced.shape[0] - 2 * min_num_samples

    # check if expected indices have a test statistic
    indices_with_results = close_reduced.index.values[min_num_samples:-min_num_samples]
    np.testing.assert_array_equal(sdfc_results.index.values, indices_with_results)

    # check no nans in results
    assert ~sdfc_results.isna().any()

    # manual calculation for first and last test statistic
    first_idx = min_num_samples
    np.testing.assert_almost_equal(
        get_test_statistic(first_idx, close_reduced), sdfc_results[0]
    )

    last_idx = close_reduced.shape[0] - min_num_samples - 1
    np.testing.assert_almost_equal(
        get_test_statistic(last_idx, close_reduced), sdfc_results[-1]
    )


@pytest.mark.parametrize("max_num_samples", [100, 400])
@pytest.mark.parametrize(
    "side_test", ["one_sided_negative", "one_sided_positive", "two_sided"]
)
def test_chu_stinchcombe_white_stat_test(
    max_num_samples: int, side_test: str, dollar_bar_dataframe: pd.DataFrame
):
    """
    Checks the following:

    1) Expected no. of computed test statistics is as expected
    2) No NaNs in output
    3) Check if the first computed statistic is the same as
       the manually computed one.
    """
    # set data
    close_reduced = get_samples(max_num_samples, dollar_bar_dataframe)

    # run test
    stat_test = ChuStinchcombeWhiteStatTest(side_test=side_test)
    test_results = stat_test.run(close_reduced)

    # check if shape test results is as expected
    assert test_results.shape[0] == max_num_samples - 1

    # check no nans
    assert ~test_results.isna().any().any()

    # manually calculate first statistic and compare with output
    t = 2
    y_diff_squared = close_reduced.diff() ** 2
    sigma_squared_t = (1 / (t - 1)) * y_diff_squared.iloc[:t].sum()
    y_t = close_reduced[t - 1]
    n = 1
    y_n = close_reduced[n - 1]

    if ONE_SIDED_NEGATIVE == side_test:
        diff_y_t_y_n = y_n - y_t
    elif TWO_SIDED == side_test:
        diff_y_t_y_n = abs(y_t - y_n)
    else:
        diff_y_t_y_n = y_t - y_n

    test_statistic = diff_y_t_y_n / np.sqrt(sigma_squared_t * (t - n))
    critical_value = np.sqrt(4.6 + np.log(t - n))
    np.testing.assert_almost_equal(
        test_results.iloc[0].values, np.array([test_statistic, critical_value])
    )
