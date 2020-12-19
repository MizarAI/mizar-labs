import numpy as np
import pandas as pd
import pytest
from mizarlabs.static import CLOSE
from mizarlabs.static import LABEL
from mizarlabs.static  import RETURN
from mizarlabs.static import T_VALUE
from mizarlabs.transformers.targets.trend_scanning import TrendScannerLabeling


@pytest.mark.parametrize("look_forward_window", [5, 10, 20])
def test_trend_scanner_transformer_num_nans(look_forward_window, dollar_bar_dataframe):
    """
    Checks if no. of NaNs is as expected
    """
    num_samples = 100
    min_sample_length = 3
    step = 1

    close_reduced = dollar_bar_dataframe[CLOSE][:num_samples]
    t_events = np.array(close_reduced.index[-30:])

    trend_scanner_transformer = TrendScannerLabeling(
        look_forward_window=look_forward_window,
        min_sample_length=min_sample_length,
        step=step,
        t_events=t_events,
    )
    trend_scanning_results = trend_scanner_transformer.transform(close_reduced)

    assert len(
        trend_scanning_results.isna().sum().unique()
    ) == 1 and trend_scanning_results.isna().sum().unique()[0] == (
        look_forward_window - 1
    )


@pytest.mark.parametrize("look_forward_window", [50, 75, 100])
def test_trend_scanner_transformer_trend(look_forward_window, dollar_bar_dataframe):
    """
    Checks the following:

    1) Trend up identified by having all values positive
    2) Trend down identified by having all values negative
    3) No trend identified by comparing mean of t-value to approximately 0
    """
    num_samples = 500
    min_sample_length = 40
    step = 1

    close_reduced = dollar_bar_dataframe[CLOSE][:num_samples]
    close_reduced.loc[close_reduced.index] = (
        np.linspace(1, 100, close_reduced.shape[0]) ** 2
    )

    trend_scanner_transformer = TrendScannerLabeling(
        look_forward_window=look_forward_window,
        min_sample_length=min_sample_length,
        step=step,
    )

    # check trend up
    trend_up = pd.Series(
        np.linspace(1, 100, close_reduced.shape[0]) ** 2, index=close_reduced.index
    )
    trend_scanning_results = trend_scanner_transformer.transform(trend_up)
    assert (trend_scanning_results.dropna()[[T_VALUE, RETURN, LABEL]] >= 0).all().all()

    # check trend down
    trend_down = pd.Series(
        1 / np.linspace(0.1, 10, close_reduced.shape[0]) ** 2, index=close_reduced.index
    )
    trend_scanning_results = trend_scanner_transformer.transform(trend_down)
    assert (trend_scanning_results.dropna()[[T_VALUE, RETURN, LABEL]] <= 0).all().all()

    # check no trend
    no_trend = pd.Series(
        1e7 + np.random.rand(close_reduced.shape[0]), index=close_reduced.index
    )
    trend_scanning_results = trend_scanner_transformer.transform(no_trend)
    assert pytest.approx(0, abs=0.6) == trend_scanning_results[T_VALUE].mean()
