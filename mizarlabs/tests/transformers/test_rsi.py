from mizarlabs.transformers.technical.rsi import BarArrivalRSIStrategy
from mizarlabs.transformers.technical.rsi import RSIConfirmation


def test_bar_arrival_rsi_strategy_predict(dollar_bar_dataframe):
    rsi_upper_threshold = 55
    rsi_lower_threshold = 45
    bar_arrival_upper_threshold = 0
    bar_arrival_lower_threshold = -0.2
    rsi_timeperiod = 25
    bar_arrival_fast_period = 500
    bar_arrival_slow_period = 200
    max_bar_arrival_mean_diff = 10000000000000
    bar_arrival_rsi_strategy = BarArrivalRSIStrategy(
        rsi_upper_threshold,
        rsi_lower_threshold,
        bar_arrival_upper_threshold,
        bar_arrival_lower_threshold,
        rsi_timeperiod,
        bar_arrival_fast_period,
        bar_arrival_slow_period,
        max_bar_arrival_mean_diff,
    )
    preds = bar_arrival_rsi_strategy.predict(dollar_bar_dataframe)
    assert set(preds).issubset(bar_arrival_rsi_strategy.classes_)


def test_bar_arrival_rsi_strategy_predict_proba(dollar_bar_dataframe):
    rsi_upper_threshold = 55
    rsi_lower_threshold = 45
    bar_arrival_upper_threshold = 0
    bar_arrival_lower_threshold = -0.2
    rsi_timeperiod = 25
    bar_arrival_fast_period = 500
    bar_arrival_slow_period = 200
    max_bar_arrival_mean_diff = 10000000000000
    bar_arrival_rsi_strategy = BarArrivalRSIStrategy(
        rsi_upper_threshold,
        rsi_lower_threshold,
        bar_arrival_upper_threshold,
        bar_arrival_lower_threshold,
        rsi_timeperiod,
        bar_arrival_fast_period,
        bar_arrival_slow_period,
        max_bar_arrival_mean_diff,
    )
    preds = bar_arrival_rsi_strategy.predict(dollar_bar_dataframe)
    pred_proba = bar_arrival_rsi_strategy.predict_proba(dollar_bar_dataframe)
    pred_to_idx_map = {0: 0, 1: 1}
    assert all(pred_proba.sum(axis=1) == 1)
    assert all(pred_proba[i, pred_to_idx_map[p]] == 1 for i, p in enumerate(preds))


def test_rsi_confirmation_predict(dollar_bar_dataframe):
    rsi_threshold = 45
    rsi_timeperiod = 25
    rsi_confirmation_period = 2
    rsi_moving_average_window = 2

    bar_arrival_rsi_strategy = RSIConfirmation(
        rsi_timeperiod,
        rsi_threshold,
        rsi_confirmation_period,
        rsi_moving_average_window,
    )
    preds = bar_arrival_rsi_strategy.predict(dollar_bar_dataframe)
    assert set(preds).issubset(bar_arrival_rsi_strategy.classes_)


def test_rsi_confirmation_predict_proba(dollar_bar_dataframe):
    rsi_threshold = 45
    rsi_timeperiod = 25
    rsi_confirmation_period = 2
    rsi_moving_average_window = 2
    bar_arrival_rsi_strategy = RSIConfirmation(
        rsi_timeperiod,
        rsi_threshold,
        rsi_confirmation_period,
        rsi_moving_average_window,
    )
    preds = bar_arrival_rsi_strategy.predict(dollar_bar_dataframe)
    pred_proba = bar_arrival_rsi_strategy.predict_proba(dollar_bar_dataframe)
    pred_to_idx_map = {0: 0, 1: 1}
    assert all(pred_proba.sum(axis=1) == 1)
    assert all(pred_proba[i, pred_to_idx_map[p]] == 1 for i, p in enumerate(preds))
