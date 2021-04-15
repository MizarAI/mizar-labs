from mizarlabs.transformers.technical.mean_reversion import MeanReversionStrategy


def test_predict(dollar_bar_dataframe):
    threshold = 0.001
    short_time_period = 5
    mean_reversion_strategy = MeanReversionStrategy(
        threshold=threshold, short_time_period=short_time_period
    )
    preds = mean_reversion_strategy.predict(dollar_bar_dataframe)
    assert set(preds).issubset(mean_reversion_strategy.classes_)

    assert all(preds[:short_time_period - 1] == 0)
    threshold = 0
    mean_reversion_strategy = MeanReversionStrategy(
        threshold=threshold, short_time_period=short_time_period
    )
    preds = mean_reversion_strategy.predict(dollar_bar_dataframe)
    assert 0 not in set(preds[short_time_period:])

    threshold = 1000
    mean_reversion_strategy = MeanReversionStrategy(
        threshold=threshold, short_time_period=short_time_period
    )
    preds = mean_reversion_strategy.predict(dollar_bar_dataframe)
    assert all(preds == 0)


def test_predict_proba(dollar_bar_dataframe):
    threshold = 0.01
    short_time_period = 5
    mean_reversion_strategy = MeanReversionStrategy(
        threshold=threshold, short_time_period=short_time_period
    )
    preds = mean_reversion_strategy.predict(dollar_bar_dataframe)
    pred_proba = mean_reversion_strategy.predict_proba(dollar_bar_dataframe)
    pred_to_idx_map = {-1: 0, 0: 1, 1: 2}
    assert all(pred_proba.sum(axis=1) == 1)
    assert all(pred_proba[i, pred_to_idx_map[p]] == 1 for i, p in enumerate(preds))