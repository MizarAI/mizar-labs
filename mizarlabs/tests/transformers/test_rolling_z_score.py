from mizarlabs.transformers.features import RollingZScoreTransformer
from mizarlabs.transformers.features import RollingZScoreTransformerFactory
import pandas as pd
import pytest
import numpy as np
from mizarlabs.model.pipeline import MizarFeatureUnion


def test_rolling_z_score_transformer_factory():
    with pytest.raises(KeyError) as exc:
        RollingZScoreTransformerFactory.get_transformer("some_bad_key", window=1)
        assert "some_bad_key" in exc
    for k in RollingZScoreTransformerFactory.types.keys():
        transformer = RollingZScoreTransformerFactory.get_transformer(k, window=1)
        assert transformer.to_series_func == RollingZScoreTransformerFactory.types[k]


def test_transform():
    x = pd.Series([1, 3, 24, 5, 31, 32, 5])
    window = 3
    rolling_z_score_transformer = RollingZScoreTransformer(
        window=window, to_series_func=lambda x: x
    )
    z_score = rolling_z_score_transformer.transform(x)
    assert np.isnan(z_score).sum() == window - 1 and all(np.isnan(z_score)[: window - 1])
    for i in range(window - 1, x.shape[0]):
        z_score[i] == (
            x[i] - np.mean(x[(i - (window - 1)) * window : (i - 1) * window])
        ) / np.std(x[(i - (window - 1)) * window : (i - 1) * window], ddof=1)


def test_feature_union(dollar_bar_dataframe):
    feature_transformer = MizarFeatureUnion(
        [
            (
                "bar_arrival_time",
                RollingZScoreTransformerFactory.get_transformer("bar_arrival_time", 2),
            ),
            (
                "buy_sell_diff",
                RollingZScoreTransformerFactory.get_transformer("buy_sell_diff", 2),
            ),
            (
                "average_buy_size",
                RollingZScoreTransformerFactory.get_transformer("average_buy_size", 2),
            ),
            (
                "high_to_low_ratio",
                RollingZScoreTransformerFactory.get_transformer("high_to_low_ratio", 2),
            ),
        ]
    )
    features_df = feature_transformer.transform(dollar_bar_dataframe)
    assert all(features_df.index == dollar_bar_dataframe.index)
    assert features_df.shape == (
        len(dollar_bar_dataframe),
        len(feature_transformer.transformer_list),
    )
