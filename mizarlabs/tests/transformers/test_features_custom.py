from mizarlabs.transformers.features.custom import RollingZScoreTransformer
from mizarlabs.transformers.features.custom import RollingZScoreTransformerFactory
import pandas as pd
import pytest
import numpy as np


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
    assert z_score.isna().sum() == window - 1 and all(z_score.isna()[: window - 1])
    for i in range(window - 1, x.shape[0]):
        z_score[i] == (
            x[i] - np.mean(x[(i - (window - 1)) * window : (i - 1) * window])
        ) / np.std(x[(i - (window - 1)) * window : (i - 1) * window], ddof=1)
