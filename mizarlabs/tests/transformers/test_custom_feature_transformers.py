from mizarlabs.transformers.features.custom import LocalMaxToCloseRatio
import numpy as np


def test_local_max_to_close_ratio(dollar_bar_dataframe):
    window=2
    transformer = LocalMaxToCloseRatio(window=window)
    feature = transformer.transform(dollar_bar_dataframe)
    for i in range(window - 1, len(feature)):
        local_max = dollar_bar_dataframe.high.iloc[i - 1:i - 1+ window].max()
        ratio = local_max / dollar_bar_dataframe.close.iloc[i]
        assert (feature[i] == ratio).all()