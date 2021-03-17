import random

import numpy as np
import pandas as pd
import pytest
from mizarlabs import static
from mizarlabs.transformers.microstructural_features.vpin import VPIN


@pytest.mark.usefixtures("dollar_bar_dataframe")
@pytest.mark.parametrize("index_to_check", list(range(50, 100)))
@pytest.mark.parametrize("window_size", [10, 30, 40])
def test_vpin(dollar_bar_dataframe, index_to_check, window_size):
    vpin = VPIN(window=window_size)
    vpin_feature = vpin.transform(dollar_bar_dataframe)

    total_volume_sum = float(
        dollar_bar_dataframe.iloc[
            index_to_check - window_size + 1 : index_to_check + 1
        ][vpin.base_asset_volume_column_name].sum()
    )

    buy_volume = dollar_bar_dataframe.iloc[
        index_to_check - window_size + 1 : index_to_check + 1
    ][vpin.base_asset_buy_volume_column_name]

    sell_volume = dollar_bar_dataframe.iloc[
        index_to_check - window_size + 1 : index_to_check + 1
    ][vpin.base_asset_sell_volume_column_name]

    diff_volume_sum = float((buy_volume - sell_volume).abs().sum())

    assert np.round(vpin_feature[index_to_check], 8) == round(
        diff_volume_sum / total_volume_sum, 8
    )


@pytest.mark.parametrize("window_size", [10, 20, 30])
def test_vpin_with_zero_volume(window_size):
    index = pd.date_range(start="2018-01-01", end="2018-05-01", freq="6h")
    buy_volume = np.array(
        [
            0
            if i in [i for i in range(int(len(index) / 3), int(len(index) * 2 / 3))]
            else int(abs(random.random()) * 1000)
            for i in range(len(index))
        ]
    )
    sell_volume = np.array(
        [
            0
            if i in [i for i in range(int(len(index) / 3), int(len(index) * 2 / 3))]
            else int(abs(random.random()) * 1000)
            for i in range(len(index))
        ]
    )
    total_volume = buy_volume + sell_volume

    no_volume_df = pd.DataFrame(
        {
            static.BASE_ASSET_VOLUME: total_volume,
            static.BASE_ASSET_SELL_VOLUME: sell_volume,
            static.BASE_ASSET_BUY_VOLUME: buy_volume,
        },
        index=index,
    )

    vpin = VPIN(window=window_size)
    vpin_feature = vpin.transform(no_volume_df)
    assert (
        len(
            np.unique(
                vpin_feature[
                    int(len(index) / 3) + window_size - 1 : int(len(index) * 2 / 3)
                ]
            )
        )
        == 1
    )
    assert (
        np.unique(
            vpin_feature[
                int(len(index) / 3) + window_size - 1 : int(len(index) * 2 / 3)
            ]
        )[0]
        == 0
    )
