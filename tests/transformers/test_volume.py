import datetime
import pandas as pd

from mizarlabs.transformers.technical.volume import BuySellImbalance


def test_exponential_weighted_moving_average_difference(
    dollar_bar_dataframe: pd.DataFrame,
):
    fast = 10
    slow = 20

    buy_sell_imbalance_quote_transformer = BuySellImbalance(
        fast=fast, slow=slow, volume_type="quote"
    )
    buy_sell_imbalance_base_transformer = BuySellImbalance(
        fast=fast, slow=slow, volume_type="base"
    )

    buy_sell_imbalance_quote = buy_sell_imbalance_quote_transformer.transform(
        dollar_bar_dataframe
    )
    buy_sell_imbalance_base = buy_sell_imbalance_base_transformer.transform(
        dollar_bar_dataframe
    )

    assert (
        buy_sell_imbalance_quote != buy_sell_imbalance_base
    ).any(), "Expected different values but got the same"
