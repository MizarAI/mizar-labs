import numpy as np
import pandas as pd
from mizarlabs import static
from mizarlabs.transformers.utils import check_missing_columns
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class VPIN(BaseEstimator, TransformerMixin):
    """
    Implement the Volume-Synchronized Probability of Informed Trading.

    We assume that the index of the dataframe in input is based on the close time

    See page 292 of Advances in Financial Machine Learning by Marcos Lopez de
    Prado for additional information.

    :param window: The window size used for the calculation of the VPIN
    :type window: int
    :param base_asset_volume_column_name: name of the column where the volume (sum of the base asset quantity per trade)
                                          of the time bar is stored, defaults to config.BASE_ASSET_VOLUME
    :type base_asset_volume_column_name: str, optional
    :param base_asset_buy_volume_column_name: name of the column where the volume (sum of the buy base asset quantity
                                              per trade) of the time bar is stored, defaults to
                                              config.BASE_ASSET_BUY_VOLUME
    :type base_asset_buy_volume_column_name: str, optional
    :param base_asset_sell_volume_column_name: name of the column where the volume (sum of the sell base asset quantity
                                               per trade) of the time bar is stored, defaults to
                                               config.BASE_ASSET_SELL_VOLUME
    :type base_asset_sell_volume_column_name: str, optional
    """

    def __init__(
        self,
        window: int = 20,
        base_asset_volume_column_name: str = static.BASE_ASSET_VOLUME,
        base_asset_buy_volume_column_name: str = static.BASE_ASSET_BUY_VOLUME,
        base_asset_sell_volume_column_name: str = static.BASE_ASSET_SELL_VOLUME,
    ):
        self.window = window
        self.base_asset_volume_column_name = base_asset_volume_column_name
        self.base_asset_buy_volume_column_name = base_asset_buy_volume_column_name
        self.base_asset_sell_volume_column_name = base_asset_sell_volume_column_name

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_missing_columns(
            X,
            [
                self.base_asset_volume_column_name,
                self.base_asset_buy_volume_column_name,
                self.base_asset_sell_volume_column_name,
            ],
        )

        total_volume_windowed_sum = X.rolling(self.window)[
            self.base_asset_volume_column_name
        ].sum()

        diff_volume = (
            (
                X[self.base_asset_buy_volume_column_name]
                - X[self.base_asset_sell_volume_column_name]
            )
            .abs()
            .rolling(
                self.window,
            )
            .sum()
        )

        vpin_feature = diff_volume / total_volume_windowed_sum

        vpin_feature.iloc[self.window :] = vpin_feature[self.window :].fillna(0)

        assert (
            ~vpin_feature[self.window :].isna().any()
        ), "No NaNs are allowed after the warm-up period of the rolling window"

        return vpin_feature.values.reshape(-1, 1)
