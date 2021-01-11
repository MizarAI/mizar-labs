import numpy as np
import pandas as pd
from mizarlabs.transformers.utils import check_missing_columns
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from mizarlabs import static


class KyleLambda(BaseEstimator, TransformerMixin):
    """
     Kyle lambda liquidity estimator (p.286).

    See page 286 of Advances in Financial Machine Learning by Marcos Lopez de
    Prado for additional information.

    :param window: The window size
    :type window: int
    :param close_column_name: The name of the close column
    :type close_column_name: str, optional
    :param base_asset_volume_column_name: The name of the base asset volume column
    :type base_asset_volume_column_name: str, optional
    """

    def __init__(
        self,
        window: int = 20,
        close_column_name: str = "close",
        base_asset_volume_column_name: str = "base_asset_volume",
    ):
        self.window = window
        self.close_column_name = close_column_name
        self.base_asset_volume_column_name = base_asset_volume_column_name

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_missing_columns(
            X, [self.close_column_name, self.base_asset_volume_column_name]
        )

        close_diff_abs_over_vol = (
            X[self.close_column_name].diff().abs()
            / X[self.base_asset_volume_column_name]
        )
        kyle_lambda = close_diff_abs_over_vol.rolling(self.window).mean()
        return kyle_lambda.values.reshape(-1, 1)


class AmihudLambda(BaseEstimator, TransformerMixin):
    """
    Amihud Lambda liquidity estimator (p.288).

    See page 288 of Advances in Financial Machine Learning by Marcos Lopez de
    Prado for additional information.

    :param window: The window size
    :type window: int
    :param close_column_name: The name of the close column
    :type close_column_name: str, optional
    :param quote_asset_volume_column_name: The name of the quote asset volume column
    :type quote_asset_volume_column_name: str, optional
    """

    def __init__(
        self,
        window: int = 20,
        close_column_name: str = static.CLOSE,
        quote_asset_volume_column_name: str = static.QUOTE_ASSET_VOLUME,
    ):
        self.window = window
        self.close_column_name = close_column_name
        self.quote_asset_volume_column_name = quote_asset_volume_column_name

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_missing_columns(
            X, [self.close_column_name, self.quote_asset_volume_column_name]
        )

        abs_log_close_diff = np.abs(np.log(X[self.close_column_name]).diff())
        abs_ret_over_quote_vol = (
            abs_log_close_diff / X[self.quote_asset_volume_column_name]
        )
        return abs_ret_over_quote_vol.rolling(self.window).mean().values.reshape(-1, 1)


class HasbrouckLambda(BaseEstimator, TransformerMixin):
    """
    Hasbrouck Lambda price impact estimator(p.289).

    See page 289 of Advances in Financial Machine Learning by Marcos Lopez de
    Prado for additional information.

    :param window: The window size
    :type window: int
    :param close_column_name: The name of the close column
    :type close_column_name: str, optional
    :param quote_asset_volume_column_name: The name of the quote asset volume column
    :type quote_asset_volume_column_name: str, optional
    """

    def __init__(
        self,
        window: int = 20,
        close_column_name: str = static.CLOSE,
        quote_asset_volume_column_name: str = static.QUOTE_ASSET_VOLUME,
    ):
        self.window = window
        self.close_column_name = close_column_name
        self.quote_asset_volume_column_name = quote_asset_volume_column_name

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_missing_columns(
            X,
            [self.close_column_name, self.quote_asset_volume_column_name],
        )

        abs_log_close_diff = np.log(X[self.close_column_name]).diff().abs()
        return (
            (abs_log_close_diff / np.sqrt(X[self.quote_asset_volume_column_name]))
            .rolling(self.window)
            .mean()
        ).values.reshape(-1, 1)
