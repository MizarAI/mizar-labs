import numpy as np
import pandas as pd
from mizarlabs import static
from mizarlabs.transformers.utils import check_missing_columns
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class RollMeasure(BaseEstimator, TransformerMixin):
    """
    Implement the roll measure which gives the estimate of effective bid-ask spread.

    See page 282 of Advances in Financial Machine Learning by Marcos Lopez de
    Prado for additional information.

    :param window: The window size
    :type window: int
    :param close_column_name: The name of the close column
    :type close_column_name: str, optional
    """

    def __init__(self, window: int = 20, close_column_name: str = static.CLOSE):
        self.window = window
        self.close_column_name = close_column_name

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_missing_columns(X, [self.close_column_name])

        close_diff = X[self.close_column_name].diff()
        close_diff_lag = close_diff.shift(1)
        return (
            2 * np.sqrt(abs(close_diff.rolling(window=self.window).cov(close_diff_lag)))
        ).values.reshape(-1, 1)


class RollImpact(RollMeasure):
    """
    Derivate from Roll Measure which takes into account dollar volume traded.

    See page 282 of Advances in Financial Machine Learning by Marcos Lopez de
    Prado for additional information.

    :param window: The window size
    :type window: int
    :param close_column_name: The name of the close column
    :type close_column_name: str, optional
    :param quote_asset_volume_column_name: The name quote asset column column
    :type quote_asset_volume_column_name: str, optional
    """

    def __init__(
        self,
        window: int = 20,
        close_column_name: str = static.CLOSE,
        quote_asset_volume_column_name: str = static.QUOTE_ASSET_VOLUME,
    ):
        super().__init__(window, close_column_name)
        self.quote_asset_volume_column_name = quote_asset_volume_column_name

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        roll_measure = super().transform(X)
        return roll_measure / X[self.quote_asset_volume_column_name].values.reshape(
            -1, 1
        )


class ParkinsonVolatility(BaseEstimator, TransformerMixin):
    """
    High low volatility estimator developed by Parkinson (1980).

    :param window: The window size
    :type window: int
    :param high_column_name: The name of the high column
    :type high_column_name: str, optional
    :param low_column_name: The name of the low column
    :type low_column_name: str, optional
    """

    def __init__(
        self,
        window: int = 20,
        high_column_name: str = static.HIGH,
        low_column_name: str = static.LOW,
    ):
        self.window = window
        self.high_column_name = high_column_name
        self.low_column_name = low_column_name

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_missing_columns(X, [self.high_column_name, self.low_column_name])

        parkinson_constant = 1 / (4 * np.log(2))

        return (
            (
                parkinson_constant
                * np.square(np.log(X[self.high_column_name] / X[self.low_column_name]))
            )
            .rolling(self.window)
            .mean()
        ).values.reshape(-1, 1)


def _get_beta(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Get beta estimate from Corwin-Schultz algorithm (p.285, Snippet 19.1).

    :param high: Series containing high prices
    :rtype high: pd.Series
    :param low: Series containing low prices
    :type low: pd.Series
    :param window: The window size of the estimator
    :rtype window: int
    :return: Beta estimates
    :rtype: pd.Series
    """
    ret = np.log(high / low)
    high_low_ret = ret ** 2
    beta = high_low_ret.rolling(window=2).sum()
    beta = beta.rolling(window=window).mean()
    return beta


def _get_gamma(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Get gamma estimate from Corwin-Schultz algorithm (p.285, Snippet 19.1).

    :param high: Series containing high prices
    :rtype high: pd.Series
    :param low: Series containing low prices
    :type low: pd.Series
    :return: Gamma estimates
    :rtype: pd.Series
    """
    high_max = high.rolling(window=2).max()
    low_min = low.rolling(window=2).min()
    gamma = np.log(high_max / low_min) ** 2
    return gamma


def _get_alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    """
    Get alpha from Corwin-_get_betaSchultz algorithm, (p.285, Snippet 19.1).

    :param high: Series containing high prices
    :rtype high: pd.Series
    :param low: Series containing low prices
    :type low: pd.Series
    :return: Alpha estimates
    :rtype: pd.Series
    """
    den = 3 - 2 * 2 ** 0.5
    alpha = (2 ** 0.5 - 1) * (beta ** 0.5) / den
    alpha -= (gamma / den) ** 0.5
    alpha[alpha < 0] = 0  # Set negative alphas to 0 (see p.727 of paper)
    return alpha


class CorwinSchultzSpread(BaseEstimator, TransformerMixin):
    """
    Corwin Schultz spread estimator.

    See page 284 of Advances in Financial Machine Learning by Marcos Lopez de
    Prado for additional information.

    :param window: The window size
    :type window: int
    :param high_column_name: The name of the high column
    :type high_column_name_get_beta: str, optional
    :param low_column_name: The name of the low column
    :type low_column_name: str, optional
    """

    def __init__(
        self,
        window: int = 20,
        high_column_name: str = static.HIGH,
        low_column_name: str = static.LOW,
    ):
        self.window = window
        self.high_column_name = high_column_name
        self.low_column_name = low_column_name

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:

        check_missing_columns(X, [self.high_column_name, self.low_column_name])

        beta = _get_beta(X[self.high_column_name], X[self.low_column_name], self.window)
        gamma = _get_gamma(X[self.high_column_name], X[self.low_column_name])
        alpha = _get_alpha(beta, gamma)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

        return spread.values.reshape(-1, 1)


class BeckersParkinsonVolatility(BaseEstimator, TransformerMixin):
    """
    Get Bekker-Parkinson volatility from gamma and beta in Corwin-Schultz algorithm, (p.286, Snippet 19.2).

    See page 284 of Advances in Financial Machine Learning by Marcos Lopez de
    Prado for additional information.

    :param window: The window size
    :type window: int
    :param high_column_name: The name of the high column
    :type high_column_name: str, optional
    :param low_column_name: The name of the low column
    :type low_column_name: str, optional
    """

    def __init__(
        self,
        window: int = 20,
        high_column_name: str = static.HIGH,
        low_column_name: str = static.LOW,
    ):
        self.window = window
        self.high_column_name = high_column_name
        self.low_column_name = low_column_name

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:

        check_missing_columns(X, [self.high_column_name, self.low_column_name])

        beta = _get_beta(X[self.high_column_name], X[self.low_column_name], self.window)
        gamma = _get_gamma(X[self.high_column_name], X[self.low_column_name])

        k2 = (8 / np.pi) ** 0.5
        den = 3 - 2 * 2 ** 0.5
        sigma = (2 ** -0.5 - 1) * beta ** 0.5 / (k2 * den)
        sigma += (gamma / (k2 ** 2 * den)) ** 0.5
        sigma[sigma < 0] = 0
        return sigma.values.reshape(-1, 1)
