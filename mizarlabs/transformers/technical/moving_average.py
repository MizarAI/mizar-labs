import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class MovingAverageCrossOver(BaseEstimator, TransformerMixin):
    """
    Moving average crossover transformer

    It transform the input data to a 1 when the the fast moving average goes
    above the slow moving average and -1 when the slow moving average goes
    above the fast moving average. When fill_between_crossovers is true, then
    the transfromation will be equal to 1 when the fast moving average is
    above the slow one and -1 viceversa.

    The transformer is specified by the fast and slow moving average
    number of bars.

    :param fast: number of bars to use for the moving average of the fast
                 moving average
    :type fast: int
    :param slow: number of bars to use for the moving average of the slow
                  moving average
    :type slow: int
    :param column_name: The name of the column in input that will be used for the
                        transformation.
    :type column_name: str
    :param fill_between_crossovers: whether or not to fill the crossover value untill
                                    the next crossover.
    :type fill_between_crossovers: bool
    """

    def __init__(
        self,
        fast: int,
        slow: int,
        column_name: str,
        fill_between_crossovers: bool = False,
    ):

        if fast > slow:
            raise ValueError("Fast should be smaller than slow")
        self.fast_ = fast
        self.slow_ = slow
        self.column_name = column_name
        self.fill_between_crossovers = fill_between_crossovers

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame):
        fast_moving_average = X[self.column_name].rolling(window=self.fast_).mean()
        slow_moving_average = X[self.column_name].rolling(window=self.slow_).mean()

        # filling the zeros with nan so that we can forward fill the nans
        # with the previous crossover value
        if self.fill_between_crossovers:
            side = pd.Series(np.nan, index=X.index)
        else:
            side = pd.Series(0, index=X.index)

        side.loc[
            self._get_down_cross_indices(
                fast_ma=fast_moving_average, slow_ma=slow_moving_average
            )
        ] = -1
        side.loc[
            self._get_up_cross_indices(
                fast_ma=fast_moving_average, slow_ma=slow_moving_average
            )
        ] = 1

        # forward filling nans with previous crossover values
        side.ffill(inplace=True)
        # filling nans at the top of the series with zeros
        side.fillna(0, inplace=True)

        return side.values.reshape(-1, 1)

    @staticmethod
    def _get_up_cross_indices(
        fast_ma: pd.Series, slow_ma: pd.Series
    ) -> pd.DatetimeIndex:
        # previous value is smaller or equal
        cond_0 = fast_ma.shift(1) <= slow_ma.shift(1)
        # current value is greater
        cond_1 = fast_ma > slow_ma
        return fast_ma[(cond_0) & (cond_1)].index

    @staticmethod
    def _get_down_cross_indices(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
        # previous value is greater or equal
        cond_0 = fast_ma.shift(1) >= slow_ma.shift(1)
        # current value is smaller
        cond_1 = fast_ma < slow_ma
        return fast_ma[(cond_0) & (cond_1)].index


class MovingAverageCrossOverPredictor(MovingAverageCrossOver):
    classes_ = [-1.0, 0.0, 1.0]
    n_classes_ = 3

    def predict(self, X: pd.DataFrame):
        return self.transform(X).flatten()

    def predict_proba(self, X: pd.DataFrame):
        predictions = self.predict(X)
        probabilities = np.zeros(shape=(len(predictions), len(self.classes_)))
        for i, class_value in enumerate(self.classes_):
            probabilities[:, i] = predictions == class_value
        return probabilities


class ExponentialWeightedMovingAverageDifference:
    def __init__(
        self,
        fast: int,
        slow: int,
        column_name: str,
        normalised: bool = True,
    ):

        if fast > slow:
            raise ValueError("Fast should be smaller than slow")
        self.fast_ = fast
        self.slow_ = slow
        self.normalised = normalised
        self.column_name = column_name

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        fast_ewma = X[self.column_name].ewm(span=self.fast_).mean()
        slow_ewma = X[self.column_name].ewm(span=self.slow_).mean()

        ewma_difference = fast_ewma - slow_ewma

        if self.normalised:
            return (ewma_difference / (fast_ewma + 1)).values.reshape(
                -1, 1
            )  # plus one to avoid zero division error
        else:
            return ewma_difference.values.reshape(-1, 1)
