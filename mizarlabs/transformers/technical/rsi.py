import numpy as np
import pandas as pd

from mizarlabs.transformers.technical.factory import TAFactory


class BarArrivalRSIStrategy:
    """
    This strategy is based on the bar arrival and so it makes sense to
    be used only on dynamic bars (dollar, volume, tick)

    This strategy only considers long positions
    """

    classes_ = [0.0, 1.0]
    n_classes_ = 2

    def __init__(
        self,
        rsi_upper_threshold: float,
        rsi_lower_threshold: float,
        bar_arrival_upper_threshold: float,
        bar_arrival_lower_threshold: float,
        rsi_timeperiod: int,
        bar_arrival_fast_period: int,
        bar_arrival_slow_period: int,
        max_bar_arrival_mean_diff: int,
    ):
        self.rsi_high = rsi_upper_threshold
        self.rsi_low = rsi_lower_threshold
        self.rsi_timeperiod = rsi_timeperiod
        self.bar_arrivals_high = bar_arrival_upper_threshold
        self.bar_arrivals_low = bar_arrival_lower_threshold

        self.rsi = TAFactory().create_transformer(
            "RSI", "close", kw_args={"timeperiod": self.rsi_timeperiod}
        )
        self.bar_arrival_fast_period = bar_arrival_fast_period
        self.bar_arrival_slow_period = bar_arrival_slow_period
        self.max_bar_arrival_mean_diff = max_bar_arrival_mean_diff

        self.fitted_ = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        bar_arrival_time = X.index.to_series().diff().dropna().astype(int)
        bar_arrival_time_rolling_mean_slow = (
            bar_arrival_time.dropna()
            .astype(int)
            .rolling(window=self.bar_arrival_slow_period)
            .mean()
        )
        bar_arrival_time_rolling_mean_fast = (
            bar_arrival_time.dropna()
            .astype(int)
            .rolling(window=self.bar_arrival_fast_period)
            .mean()
        )
        normalised_bar_arrival_rolling_mean_diff = (
            bar_arrival_time_rolling_mean_fast - bar_arrival_time_rolling_mean_slow
        ) / self.max_bar_arrival_mean_diff

        rsi = pd.Series(
            self.rsi.transform(X).flatten(),
            index=X.index,
            name="rsi",
        )

        buy_signal = (
            (rsi > self.rsi_low)
            & (rsi < self.rsi_high)
            & (normalised_bar_arrival_rolling_mean_diff < self.bar_arrivals_high)
            & (normalised_bar_arrival_rolling_mean_diff > self.bar_arrivals_low)
        )

        return buy_signal.astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        predictions = self.predict(X)
        probabilities = np.zeros(shape=(len(predictions), len(self.classes_)))
        for i, class_value in enumerate(self.classes_):
            probabilities[:, i] = predictions == class_value
        return probabilities
