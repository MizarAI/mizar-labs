import numpy as np
import pandas as pd

from mizarlabs.transformers.technical.factory import TAFactory


class MeanReversionStrategy:
    classes_ = [-1.0, 0.0, 1.0]
    n_classes_ = 3

    def __init__(self, threshold: float, short_time_period: int):
        self.threshold = threshold
        self.short_time_period = short_time_period
        self.short_ema_transformer = TAFactory().create_transformer(
            "EMA", "close", kw_args={"timeperiod": short_time_period}
        )
        self.fitted_ = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        short_ema = pd.Series(
            self.short_ema_transformer.transform(X).flatten(), index=X.index
        )

        diff_close_short_ema = (X["close"] - short_ema) / short_ema
        signal = pd.Series(0, index=X.index)
        signal.loc[(diff_close_short_ema > self.threshold)] = -1
        signal.loc[(diff_close_short_ema < -self.threshold)] = 1
        return signal.values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        predictions = self.predict(X)
        probabilities = np.zeros(shape=(len(predictions), len(self.classes_)))
        for i, class_value in enumerate(self.classes_):
            probabilities[:, i] = predictions == class_value
        return probabilities
