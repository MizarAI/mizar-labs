import numpy as np
import pandas as pd
from mizarlabs.model.bootstrapping import get_ind_matrix
from mizarlabs.static import EVENT_END_TIME
from mizarlabs.transformers.utils import check_missing_columns
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class AverageUniqueness(BaseEstimator, TransformerMixin):
    """
    Calculates the average uniqueness of samples.
    """

    def __init__(self, event_end_time_column_name: str = EVENT_END_TIME):
        self._event_end_time_column_name = event_end_time_column_name

    def transform(self, X: pd.DataFrame) -> pd.Series:
        check_missing_columns(X, [self._event_end_time_column_name])

        assert ~X[self._event_end_time_column_name].isna().any(), (
            "The expiration barrier should always have a value but "
            f" the indices {', '.join([str(index) for index in X[X[self._event_end_time_column_name].isna()].index])} contains NaNs"
        )
        assert X.index.is_unique, (
            f"Index should be unique but indices"
            f" {', '.join([str(index) for index in X[X.index.duplicated(keep='last')][:3].index])} are duplicated"
        )
        ind_mat = get_ind_matrix(
            X.copy()[self._event_end_time_column_name],
            X.copy(),
            self._event_end_time_column_name,
        )

        concurrent_events = ind_mat.sum(axis=1)

        uniqueness_matrix = ind_mat / concurrent_events.reshape(-1, 1)
        uniqueness_matrix[uniqueness_matrix == 0] = None
        average_uniqueness_array = np.nanmean(uniqueness_matrix, axis=0)

        average_uniqueness = pd.Series(average_uniqueness_array, index=X.index)

        return average_uniqueness
