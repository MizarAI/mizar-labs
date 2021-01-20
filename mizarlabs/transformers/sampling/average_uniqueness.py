import pandas as pd
from mizarlabs.model.bootstrapping import get_ind_matrix
from mizarlabs.model.bootstrapping import calc_average_uniqueness
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
        ind_mat_csc = get_ind_matrix(
            X.copy()[self._event_end_time_column_name],
            X.copy(),
            self._event_end_time_column_name,
        ).tocsc()

        average_uniqueness_array = calc_average_uniqueness(ind_mat_csc)
        average_uniqueness = pd.Series(average_uniqueness_array, index=X.index)

        return average_uniqueness
