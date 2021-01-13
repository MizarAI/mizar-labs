import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin



class SideFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        side: str,
        primary_model: BaseEstimator,
    ):
        self.side = side
        self.primary_model = primary_model

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame):
        pass
