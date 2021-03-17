import random

import numpy as np
import pandas as pd
import pytest
from mizarlabs import static
from mizarlabs.transformers.microstructural_features.second_generation import (
    AmihudLambda,
)
from mizarlabs.transformers.microstructural_features.second_generation import (
    HasbrouckLambda,
)
from mizarlabs.transformers.microstructural_features.second_generation import KyleLambda


@pytest.mark.parametrize("window", random.choices(range(10, 40), k=3))
@pytest.mark.usefixtures("dollar_bar_dataframe")
def test_kyle_lambda(dollar_bar_dataframe: pd.DataFrame, window: int):
    """
    Checks the following:
        1) if the no. of NaNs is as expected, i.e. equal to the window.
        2) if all values after the warming up period are computed.
        3) if all values are positive, as Kyle's Lambda is always positive.
    """
    kyle_lambda_transformer = KyleLambda(window)
    kyle_lambda_transformer_values = kyle_lambda_transformer.transform(
        dollar_bar_dataframe[[static.CLOSE, static.BASE_ASSET_VOLUME]].astype(float)
    )

    assert np.isnan(kyle_lambda_transformer_values).sum() == window
    assert np.isnan(kyle_lambda_transformer_values[:window]).all()
    assert all(kyle_lambda_transformer_values[window:] > 0)


@pytest.mark.parametrize("window", random.choices(range(10, 40), k=3))
@pytest.mark.usefixtures("dollar_bar_dataframe")
def test_amihud_lambda(dollar_bar_dataframe: pd.DataFrame, window: int):
    """
    Checks the following:
        1) if the no. of NaNs is as expected, i.e. equal to the window.
        2) if all values after the warming up period are computed.
        3) if all values are positive, as Kyle's Lambda is always positive.
    """
    amihud_lambda_transformer = AmihudLambda(window)
    amihud_lambda_transformer_values = amihud_lambda_transformer.transform(
        dollar_bar_dataframe[[static.CLOSE, static.QUOTE_ASSET_VOLUME]].astype(float)
    )

    assert np.isnan(amihud_lambda_transformer_values).sum() == window
    assert np.isnan(amihud_lambda_transformer_values[:window]).all()
    assert all(amihud_lambda_transformer_values[window:] > 0)


@pytest.mark.parametrize("window", random.choices(range(10, 40), k=3))
@pytest.mark.usefixtures("dollar_bar_dataframe")
def test_hasbrouck_lambda(dollar_bar_dataframe: pd.DataFrame, window: int):
    """
    Checks the following:
        1) if the no. of NaNs is as expected, i.e. equal to the window.
        2) if all values after the warming up period are computed.
        3) if all values are positive, as Kyle's Lambda is always positive.
    """
    hasbrouck_lambda_transformer = HasbrouckLambda(window)
    hasbrouck_lambda_transformer_values = hasbrouck_lambda_transformer.transform(
        dollar_bar_dataframe[[static.CLOSE, static.QUOTE_ASSET_VOLUME]].astype(float)
    )

    assert np.isnan(hasbrouck_lambda_transformer_values).sum() == window
    assert np.isnan(hasbrouck_lambda_transformer_values[:window]).all()
    assert all(hasbrouck_lambda_transformer_values[window:] > 0)
