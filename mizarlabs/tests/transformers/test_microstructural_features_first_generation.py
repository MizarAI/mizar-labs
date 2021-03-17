import random

import numpy as np
import pandas as pd
import pytest
from mizarlabs import static
from mizarlabs.transformers.microstructural_features.first_generation import (
    CorwinSchultzSpread,
)
from mizarlabs.transformers.microstructural_features.first_generation import (
    ParkinsonVolatility,
)
from mizarlabs.transformers.microstructural_features.first_generation import RollImpact
from mizarlabs.transformers.microstructural_features.first_generation import RollMeasure


@pytest.mark.parametrize("window", random.choices(range(10, 40), k=3))
@pytest.mark.usefixtures("dollar_bar_dataframe")
def test_corwin_schultz(dollar_bar_dataframe: pd.DataFrame, window: int):
    """
    Compares corwin shultz transformer output vs. manual calculation.
    """
    corwin_schultz_transformer = CorwinSchultzSpread(window)
    corwin_schultz_transformer_values = corwin_schultz_transformer.transform(
        dollar_bar_dataframe[[static.HIGH, static.LOW]].astype(float)
    )
    assert np.isnan(corwin_schultz_transformer_values).sum() == window
    assert (
        np.isnan(
            corwin_schultz_transformer_values[
                pd.Series(
                    corwin_schultz_transformer_values.flatten()
                ).first_valid_index() :
            ]
        ).sum()
        == 0
    )
    assert (
        corwin_schultz_transformer_values[
            pd.Series(corwin_schultz_transformer_values.flatten()).first_valid_index() :
        ].shape[0]
        == corwin_schultz_transformer_values.shape[0] - window
    )

    # manual calculation of the value with index equal to window
    high = dollar_bar_dataframe[static.HIGH].iloc[: window + 1].astype(float)
    low = dollar_bar_dataframe[static.LOW].iloc[: window + 1].astype(float)
    roll_max = max(high[-2:])
    roll_min = min(low[-2:])
    gamma = np.log(roll_max / roll_min) ** 2
    beta_intermediate = np.log(high / low) ** 2
    beta = np.mean(
        [i + j for i, j in zip(beta_intermediate.shift(1), beta_intermediate)][1:]
    )
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 ** 1.5) - np.sqrt(
        gamma / (3 - 2 ** 1.5)
    )
    alpha = max(0, alpha)
    corwin_schultz_manual = 2 * (np.e ** alpha - 1) / (1 + np.e ** alpha)
    assert np.round(corwin_schultz_transformer_values[window], 8) == np.round(
        corwin_schultz_manual, 8
    )


@pytest.mark.parametrize("index_to_check", random.choices(range(120, 300), k=10))
@pytest.mark.parametrize("window", random.choices(range(10, 40), k=3))
@pytest.mark.usefixtures("dollar_bar_dataframe")
def test_parkinson_volatility(
    dollar_bar_dataframe: pd.DataFrame, index_to_check: int, window: int
):
    """
    Checks if output of transformer is similar to manual calculation.
    """
    parkison_volatility_transformer = ParkinsonVolatility(window)
    parkinson_constant = 1 / (4 * np.log(2))
    high = dollar_bar_dataframe[index_to_check + 1 - window : index_to_check + 1][
        static.HIGH
    ].astype(float)
    low = dollar_bar_dataframe[index_to_check + 1 - window : index_to_check + 1][
        static.LOW
    ].astype(float)

    parkinson_volatility = float(
        np.sum(np.square(np.log(high / low)) * parkinson_constant / window)
    )

    parkinson_vol_transformer_value = parkison_volatility_transformer.transform(
        dollar_bar_dataframe[[static.HIGH, static.LOW]].astype(float)
    )[index_to_check]

    assert np.round(parkinson_volatility, 8) == np.round(
        parkinson_vol_transformer_value, 8
    )


@pytest.mark.parametrize("index_to_check", random.choices(range(120, 300), k=10))
@pytest.mark.parametrize("window", random.choices(range(10, 40), k=3))
@pytest.mark.usefixtures("dollar_bar_dataframe")
def test_roll_impact(
    dollar_bar_dataframe: pd.DataFrame, index_to_check: int, window: int
):
    """
    Checks if output of transformer is similar to manual calculation.
    """
    roll_impact = RollImpact(window=window)
    close_diff = dollar_bar_dataframe[
        index_to_check - (window + 1) : index_to_check + 1
    ][static.CLOSE].diff()[1:]
    close_diff_lag = close_diff.shift(1)[1:]

    dollar_volume = float(
        dollar_bar_dataframe.iloc[index_to_check][static.QUOTE_ASSET_VOLUME]
    )

    roll_impact_numpy = (
        2
        * np.sqrt(
            abs(
                np.cov(
                    [
                        close_diff[1:].astype(float).values,
                        close_diff_lag.astype(float).values,
                    ]
                )[0][1]
            )
        )
        / dollar_volume
    )

    roll_impact_pandas = (
        2
        * np.sqrt(
            np.abs(close_diff.astype(float)[1:].cov(close_diff_lag.astype(float)))
        )
        / dollar_volume
    )

    roll_impact_transformer = roll_impact.transform(
        dollar_bar_dataframe[[static.CLOSE, static.QUOTE_ASSET_VOLUME]].astype(float)
    )[index_to_check]

    assert np.round(roll_impact_numpy, 8) == np.round(roll_impact_transformer, 8)
    assert np.round(roll_impact_pandas, 8) == np.round(roll_impact_transformer, 8)


@pytest.mark.parametrize("index_to_check", random.choices(range(120, 300), k=10))
@pytest.mark.parametrize("window", random.choices(range(10, 40), k=3))
@pytest.mark.usefixtures("dollar_bar_dataframe")
def test_roll_measures(
    dollar_bar_dataframe: pd.DataFrame, index_to_check: int, window: int
):
    """
    Checks if output of transformer is similar to manual calculation.
    """
    roll_measure = RollMeasure(window)
    close_diff = dollar_bar_dataframe[
        index_to_check - (window + 1) : index_to_check + 1
    ][static.CLOSE].diff()[1:]
    close_diff_lag = close_diff.shift(1)[1:]

    roll_measure_numpy = 2 * np.sqrt(
        abs(
            np.cov(
                [
                    close_diff[1:].astype(float).values,
                    close_diff_lag.astype(float).values,
                ]
            )[0][1]
        )
    )

    roll_measure_pandas = 2 * np.sqrt(
        np.abs(close_diff.astype(float)[1:].cov(close_diff_lag.astype(float)))
    )

    roll_measure_transformer = roll_measure.transform(
        dollar_bar_dataframe[[static.CLOSE, static.QUOTE_ASSET_VOLUME]].astype(float)
    )[index_to_check]

    assert np.round(roll_measure_numpy, 8) == np.round(roll_measure_transformer, 8)
    assert np.round(roll_measure_pandas, 8) == np.round(roll_measure_transformer, 8)
