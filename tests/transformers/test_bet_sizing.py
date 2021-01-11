import numpy as np
import pandas as pd
import pytest
from mizarlabs.transformers.trading.bet_sizing import avg_active_signals
from mizarlabs.transformers.trading.bet_sizing import BET_SIZE
from mizarlabs.transformers.trading.bet_sizing import BetSizingFromProbabilities
from mizarlabs.transformers.trading.bet_sizing import discretise_signal
from mizarlabs.static import EVENT_END_TIME
import datetime
from mizarlabs.transformers.trading.bet_sizing import BET_SIZE
from mizarlabs.transformers.trading.bet_sizing import PREDICTION
from mizarlabs.transformers.trading.bet_sizing import PROBABILITY
from scipy.stats import norm
from mizarlabs.static import SIDE


@pytest.mark.parametrize("step_size", [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
@pytest.mark.usefixtures("dataframe_for_bet_sizing_testing_2_classes")
def test_discrete_signal(step_size, dataframe_for_bet_sizing_testing_2_classes):
    """
    Tests the discrete_signal function.
    """
    # rounding because otherwise there are small differences compared to the
    # results coming from the function
    bet_size_discrete = np.round(
        np.array(
            [
                max(-1, min(1, m_i))
                for m_i in np.round(
                    dataframe_for_bet_sizing_testing_2_classes[BET_SIZE] / step_size,
                    0,
                )
                * step_size
            ]
        ),
        4,
    )

    test_bet_discrete = discretise_signal(
        signal=dataframe_for_bet_sizing_testing_2_classes[BET_SIZE],
        step_size=step_size,
    )

    assert (bet_size_discrete == test_bet_discrete).all()


@pytest.mark.usefixtures("dataframe_for_bet_sizing_testing_2_classes")
def test_avg_active_signals(dataframe_for_bet_sizing_testing_2_classes):
    """
    Tests the avg_active_signals function
    """
    # Calculation of the average active bets.
    t_p = set(
        dataframe_for_bet_sizing_testing_2_classes[EVENT_END_TIME].dropna().to_numpy()
    )
    t_p = t_p.union(dataframe_for_bet_sizing_testing_2_classes.index.to_numpy())
    t_p = list(t_p)
    t_p.sort()
    avg_list = []
    for t_i in t_p:
        avg_list.append(
            dataframe_for_bet_sizing_testing_2_classes[
                (dataframe_for_bet_sizing_testing_2_classes.index <= t_i)
                & (
                    (dataframe_for_bet_sizing_testing_2_classes[EVENT_END_TIME] > t_i)
                    | pd.isnull(
                        dataframe_for_bet_sizing_testing_2_classes[EVENT_END_TIME]
                    )
                )
            ][BET_SIZE].mean()
        )
    avg_active = pd.Series(data=np.array(avg_list), index=t_p).fillna(0)
    test_avg_active = avg_active_signals(
        dataframe_for_bet_sizing_testing_2_classes, EVENT_END_TIME
    )

    np.testing.assert_array_almost_equal(avg_active.values, test_avg_active.values)


@pytest.mark.usefixtures("dataframe_for_bet_sizing_testing_2_classes")
def test_bet_sizing_from_probabilities_2_classes(
    dataframe_for_bet_sizing_testing_2_classes,
):
    """Testing the BetsizingFromProbabilities with 2 classes"""
    bet_sizing_from_probabilities = BetSizingFromProbabilities(num_classes=2)
    bet_size = bet_sizing_from_probabilities.transform(
        dataframe_for_bet_sizing_testing_2_classes
    )

    assert (bet_size <= 1).all()
    assert (bet_size >= -1).all()

    bet_sizing_from_probabilities = BetSizingFromProbabilities(
        num_classes=2, average_active=True
    )
    bet_size = bet_sizing_from_probabilities.transform(
        dataframe_for_bet_sizing_testing_2_classes
    )

    assert (bet_size <= 1).all()
    assert (bet_size >= -1).all()

    bet_sizing_from_probabilities = BetSizingFromProbabilities(
        num_classes=2, average_active=True, discretise=True, step_size=0.2
    )
    bet_size = bet_sizing_from_probabilities.transform(
        dataframe_for_bet_sizing_testing_2_classes
    )
    assert (bet_size <= 1).all()
    assert (bet_size >= -1).all()
    assert (set(bet_size.unique())).issubset(set(np.round(np.arange(-1, 1.2, 0.2), 2)))

    bet_sizing_from_probabilities = BetSizingFromProbabilities(
        num_classes=2, meta_labeling=True
    )
    bet_size = bet_sizing_from_probabilities.transform(
        dataframe_for_bet_sizing_testing_2_classes
    )
    assert (bet_size <= 1).all()
    assert (bet_size >= -1).all()


@pytest.mark.usefixtures("dataframe_for_bet_sizing_testing_3_classes")
def test_bet_sizing_from_probabilities_3_classes(
    dataframe_for_bet_sizing_testing_3_classes,
):
    """Testing the BetsizingFromProbabilities with 3 classes"""
    bet_sizing_from_probabilities = BetSizingFromProbabilities(num_classes=3)
    bet_size = bet_sizing_from_probabilities.transform(
        dataframe_for_bet_sizing_testing_3_classes
    )

    assert (bet_size <= 1).all()
    assert (bet_size >= -1).all()

    bet_sizing_from_probabilities = BetSizingFromProbabilities(
        num_classes=3, average_active=True
    )
    bet_size = bet_sizing_from_probabilities.transform(
        dataframe_for_bet_sizing_testing_3_classes
    )

    assert (bet_size <= 1).all()
    assert (bet_size >= -1).all()

    bet_sizing_from_probabilities = BetSizingFromProbabilities(
        num_classes=3, average_active=True, discretise=True, step_size=0.2
    )
    bet_size = bet_sizing_from_probabilities.transform(
        dataframe_for_bet_sizing_testing_3_classes
    )
    assert (bet_size <= 1).all()
    assert (bet_size >= -1).all()
    assert (set(bet_size.unique())).issubset(set(np.round(np.arange(-1, 1.2, 0.2), 2)))

    bet_sizing_from_probabilities = BetSizingFromProbabilities(
        num_classes=3, meta_labeling=True
    )
    bet_size = bet_sizing_from_probabilities.transform(
        dataframe_for_bet_sizing_testing_3_classes
    )
    assert (bet_size <= 1).all()
    assert (bet_size >= -1).all()
