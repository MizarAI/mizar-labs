import random

import numpy as np
import pandas as pd
import pytest
from mizarlabs.static import CLOSE
from mizarlabs.transformers.sample_weights import SampleWeightsByReturns
from mizarlabs.transformers.sample_weights import SampleWeightsByTimeDecay
from mizarlabs.transformers.targets.labeling import EVENT_END_TIME


@pytest.mark.usefixtures(
    "dollar_bar_target_labels", "dollar_bar_dataframe", "dollar_bar_ind_matrix"
)
@pytest.mark.parametrize("index_to_check", random.choices(range(0, 200), k=10))
def test_sample_weights_by_returns(
    dollar_bar_dataframe,
    dollar_bar_target_labels,
    dollar_bar_ind_matrix,
    dollar_bar_ind_matrix_indices,
    index_to_check,
):
    """Test sample weights by returns calculated values are correct"""

    df = dollar_bar_dataframe.merge(
        dollar_bar_target_labels, left_index=True, right_index=True
    )
    sample_weights_transformer = SampleWeightsByReturns()
    sample_weights = sample_weights_transformer.transform(df)

    assert (sample_weights).all() >= 0

    returns = np.log(df[CLOSE]).diff()

    num_concurrent_events = pd.Series(
        dollar_bar_ind_matrix.tocsc().sum(axis=1).A1,
        index=dollar_bar_ind_matrix_indices
    ).loc[df.index]

    start_time = df.index[index_to_check]
    end_time = df[EVENT_END_TIME].iloc[index_to_check]
    weight = abs(
        (
            returns.loc[start_time:end_time]
            / num_concurrent_events.loc[start_time:end_time]
        ).sum()
    )

    assert sample_weights[index_to_check] == weight


@pytest.mark.usefixtures(
    "dollar_bar_target_labels",
)
def test_sample_weights_by_time_decay(dollar_bar_target_labels):
    time_decay_weights = SampleWeightsByTimeDecay(minimum_decay_weight=1).transform(
        dollar_bar_target_labels
    )

    assert set(time_decay_weights.values) == {
        1
    }, "Time decay weights should be equal to 1 when the minimum is set to be 1"

    time_decay_weights = SampleWeightsByTimeDecay(minimum_decay_weight=0.5).transform(
        dollar_bar_target_labels
    )

    assert (
        time_decay_weights.values >= 0
    ).all(), "Time decay weights should be greater or equal to 0"

    assert (
        time_decay_weights.is_monotonic_increasing
    ), "Time decay weights should be monotonic increasing"

    time_decay_smaller_weights = SampleWeightsByTimeDecay(
        minimum_decay_weight=0.1
    ).transform(dollar_bar_target_labels)

    assert (time_decay_weights[:-1] > time_decay_smaller_weights[:-1]).all(), (
        "The time dacay weights with smaller minimum should have"
        " smaller values compared to the time decay weights with larger minimum"
    )

    time_decay_weights = SampleWeightsByTimeDecay(
        minimum_decay_weight=-0.001
    ).transform(dollar_bar_target_labels)

    assert round(time_decay_weights[0], 2) == 0, "The oldest element should be 0"
