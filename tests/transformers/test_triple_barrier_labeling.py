import numpy as np
import pandas as pd
import pytest
from mizarlabs.static import CLOSE
from mizarlabs.static import LABEL
from mizarlabs.static import SIDE
from mizarlabs.transformers.targets.labeling import TripleBarrierMethodLabeling


@pytest.mark.parametrize("spike_occurence_days", [5, 15, 30, 60])
@pytest.mark.parametrize("num_expiration_bars", [3, 7, 14, 30])
@pytest.mark.parametrize("frequency", ["d", "6h"])
@pytest.mark.parametrize("factor", [0.5, -0.5])
def test_labeling_with_syntetic_bullish_spike(
    spike_occurence_days, num_expiration_bars, frequency, factor
):
    triple_barrier = TripleBarrierMethodLabeling(
        num_expiration_bars=num_expiration_bars,
        profit_taking_factor=1,
        stop_loss_factor=1,
    )
    index = pd.date_range(start="2018-01-01", end="2018-05-01", freq=frequency)

    price = [10 for _ in range(len(index))]
    spike_occurence_series = pd.Series(price, index=index)
    spike_date = index[0] + pd.Timedelta(days=spike_occurence_days)
    spike_occurence_series.loc[spike_date] = spike_occurence_series.loc[spike_date] * (
        1 + factor
    )

    spike_occurence_df = spike_occurence_series.to_frame(CLOSE)
    spike_occurence_labels = triple_barrier.fit_transform(spike_occurence_df)

    spike_occurence_labels = spike_occurence_labels.iloc[:-num_expiration_bars]
    first_not_nan_value_index = (
        sum(spike_occurence_labels.index.date == spike_occurence_labels.index[0].date())
        + 2
    )  # the 2 is needed to allow the calculation of the ewm stdev

    spike_label = np.sign(factor)

    assert (
        spike_occurence_labels.iloc[first_not_nan_value_index].name
        == spike_occurence_labels.first_valid_index()
    ), "The first valid index does not match with first non nan index"

    assert (
        spike_occurence_labels.iloc[:first_not_nan_value_index].isna().all().all()
    ), "First part of the dataframe should not be a not number"

    spike_occurence_labels.dropna(inplace=True)

    num_bars_from_start_to_spike = spike_occurence_labels.loc[
        spike_occurence_labels.index < spike_date
    ].shape[0]
    index_mask_before_spike = spike_occurence_labels.index < spike_date
    expected_num_spike_sign_bars = min(
        num_expiration_bars, num_bars_from_start_to_spike
    )
    expected_num_neutral_bars = spike_occurence_labels.shape[0] - (
        min(num_expiration_bars, num_bars_from_start_to_spike) + 1
    )

    assert (
        spike_occurence_labels.loc[spike_date][LABEL] == -spike_label
    ), f"On the spike date the barrier class should be {-spike_label}"

    assert (
        spike_occurence_labels.loc[spike_occurence_labels[LABEL] == -spike_label].shape[
            0
        ]
        == 1
    ), f"Only one bar should be labels with {-spike_label}"

    assert (
        sum(spike_occurence_labels[LABEL] == spike_label)
        == expected_num_spike_sign_bars
    ), f"The number of bars labeled with {spike_label} should be equal to {expected_num_spike_sign_bars}"

    assert (
        spike_occurence_labels.loc[spike_occurence_labels[LABEL] == 0].shape[0]
        == expected_num_neutral_bars
    ), f"The number of bars with label 0 should be equal to {expected_num_neutral_bars}"

    assert (
        spike_occurence_labels.loc[index_mask_before_spike].iloc[
            -expected_num_spike_sign_bars:
        ][LABEL]
        == spike_label
    ).all(), f"{expected_num_spike_sign_bars} before the spike are expected to be labeled with {spike_label}"


@pytest.mark.parametrize("spike_occurence_days", [30, 60])
@pytest.mark.parametrize("num_expiration_bars", [3, 7, 14])
@pytest.mark.parametrize("frequency", ["d", "6h"])
@pytest.mark.parametrize("factor", [0.5, -0.5])
def test_triple_barrier_metalabeling(
    num_expiration_bars, spike_occurence_days, frequency, factor
):
    """
    Checking if the metalabels are correct when base model is
    always correct and when it is always wrong
    """
    triple_barrier = TripleBarrierMethodLabeling(
        num_expiration_bars=num_expiration_bars,
        profit_taking_factor=1,
        stop_loss_factor=1,
        metalabeling=True,
    )
    index = pd.date_range(start="2018-01-01", end="2018-05-01", freq=frequency)

    price = [10 for _ in range(len(index))]
    spike_occurence_series = pd.Series(price, index=index)
    spike_date = index[0] + pd.Timedelta(days=spike_occurence_days)
    spike_occurence_series.loc[spike_date] = spike_occurence_series.loc[spike_date] * (
        1 + factor
    )
    spike_occurence_bars = np.where(spike_occurence_series.index == spike_date)[0][0]

    spike_occurence_df = spike_occurence_series.to_frame(CLOSE)
    # always right base model so the results from metalabeling should
    # be always equal to 1
    spike_occurence_df[SIDE] = 0
    spike_occurence_df.loc[
        spike_occurence_df.index[
            spike_occurence_bars - num_expiration_bars : spike_occurence_bars
        ],
        SIDE,
    ] = np.sign(factor)
    spike_occurence_df.loc[:, SIDE].iloc[spike_occurence_bars] = -np.sign(factor)
    spike_occurence_labels = triple_barrier.fit_transform(spike_occurence_df)

    assert (
        spike_occurence_labels[LABEL].iloc[
            spike_occurence_bars - num_expiration_bars : spike_occurence_bars + 1
        ]
        == 1
    ).all(), "The metalabels should always be equal to 1 (correct base model) where the side is specified"

    assert spike_occurence_labels[LABEL].isna().sum() == spike_occurence_labels[
        LABEL
    ].shape[0] - (
        num_expiration_bars + 1
    ), "The metalabels should be nan Where the side is not speficied"

    # always wrong base model so the results from metalabeling should be
    # always equal to 0
    spike_occurence_df.loc[:, SIDE] = 0
    spike_occurence_df.loc[
        spike_occurence_df.index[
            spike_occurence_bars - num_expiration_bars : spike_occurence_bars
        ],
        SIDE,
    ] = -np.sign(factor)
    spike_occurence_df.loc[:, SIDE].iloc[spike_occurence_bars] = np.sign(factor)
    spike_occurence_labels = triple_barrier.fit_transform(spike_occurence_df)

    assert (
        spike_occurence_labels[LABEL].iloc[
            spike_occurence_bars - num_expiration_bars : spike_occurence_bars + 1
        ]
        == 0
    ).all(), "The metalabels should always be equal to 0 (wrong base model) where the side is specified"

    assert spike_occurence_labels[LABEL].isna().sum() == spike_occurence_labels[
        LABEL
    ].shape[0] - (
        num_expiration_bars + 1
    ), "The metalabels should be nan Where the side is not speficied"
