import numpy as np
import pandas as pd
import pytest
from mizarlabs.model.bootstrapping import _bootstrap_loop_run
from mizarlabs.model.bootstrapping import get_ind_matrix
from mizarlabs.model.bootstrapping import seq_bootstrap
from mizarlabs.static import CLOSE
from mizarlabs.transformers.targets.labeling import EVENT_END_TIME
from mizarlabs.transformers.targets.labeling import TripleBarrierMethodLabeling


@pytest.mark.usefixtures("dollar_bar_dataframe")
def test_ind_matrix(dollar_bar_dataframe: pd.DataFrame):
    """
    Tests whether the indicator matrix can be calculated.
    """
    triple_barrier = TripleBarrierMethodLabeling(
        num_expiration_bars=10, profit_taking_factor=0.2, stop_loss_factor=0.2
    )
    target_labels = triple_barrier.fit_transform(dollar_bar_dataframe[[CLOSE]])

    target_labels = target_labels.dropna()

    get_ind_matrix(target_labels[EVENT_END_TIME], dollar_bar_dataframe, EVENT_END_TIME)


@pytest.mark.usefixtures("ind_matrix")
def test_seq_boostrap(ind_matrix: np.array):
    """
    Check the shape of the indicator matrix.
    """
    bootstrapped_samples = seq_bootstrap(ind_matrix)

    assert len(bootstrapped_samples) == ind_matrix.shape[1]


@pytest.mark.usefixtures("ind_matrix")
def test_bootstrap_loop_run(ind_matrix: np.array):
    """
    Check the computed average uniqueness.
    """
    prev_concurrency = np.zeros(ind_matrix.shape[0])
    avg_uniqueness = np.ones(ind_matrix.shape[1])
    indices = np.arange(len(avg_uniqueness))

    avg_uniqueness = _bootstrap_loop_run(
        ind_matrix, prev_concurrency, avg_uniqueness, indices
    )

    np.testing.assert_array_equal(avg_uniqueness, np.array([1.0, 1.0, 1.0, 1.0]))

    prev_concurrency += ind_matrix[:, 0]

    avg_uniqueness = _bootstrap_loop_run(
        ind_matrix, prev_concurrency, avg_uniqueness, indices
    )
    assert avg_uniqueness[0] < avg_uniqueness[1]
    assert avg_uniqueness[0] < avg_uniqueness[2]
    assert avg_uniqueness[0] < avg_uniqueness[3]
    assert np.unique(avg_uniqueness).shape[0] == 2

    prev_concurrency += ind_matrix[:, 2]
    avg_uniqueness = _bootstrap_loop_run(
        ind_matrix, prev_concurrency, avg_uniqueness, indices
    )

    assert avg_uniqueness[0] < avg_uniqueness[1]
    assert avg_uniqueness[0] < avg_uniqueness[3]
    assert avg_uniqueness[2] < avg_uniqueness[1]
    assert avg_uniqueness[2] < avg_uniqueness[3]
    assert avg_uniqueness[2] == avg_uniqueness[0]
