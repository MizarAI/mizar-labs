import numpy as np
import pandas as pd
import pytest
from mizarlabs.model.bootstrapping import _calc_update_avg_unique
from mizarlabs.model.bootstrapping import get_ind_matrix
from mizarlabs.model.bootstrapping import seq_bootstrap
from mizarlabs.static import CLOSE
from mizarlabs.transformers.targets.labeling import EVENT_END_TIME
from mizarlabs.transformers.targets.labeling import TripleBarrierMethodLabeling
from scipy import sparse


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


@pytest.mark.usefixtures("ind_matrix_csc")
def test_seq_boostrap(ind_matrix: sparse.csc_matrix):
    """
    Check the shape of the indicator matrix.
    """
    bootstrapped_samples = seq_bootstrap(ind_matrix)

    assert len(bootstrapped_samples) == ind_matrix.shape[1]


@pytest.mark.usefixtures("ind_matrix_csc")
def test_calc_average_uniqueness(ind_matrix: sparse.csc_matrix):
    """
    Check the computed average uniqueness.
    """
    samples_to_update = np.array([0, 2])
    bootstrapped_samples = np.array([0])
    avg_uniqueness = _calc_update_avg_unique(
        ind_matrix, samples_to_update, bootstrapped_samples,
    )
    assert avg_uniqueness.shape[0] == samples_to_update.shape[0]
    assert avg_uniqueness[1] == 1
    assert avg_uniqueness[0] < 1

    samples_to_update = np.array([0, 2])
    bootstrapped_samples = np.array([0, 0])
    avg_uniqueness_next = _calc_update_avg_unique(
        ind_matrix, samples_to_update, bootstrapped_samples,
    )
    assert avg_uniqueness_next[0] < avg_uniqueness[0]
    assert avg_uniqueness_next[1] == avg_uniqueness[1]
