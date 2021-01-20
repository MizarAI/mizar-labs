import numpy as np
import pandas as pd
from scipy import sparse
from typing import Union, Tuple


def get_ind_matrix(
    samples_info_sets: pd.Series,
    price_bars: pd.DataFrame,
    event_end_time_column_name: str,
    return_indices: bool = False,
) -> Union[sparse.lil_matrix, Tuple[sparse.lil_matrix, pd.Timestamp]]:
    """
    Snippet 4.3, page 65, Build an Indicator Matrix
    Get indicator matrix. The book implementation uses bar_index as input,
    however there is no explanation how to form it. We decided that
    using triple_barrier_events and price bars by analogy with concurrency
    is the best option.

    :param samples_info_sets: Series indicating the start and end time of a sample, e.g. from triple barrier
    :type samples_info_sets: pd.Series
    :param price_bars: Price bars which were used to form triple barrier events or other labelling method
    :type price_bars: pd.DataFrame
    :param event_end_time_column_name: Column name
    :type event_end_time_column_name: str
    :return: Indicator binary matrix indicating what (price) bars influence the label for each observation and in
             addition in can also return the respective timestamp indices
    :rtype: Union[sparse.lil_matrix, Tuple[sparse.lil_matrix, pd.Timestamp]]
    """
    if (
        bool(samples_info_sets.isnull().values.any()) is True
        or bool(samples_info_sets.index.isnull().any()) is True
    ):
        raise ValueError("NaN values in triple_barrier_events, delete nans")

    triple_barrier_events = pd.DataFrame(
        samples_info_sets
    )  # Convert Series to DataFrame
    # Take only period covered in triple_barrier_events
    trimmed_price_bars_index = price_bars[
        (price_bars.index >= triple_barrier_events.index.min())
        & (price_bars.index <= triple_barrier_events[event_end_time_column_name].max())
    ].index

    label_endtime = triple_barrier_events[event_end_time_column_name]
    bar_index = sorted(
        list(
            {
                *triple_barrier_events.index,
                *triple_barrier_events[event_end_time_column_name],
                *trimmed_price_bars_index,
            }
        )
    )  # Drop duplicates and sort

    # Get sorted timestamps with index in sorted array
    sorted_timestamps = dict(zip(sorted(bar_index), range(len(bar_index))))

    tokenized_endtimes = np.column_stack(
        (
            label_endtime.index.map(sorted_timestamps),
            label_endtime.map(sorted_timestamps).values,
        )
    )  # Create array of arrays: [label_index_position, label_endtime_position]
    ind_mat = sparse.lil_matrix(
        (len(bar_index), len(label_endtime))
    )  # Init indicator matrix
    for sample_num, (label_index, label_endtime_index) in enumerate(tokenized_endtimes):
        ind_mat[label_index : label_endtime_index + 1, sample_num] = 1

    ind_mat_csc = ind_mat.tocsc()
    assert np.max(ind_mat_csc) == 1
    assert np.min(ind_mat_csc) == 0

    if return_indices:
        return ind_mat, bar_index
    else:
        return ind_mat


def calc_average_uniqueness(ind_mat_csc: sparse.csc_matrix) -> np.ndarray:
    """
    Calculates the average uniqueness of an indicator matrix.

    :param ind_mat_csc: indicator matrix, with size (T x N), where T is no. of timestamps and N the number of samples.
    :type ind_mat_csc: sparse.csc_matrix
    :return: array with average uniqueness per column in the indicator matrix, where a column represents a sample
    :rtype: np.ndarray
    """
    concurrent_events = ind_mat_csc.sum(axis=1)
    uniqueness_matrix = ind_mat_csc.multiply(1 / concurrent_events)
    counts = np.diff(uniqueness_matrix.tocsc().indptr)
    sums = uniqueness_matrix.tocsr().sum(axis=0).A1
    average_uniqueness_array = sums / counts
    return average_uniqueness_array


def _calc_update_avg_unique(
    ind_mat: sparse.csc_matrix,
    samples_to_update: np.ndarray,
    bootstrapped_samples: np.ndarray,
) -> np.ndarray:
    """
    Calculates the average uniqueness for a subset of samples if they were to be added to the currently bootstrap
    samples.

    :param ind_mat: indicator matrix, with size (T x N), where T is no. of timestamps and N the number of samples.
    :type ind_mat: sparse.csc_matrix
    :param samples_to_update: an arrray with indices for the indicator matrix to slice from
    :type samples_to_update: np.ndarray
    :param bootstrapped_samples: an arrray with indices representing the all the bootstrapped samples
    :type bootstrapped_samples: np.ndarray
    :return: array with average uniqueness per sample to update
    :rtype: np.ndarray
    """
    ind_mat_samples = ind_mat[:, samples_to_update]
    partial_conc_events = ind_mat[:, bootstrapped_samples].sum(axis=1)
    non_zero_partial_conc_events_idx = partial_conc_events.nonzero()[0]
    conc_events_per_sample = ind_mat_samples.todense()
    conc_events_per_sample[non_zero_partial_conc_events_idx, :] = (
        conc_events_per_sample[non_zero_partial_conc_events_idx, :]
        + partial_conc_events[non_zero_partial_conc_events_idx]
    )

    # NOTE: this type conversion is needed to do division, but is slow
    conc_events_per_sample = conc_events_per_sample.astype(np.float32)
    conc_events_per_sample[non_zero_partial_conc_events_idx, :] = (
        1 / conc_events_per_sample[non_zero_partial_conc_events_idx]
    )

    uniqueness_matrix_per_sample = ind_mat_samples.multiply(conc_events_per_sample)
    uniqueness_matrix_per_sample_csr = uniqueness_matrix_per_sample.tocsr()
    uniqueness_sums_per_sample = uniqueness_matrix_per_sample_csr.sum(axis=0).A1
    counts_per_sample = np.diff(uniqueness_matrix_per_sample_csr.tocsc().indptr)
    average_uniqueness_array_per_sample = uniqueness_sums_per_sample / counts_per_sample
    return average_uniqueness_array_per_sample


def seq_bootstrap(
    ind_mat: sparse.csc_matrix,
    sample_length: int = None,
    random_state: np.random.RandomState = None,
    update_probs_every: int = 1,
) -> np.array:
    """
    Returns a numpy array with tokenized indices of selected samples,
    which have been selected by sequential bootstrap procedure.

    :param ind_mat: indicator matrix from triple barrier events
    :type ind_mat: sparse.csc_matrix
    :param sample_length: Length of bootstrapped sample, defaults to None
    :type sample_length: int, optional
    :param random_state: random state, defaults to np.random.RandomState()
    :type random_state: np.random.RandomState, optional
    :return: numpy array with tokenized indices of selected samples
    :rtype: np.array
    """
    if random_state is None:
        random_state = np.random.RandomState()

    if sample_length is None:
        sample_length = ind_mat.shape[1]

    bootstrapped_samples = np.array([], dtype=np.intp)  # Bootstrapped samples

    avg_unique = np.ones(ind_mat.shape[1])
    prob = avg_unique / np.sum(avg_unique)
    sample_indices = np.arange(ind_mat.shape[1])
    ind_mat = ind_mat.astype(np.uint32)
    rows_impacted = np.array([], dtype=int)

    for i in range(int(np.ceil(sample_length / update_probs_every))):
        if rows_impacted.shape[0] > 0:
            rows_impacted = np.unique(rows_impacted)
            samples_impacted = ind_mat[rows_impacted, :].nonzero()[1]
            samples_to_update = np.unique(samples_impacted)
            avg_unique[samples_to_update] = _calc_update_avg_unique(
                ind_mat,
                samples_to_update,
                bootstrapped_samples,
            )
            prob = avg_unique / np.sum(avg_unique)  # Draw prob
            rows_impacted = np.array([], dtype=int)

        choices = random_state.choice(
            sample_indices,
            p=prob,
            size=min(sample_length - i * update_probs_every, update_probs_every),
        )
        bootstrapped_samples = np.concatenate((bootstrapped_samples, choices))
        rows_impacted = np.concatenate(
            (rows_impacted, ind_mat[:, choices].nonzero()[0])
        )

    return bootstrapped_samples
