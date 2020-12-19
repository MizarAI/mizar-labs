import numpy as np
import pandas as pd
import logging
from numba import jit
from numba import prange


def get_ind_matrix(
    samples_info_sets: pd.Series,
    price_bars: pd.DataFrame,
    event_end_time_column_name: str,
) -> np.ndarray:
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
    :return: Indicator binary matrix indicating what (price) bars influence the label for each observation
    :rtype: np.ndarray
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

    ind_mat = np.zeros((len(bar_index), len(label_endtime)))  # Init indicator matrix
    for sample_num, (label_index, label_endtime) in enumerate(tokenized_endtimes):
        ones_array = np.ones(
            (1, label_endtime - label_index + 1)
        )  # Ones array which corresponds to number of 1 to insert
        ind_mat[label_index : label_endtime + 1, sample_num] = ones_array

    assert np.max(ind_mat) == 1
    assert np.min(ind_mat) == 0

    return ind_mat


@jit(parallel=True, nopython=True)
def _bootstrap_loop_run(
    ind_mat: np.array,
    prev_concurrency: np.array,
    avg_unique: np.array,
    columns_indices_impacted: np.array,
) -> np.array:
    """
    Part of Sequential Bootstrapping for-loop.
    Using previously accumulated concurrency array, loops
    through all samples and generates averages uniqueness
    array of label based on previously accumulated concurrency

    :param ind_mat: Indicator matrix from get_ind_matrix function
    :type ind_mat: np.array
    :param prev_concurrency: Accumulated concurrency from previous
                             iterations of sequential bootstrapping
    :type prev_concurrency: np.array
    :param avg_unique: Array used as cache to store current average uniqueness
                       values per sample
    :type avg_unique: np.array
    :param columns_indices_impacted: Array to indicate which samples'
                                     average uniqueness is impacted after
                                     selecting a sample.
    :type columns_indices_impacted: np.array
    :return: Array with updated average uniqueness values per sample.
    :rtype: np.array
    """

    # iterate of samples that need updating, as previous iteration has a sample that impacts these
    for idx in prange(len(columns_indices_impacted)):

        i = columns_indices_impacted[idx]

        prev_average_uniqueness = 0
        number_of_elements = 0
        # select column
        reduced_mat = ind_mat[:, i]

        # select only rows with ones, as the sample only spans a specific period
        reduced_mat_ind = np.nonzero(reduced_mat)
        reduced_mat_greater_than_0 = reduced_mat[reduced_mat_ind]
        prev_concurrency_greater_than_0 = prev_concurrency[reduced_mat_ind]

        # compute average uniquness of sample recursively
        for j in range(len(reduced_mat_greater_than_0)):

            new_el = reduced_mat_greater_than_0[j] / (
                reduced_mat_greater_than_0[j] + prev_concurrency_greater_than_0[j]
            )

            average_uniqueness = (
                prev_average_uniqueness * number_of_elements + new_el
            ) / (number_of_elements + 1)

            number_of_elements += 1

            prev_average_uniqueness = average_uniqueness

        avg_unique[i] = average_uniqueness
    return avg_unique


def seq_bootstrap(
    ind_mat: np.ndarray,
    sample_length: int = None,
    warmup_samples: list = None,
    verbose: bool = False,
    random_state: np.random.RandomState = None,
) -> np.array:
    """
    Returns a numpy array with tokenized indices of selected samples,
    which have been selected by sequential bootstrap procedure.

    :param ind_mat: indicator matrix from triple barrier events
    :type ind_mat: np.ndarray
    :param sample_length: Length of bootstrapped sample, defaults to None
    :type sample_length: int, optional
    :param warmup_samples: list of previously drawn samples, defaults to None
    :type warmup_samples: list, optional
    :param verbose: flag to log standard bootstrap uniqueness vs sequential bootstrap uniqueness, defaults to False
    :type verbose: bool, optional
    :param random_state: random state, defaults to np.random.RandomState()
    :type random_state: np.random.RandomState, optional
    :return: numpy array with tokenized indices of selected samples
    :rtype: np.array
    """
    if random_state is None:
        random_state = np.random.RandomState()

    if sample_length is None:
        sample_length = ind_mat.shape[1]

    if warmup_samples is None:
        warmup_samples = []

    bootstrapped_samples = np.zeros(sample_length, dtype=int)  # Bootstrapped samples
    prev_concurrency = np.zeros(ind_mat.shape[0])  # Init with zeros (phi is empty)

    avg_unique = np.ones(ind_mat.shape[1])
    samples_to_update = np.array([], dtype=int)

    for i in range(sample_length):
        if samples_to_update.shape[0] != 0:
            avg_unique = _bootstrap_loop_run(
                ind_mat, prev_concurrency, avg_unique, samples_to_update
            )

        prob = avg_unique / np.sum(avg_unique)  # Draw prob

        if warmup_samples:
            choice = warmup_samples.pop(0)
        else:
            choice = random_state.choice(range(ind_mat.shape[1]), p=prob)

        bootstrapped_samples[i] = choice

        prev_concurrency += ind_mat[:, choice]  # Add recorded label array from ind_mat

        rows_impacted = np.where(ind_mat[:, choice] == 1)[0]
        samples_to_update = np.array(
            list(set(np.where(ind_mat[rows_impacted, :] == 1)[1])), dtype=int
        )

        if verbose is True:
            logging.info(f"Probability: {prob}")

    return bootstrapped_samples
