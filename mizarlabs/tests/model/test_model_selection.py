import numpy as np
import pandas as pd
import pytest
from mizarlabs.model.model_selection import CombPurgedKFoldCV
from mizarlabs.model.model_selection import embargo
from mizarlabs.tests.conftest import create_random_sample_set
from mizarlabs.tests.conftest import prepare_cv_object


@pytest.mark.usefixtures("time_inhomogeneous_data")
def test_split_purged_kfold_cv(time_inhomogeneous_data):
    """
    Apply split to the sample described in the docstring of
    prepare_time_inhomogeneous_cv_object, with n_splits = 4
    and n_test_splits = 2. The folds are [0 : 6], [6 : 11], [11 : 16],
    [16 : 21]. We use an embargo of zero.
    Inspection shows that the pairs test-train sets should respectively be
    [...]
    3. Train: folds 1 and 4, samples [0, 1, 2, 3, 4, 16, 17, 18, 19, 20].
    Test: folds 2 and 3, samples [6, 7, 8, 9, 10, 11, 12, 13, 14, 15].
    Sample 5 is purged from the train set.
    4. Train: folds 2 and 3, samples [7, 8, 9, 10, 11, 12, 13, 14, 15].
    Test: folds 1 and 4, samples [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20].
    Sample 6 is embargoed.
    [...]
    """

    X, pred_times, eval_times = time_inhomogeneous_data

    cv = CombPurgedKFoldCV(
        n_groups=4, n_test_splits=2, pred_times=pred_times, eval_times=eval_times
    )
    count = 0
    for train_set, test_set in cv.split(X):
        count += 1
        if count == 3:
            result_train = np.array([0, 1, 2, 3, 4, 16, 17, 18, 19, 20])
            result_test = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            assert np.array_equal(result_train, train_set)
            assert np.array_equal(result_test, test_set)
        if count == 4:
            result_train = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15])
            result_test = np.array([0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20])
            assert np.array_equal(result_train, train_set)
            assert np.array_equal(result_test, test_set)


def test_compute_test_set():
    """
    We consider a sample set of size 10 with test folds [2:4],
    [4:6] and [8:10]. The function should return the aggregated bounds
    [2:6], [8:10], as well as the corresponding test indices.
    """
    fold_bound_list = [(2, 4), (4, 6), (8, 10)]
    result1 = [(2, 6), (8, 10)]
    result2 = np.array([2, 3, 4, 5, 8, 9])

    X, pred_times, eval_times = create_random_sample_set(
        n_samples=10, time_shift="120m", randomize_times=False
    )

    cv = CombPurgedKFoldCV(n_groups=5, pred_times=pred_times, eval_times=eval_times)

    cv.X = X
    cv.indices = np.arange(X.shape[0])

    agg_fold_bound_list, test_indices = cv._compute_test_set(fold_bound_list)
    assert result1 == agg_fold_bound_list
    assert np.array_equal(result2, test_indices)


def test_zero_embargo():
    """
    Generate a 2n sample data set consisting of
    - hourly samples
    - two folds, with a test fold followed by a train fold,
      starting at sample n
    For the first assert statement, a fixed 119m window between the prediction
    and the the evaluation times. This results in sample n to be embargoed.
    For the second assert statement, the window is set to 120m, causing samples
    n and n + 1 to be embargoed.
    """
    n = 6
    test_fold_end = n

    X, pred_times, eval_times = create_random_sample_set(
        n_samples=2 * n, time_shift="119m", randomize_times=False
    )

    cv = CombPurgedKFoldCV(
        n_groups=2, n_test_splits=1, pred_times=pred_times, eval_times=eval_times
    )

    cv.embargo_td = pd.Timedelta(minutes=0)
    cv.X = X
    cv.indices = np.arange(X.shape[0])
    train_indices = cv.indices[n:]
    test_indices = cv.indices[:n]
    result = cv.indices[n + 1 :]
    assert np.array_equal(
        result, embargo(cv, train_indices, test_indices, test_fold_end)
    )

    prepare_cv_object(cv, n_samples=2 * n, time_shift="120m", randomlize_times=False)

    result = cv.indices[n + 2 :]
    assert np.array_equal(
        result, embargo(cv, train_indices, test_indices, test_fold_end)
    )


def test_nonzero_embargo():
    """
    Same with an embargo delay of 2h. two more samples have to be embargoed in
    each case.
    """
    n = 6
    test_fold_end = n
    X, pred_times, eval_times = create_random_sample_set(
        n_samples=2 * n, time_shift="119m", randomize_times=False
    )

    cv = CombPurgedKFoldCV(
        n_groups=2, n_test_splits=1, pred_times=pred_times, eval_times=eval_times
    )
    cv.X = X
    cv.indices = np.arange(X.shape[0])

    cv.embargo_td = pd.Timedelta(minutes=120)
    train_indices = cv.indices[n:]
    test_indices = cv.indices[:n]
    result = cv.indices[n + 3 :]

    assert np.array_equal(
        result, embargo(cv, train_indices, test_indices, test_fold_end)
    )

    prepare_cv_object(cv, n_samples=2 * n, time_shift="120m", randomlize_times=False)

    result = cv.indices[n + 4 :]
    assert np.array_equal(
        result, embargo(cv, train_indices, test_indices, test_fold_end)
    )


# def test_combinatorial_cross_validation_paths(
#     dollar_bar_dataframe,
#     dollar_bar_target_labels,
#     strategy_signal_pipeline_only_primary,
# ):
#
#     cv = CombPurgedKFoldCV(
#         n_groups=5,
#         n_test_splits=1,
#         pred_times=dollar_bar_target_labels.index.to_series(),
#         eval_times=dollar_bar_target_labels[EVENT_END_TIME],
#     )

# X_dict = {
#     "primary": dollar_bar_dataframe.loc[dollar_bar_target_labels.index, [CLOSE]]
# }
# combinatorial_cross_validation_paths(
#     X_dict, dollar_bar_dataframe, cv,
#     strategy_signal_pipeline_only_primary,
# )

# TODO: complete the test which depends on the signal strategy
