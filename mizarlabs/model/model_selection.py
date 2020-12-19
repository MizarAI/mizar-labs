import itertools as itt
import math
import numbers
import random
from abc import abstractmethod
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from mizarlabs.transformers.targets.labeling import LABEL
from scipy.stats import rv_continuous
from sklearn.model_selection._split import _BaseKFold


class BaseTimeSeriesCrossValidator(_BaseKFold):
    """
    Abstract class for time series cross-validation.

    Time series cross-validation requires each sample has a prediction
    time pred_time, at which the features are used to predict the response,
    and an evaluation time eval_time, at which the response is known and the
    error can be computed.
    Importantly, it means that unlike in standard sklearn cross-validation,
    the samples X, response y, pred_times and eval_times must all be pandas
    dataframe/series having the same index. It is also assumed that the
    samples are time-ordered with respect to the prediction time
    (i.e. pred_times is non-decreasing).

    Parameters:
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.

    """

    def __init__(
        self, n_splits=10, pred_times: pd.Series = None, eval_times: pd.Series = None
    ):

        super().__init__(n_splits, shuffle=False, random_state=None)

        if not isinstance(pred_times, pd.Series):
            raise ValueError("pred_times should be a pandas Series.")
        if not isinstance(eval_times, pd.Series):
            raise ValueError("eval_times should be a pandas Series.")

        self.pred_times = pred_times
        self.eval_times = eval_times
        self.indices = None

    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None):
        """
        Yield the indices of the train and test sets.

        :param X: pd.DataFrame, shape (n_samples, n_features), required
        :param y: pd.Series
        :param groups:  not used, inherited from _BaseKFold
        :return:
        """
        pass


class CombPurgedKFoldCV(BaseTimeSeriesCrossValidator):
    """
    Purged and embargoed combinatorial cross-validation.

    As described in Advances in financial machine learning,
    Marcos Lopez de Prado, 2018.

    The samples are decomposed into n_groups folds containing equal numbers of
    samples, without shuffling. In each cross validation round, n_test_splits
    folds are used as the test set, while the other folds are used as the train
    set. There are as many rounds as n_test_splits folds among the n_groups
    folds.Each sample should be tagged with a prediction time pred_time and an
    evaluation time eval_time. The split is such that the intervals
    [pred_times, eval_times] associated to samples in the train and test set do
    not overlap. (The overlapping samples are dropped.) In addition,
    an "embargo" period is defined, giving the minimal time between an
    evaluation time in the test set and a prediction time in the training set.
    This is to avoid, in the presence of temporal correlation, a contamination
    of the test set by the train set.

    Parameters:
    ----------
    n_groups : int, default=10
        Number of folds. Must be at least 2.
    n_test_splits : int, default=2
        Number of folds used in the test set. Must be at least 1.
    pred_times : pd.Series, shape (n_samples,), required
        Times at which predictions are made. pred_times.index has to
        coincide with X.index.
    eval_times : pd.Series, shape (n_samples,), required
        Times at which the response becomes available and the error can
        be computed. eval_times.index has to
        coincide with X.index.
    embargo_td : pd.Timedelta, default=0
        Embargo period (see explanations above).
    """

    default_embargo_td = pd.Timedelta(minutes=0)

    def __init__(
        self,
        n_groups=10,
        n_test_splits=2,
        pred_times: pd.Series = None,
        eval_times: pd.Series = None,
        embargo_td: pd.Timedelta = default_embargo_td,
    ):

        self.n_groups = n_groups

        n_splits = int(
            math.factorial(n_groups)
            / (math.factorial(n_test_splits) * math.factorial(n_groups - n_test_splits))
        )

        super().__init__(n_splits, pred_times, eval_times)
        if not isinstance(n_test_splits, numbers.Integral):
            raise ValueError(
                f"The number of test folds must be of "
                f"Integral type. {n_test_splits} of type "
                f"{type(n_test_splits)} was passed."
            )
        n_test_splits = int(n_test_splits)
        if n_test_splits <= 0 or n_test_splits > self.n_groups - 1:
            raise ValueError(
                f"K-fold cross-validation requires at least one "
                f"train/test split by setting n_test_splits"
                f" between 1 and n_groups - 1, "
                f"got n_test_splits = {n_test_splits}."
            )
        self.n_test_splits = n_test_splits
        if not isinstance(embargo_td, pd.Timedelta):
            raise ValueError(
                f"The embargo time should be of type Pandas "
                f"Timedelta. {embargo_td} of type "
                f"{type(embargo_td)} was passed."
            )
        if embargo_td < pd.Timedelta(minutes=0):
            raise ValueError(
                f"The embargo time should be positive, got embargo = {embargo_td}."
            )
        self.embargo_td = embargo_td

    def _test_fold_bounds(self, X: pd.DataFrame, y: pd.Series = None) -> List[Tuple]:
        """
        Calculate the bounds per each fold of the test set.

        :param X: pd.DataFrame, shape (n_samples, n_features), required
        :param y: pd.Series
        :return:
        """
        self.indices = np.arange(X.shape[0])

        # Fold boundaries
        fold_bounds = [
            (fold[0], fold[-1] + 1)
            for fold in np.array_split(self.indices, self.n_groups)
        ]
        # List of all combinations of n_test_splits folds selected to become
        # test sets
        selected_fold_bounds = list(itt.combinations(fold_bounds, self.n_test_splits))
        # In order for the first round to have its whole test set at the end
        # of the dataset
        selected_fold_bounds.reverse()
        return selected_fold_bounds

    def split(
        self, X: pd.DataFrame, y: pd.Series = None, groups=None
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield the indices of the train and test sets.

        Although the samples are passed in the form of a pandas dataframe,
        the indices returned are position indices, not labels.

        :param X: pd.DataFrame, shape (n_samples, n_features), required
        :param y: pd.Series
        :param groups: not used, inherited from _BaseKFold
        :return:
        """
        selected_fold_bounds = self._test_fold_bounds(X, y)

        for fold_bound_list in selected_fold_bounds:
            # Computes the bounds of the test set, and the corresponding
            # indices
            test_fold_bounds, test_indices = self._compute_test_set(fold_bound_list)
            # Computes the train set indices
            train_indices = self._compute_train_set(test_fold_bounds, test_indices)

            yield train_indices, test_indices

    def _compute_train_set(
        self, test_fold_bounds: List[Tuple[int, int]], test_indices: np.ndarray
    ) -> np.ndarray:
        """
        Compute the position indices of samples in the train set.

        :param test_fold_bounds: List of tuples of position indices
            Each tuple records the bounds of a block of indices in the test
            set.
        :param test_indices: np.ndarray
            A numpy array containing all the indices in the test set.
        :return: np.ndarray
            A numpy array containing all the indices in the train set.
        """
        # As a first approximation, the train set is the complement of the
        # test set
        train_indices = np.setdiff1d(self.indices, test_indices)
        # But we now have to purge and embargo
        for test_fold_start, test_fold_end in test_fold_bounds:
            # Purge
            train_indices = purge(self, train_indices, test_fold_start, test_fold_end)
            # Embargo
            train_indices = embargo(self, train_indices, test_indices, test_fold_end)
        return train_indices

    def _compute_test_set(
        self, fold_bound_list: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Compute the indices of the samples in the test set.

        :param fold_bound_list: List of tuples of position indices
            Each tuple records the bounds of the folds belonging to the
            test set.
        :return:
            test_fold_bounds: List of tuples of position indices
                Like fold_bound_list, but with the neighboring folds in the
                test set merged.
            test_indices: np.ndarray
                A numpy array containing the test indices.
        """
        test_indices = np.empty(0)
        test_fold_bounds = []
        for fold_start, fold_end in fold_bound_list:
            # Records the boundaries of the current test split
            if not test_fold_bounds or fold_start != test_fold_bounds[-1][-1]:
                test_fold_bounds.append((fold_start, fold_end))
            # If the current test split is contiguous to the previous one,
            # simply updates the endpoint
            elif fold_start == test_fold_bounds[-1][-1]:
                test_fold_bounds[-1] = (test_fold_bounds[-1][0], fold_end)
            test_indices = np.union1d(
                test_indices, self.indices[fold_start:fold_end]
            ).astype(int)
        return test_fold_bounds, test_indices


def embargo(
    cv: BaseTimeSeriesCrossValidator,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    test_fold_end: int,
) -> np.ndarray:
    """
    Apply the embargo procedure to part of the train set.

    This amounts to dropping the train set samples whose prediction time occurs
    within self.embargo_dt of the test set sample evaluation times.
    This method applies the embargo only to the part of the training set
    immediately following the end of the test set determined by test_fold_end.


    :param cv: Cross-validation class
        Needs to have the attributes cv.pred_times, cv.eval_times,
        cv.embargo_dt and cv.indices.
    :param train_indices: np.ndarray
        A numpy array containing all the indices of the samples currently
        included in the train set.
    :param test_indices: np.ndarray
        A numpy array containing all the indices of the samples in the test
        set.
    :param test_fold_end: int
        Index corresponding to the end of a test set block.
    :return:
        train_indices: np.ndarray
        The same array, with the indices subject to embargo removed.
    """
    if not hasattr(cv, "embargo_td"):
        raise ValueError(
            "The passed cross-validation object should have a "
            "member cv.embargo_td defining the embargo time."
        )
    last_test_eval_time = cv.eval_times.iloc[test_indices[:test_fold_end]].max()
    min_train_index = len(
        cv.pred_times[cv.pred_times <= last_test_eval_time + cv.embargo_td]
    )
    if min_train_index < cv.indices.shape[0]:
        allowed_indices = np.concatenate(
            (cv.indices[:test_fold_end], cv.indices[min_train_index:])
        )
        train_indices = np.intersect1d(train_indices, allowed_indices)
    return train_indices


def purge(
    cv: BaseTimeSeriesCrossValidator,
    train_indices: np.ndarray,
    test_fold_start: int,
    test_fold_end: int,
) -> np.ndarray:
    """
    Purge part of the train set.

    Given a left boundary index test_fold_start of the test set,
    this method removes from the train set all the
    samples whose evaluation time is posterior to the prediction time of the
    first test sample after the boundary.

    :param cv: Cross-validation class,
        Needs to have the attributes cv.pred_times, cv.eval_times and
        cv.indices.
    :param train_indices: np.ndarray, A numpy array containing all the indices
        of the samples currently included in the train set.
    :param test_fold_start: int, Index corresponding to the start of a test
        set block.
    :param test_fold_end: int, Index corresponding to the end of the same test
        set block.
    :return: train_indices: np.ndarray
        A numpy array containing the train indices purged at test_fold_start.
    """
    time_test_fold_start = cv.pred_times.iloc[test_fold_start]
    # The train indices before the start of the test fold, purged.
    train_indices_1 = np.intersect1d(
        train_indices, cv.indices[cv.eval_times < time_test_fold_start]
    )
    # The train indices after the end of the test fold.
    train_indices_2 = np.intersect1d(train_indices, cv.indices[test_fold_end:])
    return np.concatenate((train_indices_1, train_indices_2))


class LogUniformGen(rv_continuous):
    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)


def log_uniform(a=1, b=None):
    if b is None:
        b = np.exp(1)
    return LogUniformGen(a=a, b=b, name="logUniform")


def compute_back_test_paths(n_splits: int, n_test_splits: int) -> int:
    """
    Compute the number of backtest paths for the combinatorial crossvalidation.

    As explained in pg. 164 of De Prado book this function calculates
    the number of paths that can be used given the total number of splits
    (`n_splits`) and the test splits (`n_test_splits`).

    :param n_splits: the total number of splits
    :type n_splits: int
    :param n_test_splits: the number of splits used in the test set
    :type n_test_splits: int
    :return: number of the backtest paths
    :rtype: int
    """
    den = 1

    for i in range(1, n_test_splits):
        den = den * (n_splits - i)

    return int(den / math.factorial(n_test_splits - 1))


# TODO: adapt to signal pipeline
def combinatorial_cross_validation_paths(
    X: pd.DataFrame,
    y: pd.DataFrame,
    cv: CombPurgedKFoldCV,
    signal_pipeline,
    signal_pipeline_fit_params,
    label_column_name: str = LABEL,
) -> List[pd.DataFrame]:
    """
    Return the paths for the combinatorial cross validation analysis.

    :param X: Dataframe containing the features
    :type X: pd.DataFrame
    :param y: DataFrame containing the target and target info
    :type y: pd.DataFrame
    :param signal_pipeline: The signal pipeline we want to use for creating features
    :type signal_pipeline:
    :param signal_pipeline_fit_params: Fit params for the signal pipeline
    :type signal_pipeline_fit_params:
    :param cv: Combinatorial purged cv object
    :type cv: CombPurgedKFoldCV

    :return:
    """
    # computing the test bounds per each fold
    test_bounds = cv._test_fold_bounds(X, y[label_column_name])
    bound_pred_dict = {}

    for idx, train_test in enumerate(cv.split(X, y[label_column_name])):
        train_idx = train_test[0]
        test_idx = train_test[1]
        current_test_bounds = test_bounds[idx]

        signal_pipeline.fit(
            X.iloc[train_idx].values,
            y.iloc[train_idx][label_column_name].values,
            **signal_pipeline_fit_params,
        )

        # predict each sub test set
        for test_bound in current_test_bounds:

            # select the indeces of the current sub test set
            part_test = test_idx[
                np.where((test_idx >= test_bound[0]) & (test_idx < test_bound[1]))
            ]

            # predictions on the sub test set
            tmp_df = pd.DataFrame(index=X.index[part_test])
            predictions = signal_pipeline.predict(X.iloc[part_test].values)
            tmp_df["predictions"] = predictions
            proba = signal_pipeline.predict_proba(X.iloc[part_test].values)

            max_proba = np.max(proba, axis=1)
            tmp_df["max_proba"] = max_proba

            tmp_df = pd.merge(tmp_df, y, left_index=True, right_index=True, how="left")

            # adding the sub test set to the dictionary where each key is the
            # index start of the sub test set
            if test_bound[0] not in bound_pred_dict.keys():
                bound_pred_dict[test_bound[0]] = []

            bound_pred_dict[test_bound[0]].append(tmp_df)

    n_paths = compute_back_test_paths(cv.n_splits, cv.n_test_splits)
    paths = []

    # from the sub test computed create paths randomly picking one sub test
    # per start index.
    for _ in range(n_paths):
        path_df = pd.DataFrame()
        for k in bound_pred_dict.keys():
            random.shuffle(bound_pred_dict[k])
            tmp_df = bound_pred_dict[k].pop()
            path_df = path_df.append(tmp_df)

        paths.append(path_df.copy())

    yield paths
