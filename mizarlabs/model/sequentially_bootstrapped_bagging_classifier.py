import itertools
import numbers
from abc import ABCMeta
from abc import abstractmethod
from typing import Tuple
from typing import Union
from warnings import warn

import numpy as np
import pandas as pd
from mizarlabs.transformers.targets.labeling import EVENT_END_TIME
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble._bagging import BaseBagging
from sklearn.ensemble._base import _partition_estimators
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
from sklearn.utils import indices_to_mask
from sklearn.utils._joblib import delayed
from sklearn.utils._joblib import Parallel
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import has_fit_parameter

from .bootstrapping import get_ind_matrix
from .bootstrapping import seq_bootstrap

MAX_INT = np.iinfo(np.int32).max


def _generate_random_features(
    random_state: np.random.RandomState,
    bootstrap: bool,
    n_population: int,
    n_samples: int,
) -> np.array:
    """
    Draw randomly sampled indices.

    :param random_state: Random state objecc
    :type random_state: np.random.RandomState
    :param bootstrap: Boolean indicating whether drawing with or without replacement.
    :type bootstrap: bool
    :param n_population: The size of the set to sample from.
    :type n_population: int
    :param n_samples: The number of samples to draw.
    :type n_samples: int
    :return: array indicating which samples have been drawn.
    :rtype: np.array
    """
    # Draw sample indices
    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(
            n_population, n_samples, random_state=random_state
        )

    return indices


def _generate_bagging_indices(
    random_state: np.random.RandomState,
    bootstrap_features: bool,
    n_features: int,
    max_features: int,
    max_samples: int,
    ind_mat: np.array,
    update_probs_every: int,
) -> Tuple[np.array, np.array]:
    """
    Randomly draw feature and sample indices.

    :param random_state: Random state object.
    :type random_state: np.random.RandomState
    :param bootstrap_features: Boolean indicating whether features
                               need to be bootstrapped.
    :type bootstrap_features: bool
    :param n_features: Number of features to draw.
    :type n_features: int
    :param max_features: Max number of features available.
    :type max_features: int
    :param max_samples: Max number of samples to draw.
    :type max_samples: int
    :param ind_mat: Indicator matrix from triple barrier events
    :type ind_mat: np.array
    :param update_probs_every: only update the sampling probabilities with average uniqueness after
                               update_probs_every times, this will speed up training, but at the cost that you
                               do not sample perfectly according to the average uniqueness
    :return: tuple with two arrays indicating
             the features and samples drawn respectively.
    :rtype: Tuple[np.aray, np.array]
    """
    # Get valid random state
    random_state = check_random_state(random_state)

    # Draw indices
    feature_indices = _generate_random_features(
        random_state, bootstrap_features, n_features, max_features
    )
    sample_indices = seq_bootstrap(
        ind_mat, sample_length=max_samples, random_state=random_state, update_probs_every=update_probs_every,
    )

    return feature_indices, sample_indices


def _parallel_build_estimators(
    n_estimators: int,
    ensemble: BaseBagging,
    X: pd.DataFrame,
    y: pd.Series,
    ind_mat: sparse.csc_matrix,
    sample_weight: pd.Series,
    seeds: np.array,
    total_n_estimators: int,
    verbose: int,
    update_probs_every: int,
) -> tuple:
    """
    Private function used to build a batch of estimators within a job.

    :param n_estimators: Number of base estimators to build in batch.
    :type n_estimators: int
    :param ensemble: Ensemble model.
    :type ensemble: BaseBagging
    :param X: DataFrame with features.
    :type X: pd.DataFrame
    :param y: Series with labels.
    :type y: pd.Series
    :param ind_mat: indicator matrix from triple barrier events
    :type ind_mat: sparse.csc_matrix
    :param sample_weight: Series with sample weights.
    :type sample_weight: pd.Series
    :param seeds: Array with seeds for random state.
    :type seeds: np.array
    :param total_n_estimators: Total number of estimators in ensemble.
    :type total_n_estimators: int
    :param verbose: Indicating how much to print.
    :type verbose: int
    :param update_probs_every: only update the sampling probabilities with average uniqueness after
                               update_probs_every times, this will speed up training, but at the cost that you
                               do not sample perfectly according to the average uniqueness
    :return: Tuple with estimators, features and the estimator indices
    :rtype: tuple
    """
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.base_estimator_, "sample_weight")

    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []
    estimators_indices = []

    for i in range(n_estimators):
        if verbose > 1:
            print(
                "Building estimator %d of %d for this parallel run "
                "(total %d)..." % (i + 1, n_estimators, total_n_estimators)
            )

        random_state = np.random.RandomState(seeds[i])
        estimator = ensemble._make_estimator(append=False, random_state=random_state)

        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(
            random_state,
            bootstrap_features,
            n_features,
            max_features,
            max_samples,
            ind_mat,
            update_probs_every,
        )

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            sample_counts = np.bincount(indices, minlength=n_samples)
            curr_sample_weight *= sample_counts

            estimator.fit(X[:, features], y, sample_weight=curr_sample_weight)

        else:
            estimator.fit((X[indices])[:, features], y[indices])

        estimators.append(estimator)
        estimators_features.append(features)
        estimators_indices.append(indices)

    return estimators, estimators_features, estimators_indices


class SequentiallyBootstrappedBaseBagging(BaseBagging, metaclass=ABCMeta):
    """
    Base class for Sequentially Bootstrapped Classifier and Regressor, extension of sklearn's BaseBagging
    """

    @abstractmethod
    def __init__(
        self,
        samples_info_sets: pd.Series,
        price_bars: pd.DataFrame,
        base_estimator: BaseEstimator = None,
        n_estimators: int = 10,
        max_samples: Union[int, float] = 1.0,
        max_features: Union[int, float] = 1.0,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        warm_start: bool = False,
        n_jobs: int = None,
        random_state: Union[int, np.random.RandomState, None] = None,
        verbose: int = 0,
        event_end_time_column_name: str = EVENT_END_TIME,
        update_probs_every: int = 1,
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            bootstrap=True,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

        self.event_end_time_column_name = event_end_time_column_name
        self.samples_info_sets = samples_info_sets
        self.price_bars = price_bars
        self._ind_mat = None
        self.update_probs_every = update_probs_every

        # Used for create get ind_matrix subsample during cross-validation
        self.timestamp_int_index_mapping = pd.Series(
            index=samples_info_sets.index, data=range(self.ind_mat.shape[1])
        )

        self.X_time_index = None  # Timestamp index of X_train

    @property
    def ind_mat(self):
        if self._ind_mat is None:
            self._ind_mat = get_ind_matrix(
                self.samples_info_sets, self.price_bars, self.event_end_time_column_name
            )
        return self._ind_mat

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series = None,
    ):
        """Build a Sequentially Bootstrapped Bagging ensemble of estimators from the training
           set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        Returns
        -------
        self : object
        """
        return self._fit(X, y, self.max_samples, sample_weight=sample_weight)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_samples: Union[int, float] = None,
        max_depth: int = None,
        sample_weight: pd.Series = None,
    ):
        """Build a Sequentially Bootstrapped Bagging ensemble of estimators from the training
           set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        max_samples : int or float, optional (default=None)
            Argument to use instead of self.max_samples.
        max_depth : int, optional (default=None)expiritation
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        Returns
        -------
        self : object
        """

        assert isinstance(X, pd.DataFrame), "X should be a dataframe with time indices"
        assert isinstance(y, pd.Series), "y should be a series with time indices"
        assert isinstance(
            X.index, pd.DatetimeIndex
        ), "X index should be a DatetimeIndex"
        assert isinstance(
            y.index, pd.DatetimeIndex
        ), "y index should be a DatetimeIndex"

        random_state = check_random_state(self.random_state)
        self.X_time_index = X.index  # Remember X index for future sampling

        assert set(self.timestamp_int_index_mapping.index).issuperset(
            set(self.X_time_index)
        ), "The ind matrix timestamps should have all the timestamps in the training data"

        # Generate subsample ind_matrix (we need this during subsampling cross_validation)
        ind_mat = self.ind_mat.tocsc()

        subsampled_ind_mat = ind_mat[
            :, self.timestamp_int_index_mapping.loc[self.X_time_index]
        ]

        # Convert data (X is required to be 2d and indexable)
        X, y = check_X_y(
            X, y, ["csr", "csc"], dtype=None, force_all_finite=False, multi_output=True
        )
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        # Remap output
        n_samples, self.n_features_ = X.shape
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        # Validate max_samples
        if not isinstance(max_samples, (numbers.Integral, np.integer)):
            max_samples = int(max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        elif isinstance(self.max_features, np.float):
            max_features = self.max_features * self.n_features_
        else:
            raise ValueError("max_features must be int or float")

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available if warm_start=False")

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if anyexpiritation
            self.estimators_ = []
            self.estimators_features_ = []
            self.sequentially_bootstrapped_samples_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            n_more_estimators, self.n_jobs
        )
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose,)(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                subsampled_ind_mat,
                sample_weight,
                seeds[starts[i] : starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose,
                update_probs_every=self.update_probs_every,
            )
            for i in range(n_jobs)
        )

        # Reduce
        self.estimators_ += list(
            itertools.chain.from_iterable(t[0] for t in all_results)
        )
        self.estimators_features_ += list(
            itertools.chain.from_iterable(t[1] for t in all_results)
        )
        self.sequentially_bootstrapped_samples_ += list(
            itertools.chain.from_iterable(t[2] for t in all_results)
        )

        if self.oob_score:
            self._set_oob_score(X, y)

        self._ind_mat = None

        return self


class SequentiallyBootstrappedBaggingClassifier(
    SequentiallyBootstrappedBaseBagging, BaggingClassifier, ClassifierMixin
):
    """
    A Sequentially Bootstrapped Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset generated using
    Sequential Bootstrapping sampling procedure and then aggregate their individual predictions (
    either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.
    :param samples_info_sets: pd.Series, The information range on which each record is constructed from
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.
    :param price_bars: pd.DataFrame
        Price bars used in samples_info_sets generation
    :param base_estimator: object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.
    :param n_estimators: int, optional (default=10)
        The number of base estimators in the ensemble.
    :param max_samples: int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
        If int, then draw `max_samples` samples. If float, then draw `max_samples * X.shape[0]` samples.
    :param max_features: int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
        If int, then draw `max_features` features. If float, then draw `max_features * X.shape[1]` features.
    :param bootstrap_features: boolean, optional (default=False)
        Whether features are drawn with replacement.
    :param oob_score: bool, optional (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization error.
    :param warm_start: bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.
    :param n_jobs: int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    :param random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :param verbose: int, optional (default=0)
        Controls the verbosity when fitting and predicting.
    :param event_end_time_column_name: str, optional (default=EXPIRATION_BARRIER)
        name of the column with the expiration barrier dates.
    :param update_probs_every: int, optional (default=1)
        Only update the sampling probabilities with average uniqueness after
        update_probs_every times, this will speed up training, but at the cost that you
        do not sample perfectly according to the average uniqueness
    :ivar base_estimator_: estimator
        The base estimator from which the ensemble is grown.
    :ivar estimators_: list of estimators
        The collection of fitted base estimators.
    :ivar estimators_samples_: list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.
    :ivar estimators_features_: list of arrays
        The subset of drawn features for each base estimator.
    :ivar classes_: array of shape = [n_classes]
        The classes labels.
    :ivar n_classes_: int or list
        The number of classes.
    :ivar oob_score_: float
        Score of the training dataset obtained using an out-of-bag estimate.
    :ivar oob_decision_function_: array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.
    """

    def __init__(
        self,
        samples_info_sets: pd.Series,
        price_bars: pd.DataFrame,
        base_estimator: BaseEstimator = None,
        n_estimators: int = 10,
        max_samples: Union[int, float] = 1.0,
        max_features: Union[int, float] = 1.0,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        warm_start: bool = False,
        n_jobs: int = None,
        random_state: Union[int, np.random.RandomState, None] = None,
        verbose: int = 0,
        event_end_time_column_name: str = EVENT_END_TIME,
        update_probs_every: int = 1,
    ):
        super().__init__(
            samples_info_sets=samples_info_sets,
            price_bars=price_bars,
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            event_end_time_column_name=event_end_time_column_name,
            update_probs_every=update_probs_every,
        )

    def _validate_estimator(self):
        """
        Check the estimator and set the base_estimator_ attribute.
        """
        super(BaggingClassifier, self)._validate_estimator(
            default=DecisionTreeClassifier()
        )

    def _set_oob_score(self, X: pd.DataFrame, y: pd.Series):
        """
        Calculates and sets the out of bag score.

        :param X: DataFrame with features.
        :type X: pd.DataFrame
        :param y: Series with labels.
        :type y: pd.Series
        """
        n_samples = y.shape[0]
        n_classes_ = self.n_classes_

        predictions = np.zeros((n_samples, n_classes_))

        for estimator, samples, features in zip(
            self.estimators_,
            self.sequentially_bootstrapped_samples_,
            self.estimators_features_,
        ):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            if hasattr(estimator, "predict_proba"):
                predictions[mask, :] += estimator.predict_proba(
                    (X[mask, :])[:, features]
                )

            else:
                p = estimator.predict((X[mask, :])[:, features])
                j = 0

                for i in range(n_samples):
                    if mask[i]:
                        predictions[i, p[j]] += 1
                        j += 1

        if (predictions.sum(axis=1) == 0).any():
            warn(
                "Some inputs do not have OOB scores. "
                "This probably means too few estimators were used "
                "to compute any reliable oob estimates."
            )

        oob_decision_function = predictions / predictions.sum(axis=1)[:, np.newaxis]
        oob_score = accuracy_score(y, np.argmax(predictions, axis=1))

        self.oob_decision_function_ = oob_decision_function
        self.oob_score_ = oob_score
