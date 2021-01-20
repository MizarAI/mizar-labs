import numpy as np
import pandas as pd
import pytest
from mizarlabs.model.sequentially_bootstrapped_bagging_classifier import (
    SequentiallyBootstrappedBaggingClassifier,
)
from mizarlabs.static import CLOSE
from mizarlabs.transformers.targets.labeling import EVENT_END_TIME
from mizarlabs.transformers.targets.labeling import LABEL
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


@pytest.mark.usefixtures("dollar_bar_target_labels", "dollar_bar_dataframe")
def test_value_error_raise(
    dollar_bar_target_labels: pd.DataFrame, dollar_bar_dataframe: pd.DataFrame
):
    """
    Test various values error raise
    """

    X_train = dollar_bar_dataframe.loc[dollar_bar_target_labels.index[:100], [CLOSE]]
    y_train = dollar_bar_target_labels[LABEL].iloc[:100]

    samples_info_sets = dollar_bar_target_labels[EVENT_END_TIME]

    clf = KNeighborsClassifier()
    bagging_clf_1 = SequentiallyBootstrappedBaggingClassifier(
        base_estimator=clf,
        samples_info_sets=samples_info_sets,
        price_bars=dollar_bar_dataframe,
    )
    bagging_clf_2 = SequentiallyBootstrappedBaggingClassifier(
        base_estimator=clf,
        samples_info_sets=samples_info_sets,
        price_bars=dollar_bar_dataframe,
        max_samples=2000000,
    )
    bagging_clf_3 = SequentiallyBootstrappedBaggingClassifier(
        base_estimator=clf,
        samples_info_sets=samples_info_sets,
        price_bars=dollar_bar_dataframe,
        max_features="20",
    )
    bagging_clf_4 = SequentiallyBootstrappedBaggingClassifier(
        base_estimator=clf,
        samples_info_sets=samples_info_sets,
        price_bars=dollar_bar_dataframe,
        max_features=2000000,
    )
    bagging_clf_5 = SequentiallyBootstrappedBaggingClassifier(
        base_estimator=clf,
        samples_info_sets=samples_info_sets,
        price_bars=dollar_bar_dataframe,
        oob_score=True,
        warm_start=True,
    )
    bagging_clf_6 = SequentiallyBootstrappedBaggingClassifier(
        base_estimator=clf,
        samples_info_sets=samples_info_sets,
        price_bars=dollar_bar_dataframe,
        warm_start=True,
    )
    bagging_clf_7 = SequentiallyBootstrappedBaggingClassifier(
        base_estimator=clf,
        samples_info_sets=samples_info_sets,
        price_bars=dollar_bar_dataframe,
        warm_start=True,
    )

    # ValueError to use sample weight with classifier which doesn't support sample weights
    with np.testing.assert_raises(ValueError):
        bagging_clf_1.fit(X_train, y_train, sample_weight=np.ones_like(y_train))

    # ValueError for max_samples > X_train.shape[0]
    with np.testing.assert_raises(ValueError):
        bagging_clf_2.fit(X_train, y_train)

    # ValueError for non-int/float max_features param
    with np.testing.assert_raises(ValueError):
        bagging_clf_3.fit(X_train, y_train)

    # ValueError for max_features > X_train.shape[1]
    with np.testing.assert_raises(ValueError):
        bagging_clf_4.fit(X_train, y_train)

    # ValueError for warm_start and oob_score being True
    with np.testing.assert_raises(ValueError):
        bagging_clf_5.fit(X_train, y_train)

    # ValueError for decreasing the number of estimators when warm start is True
    with np.testing.assert_raises(ValueError):
        bagging_clf_6.fit(X_train, y_train)
        bagging_clf_6.n_estimators -= 2
        bagging_clf_6.fit(X_train, y_train)

    # ValueError for setting n_estimators to negative value
    with np.testing.assert_raises(ValueError):
        bagging_clf_7.fit(X_train, y_train)
        bagging_clf_7.n_estimators -= 1000
        bagging_clf_7.fit(X_train, y_train)


# TODO: the seed is not fixing the outputs, njobs=1 also does not help,
#       requires additional investigation to fix the seed
@pytest.mark.usefixtures(
    "X_train_perfect",
    "X_train_random",
    "y_train",
    "X_test_perfect",
    "X_test_random",
    "y_test",
    "samples_info_sets",
    "dollar_bar_dataframe",
)
def test_sb_classifier(
    X_train_perfect,
    X_train_random,
    y_train,
    X_test_perfect,
    X_test_random,
    y_test,
    samples_info_sets,
    dollar_bar_dataframe,
):
    """
    Test Sequentially Bootstrapped Bagging Classifier. Here we compare oos/oob scores to sklearn's bagging oos scores,
    test oos predictions values
    """

    # Init classifiers
    clf_base = RandomForestClassifier(
        n_estimators=1,
        criterion="entropy",
        bootstrap=False,
        class_weight="balanced_subsample",
    )

    sb_clf = SequentiallyBootstrappedBaggingClassifier(
        base_estimator=clf_base,
        max_features=1.0,
        n_estimators=20,
        samples_info_sets=samples_info_sets,
        price_bars=dollar_bar_dataframe,
        oob_score=True,
        random_state=1,
        n_jobs=1,
    )

    # X_train index should be in index mapping
    assert set(X_train_perfect.index).issubset(
        set(sb_clf.timestamp_int_index_mapping.index)
    )

    sb_clf.fit(X_train_perfect, y_train)

    # X_train index == clf X_train index
    assert all(sb_clf.X_time_index == X_train_perfect.index)

    # perfect model
    sb_clf.fit(X_train_perfect, y_train)
    oos_sb_predictions_perfect = sb_clf.predict(X_test_perfect)

    # random model
    sb_clf.fit(X_train_random, y_train)
    oos_sb_predictions_random = sb_clf.predict(X_test_random)

    # test perfect scores
    sb_precision_perfect = precision_score(y_test, oos_sb_predictions_perfect)
    sb_roc_auc_perfect = roc_auc_score(y_test, oos_sb_predictions_perfect)
    sb_accuracy_perfect = accuracy_score(y_test, oos_sb_predictions_perfect)
    sb_precision_random = precision_score(y_test, oos_sb_predictions_random)
    sb_roc_auc_random = roc_auc_score(y_test, oos_sb_predictions_random)
    sb_accuracy_random = accuracy_score(y_test, oos_sb_predictions_random)

    # test random scores
    random_score_approx = pytest.approx(0.5, 0.5)
    assert sb_precision_perfect == sb_roc_auc_perfect == sb_accuracy_perfect == 1.0
    assert sb_precision_random == random_score_approx
    assert sb_roc_auc_random == random_score_approx
    assert sb_accuracy_random == random_score_approx
    assert sb_precision_perfect > sb_precision_random
    assert sb_roc_auc_perfect > sb_roc_auc_random
    assert sb_accuracy_perfect > sb_accuracy_random


@pytest.mark.usefixtures(
    "samples_info_sets", "dollar_bar_dataframe", "X_train_perfect", "y_train"
)
def test_sb_bagging_not_tree_base_estimator(
    samples_info_sets, dollar_bar_dataframe, X_train_perfect, y_train
):
    """
    Test SB Bagging with non-tree base estimator (KNN)
    """
    clf = KNeighborsClassifier()
    sb_clf = SequentiallyBootstrappedBaggingClassifier(
        base_estimator=clf,
        samples_info_sets=samples_info_sets,
        price_bars=dollar_bar_dataframe,
    )
    sb_clf.fit(X_train_perfect, y_train)
    assert all(sb_clf.predict(X_train_perfect)[:10] == y_train[:10])


@pytest.mark.usefixtures(
    "samples_info_sets", "dollar_bar_dataframe", "X_train_perfect", "y_train"
)
def test_sb_bagging_non_sample_weights_with_verbose(
    samples_info_sets, dollar_bar_dataframe, X_train_perfect, y_train
):
    """
    Test SB Bagging with classifier which doesn't support sample_weights with verbose > 1
    """

    clf = LinearSVC()

    sb_clf = SequentiallyBootstrappedBaggingClassifier(
        base_estimator=clf,
        max_features=0.2,
        n_estimators=10,
        samples_info_sets=samples_info_sets,
        price_bars=dollar_bar_dataframe,
        oob_score=True,
        random_state=1,
        bootstrap_features=True,
        max_samples=30,
        verbose=2,
    )

    sb_clf.fit(X_train_perfect, y_train)
    assert all(sb_clf.predict(X_train_perfect)[:10] == y_train[:10])


@pytest.mark.usefixtures(
    "samples_info_sets", "dollar_bar_dataframe", "X_train_perfect", "y_train"
)
def test_sb_bagging_with_max_features(
    samples_info_sets, dollar_bar_dataframe, X_train_perfect, y_train
):
    """
    Test SB Bagging with base_estimator bootstrap = True, float max_features, max_features bootstrap = True
    :return:
    """
    clf = RandomForestClassifier(
        n_estimators=1,
        criterion="entropy",
        bootstrap=True,
        class_weight="balanced_subsample",
        max_depth=12,
    )

    sb_clf = SequentiallyBootstrappedBaggingClassifier(
        base_estimator=clf,
        max_features=0.2,
        n_estimators=10,
        samples_info_sets=samples_info_sets,
        price_bars=dollar_bar_dataframe,
        oob_score=True,
        random_state=1,
        bootstrap_features=True,
        max_samples=30,
        verbose=2,
    )

    sb_clf.fit(X_train_perfect, y_train)
    assert all(sb_clf.predict(X_train_perfect)[:10] == y_train[:10])


@pytest.mark.usefixtures(
    "samples_info_sets", "dollar_bar_dataframe", "X_train_perfect", "y_train"
)
def test_sb_bagging_float_max_samples_warm_start_true(
    samples_info_sets, dollar_bar_dataframe, X_train_perfect, y_train
):
    """
    Test SB Bagging with warm start = True and float max_samples
    """
    X_train_perfect = X_train_perfect[[LABEL] * 100]
    clf = RandomForestClassifier(
        n_estimators=1,
        criterion="entropy",
        bootstrap=False,
        class_weight="balanced_subsample",
        max_depth=12,
    )

    sb_clf = SequentiallyBootstrappedBaggingClassifier(
        base_estimator=clf,
        max_features=7,
        n_estimators=10,
        samples_info_sets=samples_info_sets,
        price_bars=dollar_bar_dataframe,
        oob_score=False,
        random_state=1,
        bootstrap_features=True,
        max_samples=0.3,
        warm_start=True,
    )

    sb_clf.fit(
        X_train_perfect,
        y_train,
        sample_weight=np.ones((X_train_perfect.shape[0],)),
    )
    sb_clf.n_estimators += 1
    sb_clf.fit(
        X_train_perfect,
        y_train,
        sample_weight=np.ones((X_train_perfect.shape[0],)),
    )
    sb_clf.n_estimators += 2
    sb_clf.fit(
        X_train_perfect,
        y_train,
        sample_weight=np.ones((X_train_perfect.shape[0],)),
    )

    assert all(sb_clf.predict(X_train_perfect)[:10] == y_train[:10])


# NOTE: rename this test if you want a test where you can run a large fake data set to
# optimise the performance of the classifier
@pytest.mark.usefixtures(
    "samples_info_sets", "dollar_bar_dataframe", "X_train_perfect", "y_train"
)
def performance_test_sb_bagging(
    samples_info_sets, dollar_bar_dataframe, X_train_perfect, y_train
):
    """
    Perfomance test of SB Bagging model
    """
    n_copies = 100
    randint_high = 20
    dollar_bar_dataframe_scaled = pd.concat([dollar_bar_dataframe for _ in range(n_copies)])
    dollar_bar_dataframe_scaled.index = pd.date_range(dollar_bar_dataframe.index[0], dollar_bar_dataframe.index[-1], periods=dollar_bar_dataframe_scaled.shape[0])
    samples_info_sets_scaled = pd.Series(
        [dollar_bar_dataframe_scaled.index[i + np.random.randint(1, randint_high)] for i in range(dollar_bar_dataframe_scaled.shape[0] - randint_high)],
    )
    samples_info_sets_scaled.index = dollar_bar_dataframe_scaled.index[:samples_info_sets_scaled.shape[0]]
    samples_info_sets_scaled.name = "event_end_time"
    n_features = 10
    X_train_scaled = pd.DataFrame(np.random.random((samples_info_sets_scaled.shape[0], n_features)), index=samples_info_sets_scaled.index)
    y_train_scaled = pd.Series(np.random.choice([-1, 1]), index=samples_info_sets_scaled.index)

    clf = DecisionTreeClassifier()

    sb_clf = SequentiallyBootstrappedBaggingClassifier(
        base_estimator=clf,
        max_features=n_features,
        n_estimators=100,
        samples_info_sets=samples_info_sets_scaled,
        price_bars=dollar_bar_dataframe_scaled,
        oob_score=False,
        random_state=1,
        bootstrap_features=True,
        max_samples=0.99,
        warm_start=True,
        n_jobs=1,
        verbose=10,
        update_probs_every=50,
    )

    sb_clf.fit(
        X_train_scaled,
        y_train_scaled,
    )
