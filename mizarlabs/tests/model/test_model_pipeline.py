import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from mizarlabs.model.pipeline import StrategySignalPipeline
from mizarlabs.static import CLOSE
from mizarlabs.transformers.targets.labeling import LABEL
from mizarlabs.transformers.technical.moving_average import (
    MovingAverageCrossOverPredictor,
)
from mizarlabs.transformers.trading.bet_sizing import BetSizingFromProbabilities
from mizarlabs.transformers.utils import IdentityTransformer


def test_pipeline_simple_model(dollar_bar_dataframe, dollar_bar_labels_and_info):
    """
    Checks if a simple model behaves properly as a MizarStrategyPipeline.
    """
    fast_window = 10
    slow_window = 20
    maco = MovingAverageCrossOverPredictor(
        fast=fast_window,
        slow=slow_window,
        column_name=CLOSE,
        fill_between_crossovers=True,
    )
    pipeline = StrategySignalPipeline(
        feature_transformers_primary_model={"primary_0": None}, align_on="primary_0"
    )

    pipeline.set_primary_model(maco)

    pipeline.predict(X_dict={"primary_0": dollar_bar_dataframe})

    pipeline.predict_proba(X_dict={"primary_0": dollar_bar_dataframe})

    pipeline.get_size(X_dict={"primary_0": dollar_bar_dataframe})

    pipeline.get_side(X_dict={"primary_0": dollar_bar_dataframe})


@pytest.mark.parametrize("primary_model_num_feature_generators", list(range(3)))
@pytest.mark.parametrize("metalabeling_model_num_feature_generators", list(range(3)))
def test_pipeline_transform(
    primary_model_num_feature_generators: int,
    metalabeling_model_num_feature_generators: int,
    dollar_bar_dataframe: pd.DataFrame,
    bar_feature_generator,
):
    """
    Unit test for transform pipeline
    """
    # create dict structure for primary model feature generator
    feature_transformers_primary_model = {
        f"primary_{i}": bar_feature_generator
        for i in range(primary_model_num_feature_generators)
    }

    # create dict structure for metalabeling model feature generator
    feature_transformers_metalabeling_model = {
        f"metalabeling_{i}": bar_feature_generator
        for i in range(metalabeling_model_num_feature_generators)
    }

    pipeline = StrategySignalPipeline(
        feature_transformers_primary_model=feature_transformers_primary_model,
        feature_transformers_metalabeling_model=feature_transformers_metalabeling_model,
        align_on="primary_0",
    )

    # transform data with pipeline
    X_features_pipeline = pipeline.transform(
        {
            **{
                f"primary_{i}": dollar_bar_dataframe
                for i in range(primary_model_num_feature_generators)
            },
            **{
                f"metalabeling_{i}": dollar_bar_dataframe
                for i in range(metalabeling_model_num_feature_generators)
            },
        }
    )

    # check expected output
    X_features_expected = bar_feature_generator.transform(dollar_bar_dataframe)
    if primary_model_num_feature_generators > 0:
        for df in X_features_pipeline["primary_model_features"].values():
            pd.testing.assert_frame_equal(df, X_features_expected)
    if metalabeling_model_num_feature_generators > 0:
        for df in X_features_pipeline["metalabeling_model_features"].values():
            pd.testing.assert_frame_equal(df, X_features_expected)


def test_metalabeling_model_with_moving_average_crossover(
    dollar_bar_dataframe: pd.DataFrame,
    bar_feature_generator,
    dollar_bar_labels_and_info: pd.DataFrame,
):
    """
    Unit test for transform pipeline with moving average cross over model
    """
    # init dummy models
    primary_model = MovingAverageCrossOverPredictor(
        fast=10, slow=40, column_name=CLOSE, fill_between_crossovers=True
    )
    metalabeling_model = RandomForestClassifier(random_state=1, n_jobs=-1)

    # create dict structure for primary model feature generator
    feature_transformers_primary_model = {"primary_0": IdentityTransformer()}

    # create dict structure for metalabeling model feature generator
    feature_transformers_metalabeling_model = {"metalabeling_0": bar_feature_generator}

    pipeline = StrategySignalPipeline(
        feature_transformers_primary_model=feature_transformers_primary_model,
        feature_transformers_metalabeling_model=feature_transformers_metalabeling_model,
        align_on="primary_0",
    )

    X_dict = {
        **{"primary_0": dollar_bar_dataframe},
        **{"metalabeling_0": dollar_bar_dataframe},
    }

    pipeline.set_primary_model(primary_model)

    x_metalabeling, y_metalabeling = pipeline.create_dataset_metalabeling(
        X_dict, dollar_bar_labels_and_info
    )

    # down-sampling, sample weights

    metalabeling_model.fit(x_metalabeling, y_metalabeling)

    pipeline.set_metalabeling_model(metalabeling_model)

    pipeline.predict(X_dict)

    #
    # bar_feature_generator.transform(dollar_bar_dataframe)
    # TODO: think what assertion we can add here
    # TODO: make model's prediction deterministic so that we can
    #  assert the predicitons are correct


def test_pipeline_transform_error_raising():
    feature_transformers_primary_model = {"some_name_0": None}
    feature_transformers_metalabeling_model = {"some_name_1": None}
    pipeline = StrategySignalPipeline(
        primary_model=RandomForestClassifier(random_state=1, n_jobs=-1),
        feature_transformers_primary_model=feature_transformers_primary_model,
        feature_transformers_metalabeling_model=feature_transformers_metalabeling_model,
        metalabeling_model=RandomForestClassifier(random_state=1, n_jobs=-1),
        embargo_td=pd.Timedelta(0),
    )
    with pytest.raises(AssertionError) as excinfo:
        pipeline.transform({"some_wrong_name_0": None, "some_wrong_name_1": None})

    assert (
        "The keys in X_dict should be exactly the same as the name of the feature "
        "generators of the primary (and metalabeling) model." in str(excinfo.value)
    )


def test_pipeline_primary_model_is_fitted(
    dollar_bar_dataframe,
    dollar_bar_labels_and_info,
    strategy_signal_pipeline_only_primary,
):

    strategy_signal_pipeline_only_primary.fit(
        X_dict={"primary_0": dollar_bar_dataframe}, y=dollar_bar_labels_and_info
    )

    check_is_fitted(strategy_signal_pipeline_only_primary.primary_model)


def test_pipeline_metalabeling_model_is_fitted(
    dollar_bar_dataframe,
    dollar_bar_labels_and_info,
    strategy_signal_pipeline_with_metalabeling,
):

    # deep copy of the model to make sure that tests are not pointing to the
    # same object when run in parallel
    strategy_signal_pipeline_with_metalabeling = copy.deepcopy(
        strategy_signal_pipeline_with_metalabeling
    )
    strategy_signal_pipeline_with_metalabeling.fit(
        X_dict={
            "primary_0": dollar_bar_dataframe,
            "metalabeling_0": dollar_bar_dataframe,
        },
        y=dollar_bar_labels_and_info,
        align_on="primary_0",
        align_how={"metalabeling_0": "mean"},
    )

    check_is_fitted(strategy_signal_pipeline_with_metalabeling.metalabeling_model)


def test_pipeline_predict_only_primary_model(
    dollar_bar_dataframe,
    strategy_signal_pipeline_only_primary_fitted,
    manual_primary_model_fitted,
    manual_X_primary,
):
    # deep copy of the model to make sure that tests are not pointing to the
    # same object when run in parallel
    strategy_signal_pipeline_only_primary_fitted = copy.deepcopy(
        strategy_signal_pipeline_only_primary_fitted
    )

    # manual predict
    y_pred_manual = pd.Series(
        manual_primary_model_fitted.predict(manual_X_primary),
        index=manual_X_primary.index,
    )

    y_pred_pipeline = strategy_signal_pipeline_only_primary_fitted.predict(
        X_dict={"primary_0": dollar_bar_dataframe}
    )
    pd.testing.assert_series_equal(
        y_pred_manual,
        y_pred_pipeline[StrategySignalPipeline._primary_model_predictions],
    )


def test_pipeline_predict_metalabeling(
    dollar_bar_dataframe,
    dollar_bar_labels_and_info,
    strategy_signal_pipeline_with_metalabeling_fitted,
    manual_X_y_primary,
):

    # deep copy of the model to make sure that tests are not pointing to the
    # same object when run in parallel
    strategy_signal_pipeline_with_metalabeling_fitted = copy.deepcopy(
        strategy_signal_pipeline_with_metalabeling_fitted
    )

    manual_X_primary, manual_y_primary = manual_X_y_primary
    # manual fit and predict
    (
        X_pred_features,
        y_metalabel,
    ) = strategy_signal_pipeline_with_metalabeling_fitted._get_pred_features_and_metalabels(
        X_primary_aligned=manual_X_primary,
        y_primary_aligned=manual_y_primary,
        y=dollar_bar_labels_and_info,
    )
    y_metalabel = y_metalabel.loc[manual_X_primary.index]
    y_pred_manual = pd.Series(
        RandomForestClassifier(random_state=1, n_jobs=-1)
        .fit(manual_X_primary, y_metalabel)
        .predict(manual_X_primary),
        index=manual_X_primary.index,
    )

    # predict pipeline with metalabeling
    y_pred_pipeline = strategy_signal_pipeline_with_metalabeling_fitted.predict(
        X_dict={
            "primary_0": dollar_bar_dataframe,
            "metalabeling_0": dollar_bar_dataframe,
        },
    )

    pd.testing.assert_series_equal(
        y_pred_manual,
        y_pred_pipeline[StrategySignalPipeline._metalabeling_model_predictions].loc[
            y_pred_manual.index
        ],
    )


def test_pipeline_predict_metalabeling_with_pred_features(
    dollar_bar_dataframe,
    dollar_bar_labels_and_info,
    strategy_signal_pipeline_with_metalabeling_and_pred_features_fitted,
    manual_X_y_primary,
    manual_primary_model_fitted,
):
    # deep copy of the model to make sure that tests are not pointing to the
    # same object when run in parallel
    strategy_signal_pipeline_with_metalabeling_and_pred_features_fitted = copy.deepcopy(
        strategy_signal_pipeline_with_metalabeling_and_pred_features_fitted
    )
    manual_primary_model_fitted = copy.deepcopy(manual_primary_model_fitted)

    manual_X_primary, manual_y_primary = manual_X_y_primary
    # manual metalabel fit
    (
        X_pred_features,
        y_metalabel,
    ) = strategy_signal_pipeline_with_metalabeling_and_pred_features_fitted._get_pred_features_and_metalabels(
        X_primary_aligned=manual_X_primary,
        y_primary_aligned=manual_y_primary,
        y=dollar_bar_labels_and_info,
    )
    X_metalabel = pd.concat([manual_X_primary, X_pred_features], axis=1).dropna()
    y_metalabel = y_metalabel.loc[X_metalabel.index]
    metalabing_model = RandomForestClassifier(random_state=1, n_jobs=-1).fit(
        X_metalabel, y_metalabel
    )
    manual_X_primary = manual_X_primary.loc[X_metalabel.index]

    # manual metalabel predict
    manual_primary_predict = manual_primary_model_fitted.predict(manual_X_primary)
    manual_primary_predict_proba = manual_primary_model_fitted.predict_proba(
        manual_X_primary
    )
    X_metalabel_predict = np.concatenate(
        [
            manual_X_primary.values,
            manual_primary_predict.reshape(-1, 1),
            manual_primary_predict_proba,
        ],
        axis=1,
    )
    y_pred_manual = pd.Series(
        metalabing_model.predict(X_metalabel_predict),
        index=X_metalabel.index,
    )

    # predict pipeline with metalabeling
    y_pred_pipeline = (
        strategy_signal_pipeline_with_metalabeling_and_pred_features_fitted.predict(
            X_dict={
                "primary_0": dollar_bar_dataframe,
                "metalabeling_0": dollar_bar_dataframe,
            },
        )
    )

    # compare output
    pd.testing.assert_series_equal(
        y_pred_manual,
        y_pred_pipeline[StrategySignalPipeline._metalabeling_model_predictions].loc[
            y_pred_manual.index
        ],
    )


def test_pipeline_predict_proba_only_primary_model(
    dollar_bar_dataframe,
    dollar_bar_labels_and_info,
    strategy_signal_pipeline_only_primary,
    manual_X_y_primary,
):
    # deep copy of the model to make sure that tests are not pointing to the
    # same object when run in parallel
    strategy_signal_pipeline_only_primary = copy.deepcopy(
        strategy_signal_pipeline_only_primary
    )

    strategy_signal_pipeline_only_primary.fit(
        X_dict={"primary_0": dollar_bar_dataframe}, y=dollar_bar_labels_and_info
    )

    check_is_fitted(strategy_signal_pipeline_only_primary.primary_model)

    manual_X_primary, manual_y_primary = manual_X_y_primary
    # manual fit and predict

    rf = RandomForestClassifier(random_state=1, n_jobs=-1)
    rf.fit(manual_X_primary, manual_y_primary)
    y_pred_manual = pd.DataFrame(
        rf.predict_proba(manual_X_primary),
        index=manual_X_primary.index,
        columns=rf.classes_,
    )

    y_pred_pipeline = strategy_signal_pipeline_only_primary.predict_proba(
        X_dict={"primary_0": dollar_bar_dataframe}
    )
    pd.testing.assert_frame_equal(
        y_pred_manual,
        y_pred_pipeline[StrategySignalPipeline._primary_model_proba].loc[
            y_pred_manual.index
        ],
    )


def test_pipeline_predict_proba_metalabeling(
    dollar_bar_dataframe,
    dollar_bar_labels_and_info,
    strategy_signal_pipeline_with_metalabeling_fitted,
    manual_X_y_primary,
):

    # deep copy of the model to make sure that tests are not pointing to the
    # same object when run in parallel
    strategy_signal_pipeline_with_metalabeling_fitted = copy.deepcopy(
        strategy_signal_pipeline_with_metalabeling_fitted
    )

    manual_X_primary, manual_y_primary = manual_X_y_primary
    # manual calculation of predicted probabilities
    (
        X_pred_features,
        y_metalabel,
    ) = strategy_signal_pipeline_with_metalabeling_fitted._get_pred_features_and_metalabels(
        X_primary_aligned=manual_X_primary,
        y_primary_aligned=manual_y_primary,
        y=dollar_bar_labels_and_info,
    )
    y_metalabel = y_metalabel.loc[manual_X_primary.index]
    rf = RandomForestClassifier(random_state=1, n_jobs=-1)
    rf.fit(manual_X_primary, y_metalabel)
    y_proba_manual = pd.DataFrame(
        rf.predict_proba(manual_X_primary),
        columns=rf.classes_,
        index=manual_X_primary.index,
    )

    # predict pipeline with metalabeling
    y_proba_pipeline = strategy_signal_pipeline_with_metalabeling_fitted.predict_proba(
        X_dict={
            "primary_0": dollar_bar_dataframe,
            "metalabeling_0": dollar_bar_dataframe,
        },
    )

    pd.testing.assert_frame_equal(
        y_proba_manual,
        y_proba_pipeline[StrategySignalPipeline._metalabeling_model_proba].loc[
            manual_X_primary.index
        ],
    )


def test_pipeline_predict_proba_metalabeling_with_pred_features(
    dollar_bar_dataframe,
    dollar_bar_labels_and_info,
    strategy_signal_pipeline_with_metalabeling_and_pred_features_fitted,
    manual_X_y_primary,
    manual_primary_model_fitted,
):
    # deep copy of the model to make sure that tests are not pointing to the
    # same object when run in parallel
    strategy_signal_pipeline_with_metalabeling_and_pred_features_fitted = copy.deepcopy(
        strategy_signal_pipeline_with_metalabeling_and_pred_features_fitted
    )
    manual_primary_model_fitted = copy.deepcopy(manual_primary_model_fitted)

    manual_X_primary, manual_y_primary = manual_X_y_primary
    # manual metalabel fit
    (
        X_pred_features,
        y_metalabel,
    ) = strategy_signal_pipeline_with_metalabeling_and_pred_features_fitted._get_pred_features_and_metalabels(
        X_primary_aligned=manual_X_primary,
        y_primary_aligned=manual_y_primary,
        y=dollar_bar_labels_and_info,
    )
    X_metalabel = pd.concat([manual_X_primary, X_pred_features], axis=1).dropna()
    y_metalabel = y_metalabel.loc[X_metalabel.index]
    metalabing_model = RandomForestClassifier(random_state=1, n_jobs=-1).fit(
        X_metalabel, y_metalabel
    )

    # manual metalabel predict
    manual_primary_predict = manual_primary_model_fitted.predict(manual_X_primary)
    manual_primary_predict_proba = manual_primary_model_fitted.predict_proba(
        manual_X_primary
    )
    X_metalabel_predict = np.concatenate(
        [
            manual_X_primary.values,
            manual_primary_predict.reshape(-1, 1),
            manual_primary_predict_proba,
        ],
        axis=1,
    )

    y_proba_manual = pd.DataFrame(
        metalabing_model.predict_proba(X_metalabel_predict),
        columns=metalabing_model.classes_,
        index=manual_X_primary.index,
    )

    # predict pipeline with metalabeling
    y_pred_pipeline = strategy_signal_pipeline_with_metalabeling_and_pred_features_fitted.predict_proba(
        X_dict={
            "primary_0": dollar_bar_dataframe,
            "metalabeling_0": dollar_bar_dataframe,
        },
    )

    # compare output
    pd.testing.assert_frame_equal(
        y_proba_manual,
        y_pred_pipeline[StrategySignalPipeline._metalabeling_model_proba].loc[
            y_proba_manual.index
        ],
    )


def test_pipeline_get_size_only_primary_model(
    dollar_bar_dataframe,
    strategy_signal_pipeline_only_primary_fitted,
    manual_primary_model_fitted,
    manual_X_primary,
):

    # deep copy of the model to make sure that tests are not pointing to the
    # same object when run in parallel
    strategy_signal_pipeline_only_primary_fitted = copy.deepcopy(
        strategy_signal_pipeline_only_primary_fitted
    )
    manual_primary_model_fitted = copy.deepcopy(manual_primary_model_fitted)

    # manual size calculation
    y_proba_manual = pd.DataFrame(
        manual_primary_model_fitted.predict_proba(manual_X_primary),
        index=manual_X_primary.index,
        columns=manual_primary_model_fitted.classes_,
    )

    size_manual = y_proba_manual.max(axis=1)

    size_pipeline = strategy_signal_pipeline_only_primary_fitted.get_size(
        X_dict={"primary_0": dollar_bar_dataframe}
    )
    pd.testing.assert_series_equal(size_manual, size_pipeline)


def test_pipeline_get_size_metalabeling(
    dollar_bar_dataframe,
    dollar_bar_labels_and_info,
    strategy_signal_pipeline_with_metalabeling_fitted,
    manual_X_y_primary,
):

    # deep copy of the model to make sure that tests are not pointing to the
    # same object when run in parallel
    strategy_signal_pipeline_with_metalabeling_fitted = copy.deepcopy(
        strategy_signal_pipeline_with_metalabeling_fitted
    )

    # manual fit and predict
    manual_X_primary, manual_y_primary = manual_X_y_primary

    (
        X_pred_features,
        y_metalabel,
    ) = strategy_signal_pipeline_with_metalabeling_fitted._get_pred_features_and_metalabels(
        X_primary_aligned=manual_X_primary,
        y_primary_aligned=manual_y_primary,
        y=dollar_bar_labels_and_info,
    )
    y_metalabel = y_metalabel.loc[manual_X_primary.index]
    rf = RandomForestClassifier(random_state=1, n_jobs=-1)
    rf.fit(manual_X_primary, y_metalabel)
    y_proba_manual = pd.DataFrame(
        rf.predict_proba(manual_X_primary),
        columns=rf.classes_,
        index=manual_X_primary.index,
    )

    size_manual = y_proba_manual[1.0]

    # predict pipeline with metalabeling
    y_size_pipeline = strategy_signal_pipeline_with_metalabeling_fitted.get_size(
        X_dict={
            "primary_0": dollar_bar_dataframe,
            "metalabeling_0": dollar_bar_dataframe,
        },
    )

    pd.testing.assert_series_equal(
        size_manual, y_size_pipeline.loc[manual_X_primary.index]
    )


def test_pipeline_get_size_primary_with_bet_sizer(
    dollar_bar_dataframe,
    dollar_bar_labels_and_info,
    bar_feature_generator,
    manual_X_primary,
    manual_primary_model_fitted,
):

    # deep copy of the model to make sure that tests are not pointing to the
    # same object when run in parallel
    manual_primary_model_fitted = copy.deepcopy(manual_primary_model_fitted)

    # init dummy models
    primary_model = RandomForestClassifier(random_state=1, n_jobs=-1)

    # create dict structure for primary model feature generator
    feature_transformers_primary_model = {"primary_0": bar_feature_generator}

    bet_sizer = BetSizingFromProbabilities(2)
    pipeline = StrategySignalPipeline(
        primary_model=primary_model,
        feature_transformers_primary_model=feature_transformers_primary_model,
        bet_sizer=bet_sizer,
    )

    pipeline.fit(
        X_dict={"primary_0": dollar_bar_dataframe},
        y=dollar_bar_labels_and_info,
    )
    size_pipeline = pipeline.get_size(X_dict={"primary_0": dollar_bar_dataframe})

    # manual calculation
    pred_proba = pd.DataFrame(
        data={
            "pred": manual_primary_model_fitted.predict(manual_X_primary),
            "prob": manual_primary_model_fitted.predict_proba(manual_X_primary).max(
                axis=1
            ),
        },
        index=manual_X_primary.index,
    )
    size_manual = bet_sizer.transform(pred_proba)

    # check if output is the same
    pd.testing.assert_series_equal(size_manual, size_pipeline)


def test_pipeline_get_size_metalabeling_with_bet_sizer(
    dollar_bar_dataframe,
    dollar_bar_labels_and_info,
    strategy_signal_pipeline_with_metalabeling_fitted,
    manual_X_y_primary,
    manual_primary_model_fitted,
):

    # deep copy of the model to make sure that tests are not pointing to the
    # same object when run in parallel
    strategy_signal_pipeline_with_metalabeling_fitted = copy.deepcopy(
        strategy_signal_pipeline_with_metalabeling_fitted
    )
    manual_primary_model_fitted = copy.deepcopy(manual_primary_model_fitted)

    manual_X_primary, manual_y_primary = manual_X_y_primary
    bet_sizer = BetSizingFromProbabilities(2, meta_labeling=True)
    strategy_signal_pipeline_with_metalabeling_fitted.bet_sizer = bet_sizer
    size_pipeline = strategy_signal_pipeline_with_metalabeling_fitted.get_size(
        X_dict={
            "primary_0": dollar_bar_dataframe,
            "metalabeling_0": dollar_bar_dataframe,
        },
    )

    # manual calculation of size
    (
        _,
        y_metalabel,
    ) = strategy_signal_pipeline_with_metalabeling_fitted._get_pred_features_and_metalabels(
        X_primary_aligned=manual_X_primary,
        y_primary_aligned=manual_y_primary,
        y=dollar_bar_labels_and_info,
    )
    y_metalabel = y_metalabel.loc[manual_X_primary.index]
    rf = RandomForestClassifier(random_state=1, n_jobs=-1)
    rf.fit(manual_X_primary, y_metalabel)

    manual_primary_side_pred = manual_primary_model_fitted.predict(manual_X_primary)
    manual_metalabel_pred = rf.predict(manual_X_primary)
    manual_metalabel_proba = rf.predict_proba(manual_X_primary)
    side_prob_pred = pd.DataFrame(
        data={
            "side": manual_primary_side_pred,
            "prob": np.max(manual_metalabel_proba, axis=1),
            "pred": manual_metalabel_pred,
        },
        index=manual_X_primary.index,
    )
    size_manual = bet_sizer.transform(side_prob_pred)

    # compare outputs
    pd.testing.assert_series_equal(
        size_manual, size_pipeline.loc[manual_X_primary.index]
    )


def test_pipeline_get_side(
    dollar_bar_dataframe,
    strategy_signal_pipeline_with_metalabeling_fitted,
    manual_X_y_primary,
    manual_primary_model_fitted,
):

    # deep copy of the model to make sure that tests are not pointing to the
    # same object when run in parallel
    strategy_signal_pipeline_with_metalabeling_fitted = copy.deepcopy(
        strategy_signal_pipeline_with_metalabeling_fitted
    )
    manual_primary_model_fitted = copy.deepcopy(manual_primary_model_fitted)

    manual_X_primary, manual_y_primary = manual_X_y_primary
    # manual predict
    side_manual = pd.Series(
        manual_primary_model_fitted.predict(manual_X_primary),
        index=manual_X_primary.index,
    )

    # predict pipeline with metalabeling
    side_pipeline = strategy_signal_pipeline_with_metalabeling_fitted.get_side(
        X_dict={
            "primary_0": dollar_bar_dataframe,
            "metalabeling_0": dollar_bar_dataframe,
        },
    )

    pd.testing.assert_series_equal(side_manual, side_pipeline.loc[side_manual.index])


def test_indices_for_fitting(
    dollar_bar_dataframe,
    strategy_signal_pipeline_only_primary,
    manual_X_y_primary,
    dollar_bar_labels_and_info,
):
    # deep copy of the model to make sure that tests are not pointing to the
    # same object when run in parallel
    strategy_signal_pipeline_only_primary_copy = copy.deepcopy(
        strategy_signal_pipeline_only_primary
    )

    manual_X_primary, manual_y_primary = manual_X_y_primary
    downsampled_indices = (
        dollar_bar_dataframe.sample(
            n=100,
            replace=False,
        )
        .sort_index()
        .index
    )

    intersected_downsampled_indices = sorted(
        list(set(manual_X_primary.index).intersection(set(downsampled_indices)))
    )

    manual_X_primary_downsampled = manual_X_primary.loc[intersected_downsampled_indices]

    dollar_bar_labels_and_info_downsampled = dollar_bar_labels_and_info.loc[
        manual_X_primary_downsampled.index
    ]

    rf = RandomForestClassifier(random_state=1, n_jobs=-1)

    rf.fit(manual_X_primary_downsampled, dollar_bar_labels_and_info_downsampled[LABEL])
    manual_predictions = rf.predict(manual_X_primary)
    manual_proba = rf.predict_proba(manual_X_primary)

    strategy_signal_pipeline_only_primary_copy.fit(
        X_dict={"primary_0": dollar_bar_dataframe},
        y=dollar_bar_labels_and_info,
        downsampled_indices_primary=downsampled_indices,
    )

    pipeline_predictions = strategy_signal_pipeline_only_primary_copy.predict(
        X_dict={"primary_0": dollar_bar_dataframe}
    )

    pipeline_proba = strategy_signal_pipeline_only_primary_copy.predict_proba(
        X_dict={"primary_0": dollar_bar_dataframe}
    )

    np.testing.assert_array_equal(
        manual_predictions,
        pipeline_predictions["primary_model_predictions"]
        .loc[manual_X_primary.index]
        .values,
    )

    np.testing.assert_array_equal(
        manual_proba,
        pipeline_proba["primary_model_proba"].loc[manual_X_primary.index].values,
    )


def test_align_on_indices(
    time_bar_dataframe,
    dollar_bar_dataframe,
    bar_feature_generator,
    dollar_bar_labels_and_info,
):
    primary_model = RandomForestClassifier(random_state=1, n_jobs=-1)
    metalabeling_model = RandomForestClassifier(random_state=1, n_jobs=-1)
    # create dict structure for primary model feature generator
    feature_transformers_primary_model = {
        "primary_0": bar_feature_generator,
        "primary_1": bar_feature_generator,
    }

    feature_transformers_metalabeling_model = {
        "metalabeling_0": bar_feature_generator,
        "metalabeling_1": bar_feature_generator,
    }

    pipeline = StrategySignalPipeline(
        primary_model=primary_model,
        metalabeling_model=metalabeling_model,
        feature_transformers_primary_model=feature_transformers_primary_model,
        feature_transformers_metalabeling_model=feature_transformers_metalabeling_model,
        embargo_td=pd.Timedelta(days=5),
        metalabeling_use_predictions_primary_model=False,
        metalabeling_use_proba_primary_model=False,
    )

    X_dict = {
        "primary_0": dollar_bar_dataframe,
        "primary_1": time_bar_dataframe,
        "metalabeling_0": dollar_bar_dataframe,
        "metalabeling_1": time_bar_dataframe,
    }

    pipeline.fit(
        X_dict=X_dict,
        y=dollar_bar_labels_and_info,
        align_on="primary_0",
        align_how={
            "primary_1": "mean",
            "metalabeling_0": "mean",
            "metalabeling_1": "mean",
        },
    )
    X_dict_transformed = pipeline.transform(X_dict)
    X_dict_aligned = pipeline._align_on(copy.deepcopy(X_dict_transformed))

    pd.testing.assert_frame_equal(
        X_dict_aligned[pipeline._primary_model_features]["primary_0"],
        X_dict_transformed[pipeline._primary_model_features]["primary_0"],
    )

    pd.testing.assert_series_equal(
        X_dict_aligned[pipeline._primary_model_features]["primary_1"].index.to_series(),
        X_dict_transformed[pipeline._primary_model_features][
            "primary_0"
        ].index.to_series(),
    )

    pd.testing.assert_series_equal(
        X_dict_aligned[pipeline._metalabeling_model_features][
            "metalabeling_0"
        ].index.to_series(),
        X_dict_transformed[pipeline._primary_model_features][
            "primary_0"
        ].index.to_series(),
    )

    pd.testing.assert_series_equal(
        X_dict_aligned[pipeline._metalabeling_model_features][
            "metalabeling_1"
        ].index.to_series(),
        X_dict_transformed[pipeline._primary_model_features][
            "primary_0"
        ].index.to_series(),
    )


@pytest.mark.parametrize("how", ["mean", "max", "min"])
def test_align_on_only_primary(strategy_signal_pipeline_only_primary, how):
    # deep copy of the model to make sure that tests are not pointing to the
    # same object when run in parallel
    strategy_signal_pipeline_only_primary = copy.deepcopy(
        strategy_signal_pipeline_only_primary
    )

    how_methods = {
        "mean": np.mean,
        "min": np.min,
        "max": np.max,
    }

    strategy_signal_pipeline_only_primary.align_on_ = "primary_0"
    strategy_signal_pipeline_only_primary.align_how_ = {"primary_1": how}

    primary_0 = pd.DataFrame(
        {1: 1, 2: 1}, index=pd.date_range(start="1/2/2020", end="1/2/2020", freq="1D")
    )

    primary_1 = pd.DataFrame(
        {1: [1, 3], 2: [0.5, 1.5]},
        index=pd.date_range(start="1/1/2020", periods=2, freq="12H"),
    )

    primary_1.index += pd.Timedelta(minutes=1)

    X_dict_features = {
        strategy_signal_pipeline_only_primary._primary_model_features: {
            "primary_0": primary_0,
            "primary_1": primary_1,
        },
        strategy_signal_pipeline_only_primary._metalabeling_model_features: {},
    }

    X_dict_aligned = strategy_signal_pipeline_only_primary._align_on(X_dict_features)

    np.testing.assert_array_equal(
        X_dict_aligned[strategy_signal_pipeline_only_primary._primary_model_features][
            "primary_1"
        ].values,
        how_methods[how](primary_1, axis=0).to_frame().T.values,
        err_msg=f"The alignment with the method {how} did not produced the expected outcome",
    )

    primary_0 = pd.DataFrame(
        {1: [1, 3], 2: [0.5, 1.5]},
        index=pd.date_range(start="1/1/2020", periods=2, freq="12H"),
    )

    primary_1 = pd.DataFrame(
        {1: [1], 2: [0.5]}, index=pd.date_range(start="1/1/2020", periods=1, freq="1D")
    )

    X_dict_features = {
        strategy_signal_pipeline_only_primary._primary_model_features: {
            "primary_0": primary_0,
            "primary_1": primary_1,
        },
        strategy_signal_pipeline_only_primary._metalabeling_model_features: {},
    }

    X_dict_aligned = strategy_signal_pipeline_only_primary._align_on(X_dict_features)

    np.testing.assert_array_equal(
        X_dict_aligned[strategy_signal_pipeline_only_primary._primary_model_features][
            "primary_1"
        ].values,
        np.array([[1.0, 0.5], [1.0, 0.5]]),
        err_msg="The values for the primary 1 should be forward filled",
    )

    primary_0 = pd.DataFrame(
        {1: [1.0, 3.0], 2: [0.5, 1.5]},
        index=pd.date_range(start="1/1/2020", periods=2, freq="12H"),
    )

    primary_1 = pd.DataFrame(
        {1: [1.0, 3.0], 2: [0.5, 1.5]},
        index=pd.date_range(start="1/1/2020", periods=2, freq="12H"),
    )

    X_dict_features = {
        strategy_signal_pipeline_only_primary._primary_model_features: {
            "primary_0": primary_0,
            "primary_1": primary_1,
        },
        strategy_signal_pipeline_only_primary._metalabeling_model_features: {},
    }

    X_dict_aligned = strategy_signal_pipeline_only_primary._align_on(X_dict_features)

    # The primary 1 before alignment and after should be exactly the same
    pd.testing.assert_frame_equal(
        X_dict_aligned[strategy_signal_pipeline_only_primary._primary_model_features][
            "primary_1"
        ],
        primary_1,
    )
