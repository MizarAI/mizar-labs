import datetime
import string
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
from numpy.random import randn
from scipy import sparse
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier

from mizarlabs.model.bootstrapping import get_ind_matrix
from mizarlabs.model.model_selection import BaseTimeSeriesCrossValidator
from mizarlabs.model.pipeline import MizarFeatureUnion
from mizarlabs.model.pipeline import StrategySignalPipeline
from mizarlabs.static import CLOSE
from mizarlabs.static import EVENT_END_TIME
from mizarlabs.static import FIRST_TIMESTAMP
from mizarlabs.static import LABEL
from mizarlabs.static import SIDE
from mizarlabs.static import TIMESTAMP
from mizarlabs.transformers.microstructural_features.vpin import VPIN
from mizarlabs.transformers.targets.labeling import TripleBarrierMethodLabeling
from mizarlabs.transformers.technical.moving_average import MovingAverageCrossOver
from mizarlabs.transformers.technical.moving_average import (
    MovingAverageCrossOverPredictor,
)
from mizarlabs.transformers.trading.bet_sizing import BET_SIZE
from mizarlabs.transformers.trading.bet_sizing import PREDICTION
from mizarlabs.transformers.trading.bet_sizing import PROBABILITY


def prepare_cv_object(
    cv: BaseTimeSeriesCrossValidator,
    n_samples: int,
    time_shift: str,
    randomlize_times: bool,
):
    X, pred_times, eval_times = create_random_sample_set(
        n_samples=n_samples, time_shift=time_shift, randomize_times=randomlize_times
    )
    cv.X = X
    cv.pred_times = pred_times
    cv.eval_times = eval_times
    cv.indices = np.arange(X.shape[0])


def create_test_df(n_samples: int, n_columns: int, freq: str):
    index = pd.date_range(start="2018-01-01", periods=n_samples, freq=freq)
    data = {
        col: [randn() for _ in range(n_samples)]
        for col in string.ascii_uppercase[:n_columns]
    }
    test_df = pd.DataFrame(data, index=index)
    return test_df


def create_random_sample_set(
    n_samples, time_shift="120m", randomize_times=False, freq="60T"
):
    # Create artificial data
    n_columns = 3
    test_df = create_test_df(n_samples, n_columns, freq)

    # Turn the index into a column labeled 'index'
    test_df = test_df.reset_index()
    if randomize_times:
        n_columns = 1
        # Subtract and adds random time deltas to the index column, to create
        # the prediction and evaluation times
        rand_fact = (
            create_test_df(n_samples, n_columns, freq)
            .reset_index(drop=True)
            .squeeze()
            .iloc[: len(test_df)]
            .abs()
        )
        test_df["index"] = test_df["index"].subtract(
            rand_fact.apply(lambda x: x * pd.Timedelta(time_shift))
        )
        rand_fact = (
            create_test_df(n_samples, n_columns, freq)
            .reset_index(drop=True)
            .squeeze()
            .iloc[: len(test_df)]
            .abs()
        )
        test_df["index2"] = test_df["index"].add(
            rand_fact.apply(lambda x: x * pd.Timedelta(time_shift))
        )
    else:
        test_df["index2"] = test_df["index"].apply(
            lambda x: x + pd.Timedelta(time_shift)
        )
    # Sort the data frame by prediction time
    test_df = test_df.sort_values("index")
    X = test_df[["A", "B", "C"]]
    pred_times = test_df["index"]
    exit_times = test_df["index2"]
    return X, pred_times, exit_times


@pytest.fixture
def time_bar_dataframe() -> pd.DataFrame:
    time_bar_dataframe = pd.read_pickle(
        Path(__file__).parent / "data/bar_dataframe.pkl"
    )
    object_columns = time_bar_dataframe.select_dtypes(include="object").columns
    time_bar_dataframe.loc[:, object_columns] = time_bar_dataframe.loc[
        :, object_columns
    ].astype(float)
    time_bar_dataframe.set_index(
        time_bar_dataframe.index.tz_convert(None), inplace=True
    )

    return time_bar_dataframe


@pytest.fixture
def samples_info_sets(dollar_bar_target_labels: pd.DataFrame):
    return dollar_bar_target_labels[EVENT_END_TIME]


@pytest.fixture
def dollar_bar_labels_and_info(dollar_bar_dataframe):
    # create labels
    tbml = TripleBarrierMethodLabeling(
        num_expiration_bars=5, profit_taking_factor=0.1, stop_loss_factor=0.1
    )
    labels_and_info = tbml.transform(dollar_bar_dataframe)
    return labels_and_info


@pytest.fixture
def dollar_bar_dataframe() -> pd.DataFrame:
    dollar_bar_dataframe = pd.read_pickle(
        Path(__file__).parent / "data/dollar_bar_dataframe.pkl"
    )

    object_columns = dollar_bar_dataframe.select_dtypes(include="object").columns
    dollar_bar_dataframe.loc[:, object_columns] = dollar_bar_dataframe.loc[
        :, object_columns
    ].astype(float)

    dollar_bar_dataframe.set_index(
        pd.to_datetime(
            dollar_bar_dataframe[FIRST_TIMESTAMP],
            utc=True,
            unit="ms",
        ),
        inplace=True,
    )
    dollar_bar_dataframe.index.rename(TIMESTAMP, inplace=True)
    dollar_bar_dataframe.set_index(
        dollar_bar_dataframe.index.tz_convert(None), inplace=True
    )

    return dollar_bar_dataframe


@pytest.fixture(params=[0, 10, 20, 30, 40, 50])
def dataframe_for_bet_sizing_testing_2_classes(request):
    size_factor = 1000

    prob_arr = np.array([0.711, 0.898, 0.992, 0.595, 0.544, 0.775] * size_factor)
    side_arr = np.array([1, 1, -1, 1, -1, 1] * size_factor)
    dates = np.array(
        [
            datetime.datetime(2000, 1, 1) + i * datetime.timedelta(days=1)
            for i in range(len(prob_arr))
        ]
    )

    shift_list = [0.5, 1, 2, 1.5, 0.8, 0.2] * 1000
    shift_dt = np.array([datetime.timedelta(days=d) for d in shift_list])
    dates_shifted = dates + shift_dt
    if request.param != 0:
        dates_shifted[-request.param :] = np.nan
    # Calculate the test statistic and bet size.
    z_test = (prob_arr - 0.5) / (prob_arr * (1 - prob_arr)) ** 0.5
    m_signal = side_arr * (2 * norm.cdf(z_test) - 1)

    df = pd.DataFrame.from_dict(
        {
            PROBABILITY: prob_arr,
            SIDE: side_arr,
            PREDICTION: side_arr,
            EVENT_END_TIME: dates_shifted,
            BET_SIZE: m_signal,
        }
    )
    return df.set_index(dates)


@pytest.fixture(params=[0, 10, 20, 30, 40, 50])
def dataframe_for_bet_sizing_testing_3_classes(request):
    size_factor = 1000

    prob_arr = np.array([0.711, 0.898, 0.992, 0.595, 0.544, 0.775] * size_factor)
    side_arr = np.array([1, 0, -1, 0, -1, 1] * size_factor)
    dates = np.array(
        [
            datetime.datetime(2000, 1, 1) + i * datetime.timedelta(days=1)
            for i in range(len(prob_arr))
        ]
    )

    shift_list = [0.5, 1, 2, 1.5, 0.8, 0.2] * 1000
    shift_dt = np.array([datetime.timedelta(days=d) for d in shift_list])
    dates_shifted = dates + shift_dt
    if request.param != 0:
        dates_shifted[-request.param :] = np.nan
    # Calculate the test statistic and bet size.
    z_test = (prob_arr - 0.5) / (prob_arr * (1 - prob_arr)) ** 0.5
    m_signal = side_arr * (2 * norm.cdf(z_test) - 1)

    df = pd.DataFrame.from_dict(
        {
            PROBABILITY: prob_arr,
            SIDE: side_arr,
            PREDICTION: side_arr,
            EVENT_END_TIME: dates_shifted,
            BET_SIZE: m_signal,
        }
    )
    return df.set_index(dates)


@pytest.fixture(scope="function")
def moving_average_crossover_fast_slow_one_up_crossover():
    index = pd.date_range("2020-01-01", periods=8, freq="d")

    fast = pd.Series([1, 1, 1, 1, 2, 3, 3, 4], index=index, name="fast")
    slow = pd.Series([2, 2, 2, 2, 1, 1, 2, 2], index=index, name="slow")

    return pd.DataFrame({"fast": fast, "slow": slow})


@pytest.fixture(scope="function")
def moving_average_crossover_fast_slow_one_down_crossover():
    index = pd.date_range("2020-01-01", periods=8, freq="d")

    fast = pd.Series([2, 2, 2, 2, 1, 1, 2, 2], index=index, name="fast")
    slow = pd.Series([1, 1, 1, 1, 2, 3, 3, 4], index=index, name="slow")

    return pd.DataFrame({"fast": fast, "slow": slow})


@pytest.fixture(scope="function")
def moving_average_crossover_fast_slow_multiple_crossovers():
    index = pd.date_range("2020-01-01", periods=18, freq="d")

    slow = pd.Series(
        [2, 2, 2, 2, 1, 1, 2, 2, 3, 3, 2, 1, 2, 3, 4, 3, 2, 1], index=index, name="slow"
    )
    fast = pd.Series(
        [1, 1, 1, 1, 2, 3, 3, 4, 2, 1, 2, 2, 3, 2, 1, 2, 3, 4], index=index, name="fast"
    )

    return pd.DataFrame({"fast": fast, "slow": slow})


@pytest.fixture(scope="function")
def moving_average_cross_over_transformer():
    fast = 10
    slow = 40
    return MovingAverageCrossOver(
        fast=fast, slow=slow, column_name=CLOSE, fill_between_crossovers=True
    )


@pytest.fixture(scope="function")
def moving_average_cross_over_predictor():
    fast = 10
    slow = 40
    return MovingAverageCrossOverPredictor(
        fast=fast, slow=slow, column_name=CLOSE, fill_between_crossovers=True
    )


@pytest.fixture
def dollar_bar_target_labels(dollar_bar_dataframe: pd.DataFrame) -> pd.DataFrame:
    tbml = TripleBarrierMethodLabeling(
        num_expiration_bars=10,
        profit_taking_factor=0.2,
        stop_loss_factor=0.2,
    )
    target_labels = tbml.transform(dollar_bar_dataframe[[CLOSE]])
    return target_labels.dropna()


@pytest.fixture
def dollar_bar_ind_matrix(dollar_bar_target_labels: pd.DataFrame) -> np.ndarray:
    indicators_matrix = get_ind_matrix(
        samples_info_sets=dollar_bar_target_labels[EVENT_END_TIME],
        price_bars=dollar_bar_target_labels,
        event_end_time_column_name=EVENT_END_TIME,
    )

    return indicators_matrix


@pytest.fixture
def dollar_bar_ind_matrix_indices(
    dollar_bar_target_labels: pd.DataFrame,
) -> List[pd.Timestamp]:
    _, dollar_bar_ind_matrix_indices = get_ind_matrix(
        samples_info_sets=dollar_bar_target_labels[EVENT_END_TIME],
        price_bars=dollar_bar_target_labels,
        event_end_time_column_name=EVENT_END_TIME,
        return_indices=True,
    )

    return dollar_bar_ind_matrix_indices


@pytest.fixture
def ind_matrix():
    ind_matrix = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ]
    )
    return sparse.lil_matrix(ind_matrix)


@pytest.fixture
def ind_matrix_csc(ind_matrix):
    return ind_matrix.tocsc()


@pytest.fixture
def X_train_perfect(
    dollar_bar_dataframe: pd.DataFrame, dollar_bar_target_labels: pd.DataFrame
):
    idx = 100
    X_train_perfect = dollar_bar_target_labels[[LABEL]].iloc[:idx]
    return X_train_perfect


@pytest.fixture
def X_train_random(
    dollar_bar_dataframe: pd.DataFrame, dollar_bar_target_labels: pd.DataFrame
):
    idx = 100
    return pd.DataFrame(
        np.random.random(idx),
        index=dollar_bar_target_labels.index[:idx],
        columns=["random_feature"],
    )


@pytest.fixture
def y_train(dollar_bar_dataframe: pd.DataFrame, dollar_bar_target_labels: pd.DataFrame):
    idx = 100
    return dollar_bar_target_labels[LABEL].iloc[:idx]


@pytest.fixture
def X_test_perfect(
    dollar_bar_dataframe: pd.DataFrame, dollar_bar_target_labels: pd.DataFrame
):
    idx = 100
    return dollar_bar_target_labels[[LABEL]].iloc[-idx:]


@pytest.fixture
def X_test_random(
    dollar_bar_dataframe: pd.DataFrame, dollar_bar_target_labels: pd.DataFrame
):
    idx = 100
    return pd.DataFrame(
        np.random.random(idx),
        index=dollar_bar_target_labels.index[-idx:],
        columns=["random_feature"],
    )


@pytest.fixture
def y_test(dollar_bar_dataframe: pd.DataFrame, dollar_bar_target_labels: pd.DataFrame):
    idx = 100
    return dollar_bar_target_labels[[LABEL]].iloc[-idx:]


@pytest.fixture
def time_inhomogeneous_data():
    """
    Creates a sample set consisting in 11 samples at 2h intervals,
    spanning 20h, as well as 10 samples at 59m intervals,
    with the first samples of each group occurring at the same time.
    pred_times and eval_times have the following values:
                pred_times          eval_times
    0  2000-01-01 00:00:00 2000-01-01 01:00:00
    1  2000-01-01 00:00:00 2000-01-01 01:00:00
    2  2000-01-01 00:59:00 2000-01-01 01:59:00
    3  2000-01-01 01:58:00 2000-01-01 02:58:00
    4  2000-01-01 02:00:00 2000-01-01 03:00:00
    5  2000-01-01 02:57:00 2000-01-01 03:57:00
    6  2000-01-01 03:56:00 2000-01-01 04:56:00
    7  2000-01-01 04:00:00 2000-01-01 05:00:00
    8  2000-01-01 04:55:00 2000-01-01 05:55:00
    9  2000-01-01 05:54:00 2000-01-01 06:54:00
    10 2000-01-01 06:00:00 2000-01-01 07:00:00
    11 2000-01-01 06:53:00 2000-01-01 07:53:00
    12 2000-01-01 07:52:00 2000-01-01 08:52:00
    13 2000-01-01 08:00:00 2000-01-01 09:00:00
    14 2000-01-01 08:51:00 2000-01-01 09:51:00
    15 2000-01-01 10:00:00 2000-01-01 11:00:00
    16 2000-01-01 12:00:00 2000-01-01 13:00:00
    17 2000-01-01 14:00:00 2000-01-01 15:00:00
    18 2000-01-01 16:00:00 2000-01-01 17:00:00
    19 2000-01-01 18:00:00 2000-01-01 19:00:00
    20 2000-01-01 20:00:00 2000-01-01 21:00:00
    """

    X1, pred_times1, eval_times1 = create_random_sample_set(
        n_samples=11, time_shift="1H", freq="2H"
    )
    X2, pred_times2, eval_times2 = create_random_sample_set(
        n_samples=10, time_shift="1H", freq="59T"
    )
    data1 = pd.concat([X1, pred_times1, eval_times1], axis=1)
    data2 = pd.concat([X2, pred_times2, eval_times2], axis=1)
    data = pd.concat([data1, data2], axis=0, ignore_index=True)
    data = data.sort_values(by=data.columns[3])
    data = data.reset_index(drop=True)
    X = data.iloc[:, 0:3]
    pred_times = data.iloc[:, 3]
    eval_times = data.iloc[:, 4]

    return X, pred_times, eval_times


@pytest.fixture
def strategy_signal_pipeline_only_primary(bar_feature_generator):
    # create dict structure for primary model feature generator
    feature_transformers_primary_model = {"primary_0": bar_feature_generator}

    pipeline = StrategySignalPipeline(
        feature_transformers_primary_model=feature_transformers_primary_model,
        align_on="primary_0",
        align_how={"primary_0": "mean"},
    )
    return pipeline


@pytest.fixture
def strategy_signal_pipeline_only_primary_fitted(
    strategy_signal_pipeline_only_primary,
    bar_feature_generator,
    x_dict_primary,
    dollar_bar_labels_and_info,
):
    # init dummy models
    primary_model = RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=2)

    X, y = strategy_signal_pipeline_only_primary.create_dataset_primary(
        x_dict_primary, dollar_bar_labels_and_info["label"]
    )

    primary_model.fit(X, y)

    return strategy_signal_pipeline_only_primary.set_primary_model(primary_model)


@pytest.fixture()
def bar_feature_generator_longer_warmup(dollar_bar_dataframe):
    # feature generator
    fast_window = 20
    slow_window = 100
    maco = MovingAverageCrossOver(
        fast=fast_window,
        slow=slow_window,
        column_name=CLOSE,
        fill_between_crossovers=True,
    )
    vpin = VPIN(window=100)
    bar_feature_generator = MizarFeatureUnion(
        [("MACO", maco), ("VPIN", vpin)], n_jobs=-1
    )
    return bar_feature_generator


@pytest.fixture
def bar_feature_generator(dollar_bar_dataframe):
    # feature generator
    fast_window = 10
    slow_window = 20
    maco = MovingAverageCrossOver(
        fast=fast_window,
        slow=slow_window,
        column_name=CLOSE,
        fill_between_crossovers=True,
    )
    vpin = VPIN()
    bar_feature_generator = MizarFeatureUnion(
        [("MACO", maco), ("VPIN", vpin)], n_jobs=-1
    )
    return bar_feature_generator


@pytest.fixture
def strategy_signal_pipeline_with_metalabeling(bar_feature_generator):
    feature_transformers_primary_model = {
        "primary_0": bar_feature_generator,
    }
    feature_transformers_metalabeling_model = {"metalabeling_0": bar_feature_generator}

    pipeline = StrategySignalPipeline(
        feature_transformers_primary_model=feature_transformers_primary_model,
        feature_transformers_metalabeling_model=feature_transformers_metalabeling_model,
        metalabeling_use_predictions_primary_model=False,
        metalabeling_use_proba_primary_model=False,
        align_on="primary_0",
        align_how={"metalabeling_0": "mean"},
    )
    return pipeline


@pytest.fixture
def strategy_signal_pipeline_with_metalabeling_fitted(
    strategy_signal_pipeline_with_metalabeling,
    x_dict_primary_metalabeling,
    dollar_bar_labels_and_info,
):
    primary_model = RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=2)
    metalabeling_model = RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=2)
    (
        X_primary,
        y_primary,
    ) = strategy_signal_pipeline_with_metalabeling.create_dataset_primary(
        x_dict_primary_metalabeling, dollar_bar_labels_and_info["label"]
    )
    primary_model.fit(X_primary, y_primary)
    strategy_signal_pipeline_with_metalabeling.set_primary_model(primary_model)
    (
        X_meta,
        y_meta,
    ) = strategy_signal_pipeline_with_metalabeling.create_dataset_metalabeling(
        x_dict_primary_metalabeling, dollar_bar_labels_and_info["label"]
    )
    metalabeling_model.fit(X_meta, y_meta)
    strategy_signal_pipeline_with_metalabeling.set_metalabeling_model(
        metalabeling_model
    )
    return strategy_signal_pipeline_with_metalabeling


@pytest.fixture
def strategy_signal_pipeline_with_metalabeling_longer_warm_up_primary(
    bar_feature_generator, bar_feature_generator_longer_warmup
):
    feature_transformers_primary_model = {
        "primary_0": bar_feature_generator_longer_warmup,
    }
    feature_transformers_metalabeling_model = {"metalabeling_0": bar_feature_generator}

    pipeline = StrategySignalPipeline(
        feature_transformers_primary_model=feature_transformers_primary_model,
        feature_transformers_metalabeling_model=feature_transformers_metalabeling_model,
        metalabeling_use_predictions_primary_model=False,
        metalabeling_use_proba_primary_model=False,
        align_on="primary_0",
        align_how={"metalabeling_0": "mean"},
    )
    return pipeline


@pytest.fixture
def strategy_signal_pipeline_with_metalabeling_longer_warm_up_primary_fitted(
    strategy_signal_pipeline_with_metalabeling_longer_warm_up_primary,
    x_dict_primary_metalabeling,
    dollar_bar_labels_and_info,
):
    primary_model = RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=2)
    metalabeling_model = RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=2)
    (
        X_primary,
        y_primary,
    ) = strategy_signal_pipeline_with_metalabeling_longer_warm_up_primary.create_dataset_primary(
        x_dict_primary_metalabeling, dollar_bar_labels_and_info["label"]
    )
    primary_model.fit(X_primary, y_primary)
    strategy_signal_pipeline_with_metalabeling_longer_warm_up_primary.set_primary_model(
        primary_model
    )
    (
        X_meta,
        y_meta,
    ) = strategy_signal_pipeline_with_metalabeling_longer_warm_up_primary.create_dataset_metalabeling(
        x_dict_primary_metalabeling, dollar_bar_labels_and_info["label"]
    )
    metalabeling_model.fit(X_meta, y_meta)
    strategy_signal_pipeline_with_metalabeling_longer_warm_up_primary.set_metalabeling_model(
        metalabeling_model
    )
    return strategy_signal_pipeline_with_metalabeling_longer_warm_up_primary


@pytest.fixture
def strategy_signal_pipeline_with_metalabeling_longer_warm_up_metalabeling(
    bar_feature_generator, bar_feature_generator_longer_warmup
):
    feature_transformers_primary_model = {
        "primary_0": bar_feature_generator,
    }
    feature_transformers_metalabeling_model = {
        "metalabeling_0": bar_feature_generator_longer_warmup
    }

    pipeline = StrategySignalPipeline(
        feature_transformers_primary_model=feature_transformers_primary_model,
        feature_transformers_metalabeling_model=feature_transformers_metalabeling_model,
        metalabeling_use_predictions_primary_model=False,
        metalabeling_use_proba_primary_model=False,
        align_on="primary_0",
        align_how={"metalabeling_0": "mean"},
    )
    return pipeline


@pytest.fixture
def strategy_signal_pipeline_with_metalabeling_longer_warm_up_metalabeling_fitted(
    strategy_signal_pipeline_with_metalabeling_longer_warm_up_metalabeling,
    x_dict_primary_metalabeling,
    dollar_bar_labels_and_info,
):
    primary_model = RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=2)
    metalabeling_model = RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=2)
    (
        X_primary,
        y_primary,
    ) = strategy_signal_pipeline_with_metalabeling_longer_warm_up_metalabeling.create_dataset_primary(
        x_dict_primary_metalabeling, dollar_bar_labels_and_info["label"]
    )
    primary_model.fit(X_primary, y_primary)
    strategy_signal_pipeline_with_metalabeling_longer_warm_up_metalabeling.set_primary_model(
        primary_model
    )
    (
        X_meta,
        y_meta,
    ) = strategy_signal_pipeline_with_metalabeling_longer_warm_up_metalabeling.create_dataset_metalabeling(
        x_dict_primary_metalabeling, dollar_bar_labels_and_info["label"]
    )
    metalabeling_model.fit(X_meta, y_meta)
    strategy_signal_pipeline_with_metalabeling_longer_warm_up_metalabeling.set_metalabeling_model(
        metalabeling_model
    )
    return strategy_signal_pipeline_with_metalabeling_longer_warm_up_metalabeling


@pytest.fixture
def x_dict_primary(dollar_bar_dataframe):
    return {"primary_0": dollar_bar_dataframe}


@pytest.fixture
def x_dict_primary_metalabeling(dollar_bar_dataframe):
    return {"primary_0": dollar_bar_dataframe, "metalabeling_0": dollar_bar_dataframe}


@pytest.fixture
def strategy_signal_pipeline_with_metalabeling_and_pred_features(bar_feature_generator):
    # create dict structure for primary model feature generator
    # and metalabeling_model
    feature_transformers_primary_model = {
        "primary_0": bar_feature_generator,
    }
    feature_transformers_metalabeling_model = {"metalabeling_0": bar_feature_generator}

    pipeline = StrategySignalPipeline(
        feature_transformers_primary_model=feature_transformers_primary_model,
        feature_transformers_metalabeling_model=feature_transformers_metalabeling_model,
        metalabeling_use_predictions_primary_model=True,
        metalabeling_use_proba_primary_model=True,
        align_on="primary_0",
        align_how={"metalabeling_0": "mean"},
    )
    return pipeline


@pytest.fixture
def strategy_signal_pipeline_with_metalabeling_and_pred_features_fitted(
    strategy_signal_pipeline_with_metalabeling_and_pred_features,
    x_dict_primary_metalabeling,
    dollar_bar_labels_and_info,
):
    primary_model = RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=2)
    metalabeling_model = RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=2)
    (
        X_primary,
        y_primary,
    ) = strategy_signal_pipeline_with_metalabeling_and_pred_features.create_dataset_primary(
        x_dict_primary_metalabeling, dollar_bar_labels_and_info["label"]
    )
    primary_model.fit(X_primary, y_primary)
    strategy_signal_pipeline_with_metalabeling_and_pred_features.set_primary_model(
        primary_model
    )
    (
        X_meta,
        y_meta,
    ) = strategy_signal_pipeline_with_metalabeling_and_pred_features.create_dataset_metalabeling(
        x_dict_primary_metalabeling, dollar_bar_labels_and_info["label"]
    )
    metalabeling_model.fit(X_meta, y_meta)
    return strategy_signal_pipeline_with_metalabeling_and_pred_features.set_metalabeling_model(
        metalabeling_model
    )


@pytest.fixture
def manual_X_y_primary(manual_X_primary, manual_y_primary):
    manual_y_primary.dropna(inplace=True)
    manual_X_primary = manual_X_primary.loc[manual_y_primary.index]
    return manual_X_primary, manual_y_primary


@pytest.fixture
def manual_X_primary(dollar_bar_dataframe, bar_feature_generator):
    return bar_feature_generator.transform(dollar_bar_dataframe).dropna()


@pytest.fixture
def manual_y_primary(dollar_bar_labels_and_info, manual_X_primary):
    return dollar_bar_labels_and_info.loc[manual_X_primary.index, "label"].dropna()


@pytest.fixture
def manual_primary_model_fitted(manual_X_y_primary):
    return RandomForestClassifier(random_state=1, max_depth=2).fit(
        manual_X_y_primary[0], manual_X_y_primary[1]
    )
