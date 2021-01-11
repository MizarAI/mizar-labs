import datetime
import pandas as pd

from mizarlabs.transformers.technical.moving_average import ExponentialWeightedMovingAverageDifference


def test_moving_average_crossover_only_one_up(
    moving_average_crossover_fast_slow_one_up_crossover,
    moving_average_cross_over_transformer,
):
    """
    Test the moving overage crossver when there is only one up crossover

    :param moving_average_crossover_fast_slow_one_up_crossover: Dataframe with
        fast and slow moving averages
    :type moving_average_crossover_fast_slow_one_up_crossover: pd.DataFrame
    :param moving_average_cross_over_transformer: Moving average transformer
    :type moving_average_cross_over_transformer: MovingAverageCrossOver
    :return None
    """
    fast, slow = (
        moving_average_crossover_fast_slow_one_up_crossover["fast"],
        moving_average_crossover_fast_slow_one_up_crossover["slow"],
    )

    cross_over_up_indices = moving_average_cross_over_transformer._get_up_cross_indices(
        fast, slow
    )
    cross_over_down_indices = (
        moving_average_cross_over_transformer._get_down_cross_indices(fast, slow)
    )

    assert len(cross_over_up_indices) == 1, "There should be only one up crossover"
    assert cross_over_up_indices[0].date() == datetime.date(year=2020, month=1, day=5)
    assert len(cross_over_down_indices) == 0, "There should not be down crossover"


def test_moving_average_crossover_only_one_down(
    moving_average_crossover_fast_slow_one_down_crossover,
    moving_average_cross_over_transformer,
):
    """
    Test the moving overage crossver when there is only one down crossover

    :param moving_average_crossover_fast_slow_one_down_crossover: Dataframe with
        fast and slow moving averages
    :type moving_average_crossover_fast_slow_one_down_crossover: pd.DataFrame
    :param moving_average_cross_over_transformer: Moving average transformer
    :type moving_average_cross_over_transformer: MovingAverageCrossOver
    :return None
    """
    fast, slow = (
        moving_average_crossover_fast_slow_one_down_crossover["fast"],
        moving_average_crossover_fast_slow_one_down_crossover["slow"],
    )

    cross_over_up_indices = moving_average_cross_over_transformer._get_up_cross_indices(
        fast, slow
    )
    cross_over_down_indices = (
        moving_average_cross_over_transformer._get_down_cross_indices(fast, slow)
    )

    assert len(cross_over_down_indices) == 1, "There should be only one down crossover"
    assert cross_over_down_indices[0].date() == datetime.date(year=2020, month=1, day=5)

    assert len(cross_over_up_indices) == 0, "There should not be up crossover"


def test_moving_average_crossover_multiple_crossovers(
    moving_average_crossover_fast_slow_multiple_crossovers,
    moving_average_cross_over_transformer,
):
    """
    Test the moving overage crossver when there are multiple crossovers

    :param moving_average_crossover_fast_slow_multiple_crossovers: Dataframe
        with fast and slow moving averages
    :type moving_average_crossover_fast_slow_multiple_crossovers: pd.DataFrame
    :param moving_average_cross_over_transformer: Moving average transformer
    :type moving_average_cross_over_transformer: MovingAverageCrossOver
    :return None
    """
    fast, slow = (
        moving_average_crossover_fast_slow_multiple_crossovers["fast"],
        moving_average_crossover_fast_slow_multiple_crossovers["slow"],
    )

    cross_over_up_indices = moving_average_cross_over_transformer._get_up_cross_indices(
        fast, slow
    )
    cross_over_down_indices = (
        moving_average_cross_over_transformer._get_down_cross_indices(fast, slow)
    )

    assert len(cross_over_down_indices) == 2, "There should be 2 down crossover"
    assert len(cross_over_up_indices) == 3, "There should be 2 up crossover"

    assert cross_over_up_indices[0].date() == datetime.date(year=2020, month=1, day=5)
    assert cross_over_up_indices[1].date() == datetime.date(year=2020, month=1, day=12)
    assert cross_over_up_indices[2].date() == datetime.date(year=2020, month=1, day=17)

    assert cross_over_down_indices[0].date() == datetime.date(year=2020, month=1, day=9)
    assert cross_over_down_indices[1].date() == datetime.date(
        year=2020, month=1, day=14
    )


def test_moving_average_predictor_predict_proba(
    dollar_bar_dataframe, moving_average_cross_over_predictor
):

    """
    Test the moving overage probabilites shape

    :param dollar_bar_dataframe: Dataframe with close columne
    :type dollar_bar_dataframe: pd.DataFrame
    :param moving_average_cross_over_predictor: Moving average predictor
    :type moving_average_cross_over_predictor: MovingAverageCrossOverPredictor
    :return None
    """
    probabilities = moving_average_cross_over_predictor.predict_proba(
        dollar_bar_dataframe
    )
    assert (
        probabilities.shape[1] == moving_average_cross_over_predictor.n_classes_
    ), "number of predictions columns should be equal to the number of classes"


def test_moving_average_predictor_predict(
    dollar_bar_dataframe, moving_average_cross_over_predictor
):
    """
    Test the moving overage predictions values

    :param dollar_bar_dataframe: Dataframe with close columne
    :type dollar_bar_dataframe: pd.DataFrame
    :param moving_average_cross_over_predictor: Moving average predictor
    :type moving_average_cross_over_predictor: MovingAverageCrossOverPredictor
    :return None
    """
    predictions = moving_average_cross_over_predictor.predict(dollar_bar_dataframe)
    assert set(predictions) == {-1.0, 1.0, 0.0}, "Only 1, -1 and 0 are expected"

def test_exponential_weighted_moving_average_difference(
    dollar_bar_dataframe: pd.DataFrame
):
    fast = 10
    slow = 20
    ewma_diff_norm = ExponentialWeightedMovingAverageDifference(fast=fast, slow=slow, column_name="close", normalised=True)
    ewma_diff_unnorm = ExponentialWeightedMovingAverageDifference(fast=fast, slow=slow, column_name="close", normalised=False)
    norm_out = ewma_diff_norm.transform(dollar_bar_dataframe)
    unnorm_out = ewma_diff_unnorm.transform(dollar_bar_dataframe)

    assert (norm_out != unnorm_out).any(), "Expected different values but got the same"