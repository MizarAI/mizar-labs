from abc import abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
from numba import jit
from numba import prange
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from mizarlabs.static import CLOSE
from mizarlabs.static import DAILY_VOL
from mizarlabs.static import EVENT_END_TIME
from mizarlabs.static import LABEL
from mizarlabs.static import PROFIT_TAKING
from mizarlabs.static import RETURN
from mizarlabs.static import SIDE
from mizarlabs.static import STOP_LOSS
from mizarlabs.static import TIMESTAMP_UNIT
from mizarlabs.transformers.utils import check_missing_columns
from mizarlabs.transformers.utils import convert_to_timestamp

__all__ = [
    "BaseLabeling",
    "TripleBarrierMethodLabeling",
    "triple_barrier_labeling",
    "get_labels",
    "get_daily_vol",
]


class BaseLabeling(BaseEstimator, TransformerMixin):
    """Base class for labeling."""

    def __init__(self, n_expiration_bars: int):
        self.n_expiration_bars = n_expiration_bars

    def fit(self, y):
        """
        Fit the model (just for sklearn compatibility).

        :param x:
        :param y:
        :return:
        """
        return self

    def transform(self, y: pd.DataFrame) -> pd.DataFrame:
        """
        Return a dataframe with the target labelled.

        :param y:
        :return:
        """
        self._check_input(y)
        output = self._transform(y)
        self._check_output(output)
        return output

    @abstractmethod
    def _check_output(self, output_df: pd.DataFrame):
        pass

    @abstractmethod
    def _check_input(self, input_series: pd.DataFrame):
        pass

    @abstractmethod
    def _transform(self, y: pd.DataFrame) -> pd.DataFrame:
        pass


# TODO: think about a triple barrier labeling with no
#  volatility adjustment
class TripleBarrierMethodLabeling(BaseLabeling):
    """
    Implements the triple barrier method used to label the target.

    See page 45 of Advances in Financial Machine Learning by Marcos Lopez de Prado
    for additional information.

    :param num_expiration_bars: Max number of bars from the position taking to the position closing.
    :type num_expiration_bars: int
    :param profit_taking_factor: The factor that multiplies the volatility for
        the creation of the horizontal upper barrier
    :type profit_taking_factor: float
    :param stop_loss_factor: The factor that multiplies the volatility for the
        creation of the horizontal lower barrier
    :type stop_loss_factor: float
    :param metalabeling: Whether metalabeling is activated
    :type metalabeling: bool, optional
    :param close_column_name: The name of the close column
    :type close_column_name: str, optional
    :param side_column_name: The name of the side column (metalabeling)
    :type side_column_name: str, optional
    :param volatility_window: The number of bars used for the volatility calculation
    :type volatility_window: int, optional
    :param expiration_label: Labels with 0 are returned to indicate expiration / vertical barrier has been hit
    :type expiration_label: bool, optional
    """

    def __init__(
        self,
        num_expiration_bars: int,
        profit_taking_factor: float,
        stop_loss_factor: float,
        metalabeling: bool = False,
        close_column_name: str = CLOSE,
        side_column_name: str = SIDE,
        volatility_window: int = 100,
        expiration_label: bool = False,
    ):
        super().__init__(num_expiration_bars)
        self.profit_taking_factor = profit_taking_factor
        self.stop_loss_factor = stop_loss_factor
        self.metalabeling = metalabeling
        self.close_column_name = close_column_name
        self.side_column_name = side_column_name
        self.volatility_window = volatility_window
        self.expiration_label = expiration_label
        if self.metalabeling:
            assert (
                self.side_column_name
            ), "Need to to set side_column_name when meta-labeling is selected"

    def fit(self, X, y=None, **fit_params):
        return self

    def _check_output(self, output_df: pd.DataFrame):
        check_missing_columns(output_df, [EVENT_END_TIME, DAILY_VOL, RETURN, LABEL])

    def _check_input(self, input_df: pd.DataFrame):
        if self.metalabeling:
            check_missing_columns(
                input_df, [self.close_column_name, self.side_column_name]
            )
        else:
            check_missing_columns(input_df, [self.close_column_name])

        assert isinstance(input_df, pd.DataFrame), (
            "Please provide only a Pandas dataframe with close prices and"
            " side (in case of metalabeling) with datetime indices."
        )

    def _transform(self, y: pd.DataFrame) -> pd.DataFrame:
        """
        Return a dataframe with the target labelled.

        :param y: The dataframe with the close and side
        :type y: pd.DataFrame
        :return: Dataframe containing labeled target
        :rtype: pd.DataFrame
        """
        barriers_info_df = pd.DataFrame(index=y.index)

        barriers_info_df[EVENT_END_TIME] = (
            pd.Series(barriers_info_df.index).shift(-self.n_expiration_bars).values
        )

        barriers_info_df[DAILY_VOL] = get_daily_vol(
            y[self.close_column_name], self.volatility_window
        )
        if SIDE not in y.columns:
            barriers_info_df[self.side_column_name] = 1
        else:
            barriers_info_df[self.side_column_name] = y[self.side_column_name]

        barriers_df = triple_barrier_labeling(
            close=y[self.close_column_name],
            barrier_info_df=barriers_info_df,
            profit_taking_factor=self.profit_taking_factor,
            stop_loss_factor=self.stop_loss_factor,
        )

        barriers_info_df[EVENT_END_TIME] = barriers_df.dropna(how="all").min(axis=1)

        bins = get_labels(
            barriers_df,
            barriers_info_df,
            y[self.close_column_name],
            metalabeling=self.metalabeling,
            expiration_label=self.expiration_label,
        )

        barriers_info_df.drop(SIDE, axis=1, inplace=True)

        output_df = pd.merge(
            barriers_info_df[[EVENT_END_TIME, DAILY_VOL]],
            bins,
            left_index=True,
            right_index=True,
            how="left",
        )

        output_df.loc[output_df[DAILY_VOL].isna(), EVENT_END_TIME] = pd.NaT
        output_df.loc[output_df[DAILY_VOL].isna(), RETURN] = np.nan
        output_df.loc[output_df[DAILY_VOL].isna(), LABEL] = np.nan
        return output_df


def triple_barrier_labeling(
    close: pd.Series,
    barrier_info_df: pd.DataFrame,
    profit_taking_factor: float,
    stop_loss_factor: float,
) -> pd.DataFrame:
    """
    Calculate the first hit on the stop loss and profit taking barrier.

    As described in Advances in financial machine learning,
    Marcos Lopez de Prado, 2018.

    :param close: Series of prices.
    :type close: pd.Series
    :param barrier_info_df: Info for creating the barriers
    :type barrier_info_df: pd.DataFrame
    :param profit_taking_factor: The factor that multiplies the volatility for
        the creation of the horizontal upper barrier
    :type profit_taking_factor: float
    :param stop_loss_factor: The factor that multiplies the volatility for the
        creation of the horizontal lower barrier
    :type stop_loss_factor: float
    :return: Dataframe containing the first hit for each of the barriers
    :rtype: pd.DataFrame
    """
    check_missing_columns(barrier_info_df, [EVENT_END_TIME, DAILY_VOL, SIDE])

    if profit_taking_factor < 0 and stop_loss_factor < 0:
        raise ValueError("Stop loss and profit taking factors should be greater than 0")

    barriers_df = barrier_info_df[[EVENT_END_TIME]].copy(deep=True)

    # Creating the profit taking barriers. If the profit taking factor is 0
    # then there won't be any profit taking barriers.
    if profit_taking_factor > 0:
        profit_taking_barriers = profit_taking_factor * barrier_info_df[DAILY_VOL]
    else:
        profit_taking_barriers = pd.Series(index=barrier_info_df.index)  # NaNs

    # Creating the stop loss barriers. If the stop loss factor is 0
    # then there won't be any stop loss barriers.
    if stop_loss_factor > 0:
        stop_loss_barriers = -stop_loss_factor * barrier_info_df[DAILY_VOL]
    else:
        stop_loss_barriers = pd.Series(index=barrier_info_df.index)  # NaNs

    (
        profit_taking_barriers_timestamps,
        stop_loss_barriers_timestamps,
    ) = get_horizontal_barriers_hit(
        convert_to_timestamp(
            barrier_info_df[EVENT_END_TIME].fillna(close.index[-1]).values
        ),
        convert_to_timestamp(barrier_info_df.index.values),
        close.values,
        barrier_info_df[SIDE].values,
        stop_loss_barriers.values,
        profit_taking_barriers.values,
    )

    profit_taking_barriers_timestamps = pd.Series(
        profit_taking_barriers_timestamps,
        dtype=pd.Int64Dtype(),
        index=barriers_df.index,
    ).replace(0, np.nan)

    stop_loss_barriers_timestamps = pd.Series(
        stop_loss_barriers_timestamps, dtype=pd.Int64Dtype(), index=barriers_df.index
    ).replace(0, np.nan)

    barriers_df.loc[:, STOP_LOSS] = pd.to_datetime(
        stop_loss_barriers_timestamps, unit=TIMESTAMP_UNIT
    )
    barriers_df.loc[:, PROFIT_TAKING] = pd.to_datetime(
        profit_taking_barriers_timestamps, unit=TIMESTAMP_UNIT
    )

    assert set(barriers_df.columns) == {EVENT_END_TIME, STOP_LOSS, PROFIT_TAKING}

    return barriers_df


@jit(parallel=True, nopython=True)
def get_horizontal_barriers_hit(
    expiration_barrier_timestamps: np.ndarray,
    index_timestamps: np.ndarray,
    close: np.ndarray,
    side: np.ndarray,
    stop_loss_barriers: np.ndarray,
    profit_taking_barriers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the horizontal barriers hits.

    :param expiration_barrier_timestamps: Array with the timestamps of the vertical barrier
    :type expiration_barrier_timestamps: np.ndarray
    :param index_timestamps: Array with the timestamps of the position taking
    :type index_timestamps: np.ndarray
    :param close: Array with the price of the close
    :type close: np.ndarray
    :param side: Array with the side of the position (1, -1)
    :type side: np.ndarray
    :param stop_loss_barriers: Array with the stop loss barriers values
    :type stop_loss_barriers: np.ndarray
    :param profit_taking_barriers: Array with teh profit taking barriers values
    :type profit_taking_barriers: np.ndarray
    :return: Two arrays containing when the profit taking and the stop loss
             barriers have been hit
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    stop_loss_barriers_timestamps = np.zeros_like(
        expiration_barrier_timestamps, dtype=np.float64
    )
    profit_taking_barriers_timestamps = np.zeros_like(
        expiration_barrier_timestamps, dtype=np.float64
    )

    for i in prange(len(index_timestamps)):
        # creating mask between the position taking date and expiration barrier
        # date
        within_barrier_mask = np.greater_equal(
            index_timestamps, index_timestamps[i]
        ) * np.less(index_timestamps, expiration_barrier_timestamps[i])

        # selecting all the prices between the position taking date
        # and expiration barrier date
        close_prices_within_barrier = close[within_barrier_mask]

        # Calculating returns within the prices in the barriers
        returns_within_barrier = (close_prices_within_barrier / close[i] - 1) * side[i]

        # Calculating the first date when the stop loss barrier is hit.
        # If the barrier is not hit, the result is nan
        stop_loss_mask = np.less(returns_within_barrier, stop_loss_barriers[i])

        if np.any(stop_loss_mask):
            stop_loss_barriers_timestamps[i] = np.min(
                index_timestamps[within_barrier_mask][stop_loss_mask]
            )

        # Calculating the first date when the profit taking barrier is hit.
        # If the barrier is not hit, the result is nan
        profit_taking_mask = np.greater(
            returns_within_barrier, profit_taking_barriers[i]
        )
        if np.any(profit_taking_mask):
            profit_taking_barriers_timestamps[i] = np.min(
                index_timestamps[within_barrier_mask][profit_taking_mask]
            )

    return profit_taking_barriers_timestamps, stop_loss_barriers_timestamps


def get_labels(
    barriers_df: pd.DataFrame,
    barriers_info_df: pd.DataFrame,
    close: pd.Series,
    metalabeling: bool,
    expiration_label: bool = False,
) -> pd.DataFrame:
    """
    Calculate returns and assign return classes based on the first touched bar.

    Case 1: ('side' not in barriers_info_df): bin in (-1,1) <-label by price action
    Case 2: ('side' in barriers_info_df): bin in (0,1) <-label by pnl (meta-labeling)

    :param barriers_df: dataframe with datetime when barriers are hit
    :type barriers_df: pd.DataFrame
    :param barriers_info_df: Info for creating the barriers
    :type barriers_info_df: pd.DataFrame
    :param close: Series of prices.
    :type close: pd.Series
    :param metalabeling: Whether or not metalabelign is activated
    :type metalabeling: bool

    :return: Dataframe containing event
    :rtype: pd.DataFrame
    """
    check_missing_columns(barriers_info_df, [EVENT_END_TIME, DAILY_VOL, SIDE])

    # selecting bars that have a closed position
    events_without_na = barriers_info_df.dropna(subset=[EVENT_END_TIME])
    list_existing_dates = events_without_na.index.union(
        events_without_na[EVENT_END_TIME].values
    ).drop_duplicates()
    px = close.reindex(list_existing_dates, method="bfill")

    # creating dataframe with start position date
    output_df = pd.DataFrame(index=events_without_na.index)
    # calculate the returns
    output_df[RETURN] = (
        px.loc[events_without_na[EVENT_END_TIME].values].values
        / px.loc[events_without_na.index]
        - 1
    )
    returns = (
        close.loc[events_without_na[EVENT_END_TIME]].values
        / close.loc[events_without_na.index]
        - 1
    ).rename(RETURN)

    pd.testing.assert_series_equal(returns, output_df[RETURN])

    if metalabeling:
        # metalabeling labels are always only 1 and 0. 1 is assigned when the
        # base model is correct while 0 is assigned when the base model is not
        # correct. When the base model does not take a position or the returns
        #  are 0 then the metalabel will be na
        output_df[RETURN] *= events_without_na[SIDE]
        output_df.loc[output_df[RETURN] > 0, LABEL] = 1
        output_df.loc[output_df[RETURN] < 0, LABEL] = 0
        output_df.loc[output_df[RETURN] == 0, LABEL] = np.nan
        # When metalabeling is not activated then the labels can be 1 or -1.
        # 1 is when the returns are positive and -1 when the returns are
        # negative
    else:
        output_df[LABEL] = np.sign(output_df[RETURN])

    if expiration_label:
        expired_events = barriers_df.loc[events_without_na.index].loc[
            barriers_df.stop_loss.isna() & barriers_df.profit_taking.isna()
        ]
        output_df.loc[expired_events.index, LABEL] = 0

    assert set(output_df.columns) == {RETURN, LABEL}

    return output_df


def get_daily_vol(close: pd.Series, ewm_span: int = 100) -> pd.Series:
    """
    Estimate the daily volatility.

    :param close: Contains the close price
    :type close: pd.Series
    :param ewm_span: The span of the standard deviation
    :type ewm_span: int
    :return: The daily volatility per each bar
    :rtype: pd.Series
    """
    # find nearest 1 day apart close price
    indices_with_date_diff = close.index.searchsorted(
        close.index - pd.Timedelta(days=1)
    )
    indices_filtered = indices_with_date_diff[indices_with_date_diff > 0]

    # creating series with the date of the nearest 1 day apart  close price
    shifted_close = pd.Series(
        close.index[indices_filtered - 1],
        index=close.index[close.shape[0] - indices_filtered.shape[0] :],
    )
    # calculating returns between close and nearest 1 day apart  close
    returns = close.loc[shifted_close.index] / close.loc[shifted_close].values - 1
    daily_vol = returns.ewm(span=ewm_span).std()
    return daily_vol
