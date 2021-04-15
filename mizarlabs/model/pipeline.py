import logging
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y

from mizarlabs.static import CLOSE
from mizarlabs.static import NUMBER_EXPIRATION_BARS_COLUMN
from mizarlabs.static import PROFIT_TAKING
from mizarlabs.static import SIDE
from mizarlabs.static import SIZE
from mizarlabs.static import STOP_LOSS
from mizarlabs.transformers.targets.labeling import get_daily_vol
from mizarlabs.transformers.trading.bet_sizing import BetSizingFromProbabilities
from mizarlabs.transformers.utils import check_missing_columns


class MizarFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        Xs = super().fit_transform(X, y=None, **fit_params)

        return pd.DataFrame(Xs, index=X.index)

    def transform(self, X):
        Xs = super().transform(X)

        return pd.DataFrame(Xs, index=X.index)


class MizarPipeline(Pipeline):
    """
    Implementation of pipeline that allows sample_weight as a fit argument
    """

    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
        return super().fit(X, y, **fit_params)


class StrategySignalPipeline:
    """
    A trading strategy.

    A trading strategy can include machine learning models or simple technical
    indicator transformers. From their outputs the strategy decides whether or
    not to take a position and its size.

    This strategy must have a primary model. A metalabeling model and
    a bet sizer are optional.

    The simplest setting includes only a primary model. In this case the side
    is calculated with the predict of the primary model, while the size is
    calculated from the predict_proba of the primary model.

    Adding a bet sizer means that the size is calculated from the bet sizer
    and not anymore from the primary model probabilities. The bet sizer
    calculates the bet size from the probabilities of the primary model
    predictions.

    When metalabeling model is set then the size comes from the metalabeling
    model, unless a bet sizer is set and in this case the bet sizer calculates
    the size from the probabilites provided by the metalabeling model

    :param feature_transformers_primary_model: The feature transformer that
                                               transforms the data for the
                                               primary model
    :type feature_transformers_primary_model: TransformerMixin
    :param feature_transformers_metalabeling_model: The feature transformer that
                                                    transforms the data for the
                                                    metalabeling model
    :type feature_transformers_metalabeling_model: TransformerMixin

    :param metalabeling_use_proba_primary_model: Whether to use probabilities
                                                 of the primary model as
                                                 features in the metalabeling
                                                 model
    :type metalabeling_use_proba_primary_model: bool
    :param metalabeling_use_predictions_primary_model: Whether to use
                                                       predictions of the
                                                       primary model as feature
                                                       in the metalabeling
                                                       model
    :param bet_sizer: The transformer to use for the calculation of the bet size
    :type bet_sizer: BetSizingFromProbabilities
    """

    _primary_model_features = "primary_model_features"
    _primary_model_predictions = "primary_model_predictions"
    _primary_model_proba = "primary_model_proba"
    _metalabeling_model_features = "metalabeling_model_features"
    _metalabeling_model_predictions = "metalabeling_model_predictions"
    _metalabeling_model_proba = "metalabeling_model_proba"

    _align_methods = {
        "mean": np.mean,
        "min": np.min,
        "max": np.max,
    }

    def __init__(
        self,
        feature_transformers_primary_model: Union[
            Dict[str, Union[TransformerMixin, None]]
        ],
        align_on: str,
        align_how: Dict[str, str],
        feature_transformers_metalabeling_model: Union[
            Dict[str, Union[TransformerMixin, None]]
        ] = None,
        metalabeling_use_proba_primary_model: bool = True,
        metalabeling_use_predictions_primary_model: bool = True,
        bet_sizer: Union[BetSizingFromProbabilities, None] = None,
    ):
        self.primary_model = None
        self.metalabeling_model = None

        self.feature_transformers_primary_model = (
            feature_transformers_primary_model
            if feature_transformers_primary_model
            else {}
        )

        self.align_on_ = align_on
        self.align_how_ = align_how

        self.feature_transformers_metalabeling_model = (
            feature_transformers_metalabeling_model
            if feature_transformers_metalabeling_model
            else {}
        )

        self.metalabeling_use_proba_primary_model = metalabeling_use_proba_primary_model
        self.metalabeling_use_predictions_primary_model = (
            metalabeling_use_predictions_primary_model
        )
        self.bet_sizer = bet_sizer

    def set_primary_model(self, primary_model: BaseEstimator):
        self.primary_model = primary_model
        return self

    def set_metalabeling_model(self, metalabeling_model: BaseEstimator):
        self.metalabeling_model = metalabeling_model
        return self

    def _align_on(
        self, X_features_dict: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Aligns the dataframes with features with the prespecified
        leading dataframe with features.

        :param X_features_dict: dictionary with features for the primary (and
                                metalabeling model if set) with keys
                                "primary_model_features" and
                                "metalabeling_model_features".
        :type X_features_dict: Dict[str, Dict[str, pd.DataFrame]]
        :return: dictionary with aligned features for the primary (and
                 metalabeling model if set) with keys "primary_model_features"
                 and "metalabeling_model_features".
        :rtype: Dict[str, Dict[str, pd.DataFrame]]
        """
        if not self.align_on_:
            return X_features_dict

        alignment_index = X_features_dict[self._primary_model_features][
            self.align_on_
        ].index

        for model_features_key in [
            self._primary_model_features,
            self._metalabeling_model_features,
        ]:
            if model_features_key is None:
                continue

            for feature_key in X_features_dict[model_features_key].keys():
                if feature_key != self.align_on_:

                    feature_df = X_features_dict[model_features_key][feature_key]

                    # -np.inf equivalent of datetime
                    previous_index = pd.to_datetime("1-1-1700", utc=feature_df.index.tz)

                    aligned_feature_df = pd.DataFrame(
                        columns=feature_df.columns, index=alignment_index
                    )

                    diff_days_start = (feature_df.index[0] - alignment_index[0]).days

                    if (
                        abs(diff_days_start)
                        > (alignment_index[-1] - alignment_index[0]).days / 10
                    ):
                        earlier_later = "later" if diff_days_start > 0 else "earlier"

                        logging.warning(
                            f"The start date of the features dataset"
                            f" {feature_key} starts "
                            f"{(alignment_index[-1] - alignment_index[0]).days} "
                            f"days {earlier_later}"
                            f" than the alignment data"
                        )

                    for index in alignment_index:
                        feature_df_subset = feature_df.loc[
                            (feature_df.index > previous_index)
                            & (feature_df.index <= index)
                        ]
                        if feature_df_subset.empty:
                            continue

                        elif feature_df_subset.shape[0] == 1:
                            aligned_feature_df.loc[index] = feature_df_subset.values

                        else:
                            aligned_feature_df.loc[index] = self._align_methods[
                                self.align_how_[feature_key]
                            ](feature_df_subset.values, axis=0)
                        previous_index = index

                    X_features_dict[model_features_key][
                        feature_key
                    ] = aligned_feature_df.ffill()

        return X_features_dict

    def _get_pred_proba_features(
        self,
        primary_model: BaseEstimator,
        X_pred_features: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> pd.DataFrame:
        """
        Get the probabilities of the primary model predictions to be used in
        the metalabeling model as features

        :param primary_model: The primary model object
        :type primary_model: BaseEstimator
        :param X_pred_features: DataFrame with predictions from the primary model,
                                which will be used as features for the metalabeling
                                model.
        :type X_pred_features: pd.DataFrame
        :param X_val: Features of the primary model used for validation during
                      cross validation.
        :type X_val: pd.DataFrame
        :param y_val: Class labels of the primary used for validation during
                      cross validation.
        :type y_val: pd.Series
        :return: DataFrame with predictions and probabilities from the primary
                 model to be used as features in the metalabeling model.
        :rtype: pd.DataFrame
        """
        if not hasattr(primary_model, "predict_proba"):
            raise AttributeError("Primary model does not have a predict_proba method.")

        y_pred_proba_val = primary_model.predict_proba(X_val)
        pred_proba_cols = [
            f"primary_side_pred_proba_{i}" for i in range(y_pred_proba_val.shape[1])
        ]

        if not set(pred_proba_cols).issubset(set(X_pred_features.columns)):
            X_pred_proba = pd.DataFrame(
                y_pred_proba_val,
                index=y_val.index,
                columns=[
                    f"primary_side_pred_proba_{i}"
                    for i in range(y_pred_proba_val.shape[1])
                ],
            )
            X_pred_features = pd.concat([X_pred_features, X_pred_proba], axis=1)
        else:
            X_pred_features.loc[y_val.index, pred_proba_cols] = y_pred_proba_val

        return X_pred_features

    @staticmethod
    def _check_y_metalabel(y_metalabel: pd.Series):
        """
        Check whether metalabels can be used for training the metalabeling
        model.

        :param y_metalabel: Metalabels calculated from the primary model
        :type y_metalabel: pd.Series
        :return:
        """
        unique_metalabels = np.unique(y_metalabel)
        if len(unique_metalabels) == 1:
            sentence_string = "incorrect" if 0.0 in unique_metalabels else "correct"
            raise ValueError(
                f"The primary model is always {sentence_string} "
                f"and so it is not possible to train the "
                f"metalabeling model"
            )

    @staticmethod
    def _align_X_dict_and_y(
        X_dict: Dict[str, pd.DataFrame], y: pd.Series, drop_na: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Align the features in X and the target in y

        :param X_dict: A dictionary containing features for the primary and
                       metalabeling model
        :type X_dict: Dict[str, pd.DataFrame]
        :param y: The target to which the features need to be aligned with
        :type y: pd.Series

        :return: Features, target and sample weights aligned
        :rtype: Tuple[pd.DataFrame, pd.Series, pd.Series]
        """

        if drop_na:
            y_no_nans = y.dropna()
            X = pd.concat([df.loc[y_no_nans.index] for df in X_dict.values()], axis=1)
            X_aligned = X.dropna()
            y_aligned = y.loc[X_aligned.index]
            check_X_y(X_aligned, y_aligned)
            return X_aligned, y_aligned
        else:
            X = pd.concat([df for df in X_dict.values()], axis=1)
            return X, y

    def transform(
        self,
        X_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Runs the feature transformers (if available) on the data.

        :param X_dict: A dictionary containing features for the primary and
                       metalabeling model
        :type X_dict: Dict[str, pd.DataFrame]
        :return: Dictionary containing the features per each model transformed
        :rtype: Dict[str, Dict[str, pd.DataFrame]]
        """
        expected_keys = {
            *self.feature_transformers_primary_model.keys(),
            *self.feature_transformers_metalabeling_model.keys(),
        }
        assert expected_keys == set(X_dict.keys()), (
            f"The keys in X_dict should be exactly the same as the name "
            f"of the feature generators of the primary (and metalabeling) model. "
            f"Expected {expected_keys}, but got {set(X_dict.keys())}"
        )
        return {
            self._primary_model_features: {
                name_transformer: transformer.transform(X_dict[name_transformer])
                if transformer
                else X_dict[name_transformer]
                for name_transformer, transformer in self.feature_transformers_primary_model.items()
            },
            self._metalabeling_model_features: {
                name_transformer: transformer.transform(X_dict[name_transformer])
                if transformer
                else X_dict[name_transformer]
                for name_transformer, transformer in self.feature_transformers_metalabeling_model.items()
            },
        }

    def _get_features_for_metalabeling(
        self,
        X_primary_aligned: pd.DataFrame,
        X_features_dict: Dict[str, Dict[str, pd.DataFrame]],
    ) -> pd.DataFrame:
        """
        Create the aligned dataset for the metalabeling model adding the
        predictions and probabilities from the primary model

        :param X_primary_aligned: The dateset aligned for the primary model
        :type X_primary_aligned: pd.DataFrame
        :param X_features_dict: Dict containing the transformed features for the
                                primary and metalabeling model
        :type X_features_dict: Dict[str, Dict[str, pd.DataFrame]]
        :return: Aligned dataset for the metalabeling model
        :rtype: pd.DataFrame
        """
        X_metalabel_aligned = pd.concat(
            [df for df in X_features_dict[self._metalabeling_model_features].values()],
            axis=1,
        )
        X_metalabel_aligned.dropna(inplace=True)
        # add primary model predict values for metalabel model as features
        if self.metalabeling_use_predictions_primary_model:
            X_metalabel_aligned["primary_side_pred"] = pd.Series(
                self.primary_model.predict(X_primary_aligned),
                index=X_primary_aligned.index,
            )
        # add primary model predict proba values for metalabel model as features
        if self.metalabeling_use_proba_primary_model:
            X_primary_proba_features = pd.DataFrame(
                self.primary_model.predict_proba(X_primary_aligned),
                index=X_primary_aligned.index,
                columns=[
                    f"primary_side_pred_proba_{i}"
                    for i in range(self.primary_model.n_classes_)
                ],
            )
            # to avoid the creation of unwanted nans the join between the the
            # metalabeling aligned and the primary model predictions is inner.
            # Without using an inner join nan values will be created when
            # the indices of the two dataframe are not the same
            X_metalabel_aligned = pd.concat(
                [X_metalabel_aligned, X_primary_proba_features], axis=1, join="inner"
            )

        if X_metalabel_aligned.isna().any().any():
            raise ValueError(
                "Nan values in the metalabeling aligned dataset "
                "are not allowed. Please check your metalabeling "
                "model does not create any nan values when "
                "it predicts"
            )
        return X_metalabel_aligned

    def predict(
        self,
        X_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """
        Predict the classes for the primary and metalabeling model

        :param X_dict: A dictionary containing features for the primary and
                       meta-labeling model
        :type X_dict: Dict[str, pd.DataFrame]
        :return: Predicted probabilities for primary and metalabeling model
        :rtype: Dict[str, pd.DataFrame]
        """
        if self.metalabeling_model:
            check_is_fitted(self.metalabeling_model)

        target_index = X_dict[self.align_on_].index

        for df_name, df in X_dict.items():
            if target_index[0] > df.index[-1]:
                raise ValueError(
                    f"There is no intersection between {df_name} and {self.align_on_}"
                )

        X_features_dict = self.transform(X_dict)
        X_features_dict = self._align_on(X_features_dict)
        X_primary_aligned = pd.concat(
            [df for df in X_features_dict[self._primary_model_features].values()],
            axis=1,
        )
        X_primary_aligned.dropna(inplace=True)

        # get predictions primary model
        pred_primary = pd.Series(
            self.primary_model.predict(X_primary_aligned), index=X_primary_aligned.index
        )
        predictions = {
            self._primary_model_predictions: pred_primary,
            self._metalabeling_model_predictions: None,
        }

        if self.metalabeling_model:
            X_metalabel_aligned = self._get_features_for_metalabeling(
                X_primary_aligned, X_features_dict
            )
            pred_metalabeling = pd.Series(
                self.metalabeling_model.predict(X_metalabel_aligned),
                index=X_metalabel_aligned.index,
            )
            predictions[self._metalabeling_model_predictions] = pred_metalabeling

        return predictions

    def predict_proba(self, X_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Predict the probabilities for the primary and metalabeling model

        :param X_dict: A dictionary containing features for the primary and
                       meta-labeling model
        :type X_dict: Dict[str, pd.DataFrame]
        :return: Predicted probabilities for primary and metalabeling model
        :rtype: Dict[str, pd.DataFrame]
        """

        # transforming and aligning X_dict
        X_features_dict = self.transform(X_dict)
        X_features_dict = self._align_on(X_features_dict)

        # concatenating all the features dataframes in
        # one single dataframe that can be used for
        # predictions
        X_primary_aligned = pd.concat(
            [df for df in X_features_dict[self._primary_model_features].values()],
            axis=1,
        )
        X_primary_aligned.dropna(inplace=True)

        # get probabilities from primary model
        prob_primary_df = pd.DataFrame(
            self.primary_model.predict_proba(X_primary_aligned),
            index=X_primary_aligned.index,
            columns=self.primary_model.classes_,
        )

        # creating the probabilities dictionary
        # with the probabilites from the primary model
        probabilities = {
            self._primary_model_proba: prob_primary_df,
            self._metalabeling_model_proba: None,
        }

        # if the metalabeling model is present
        # then the metalabeling probabilities are
        # added to the probabilities dictionary
        if self.metalabeling_model:
            check_is_fitted(self.metalabeling_model)

            # aligning metalabeling model's features
            X_metalabel_aligned = self._get_features_for_metalabeling(
                X_primary_aligned, X_features_dict
            )

            # get probabilities from metalabeling model
            pred_metalabeling = pd.DataFrame(
                self.metalabeling_model.predict_proba(X_metalabel_aligned),
                index=X_metalabel_aligned.index,
                columns=self.metalabeling_model.classes_,
            )
            # adding the probabilities to the dictionary
            probabilities[self._metalabeling_model_proba] = pred_metalabeling

        return probabilities

    def get_side_and_size(self, X_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate the side and size of the position

        :param X_dict: A dictionary containing features for the primary and
                       meta-labeling model
        :type X_dict: Dict[str, pd.DataFrame]
        :return: The sizes of the positions
        :rtype: pd.Series
        """

        side = self.predict(X_dict=X_dict)[self._primary_model_predictions]

        predicted_proba_dict = self.predict_proba(X_dict=X_dict)
        predicted_proba = predicted_proba_dict[
            self._metalabeling_model_proba
            if self.metalabeling_model
            else self._primary_model_proba
        ]

        # TODO: how to implement average_active option in bet_sizer
        # requires event_end_time_column_name
        if self.bet_sizer:
            pred_and_proba = pd.DataFrame(
                data={
                    "prob": predicted_proba.max(axis=1).values,
                    "pred": np.array(
                        [
                            predicted_proba.columns[i]
                            for i in np.argmax(predicted_proba.values, axis=1)
                        ]
                    ),
                },
                index=predicted_proba.index,
            )
            if self.metalabeling_model:
                predicted_side = self.predict(X_dict=X_dict)[
                    self._primary_model_predictions
                ]
                pred_and_proba["side"] = predicted_side
            size = self.bet_sizer.transform(pred_and_proba)

        else:
            if self.metalabeling_model:
                size = predicted_proba[1.0]
            else:
                size = predicted_proba.max(axis=1)

        return pd.DataFrame({"side": side, "size": size}).dropna()

    def create_dataset_metalabeling(
        self,
        X_dict: Dict[str, pd.DataFrame],
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:

        """
        Produce data set for metalabeling model fitting.

        The primary model is expected to be already set in the pipeline

        :param X_dict: Dictionary containing all the features for the data
                       for the primary and metalabeling model. The data can be
                       bar and/or tick data
        :type X_dict:  Dict[str, pd.DataFrame]
        :param y: Series with class labels for the primary model
        :type y: pd.Series
        :return: The strategy signal pipeline
        :rtype: StrategySignalPipeline
        """
        assert set(np.unique(y.dropna())).issubset({-1, 0, 1})

        # transforming the X_dict using the specified transformers
        X_features_dict = self.transform(X_dict)

        # Aligning the dictionaries to the data from which the
        # target has been constructed
        X_features_dict = self._align_on(X_features_dict)

        # align X_dict a dict consisting of dataframes with features and y
        (X_primary_aligned, y_primary_aligned,) = self._align_X_dict_and_y(
            X_features_dict[self._primary_model_features],
            y,
        )

        X_metalabel = self._get_features_for_metalabeling(
            X_primary_aligned, X_features_dict
        )
        y_metalabel = (
            self.primary_model.predict(X_primary_aligned) == y_primary_aligned
        ).astype(float)

        return (
            X_metalabel.loc[X_metalabel.index.intersection(y_metalabel.index)],
            y_metalabel.loc[X_metalabel.index.intersection(y_metalabel.index)],
        )

    def create_dataset_primary(
        self,
        X_dict: Dict[str, pd.DataFrame],
        y: pd.Series,
        drop_na: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Produce data set for primary model fitting

        :param X_dict: Dictionary containing all the features for the data
                       for the primary and metalabeling model. The data can be
                       bar and/or tick data
        :type X_dict:  Dict[str, pd.DataFrame]
        :param y: Series with class labels for the primary model
        :type y: pd.Series
        :param drop_na:
        :type drop_na: bool
        :return: The strategy signal pipeline
        :rtype: StrategySignalPipeline
        """
        assert set(np.unique(y.dropna())).issubset({-1, 0, 1})

        # transforming the X_dict using the specified transformers
        X_features_dict = self.transform(X_dict)

        # Aligning the dictionaries to the data from which the
        # target has been constructed
        X_features_dict = self._align_on(X_features_dict)

        # align X_dict a dict consisting of dataframes with features and y
        (X_aligned, y_aligned,) = self._align_X_dict_and_y(
            X_features_dict[self._primary_model_features], y, drop_na
        )

        return X_aligned, y_aligned


class StrategyTrader:
    """
    What is my purpose?

    Interacts with the data provider,
    use the strategy pipeline to make a prediction,
    Based on a prediction produces all the information to create a position
    (side, size, expiration, profit taking, stop loss)
    """

    def __init__(
        self,
        strategy_pipeline: StrategySignalPipeline,
        # min number of bars needed to create a prediction
        # (take max of the min of primary and meta-labeling model)
        min_num_bars: int,
        num_expiration_bars: int,  # when does trade expire
        stop_loss_factor: Union[float, None] = None,
        profit_taking_factor: Union[float, None] = None,
        volatility_window: int = 100,
    ):

        self.strategy_pipeline = strategy_pipeline
        self.min_num_bars = min_num_bars
        self.num_expiration_bars = num_expiration_bars
        self.volatility_window = volatility_window
        self.stop_loss_factor = stop_loss_factor
        self.profit_taking_factor = profit_taking_factor
        self.strategy_valid = None
        # TODO: think about a triple barrier labeling with no
        #  volatility adjustment -> check labeling.py

    def _check_X_dict(self, X_dict: Dict[str, pd.DataFrame]):
        """

        :param X_dict: A dictionary containing features for the primary and
                       meta-labeling model
        :type X_dict: Dict[str, pd.DataFrame]
        """
        for df_key, df in X_dict.items():
            if not df.index.is_unique:
                raise ValueError(f"The dataframe {df_key} has duplicated indices")

    def _check_signal_has_valid_output(self, signal_df: pd.DataFrame) -> None:
        """
        Check whether the signal has valid values

        :param signal_df: dataframe containing the size and side of the signal
        :type signal_df: pd.DataFrame
        :return: None
        :rtype: None
        """
        # checking if output of strategy is what we expect
        size_unexpected_values = (
            not (0 <= signal_df[SIZE]).all() or not (signal_df[SIZE] <= 1).all()
        )
        side_unexpected_values = (
            not (-1 <= signal_df[SIDE]).all() or not (signal_df[SIDE] <= 1).all()
        )

        if size_unexpected_values or side_unexpected_values:
            self.strategy_valid = False
        else:
            self.strategy_valid = True

    def create_signal(self, X_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create the signal info dataframe (size and side)

        :param X_dict: A dictionary containing features for the primary and
                       meta-labeling model
        :type X_dict: Dict[str, pd.DataFrame]
        :return: dataframe with size and side
        :rtype: pd.DataFrame
        """
        self._check_X_dict(X_dict)

        # calculating side and size from strategy pipeline
        signal_df = self.strategy_pipeline.get_side_and_size(X_dict)
        self._check_signal_has_valid_output(signal_df)

        return signal_df

    def create_strategy_bars(self, X_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create the dataframe with the strategy bars information (stoploss,
        take profit and expiration)

        :param X_dict: A dictionary containing features for the primary and
                       meta-labeling model
        :type X_dict: Dict[str, pd.DataFrame]
        :return: dataframe with stop loss, taking profit and expiration
        :rtype: pd.DataFrame
        """
        self._check_X_dict(X_dict)

        # target dataframe is extracted so that the volatility can be
        # calculated
        strategy_bars_df = X_dict[self.strategy_pipeline.align_on_]
        volatility = get_daily_vol(strategy_bars_df[CLOSE], self.volatility_window)

        # adding stop loss and profit taking
        strategy_bars_df[STOP_LOSS] = volatility * self.stop_loss_factor
        strategy_bars_df[PROFIT_TAKING] = volatility * self.profit_taking_factor

        # adding number of expiration bars
        strategy_bars_df[NUMBER_EXPIRATION_BARS_COLUMN] = self.num_expiration_bars
        return strategy_bars_df[
            [STOP_LOSS, PROFIT_TAKING, NUMBER_EXPIRATION_BARS_COLUMN]
        ]

    def create_position(self, X_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a dataframe that can be used to evaluate the strategy.

        The dataframe contains close, stop_loss, profit_taking,
        number of expiration bars, posiion size and side.

        :param X_dict: A dictionary containing features for the primary and
                       meta-labeling model
        :type X_dict: Dict[str, pd.DataFrame]
        :return: The strategy positions and related informations
        """
        # TODO: compare the predictions of the primary with the metalabeling

        # calculating side and size from strategy pipeline
        signal_df = self.create_signal(X_dict)
        strategy_bars_df = self.create_strategy_bars(X_dict)

        position_df = pd.concat([signal_df, strategy_bars_df], axis=1)

        # check all the expected column are in position_df
        check_missing_columns(
            position_df,
            [STOP_LOSS, PROFIT_TAKING, NUMBER_EXPIRATION_BARS_COLUMN, SIZE, SIDE],
        )

        # Checking if position_df has unique indices
        if not position_df.index.is_unique:
            raise ValueError("The position dataframe has duplicated indices")

        return position_df
