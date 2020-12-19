from copy import deepcopy
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import logging
from mizarlabs.model.model_selection import CombPurgedKFoldCV
from mizarlabs.static import CLOSE
from mizarlabs.static import EVENT_END_TIME
from mizarlabs.static import NUMBER_EXPIRATION_BARS_COLUMN
from mizarlabs.static import PROFIT_TAKING
from mizarlabs.static import SIDE
from mizarlabs.static import SIZE
from mizarlabs.static import STOP_LOSS
from mizarlabs.transformers.targets.labeling import get_daily_vol
from mizarlabs.transformers.trading.bet_sizing import BetSizingFromProbabilities
from mizarlabs.transformers.utils import check_missing_columns
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y


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

    :param primary_model: The primary model estimator
    :type primary_model: BaseEstimator
    :param feature_transformers_primary_model: The feature transformer that
                                               transforms the data for the
                                               primary model
    :type feature_transformers_primary_model: TransformerMixin
    :param feature_transformers_metalabeling_model: The feature transformer that
                                                    transforms the data for the
                                                    metalabeling model
    :type feature_transformers_metalabeling_model: TransformerMixin
    :param metalabeling_model: The metalabeling model estimator
    :type metalabeling_model: BaseEstimator
    :param cpcv_num_groups: The number of groups for the combinatorial cross
                            validation used for the metalabels calculation
    :type cpcv_num_groups: int
    :param embargo_td: The number of days to use for the embargo of
                       combinatorial cross validation
    :type embargo_td: pd.Timedelta
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
    # TODO: add median to the align methods. The issue is that median does not
    #  preserve the type
    _align_methods = {
        "mean": np.mean,
        "min": np.min,
        "max": np.max,
    }

    def __init__(
        self,
        primary_model: BaseEstimator,
        feature_transformers_primary_model: Union[
            Dict[str, Union[TransformerMixin, None]]
        ],
        feature_transformers_metalabeling_model: Union[
            Dict[str, Union[TransformerMixin, None]]
        ] = None,
        metalabeling_model: Union[BaseEstimator, None] = None,
        cpcv_num_groups: int = 6,
        embargo_td: Union[None, pd.Timedelta] = None,
        metalabeling_use_proba_primary_model: bool = True,
        metalabeling_use_predictions_primary_model: bool = True,
        bet_sizer: Union[BetSizingFromProbabilities, None] = None,
    ):
        self.primary_model = primary_model

        self.feature_transformers_primary_model = (
            feature_transformers_primary_model
            if feature_transformers_primary_model
            else {}
        )

        self.feature_transformers_metalabeling_model = (
            feature_transformers_metalabeling_model
            if feature_transformers_metalabeling_model
            else {}
        )
        self.metalabeling_model = metalabeling_model
        self.cpcv_num_groups = cpcv_num_groups

        if self.metalabeling_model is not None and not isinstance(
            embargo_td, pd.Timedelta
        ):
            raise ValueError(
                "Metalabeling model has been provided, so please enter an "
                "embargo_td (pandas.TimeDelta object)."
            )
        self._cv = None
        self.embargo_td = embargo_td
        self.metalabeling_use_proba_primary_model = metalabeling_use_proba_primary_model
        self.metalabeling_use_predictions_primary_model = (
            metalabeling_use_predictions_primary_model
        )
        self.bet_sizer = bet_sizer
        self.downsampled_indices_primary = None
        self.downsampled_indices_metalabeling = None
        self.sample_weight_primary = None
        self.sample_weight_metalabeling = None

    def _down_sample(
        self,
        X_feature_dict: Dict[str, pd.DataFrame],
        y: pd.DataFrame,
        sample_weight: pd.Series = None,
        downsampled_indices: pd.DatetimeIndex = None,
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.Series]:
        """
        Downsample X_features, y and sample weight based on the provided
        downsampled indices.

        :param X_feature_dict: dictionary with features for the primary (and
                               metalabeling model if set) with keys
                               "primary_model_features" and
                               "metalabeling_model_features".
        :type X_feature_dict: Dict[str, pd.DataFrame]
        :param y: Pandas DataFrame with class labels for the primary model,
                  where the index is the start_time of the event and the column
                  "event_end_time" indicates when the event has ended.
        :type y: pd.DataFrame
        :param sample_weight: weights to assign to samples in the dataset during
                              fitting, defaults to None
        :type sample_weight: pd.Series, optional
        :param downsampled_indices: series to indicate which samples have been
                                    selected for downsampling, defaults to None
        :type downsampled_indices: pd.DatetimeIndex, optional
        :return: tuple with down samples features X, labels y and sample
                 weights, respectively.
        :rtype: Tuple[Dict[str, pd.DataFrame], pd.DataFrame,
                      pd.Series]
        """
        # Getting only the intersected indices between y and the downsampled
        # indices. This is necessary because during combinatorial cross
        # validation the size of X and y will be reduced and the downsampled
        # indices might not be included
        intersected_downsampled_indices = list(
            sorted(set(y.index).intersection(set(downsampled_indices)))
        )
        y = y.loc[intersected_downsampled_indices]

        if sample_weight:
            sample_weight = sample_weight.loc[intersected_downsampled_indices]

        for model_key in X_feature_dict.keys():
            for dataset_key in X_feature_dict[model_key].keys():
                X_feature_dict[model_key][dataset_key] = X_feature_dict[model_key][
                    dataset_key
                ].loc[intersected_downsampled_indices]

        return X_feature_dict, y, sample_weight

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

    # TODO: downsampled_indices should be per model and not global.
    #  The pipeline won't fit well the use case when is composed of
    #  a model with sequential information  (e.g. moving averages)
    #  and a model that needs the down sampling for (computational or
    #  overfitting) (the same should apply for the sample_weight argument)
    def fit(
        self,
        X_dict: Dict[str, pd.DataFrame],
        y: pd.DataFrame,
        label_column_name: str = "label",
        sample_weight_primary: pd.Series = None,
        sample_weight_metalabeling: pd.Series = None,
        downsampled_indices_primary: pd.DatetimeIndex = None,
        downsampled_indices_metalabeling: pd.DatetimeIndex = None,
        align_on: str = None,
        align_how: Dict[str, str] = None,
    ):
        """
        Fit the strategy pipeline

        :param X_dict: Dictionary containing all the features for the data
                       for the primary and metalabeling model. The data can be
                       bar and/or tick data
        :type X_dict:  Dict[str, pd.DataFrame]
        :param y: Pandas DataFrame with class labels for the primary model,
                  where the index is the start_time of the event and the column
                  "event_end_time" indicates when the event has ended.
        :type y: pd.DataFrame
        :param label_column_name: The name of the column containing the label
        :type label_column_name: str
        :param sample_weight_primary: The sample weights for training the primary model
        :type sample_weight_primary: pd.Series
        :param sample_weight_metalabeling: The sample weights for training the metalabeling model
        :type sample_weight_metalabeling: pd.Series
        :param downsampled_indices_primary: The indices to use for training the
                                            primary model, they should be a subset
                                            of the indices in the dataframes
        :type downsampled_indices_primary: pd.DatetimeIndex
        :param downsampled_indices_metalabeling: The indices to use for training the
                                            metalabeling model, they should be a subset
                                            of the indices in the dataframes
        :type downsampled_indices_metalabeling: pd.DatetimeIndex
        :param align_on: The name of the input dataframe we want to align the
                         other dataframes to
        :type align_on: str
        :param align_how: The methodologies to use for alignment of the
                          dataframes
        :type align_how: Dict[str, str]
        :return: The strategy signal pipeline
        :rtype: StrategySignalPipeline
        """
        assert set(np.unique(y[label_column_name].dropna())).issubset({-1, 0, 1})

        # setting the align_on and align_how variable
        self._set_align_on(X_dict, align_on, align_how)

        # check if the align_on and align_how variable are set correctly
        self._check_align_on(X_dict, self.align_on_, self.align_how_)

        # transforming the X_dict using the specified transformers
        X_features_dict = self.transform(X_dict)

        # Aligning the dictionaries to the data from which the
        # target has been constructed
        X_features_dict = self._align_on(X_features_dict)

        # if downsample_indices is different than None then
        # the attribute downsample_indices is updated with the new value.
        # This help to refit the model keeping the information of the
        # downsampled indices
        if downsampled_indices_primary is not None:
            self.downsampled_indices_primary = downsampled_indices_primary
        if downsampled_indices_metalabeling is not None:
            self.downsampled_indices_metalabeling = downsampled_indices_metalabeling

        # if sample_weight is different than None then the attribute
        # sample_weight is updated with the new value.
        # This help to refit the model keeping the information of the
        # sample weight
        if sample_weight_primary is not None:
            self.sample_weight_primary = sample_weight_primary
        if sample_weight_metalabeling is not None:
            self.sample_weight_metalabeling = sample_weight_metalabeling

        # If downsampled_indices_primary is set2 for the primary model indices
        # then we reduced the input data to the selected indices
        if self.downsampled_indices_primary is not None:
            (
                X_features_dict_primary,
                y_primary,
                sample_weight_primary,
            ) = self._down_sample(
                X_features_dict[self._primary_model_features],
                y,
                self.sample_weight_primary,
                self.downsampled_indices_primary,
            )
        else:
            X_features_dict_primary, y_primary, sample_weight_primary = (
                X_features_dict[self._primary_model_features],
                y,
                self.sample_weight_primary,
            )

        # If downsampled_indices_primary is set2 for the metalabeling model indices
        # then we reduced the input data to the selected indices
        if self.downsampled_indices_metalabeling is not None:
            (
                X_features_dict_metalabeling,
                y_metalabeling,
                sample_weight_metalabeling,
            ) = self._down_sample(
                X_features_dict[self._metalabeling_model_features],
                y,
                self.sample_weight_metalabeling,
                self.downsampled_indices_metalabeling,
            )
        else:
            X_features_dict_metalabeling, y_metalabeling, sample_weight_metalabeling = (
                X_features_dict[self._metalabeling_model_features],
                y,
                self.sample_weight_metalabeling,
            )

        # align X_dict a dict consisting of dataframes with features and y
        (
            X_primary_aligned,
            y_primary_aligned,
            sample_weight_primary_aligned,
        ) = self._align_X_dict_and_y(
            X_features_dict_primary,
            y_primary[label_column_name],
            sample_weight=sample_weight_primary,
        )

        # fit primary model
        self.primary_model.fit(
            X_primary_aligned,
            y_primary_aligned,
            sample_weight=sample_weight_primary_aligned,
        )

        # If metalabeing exists then we fit it
        if self.metalabeling_model is not None:
            self._fit_metalabeling_model(
                X_features_dict_metalabeling,
                X_primary_aligned,
                y_metalabeling,
                y_primary_aligned,
            )
        return self

    def _set_cv(self, y: pd.DataFrame, y_primary_aligned: pd.Series) -> None:
        """
        Initialise the combinatorial cross validation object

        :param y: Pandas DataFrame with class labels for the primary model, where
                  the index is the start_time of the event and the column
                  "event_end_time" indicates when the event has ended.
        :type y: pd.DataFrame
        :param y_primary_aligned: The primary model labels aligned with
                                  the features.
        :type y_primary_aligned: pd.Series
        :return: None
        """
        # setting this higher than 1 will result in duplicate cv metalabels
        cpcv_num_test = 1
        pred_times = y.loc[y_primary_aligned.index].index.to_series()
        eval_times = y.loc[y_primary_aligned.index, EVENT_END_TIME]
        self._cv = CombPurgedKFoldCV(
            n_groups=self.cpcv_num_groups,
            n_test_splits=cpcv_num_test,
            pred_times=pred_times,
            eval_times=eval_times,
            embargo_td=self.embargo_td,
        )

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

    def _get_pred_features_and_metalabels(
        self,
        X_primary_aligned: pd.DataFrame,
        y: pd.DataFrame,
        y_primary_aligned: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Computes the cross validated predictions and probabilities of primary model
        and the cross validated metalabels.

        The features and the labels will be used to fit the metalabeling model.

        :param X_primary_aligned: DataFrame with features of the primary model.
        :type X_primary_aligned: pd.DataFrame
        :param y: Pandas DataFrame with class labels for the primary model, where
                  the index is the start_time of the event and the column
                  "event_end_time" indicates when the event has ended.
        :type y: pd.DataFrame
        :param y_primary_aligned: Pandas Series with class labels to fit the primary model,
                                  which are aligned with the features in the primary model.
        :type y_primary_aligned: pd.Series
        :return: Tuple with the predictive features to be used in the metalabeling model,
                 and the metalabels, which are used as class labels when fitting the
                 metalabeling model.
        :rtype: Tuple[pd.DataFrame, pd.Series]
        """

        # prepare for cv
        self._set_cv(y, y_primary_aligned)
        primary_model_copy = deepcopy(self.primary_model)
        y_metalabel = pd.Series(dtype=np.float64)

        if self.metalabeling_use_predictions_primary_model:
            columns = ["primary_side_pred"]
        else:
            columns = []

        X_pred_features = pd.DataFrame(columns=columns, index=y_primary_aligned.index)
        # fitting and predicting the primary model on
        # the cv splits for the creation of the
        # metalabeling labels
        for train_index, val_index in self._cv.split(
            X=X_primary_aligned, y=y_primary_aligned
        ):
            X_train = X_primary_aligned.iloc[train_index]
            y_train = y_primary_aligned.iloc[train_index]
            X_val = X_primary_aligned.iloc[val_index]
            y_val = y_primary_aligned.iloc[val_index]
            primary_model_copy.fit(X_train, y_train)

            y_pred_val = primary_model_copy.predict(X_val)
            # create features based on pred and/or pred_proba
            if self.metalabeling_use_predictions_primary_model:
                X_pred_features.loc[y_val.index, "primary_side_pred"] = y_pred_val

            if self.metalabeling_use_proba_primary_model:
                X_pred_features = self._get_pred_proba_features(
                    primary_model_copy, X_pred_features, X_val, y_val
                )

            # create metalabels of split
            y_metalabel_cv = (y_pred_val == y_val).astype(float)
            y_metalabel = pd.concat([y_metalabel, y_metalabel_cv])

        # sort indices
        X_pred_features.sort_index(inplace=True)
        y_metalabel.sort_index(inplace=True)

        self._check_y_metalabel(y_metalabel)

        return X_pred_features, y_metalabel

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

    def _fit_metalabeling_model(
        self,
        X_features_dict: Dict[str, pd.DataFrame],
        X_primary_aligned: pd.DataFrame,
        y: pd.DataFrame,
        y_primary_aligned: pd.Series,
    ) -> None:
        """
        Fit the metalabeling model

        :param X_features_dict: Dict containing the transformed features for the
                                primary and metalabeling model
        :type X_features_dict: Dict[str, pd.DataFrame]
        :param X_primary_aligned: DataFrame with features of the primary model.
        :type X_primary_aligned: pd.DataFrame
        :param y: target with its info not aligned
        :type y: pd.DataFrame
        :param y_primary_aligned: primary target aligned
        :type y_primary_aligned: pd.Series
        """
        # get the predictions from the primary model
        X_pred_features, y_metalabel = self._get_pred_features_and_metalabels(
            X_primary_aligned, y, y_primary_aligned
        )
        X_features_dict["primary_prediction_features"] = X_pred_features

        # align features and sample weight with target
        (
            X_metalabel_aligned,
            y_metalabel_aligned,
            sample_weight_metalabel_aligned,
        ) = self._align_X_dict_and_y(X_features_dict, y_metalabel)

        self.metalabeling_model.fit(
            X_metalabel_aligned,
            y_metalabel_aligned,
            sample_weight=sample_weight_metalabel_aligned,
        )

    @staticmethod
    def _align_X_dict_and_y(
        X_dict: Dict[str, pd.DataFrame],
        y: pd.Series,
        sample_weight: Union[None, pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Align the features in X and the target in y

        :param X_dict: A dictionary containing features for the primary and
                       metalabeling model
        :type X_dict: Dict[str, pd.DataFrame]
        :param y: The target to which the features need to be aligned with
        :type y: pd.Series
        :param sample_weight: The sample weights
        :type sample_weight: pd.Series
        :return: Features, target and sample weights aligned
        :rtype: Tuple[pd.DataFrame, pd.Series, pd.Series]
        """
        y_no_nans = y.dropna()
        # TODO: will not work with bars with different indices (volume vs dollar for example)
        # probably need to use methods from df.index (index search sorted or something)
        X = pd.concat([df.loc[y_no_nans.index] for df in X_dict.values()], axis=1)
        X_aligned = X.dropna()
        y_aligned = y.loc[X_aligned.index]
        check_X_y(X_aligned, y_aligned)

        if sample_weight is not None:
            sample_weight = sample_weight.loc[y_aligned.index]

        return X_aligned, y_aligned, sample_weight

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

    def _set_align_on(
        self,
        X_dict: Dict[str, pd.DataFrame],
        align_on: Union[None, str] = None,
        align_how: Union[Dict[str, str], None] = None,
    ) -> None:
        """
        Set the alignment variables (align_on_ and align_how) to the pipeline
        instance.

        :param X_dict: Dictionary containing the dataframe for the primary and
                       metalabeling model
        :type X_dict: Dict[str, pd.DataFrame]
        :param align_on: The key of the dataframe that leads the alignment of
                         the other dataframes
        :type align_on: str
        :param align_how: The methods to use for the alignment
        :type align_how: Dict[str, str]
        :return: None
        """
        # If align_on is specified we save it in the object along with
        # the align_how dict otherwise we expect to have only one element
        # in X_dict key and we save the key in align_on
        if align_on:
            self.align_on_ = align_on
            self.align_how_ = align_how
        # if self.align_on_ isn't set and align_on is None and
        # X_dict has only one key then we set align_on
        # with the first key of the dict
        elif (
            align_on is None
            and not hasattr(self, "align_on_")
            and len(X_dict.keys()) == 1
        ):
            self.align_on_ = list(X_dict.keys())[0]
            self.align_how_ = None

        elif align_on is None and hasattr(self, "align_on_"):
            logging.info("align_on_ is already set")

    def _check_align_on(
        self,
        X_dict: Dict[str, pd.DataFrame],
        align_on: Union[None, str] = None,
        align_how: Dict[str, str] = None,
    ) -> None:
        """
        Checks if align on is set correctly, otherwise raise ValueError

        :param X_dict: A dictionary containing features for the primary and
                       metalabeling model
        :type X_dict: Dict[str, pd.DataFrame]
        :param align_on: The name of the key in X_dict on which the other
                         dataframes will be aligned on.
        :return: None
        """
        n_sources = len(X_dict.keys())

        # Nothing to check, there is only one data source which is assumed to
        # be the target datasource
        if n_sources == 1:
            return

        if not align_on and (n_sources) > 1:
            raise ValueError(
                f"{n_sources} data sources detected, please specify align_on to indicate how to align the features."
            )

        if align_on and align_on not in X_dict.keys():
            raise ValueError(
                f"Value of align_on {align_on}, should be "
                f"equal to the one of the keys in X_dict {list(X_dict.keys())}."
            )

        if align_on and not align_how:
            raise ValueError("align_how is not specified")

        # TODO: this message won't be very readable when align_how has more keys
        #  than expected.
        if align_on and not (set(X_dict.keys()) - {align_on}) == set(align_how.keys()):
            raise ValueError(
                f"align_how does not have the key/s {(set(X_dict.keys()) - {align_on}) - set(align_how.keys())}"
            )

        if align_on and not set(self._align_methods.keys()).issuperset(
            set(align_how.values())
        ):
            raise ValueError(
                f"The method/s {', '.join(set(align_how.values()) - set(self._align_methods.keys()))} is not implemented."
                f" The allowed alignment methods are {', '.join(self._align_methods.keys())}"
            )

    def _get_features_for_metalabeling_prediction(
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
        check_is_fitted(self.primary_model)
        if self.metalabeling_model:
            check_is_fitted(self.metalabeling_model)

        self._check_align_on(X_dict, self.align_on_, self.align_how_)

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
            X_metalabel_aligned = self._get_features_for_metalabeling_prediction(
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
        # check if the primary model is fitted
        check_is_fitted(self.primary_model)

        # check align_on and how are set correctly
        self._check_align_on(X_dict, self.align_on_, self.align_how_)

        # transforming and aligning X_dict
        X_features_dict = self.transform(X_dict)
        X_features_dict = self._align_on(X_features_dict)

        # concatening all the features dataframes in
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
            X_metalabel_aligned = self._get_features_for_metalabeling_prediction(
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

    def get_size(self, X_dict: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        Calculate the size of the position

        :param X_dict: A dictionary containing features for the primary and
                       meta-labeling model
        :type X_dict: Dict[str, pd.DataFrame]
        :return: The sizes of the positions
        :rtype: pd.Series
        """
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
            return self.bet_sizer.transform(pred_and_proba)

        else:
            if self.metalabeling_model:
                return predicted_proba[1.0]
            else:
                return predicted_proba.max(axis=1)

    def get_side(self, X_dict: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        Calculate the side of the position

        :param X_dict: A dictionary containing features for the primary and
                       meta-labeling model
        :type X_dict: Dict[str, pd.DataFrame]
        :return: The sides of the positions
        :rtype: pd.Series
        """
        return self.predict(X_dict=X_dict)[self._primary_model_predictions]


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
        # Check if the strategy pipeline is fitted
        check_is_fitted(strategy_pipeline, msg="Strategy pipeline should be fitted")

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
        signal_df = self.strategy_pipeline.get_side(X_dict).to_frame(name=SIDE)
        signal_df[SIZE] = self.strategy_pipeline.get_size(X_dict)

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
