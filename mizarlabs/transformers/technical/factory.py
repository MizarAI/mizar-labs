import re
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import attr
import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import FunctionTransformer


@attr.s
class TAFactory:
    """
    Factory that creates sklearn transformers for function available in the TA-lib package.
    """

    ta_func_dict = talib.__dict__
    _real = "real"

    @staticmethod
    def get_function_groups() -> Dict[str, str]:
        """
        Returns a dictionary with TA-lib function names specified per group.

        :return: the key-value pairs are group names by list of function names
        :rtype: Dict[str, str]
        """
        return talib.get_function_groups()

    @staticmethod
    def get_functions() -> List[str]:
        """
        Returns all the available function names in TA-lib package.

        :return: list with function names in the TA-lib package.
        :rtype: List[str]
        """
        return talib.get_functions()

    def get_args_kwargs(self, func_name: str) -> Tuple[List[str]]:
        """
        Returns a tuple with a list of valid argument names and keyword argument names for the TA function.

        :param func_name: function name in the TA-lib package
        :type func_name: str
        :return: first element is the list with valid argument names and the second element is the list with
                 valid keyword argument names.
        :rtype: Tuple[List[str]]
        """
        ta_func = self.get_ta_func(func_name)
        signature = re.search(r"\((.+)\)", ta_func.__doc__).group(1)

        args_kwargs_names = signature.split("[, ")
        if len(args_kwargs_names) == 1:
            args_names, kwargs_names = args_kwargs_names[0], None
        else:
            args_names, kwargs_names = args_kwargs_names

        args_names_list = args_names.split(", ")

        if kwargs_names is not None:
            kwargs_names_list = kwargs_names.strip("=?]").split("=?, ")
        else:
            kwargs_names_list = None
        return args_names_list, kwargs_names_list

    def get_ta_func(self, func_name: str) -> callable:
        """
        Returns the TA function callable.

        :param func_name: name of the function in TA-lib package
        :type func_name: str
        :raises ValueError: if provided name does not exist as a function in the TA-lib package an error is raised
        :return: the TA function which creates TA features
        :rtype: callable
        """
        if func_name not in self.ta_func_dict.keys():
            raise ValueError(
                f"The func_name {func_name} is not available. Please check if the name is spelled correctly."
            )
        ta_func = self.ta_func_dict.get(func_name)
        return ta_func

    def create_transformer(
        self,
        func_name: str,
        col_to_transform: Union[List[str], str] = None,
        col_to_transform_mapping: Dict[str, str] = None,
        kw_args: dict = None,
    ) -> FunctionTransformer:
        """
        Creates a transformer for a given function available in the TA-lib package.

        :param func_name: name of function in TA-lib package
        :type func_name: str
        :param col_to_transform: name of the column in dataframe to transform, defaults to None
        :type col_to_transform: Union[List[str], str], optional
        :param col_to_transform_mapping: remapping of names in dataframe to valid arg names in the TA function,
                                         defaults to None
        :type col_to_transform_mapping: Dict[str, str], optional
        :param kw_args: kwargs for the TA function, defaults to None
        :type kw_args: dict, optional
        :raises TypeError: transformer only works on pandas DataFrames
        :raises ValueError: transformer expect certain columns to be available in the dataframe to transform
        :return: scikit-learn FunctionTransformer version of the TA function
        :rtype: FunctionTransformer
        """

        ta_func = self.get_ta_func(func_name)

        args_names_list, kwargs_names_list = self.get_args_kwargs(func_name)

        args_names_list = self._check_and_set_args(
            func_name, args_names_list, col_to_transform, col_to_transform_mapping
        )

        kw_args = self._check_and_set_kwargs(func_name, kw_args, kwargs_names_list)

        def ta_func_transformed(X: pd.DataFrame) -> np.ndarray:
            if not isinstance(X, pd.DataFrame):
                raise TypeError(
                    f"X must be a pandas DataFrame, but got type {type(X)}."
                )
            if not set(X.columns).issuperset(set(args_names_list)):
                raise ValueError(
                    f"Expected the following columns in the dataframe: {set(args_names_list) - set(X.columns)}"
                )
            args = [X[i] for i in args_names_list]
            output = ta_func(*args, **kw_args)

            # parse output to 2-d arrays to enable concatenating
            if isinstance(output, tuple):
                output = pd.concat(output, axis=1).values

            if output.ndim != 2:
                if isinstance(output, pd.Series):
                    output = output.values
                output = output.reshape(-1, 1)

            return output

        return FunctionTransformer(ta_func_transformed)

    def _check_and_set_args(
        self,
        func_name: str,
        args_names_list: List[str],
        col_to_transform: Union[str, None],
        col_to_transform_mapping: Union[str, None],
    ) -> List[str]:
        """
        Checks and sets the args for the TA function.

        :param func_name: name of the TA function
        :type func_name: str
        :param args_names_list: list with valid argument names
        :type args_names_list: List[str]
        :param col_to_transform: if the TA function takes in an array this will specify which column to transform in
                                 the dataframe
        :type col_to_transform: Union[str, None]
        :param col_to_transform_mapping: maps valid argument names to column names in the dataframe to transform
        :type col_to_transform_mapping: Union[str, None]
        :raises ValueError: column name to transform is not provided, i.e. transformer does not know which column
                            in the dataframe to transform
        :raises ValueError: transformer takes in two arrays, so both column need to be specified
        :raises TypeError: the column mapping is not a dictionary
        :raises ValueError: the mapping contains names that are not valid argument names
        :return: list with valid column names to extract from the dataframe to transform
        :rtype: List[str]
        """
        # ta func takes in an array and transforms it directly
        # since X is a DataFrame with multiple columns, it must be
        # specified which column is being transformed
        if self._real in args_names_list:
            if not isinstance(col_to_transform, str):
                raise ValueError(
                    f"The function {func_name} requires col_to_transform to be specified."
                )
            idx_real = args_names_list.index(self._real)
            args_names_list[idx_real] = col_to_transform

        # functions that take in two arrays need to be specified as a list with two column names
        if all(f"real{i}" in args_names_list for i in range(2)):
            if not (
                isinstance(col_to_transform, list)
                and all(isinstance(i, str) for i in col_to_transform)
            ):
                raise ValueError(
                    f"Selected function {func_name} requires two columns to be "
                    "specified to apply the transformation on, please provide a list with column names. "
                )
            for i in range(2):
                idx_real = args_names_list.index(f"real{i}")
                args_names_list[idx_real] = col_to_transform[i]

        # some expected columns can have different names in the dataframe
        # this part will map the expected name in the ta func to the name in the dataframe
        if col_to_transform_mapping is not None:
            if not isinstance(col_to_transform_mapping, dict):
                raise TypeError(
                    f"Expected a dictionary for col_to_transform_mapping, but got {type(col_to_transform_mapping)}."
                )
            if set(col_to_transform_mapping.keys()).issubset(set(args_names_list)):
                for i, arg_name_i in enumerate(args_names_list):
                    if arg_name_i in col_to_transform_mapping.keys():
                        args_names_list[i] = col_to_transform_mapping[arg_name_i]
            else:
                raise ValueError(
                    f"The function {func_name} does not accept "
                    f"{set(col_to_transform_mapping.keys()) - set(args_names_list)} as arguments. "
                    f"The only valid arguments are {args_names_list}."
                )
        return args_names_list

    @staticmethod
    def _check_and_set_kwargs(
        func_name: str,
        kw_args: Union[None, Dict[str, any]],
        kwargs_names_list: List[str],
    ) -> Dict[str, Any]:
        """
        Checks if the provided kwargs for the TA function are valid.

        :param func_name: name of the TA function
        :type func_name: str
        :param kw_args: dictionary with kwargs that are being set for the TA function
        :type kw_args: Union[None, Dict[str, any]]
        :param kwargs_names_list: list with valid kwarg names
        :type kwargs_names_list: List[str]
        :raises error: if the TA function has no kwargs or kwargs do not exist errors are raised
        :raises ValueError: TA function has no kwargs, but kwargs are being set.
        :raises ValueError: TA function does not have all the kwargs that are being set.
        :return: Dictionary with valid kwargs for the TA function.
        :rtype: Dict[str, Any]
        """
        # raise error if kwargs are being set, while there are no kwargs or some do not exist
        if kwargs_names_list is None and kw_args is not None:
            raise ValueError(
                f"Selected function {func_name} has no key word arguments, but got {kw_args}."
            )
        if kwargs_names_list is not None and kw_args is not None:
            if not set(kw_args.keys()).issubset(set(kwargs_names_list)):
                raise ValueError(
                    f"The kwargs {set(kw_args.keys()) - set(kwargs_names_list)} are not valid kwargs "
                    f"for the function {func_name}. Valid kwargs are: {kwargs_names_list}."
                )

        if kw_args is None:
            kw_args = dict()

        return kw_args
