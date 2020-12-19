from string import ascii_letters
from string import digits

import numpy as np
import pandas as pd
import pytest
import talib
from mizarlabs.transformers.technical.factory import TAFactory
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer


@pytest.mark.parametrize("func_name", talib.get_functions())
def test_create_transformer(dollar_bar_dataframe: pd.DataFrame, func_name: str):
    """
    Checks if the transformer can be created and if it can transform a bar dataframe into a feature array.
    """
    # the MAVP indicator takes in a special array with integers
    if func_name == "MAVP":
        dollar_bar_dataframe["periods"] = 5
    transformer = create_transformer(func_name)
    X_transformed = transformer.transform(dollar_bar_dataframe)
    if not isinstance(X_transformed, np.ndarray):
        raise TypeError(f"Expected a numpy array, but got {type(X_transformed)}.")


def create_transformer(func_name: str) -> FunctionTransformer:
    """
    Helper function to create a FunctionTransformer.
    """
    ta_factory = TAFactory()
    args, _ = ta_factory.get_args_kwargs(func_name)

    # set the kwargs correctly depending on the selected function to test
    if "volume" in args:
        kwargs = {
            "col_to_transform": "close",
            "col_to_transform_mapping": {"volume": "base_asset_volume"},
        }
    elif "real0" in args and "real1" in args:
        kwargs = {"col_to_transform": ["open", "close"]}
    elif "real" in args:
        kwargs = {"col_to_transform": "close"}
    else:
        kwargs = {}
    transformer = ta_factory.create_transformer(func_name, **kwargs)
    return transformer


def test_mass_feature_generation(dollar_bar_dataframe: pd.DataFrame):
    """
    Checks if all the transformers can be combined in a FeatureUnion to create features simultaneously
    """
    # the MAVP indicator takes in a special array with integers
    dollar_bar_dataframe["periods"] = 5
    ta_factory = TAFactory()
    ta_funcs = ta_factory.get_functions()
    feature_generator = FeatureUnion(
        [(name, create_transformer(name)) for name in ta_funcs], n_jobs=-1
    )
    X_transformed = feature_generator.transform(dollar_bar_dataframe)
    assert isinstance(X_transformed, np.ndarray)


def test_get_function_groups():
    """
    Checks if function by group dictionary is correct.
    """
    assert talib.get_function_groups() == TAFactory().get_function_groups()


def test_get_functions():
    """
    Checks if available functions are as expected.
    """
    assert talib.get_functions() == TAFactory().get_functions()


def test_get_args_kwargs():
    """
    Checks if args and kwargs are valid var names
    """
    funcs = talib.get_functions()
    allowed_chars = set(ascii_letters + digits + "_")
    for func_name in funcs:
        args, kwargs = TAFactory().get_args_kwargs(func_name)
        assert all(set(i).issubset(allowed_chars) for i in args)
        if kwargs:
            assert all(set(i).issubset(allowed_chars) for i in kwargs)


def test_get_ta_func():
    """
    Checks if errors are raised correctly.
    """
    with pytest.raises(ValueError):
        TAFactory().get_ta_func("")

    for func in talib.get_functions():
        assert talib.__dict__.get(func) == TAFactory().get_ta_func(func)


def test_check_and_set_args():
    """
    Checks if errors are raised correctly.
    """
    with pytest.raises(ValueError, match=r".*col_to_transform.*"):
        TAFactory()._check_and_set_args("SMA", ["real"], None, None)

    col_to_transform = "col"
    args_names_list = TAFactory()._check_and_set_args(
        "SMA", ["real"], col_to_transform, None
    )
    assert args_names_list == [col_to_transform]

    with pytest.raises(ValueError, match=r".*two columns.*"):
        TAFactory()._check_and_set_args("CORR", ["real0", "real1"], None, None)

    col_to_transform = ["col0", "col1"]
    args_names_list = TAFactory()._check_and_set_args(
        "CORR", ["real0", "real1"], col_to_transform, None
    )
    assert col_to_transform == args_names_list

    with pytest.raises(TypeError, match=r"Expected a dictionary.*"):
        TAFactory()._check_and_set_args(
            "VWAP",
            ["real", "volume"],
            col_to_transform="col",
            col_to_transform_mapping="",
        )

    with pytest.raises(ValueError, match=r".*only valid arguments.*"):
        TAFactory()._check_and_set_args(
            "VWAP",
            ["real", "volume"],
            col_to_transform="col",
            col_to_transform_mapping={"vol": "col0"},
        )

    args_names_list = TAFactory()._check_and_set_args(
        "VWAP",
        ["real", "volume"],
        col_to_transform="col",
        col_to_transform_mapping={"volume": "col0"},
    )
    assert ["col", "col0"] == args_names_list


def test_check_and_set_kwargs():
    """
    Checks if errors are raised correctly.
    """
    with pytest.raises(ValueError, match=r"Selected function.*"):
        TAFactory()._check_and_set_kwargs("SMA", kw_args={}, kwargs_names_list=None)

    with pytest.raises(ValueError, match=r"The kwargs.*"):
        TAFactory()._check_and_set_kwargs(
            "SMA", kw_args={"blah": 1}, kwargs_names_list=["blahh"]
        )

    kwargs = TAFactory()._check_and_set_kwargs(
        "SMA", kw_args={"blah": 1}, kwargs_names_list=["blah"]
    )
    assert kwargs == {"blah": 1}

    kwargs = TAFactory()._check_and_set_kwargs(
        "SMA", kw_args=None, kwargs_names_list=["blah"]
    )
    assert kwargs == dict()


def test_create_transformer_with_kwargs():
    """
    Check if kwarg is set correctly and results in a similar transformation
    compared to directly used the talib package.
    """
    func_name = "SMA"
    X = pd.DataFrame(np.arange(50), columns=["some_col"])
    transformer = TAFactory().create_transformer(
        func_name, "some_col", kw_args={"timeperiod": 5}
    )
    X_transformed = transformer.transform(X)
    np.testing.assert_array_equal(
        X_transformed.flatten(), talib.SMA(np.arange(50).astype(float), timeperiod=5)
    )


if __name__ == "__main__":
    pytest.main([__file__])
