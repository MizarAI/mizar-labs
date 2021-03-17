import pandas as pd
from mizarlabs.transformers.sampling.down_sampling import CUSUMFilter


def test_CUSUM_filter():
    """
    CUSUM filter should only select the sample when the price spike occurs (up- or downwards)
    """
    frequency = "d"
    index = pd.date_range(start="2018-01-01", end="2018-05-01", freq=frequency)

    price = [1 if i < int(len(index) / 2) else 3 for i in range(len(index))]
    one_price_increase = pd.Series(price, index=index)

    price = [3 if i < int(len(index) / 2) else 1 for i in range(len(index))]
    one_price_decrease = pd.Series(price, index=index)

    cusum_filter = CUSUMFilter(1)
    filtered_indices = cusum_filter.transform(one_price_increase)

    assert len(filtered_indices) == 1
    assert filtered_indices[0] == index[int(len(index) / 2)]

    filtered_indices = cusum_filter.transform(one_price_decrease)

    assert len(filtered_indices) == 1
    assert filtered_indices[0] == index[int(len(index) / 2)]
