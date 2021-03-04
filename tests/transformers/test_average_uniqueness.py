import pandas as pd
import pytest

from mizarlabs.static import CLOSE
from mizarlabs.transformers.sampling.average_uniqueness import AverageUniqueness
from mizarlabs.transformers.targets.labeling import TripleBarrierMethodLabeling


@pytest.mark.usefixtures("dollar_bar_dataframe")
def test_average_uniqueness(dollar_bar_dataframe: pd.DataFrame):
    """
    Check whether average uniqueness is the same as the manual calculation.
    """
    triple_barrier = TripleBarrierMethodLabeling(
        num_expiration_bars=3, profit_taking_factor=0.1, stop_loss_factor=0.1
    )
    target_labels = triple_barrier.fit_transform(dollar_bar_dataframe[[CLOSE]])

    target_labels = target_labels.dropna()

    avg_uniqueness_transformer = AverageUniqueness()
    avg_uniqueness = avg_uniqueness_transformer.transform(target_labels)
    assert avg_uniqueness.iloc[22] == 13 / 36


if __name__ == "__main__":
    pytest.main([__file__])
