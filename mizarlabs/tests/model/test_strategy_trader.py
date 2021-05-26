from unittest.mock import MagicMock
from mizarlabs.model.pipeline import StrategyTrader
import pandas as pd
from mizarlabs.static import STOP_LOSS
from mizarlabs.static import PROFIT_TAKING


def test_create_strategy_bars_static_pnl_taking(dollar_bar_dataframe: pd.DataFrame):
    strategy_pipeline = MagicMock()
    strategy_pipeline.align_on_ = 'bar'
    X_dict = {"bar": dollar_bar_dataframe}
    strategy_trader = StrategyTrader(
        strategy_pipeline=strategy_pipeline,
        min_num_bars=1,
        num_expiration_bars=1,
        stop_loss_factor=0.21,
        profit_taking_factor=0.19,
        volatility_adjusted_stop_loss=False,
    )
    strategy_bars = strategy_trader.create_strategy_bars(X_dict)
    assert (strategy_bars[STOP_LOSS].values == 0.21).all()
    assert (strategy_bars[PROFIT_TAKING].values == 0.19).all()
