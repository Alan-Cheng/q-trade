"""
因子模組
提供選股因子和擇時因子
"""

# 從子模組導出常用類別
from .timing import (
    BaseBuyFactor,
    BaseSellFactor,
    BreakoutBuyFactor,
    BreakdownSellFactor,
    AtrStopSellFactor,
    CloseAtrStopSellFactor,
    RiskStopSellFactor,
)
from .selection import (
    StockPickerBase,
    RegressAnglePicker,
    RPSPicker,
)

__all__ = [
    # 擇時因子
    'BaseBuyFactor',
    'BaseSellFactor',
    'BreakoutBuyFactor',
    'BreakdownSellFactor',
    'AtrStopSellFactor',
    'CloseAtrStopSellFactor',
    'RiskStopSellFactor',
    # 選股因子
    'StockPickerBase',
    'RegressAnglePicker',
    'RPSPicker',
]

