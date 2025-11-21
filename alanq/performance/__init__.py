"""
績效評估模組
提供交易績效指標計算和顯示功能
"""

from .performance_metrics import PerformanceMetrics

# 導出基本指標類別
from .basic import (
    BaseBasicMetric,
    TotalReturnMetric,
    AnnualReturnMetric,
    VolatilityMetric,
    SharpeMetric,
    MaxDrawdownMetric,
    FinalEquityMetric,
)

# 導出詳細指標類別
from .detail import (
    BaseDetailedMetric,
    BasicTradeStatsMetric,
    WinRateMetric,
    ProfitLossStatsMetric,
    ExtremeTradesMetric,
    HoldingStatsMetric,
    ConsecutiveTradesMetric,
)

__all__ = [
    'PerformanceMetrics',
    # 基本指標
    'BaseBasicMetric',
    'TotalReturnMetric',
    'AnnualReturnMetric',
    'VolatilityMetric',
    'SharpeMetric',
    'MaxDrawdownMetric',
    'FinalEquityMetric',
    # 詳細指標
    'BaseDetailedMetric',
    'BasicTradeStatsMetric',
    'WinRateMetric',
    'ProfitLossStatsMetric',
    'ExtremeTradesMetric',
    'HoldingStatsMetric',
    'ConsecutiveTradesMetric',
]

