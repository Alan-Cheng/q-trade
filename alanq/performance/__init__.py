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
    # 新的單一指標類別（推薦使用）
    WinRateMetric,
    TotalTradesMetric,
    WinningTradesMetric,
    LosingTradesMetric,
    ProfitLossRatioMetric,
    AvgProfitMetric,
    AvgLossMetric,
    NetProfitMetric,
    MaxSingleProfitMetric,
    MaxSingleLossMetric,
    AvgHoldingDaysMetric,
    AvgReturnMetric,
    MaxConsecutiveWinsMetric,
    MaxConsecutiveLossesMetric,
    # 舊的合併類別（已廢棄，保留作為向後相容）
    BasicTradeStatsMetric,
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
    # 詳細指標（新的單一指標類別）
    'BaseDetailedMetric',
    'WinRateMetric',
    'TotalTradesMetric',
    'WinningTradesMetric',
    'LosingTradesMetric',
    'ProfitLossRatioMetric',
    'AvgProfitMetric',
    'AvgLossMetric',
    'NetProfitMetric',
    'MaxSingleProfitMetric',
    'MaxSingleLossMetric',
    'AvgHoldingDaysMetric',
    'AvgReturnMetric',
    'MaxConsecutiveWinsMetric',
    'MaxConsecutiveLossesMetric',
    # 詳細指標（舊的合併類別，已廢棄）
    'BasicTradeStatsMetric',
    'ProfitLossStatsMetric',
    'ExtremeTradesMetric',
    'HoldingStatsMetric',
    'ConsecutiveTradesMetric',
]

