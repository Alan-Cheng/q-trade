"""
詳細績效指標模組（基於交易紀錄）
"""

from .base_detailed_metric import BaseDetailedMetric
from .win_rate_metric import WinRateMetric
from .total_trades_metric import TotalTradesMetric
from .winning_trades_metric import WinningTradesMetric
from .losing_trades_metric import LosingTradesMetric
from .profit_loss_ratio_metric import ProfitLossRatioMetric
from .avg_profit_metric import AvgProfitMetric
from .avg_loss_metric import AvgLossMetric
from .net_profit_metric import NetProfitMetric
from .max_single_profit_metric import MaxSingleProfitMetric
from .max_single_loss_metric import MaxSingleLossMetric
from .avg_holding_days_metric import AvgHoldingDaysMetric
from .avg_return_metric import AvgReturnMetric
from .max_consecutive_wins_metric import MaxConsecutiveWinsMetric
from .max_consecutive_losses_metric import MaxConsecutiveLossesMetric

# 保留舊的合併類別作為向後相容（已廢棄）
from .basic_trade_stats_metric import BasicTradeStatsMetric
from .profit_loss_stats_metric import ProfitLossStatsMetric
from .extreme_trades_metric import ExtremeTradesMetric
from .holding_stats_metric import HoldingStatsMetric
from .consecutive_trades_metric import ConsecutiveTradesMetric

__all__ = [
    'BaseDetailedMetric',
    # 新的單一指標類別（推薦使用）
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
    # 舊的合併類別（已廢棄，保留作為向後相容）
    'BasicTradeStatsMetric',
    'ProfitLossStatsMetric',
    'ExtremeTradesMetric',
    'HoldingStatsMetric',
    'ConsecutiveTradesMetric',
]

