import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class AvgProfitMetric(BaseDetailedMetric):
    """
    平均獲利指標
    計算公式：獲利交易的平均獲利金額
    """
    metric_name = '平均獲利'
    higher_is_better = True
    target = None
    description = '平均獲利'
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算平均獲利"""
        if len(trades) == 0:
            return {'avg_profit': 0}
        
        winning_trades = trades[trades['pnl'] > 0]
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        
        return {'avg_profit': avg_win}

