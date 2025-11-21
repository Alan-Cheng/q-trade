import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class MaxConsecutiveWinsMetric(BaseDetailedMetric):
    """
    最大連續獲利次數指標
    """
    metric_name = '最大連續獲利次數'
    higher_is_better = True
    target = None
    description = '最大連續獲利次數'
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算最大連續獲利次數"""
        if len(trades) == 0:
            return {'max_consecutive_wins': 0}
        
        trades_sorted = trades.sort_values('entry_date')
        is_profit = (trades_sorted['pnl'] > 0).astype(int)
        
        max_wins = 0
        current_wins = 0
        for profit in is_profit:
            if profit == 1:
                current_wins += 1
                max_wins = max(max_wins, current_wins)
            else:
                current_wins = 0
        
        return {'max_consecutive_wins': max_wins}

