import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class MaxConsecutiveLossesMetric(BaseDetailedMetric):
    """
    最大連續虧損次數指標
    """
    metric_name = '最大連續虧損次數'
    higher_is_better = False  # 連續虧損次數越少越好
    target = None
    description = '最大連續虧損次數'
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算最大連續虧損次數"""
        if len(trades) == 0:
            return {'max_consecutive_losses': 0}
        
        trades_sorted = trades.sort_values('entry_date')
        is_loss = (trades_sorted['pnl'] < 0).astype(int)
        
        max_losses = 0
        current_losses = 0
        for loss in is_loss:
            if loss == 1:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        
        return {'max_consecutive_losses': max_losses}

