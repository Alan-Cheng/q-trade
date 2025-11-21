import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class MaxSingleLossMetric(BaseDetailedMetric):
    """
    最大單筆虧損指標
    """
    metric_name = '最大單筆虧損'
    higher_is_better = False  # 虧損越小越好
    target = None
    description = '最大單筆虧損'
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算最大單筆虧損"""
        if len(trades) == 0:
            return {'max_single_loss': 0}
        
        return {'max_single_loss': trades['pnl'].min()}

