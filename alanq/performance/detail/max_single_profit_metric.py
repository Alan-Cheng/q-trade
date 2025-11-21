import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class MaxSingleProfitMetric(BaseDetailedMetric):
    """
    最大單筆獲利指標
    """
    metric_name = '最大單筆獲利'
    higher_is_better = True
    target = None
    description = '最大單筆獲利'
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算最大單筆獲利"""
        if len(trades) == 0:
            return {'max_single_profit': 0}
        
        return {'max_single_profit': trades['pnl'].max()}

