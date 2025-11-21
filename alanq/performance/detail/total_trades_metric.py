import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class TotalTradesMetric(BaseDetailedMetric):
    """
    總交易次數指標
    """
    metric_name = '總交易次數'
    higher_is_better = None  # 交易次數沒有好壞之分
    target = None
    description = '總交易次數'
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算總交易次數"""
        return {'total_trades': len(trades)}

