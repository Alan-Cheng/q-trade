import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class LosingTradesMetric(BaseDetailedMetric):
    """
    虧損交易次數指標
    """
    metric_name = '虧損交易次數'
    higher_is_better = False  # 虧損交易次數越少越好
    target = None
    description = '虧損交易次數'
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算虧損交易次數"""
        if len(trades) == 0:
            return {'losing_trades': 0}
        
        return {'losing_trades': len(trades[trades['pnl'] < 0])}

