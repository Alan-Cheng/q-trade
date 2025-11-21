import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class WinningTradesMetric(BaseDetailedMetric):
    """
    獲利交易次數指標
    """
    metric_name = '獲利交易次數'
    higher_is_better = True
    target = None
    description = '獲利交易次數'
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算獲利交易次數"""
        if len(trades) == 0:
            return {'winning_trades': 0}
        
        return {'winning_trades': len(trades[trades['pnl'] > 0])}

