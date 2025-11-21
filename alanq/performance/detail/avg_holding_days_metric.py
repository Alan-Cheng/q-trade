import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class AvgHoldingDaysMetric(BaseDetailedMetric):
    """
    平均持倉天數指標
    """
    metric_name = '平均持倉天數'
    higher_is_better = None  # 持倉天數沒有好壞之分
    target = None
    description = '平均持倉天數'
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算平均持倉天數"""
        if len(trades) == 0:
            return {'avg_holding_days': 0}
        
        avg_holding = trades['holding_days'].mean() if 'holding_days' in trades.columns else 0
        return {'avg_holding_days': avg_holding}

