import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class AvgReturnMetric(BaseDetailedMetric):
    """
    平均報酬率指標
    """
    metric_name = '平均報酬率'
    higher_is_better = True
    target = None
    description = '平均報酬率'
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算平均報酬率"""
        if len(trades) == 0:
            return {'avg_return': 0}
        
        avg_return = trades['return_pct'].mean()
        return {'avg_return': avg_return}

