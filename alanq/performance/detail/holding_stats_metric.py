import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class HoldingStatsMetric(BaseDetailedMetric):
    """
    持倉統計指標
    計算：平均持倉天數、平均報酬率
    """
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算持倉統計"""
        if len(trades) == 0:
            return {
                'avg_holding_days': 0,
                'avg_return': 0,
            }
        
        avg_holding = trades['holding_days'].mean() if 'holding_days' in trades.columns else 0
        avg_return = trades['return_pct'].mean()
        
        return {
            'avg_holding_days': avg_holding,
            'avg_return': avg_return,
        }

