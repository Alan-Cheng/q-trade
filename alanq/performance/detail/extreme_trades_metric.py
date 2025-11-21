import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class ExtremeTradesMetric(BaseDetailedMetric):
    """
    極端交易指標
    計算：最大單筆獲利、最大單筆虧損
    """
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算極端交易"""
        if len(trades) == 0:
            return {
                'max_single_profit': 0,
                'max_single_loss': 0,
            }
        
        return {
            'max_single_profit': trades['pnl'].max(),
            'max_single_loss': trades['pnl'].min(),
        }

