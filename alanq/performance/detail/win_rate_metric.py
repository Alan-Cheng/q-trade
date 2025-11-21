import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class WinRateMetric(BaseDetailedMetric):
    """
    勝率指標
    計算公式：獲利交易次數 / 總交易次數
    """
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算勝率"""
        if len(trades) == 0:
            return {'win_rate': 0}
        
        winning = existing_stats.get('winning_trades', len(trades[trades['pnl'] > 0]))
        total = existing_stats.get('total_trades', len(trades))
        
        win_rate = winning / total if total > 0 else 0
        return {'win_rate': win_rate}

