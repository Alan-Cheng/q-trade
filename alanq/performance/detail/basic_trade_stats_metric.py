import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class BasicTradeStatsMetric(BaseDetailedMetric):
    """
    基本交易統計指標
    計算：總交易次數、獲利交易次數、虧損交易次數、持平交易次數
    """
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算基本交易統計"""
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'breakeven_trades': 0,
            }
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(trades[trades['pnl'] > 0]),
            'losing_trades': len(trades[trades['pnl'] < 0]),
            'breakeven_trades': len(trades[trades['pnl'] == 0]),
        }

