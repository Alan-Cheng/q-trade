import numpy as np
import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class ProfitLossStatsMetric(BaseDetailedMetric):
    """
    獲利虧損統計指標
    計算：平均獲利、平均虧損、盈虧比、總獲利、總虧損、淨獲利
    """
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算獲利虧損統計"""
        if len(trades) == 0:
            return {
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_loss_ratio': 0,
                'total_profit': 0,
                'total_loss': 0,
                'net_profit': 0,
            }
        
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf
        
        total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        
        return {
            'avg_profit': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': total_profit - total_loss,
        }

