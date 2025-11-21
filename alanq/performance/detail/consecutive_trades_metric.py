import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class ConsecutiveTradesMetric(BaseDetailedMetric):
    """
    連續交易指標
    計算：最大連續獲利次數、最大連續虧損次數
    """
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算連續交易統計"""
        if len(trades) == 0:
            return {
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
            }
        
        trades_sorted = trades.sort_values('entry_date')
        is_profit = (trades_sorted['pnl'] > 0).astype(int)
        is_loss = (trades_sorted['pnl'] < 0).astype(int)
        
        # 計算連續獲利
        max_wins = 0
        current_wins = 0
        for profit in is_profit:
            if profit == 1:
                current_wins += 1
                max_wins = max(max_wins, current_wins)
            else:
                current_wins = 0
        
        # 計算連續虧損
        max_losses = 0
        current_losses = 0
        for loss in is_loss:
            if loss == 1:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        
        return {
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses,
        }

