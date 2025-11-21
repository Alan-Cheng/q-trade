import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class NetProfitMetric(BaseDetailedMetric):
    """
    淨獲利指標
    計算公式：總獲利 - 總虧損
    """
    metric_name = '淨獲利'
    higher_is_better = True
    target = None
    description = '淨獲利'
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算淨獲利"""
        if len(trades) == 0:
            return {'net_profit': 0}
        
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        
        return {'net_profit': total_profit - total_loss}

