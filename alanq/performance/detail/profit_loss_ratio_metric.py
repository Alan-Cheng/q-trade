import numpy as np
import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class ProfitLossRatioMetric(BaseDetailedMetric):
    """
    盈虧比指標
    計算公式：平均獲利 / 平均虧損
    """
    metric_name = '盈虧比'
    higher_is_better = True
    target = 2.0
    description = '盈虧比'
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算盈虧比"""
        if len(trades) == 0:
            return {'profit_loss_ratio': 0}
        
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf
        
        return {'profit_loss_ratio': profit_loss_ratio}

