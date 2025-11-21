import pandas as pd
from typing import Dict
from .base_detailed_metric import BaseDetailedMetric

class AvgLossMetric(BaseDetailedMetric):
    """
    平均虧損指標
    計算公式：虧損交易的平均虧損金額（絕對值）
    """
    metric_name = '平均虧損'
    higher_is_better = False  # 虧損越小越好
    target = None
    description = '平均虧損'
    
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算平均虧損"""
        if len(trades) == 0:
            return {'avg_loss': 0}
        
        losing_trades = trades[trades['pnl'] < 0]
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        
        return {'avg_loss': avg_loss}

