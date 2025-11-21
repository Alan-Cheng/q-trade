import numpy as np
import pandas as pd
from typing import Dict
from .base_basic_metric import BaseBasicMetric

class MaxDrawdownMetric(BaseBasicMetric):
    """
    最大回撤指標
    計算公式：權益曲線相對累積最高點的最大跌幅
    """
    higher_is_better = False  # 回撤越小越好
    target = 0.0
    description = '最大回撤'
    
    def calculate(self, equity_curve: pd.Series, 
                  initial_capital: float,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算最大回撤"""
        if equity_curve is None or equity_curve.empty:
            return {self.metric_name: np.nan}
        
        roll_max = equity_curve.cummax()
        drawdown = equity_curve / roll_max - 1.0
        max_drawdown = drawdown.min()
        
        return {self.metric_name: max_drawdown}

