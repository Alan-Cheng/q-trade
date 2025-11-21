import numpy as np
import pandas as pd
from typing import Dict
from .base_basic_metric import BaseBasicMetric

class TotalReturnMetric(BaseBasicMetric):
    """
    總報酬率指標
    計算公式：(最終權益 / 初始資金) - 1
    """
    
    def calculate(self, equity_curve: pd.Series, 
                  initial_capital: float,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算總報酬率"""
        if equity_curve is None or equity_curve.empty or initial_capital is None:
            return {self.metric_name: np.nan}
        
        total_return = (equity_curve.iloc[-1] / initial_capital) - 1
        return {self.metric_name: total_return}

