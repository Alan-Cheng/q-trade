import numpy as np
import pandas as pd
from typing import Dict
from .base_basic_metric import BaseBasicMetric

class VolatilityMetric(BaseBasicMetric):
    """
    年化波動率指標
    計算公式：日報酬率標準差 * sqrt(252)
    """
    higher_is_better = False  # 波動率越小越好
    target = 0.15
    description = '年化波動率'
    
    def calculate(self, equity_curve: pd.Series, 
                  initial_capital: float,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算年化波動率"""
        if equity_curve is None or equity_curve.empty:
            return {self.metric_name: np.nan}
        
        daily_returns = equity_curve.pct_change().dropna()
        
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            volatility = daily_returns.std() * np.sqrt(252)
        else:
            volatility = np.nan
        
        return {self.metric_name: volatility}

