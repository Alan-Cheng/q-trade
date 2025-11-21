import numpy as np
import pandas as pd
from typing import Dict
from .base_basic_metric import BaseBasicMetric

class SharpeMetric(BaseBasicMetric):
    """
    Sharpe 比率指標
    計算公式：(日報酬率平均值 / 日報酬率標準差) * sqrt(252)
    """
    
    def calculate(self, equity_curve: pd.Series, 
                  initial_capital: float,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算 Sharpe 比率"""
        if equity_curve is None or equity_curve.empty:
            return {self.metric_name: np.nan}
        
        daily_returns = equity_curve.pct_change().dropna()
        
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = np.nan
        
        return {self.metric_name: sharpe}

