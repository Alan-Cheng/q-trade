import numpy as np
import pandas as pd
from typing import Dict
from .base_basic_metric import BaseBasicMetric

class AnnualReturnMetric(BaseBasicMetric):
    """
    年化報酬率指標
    計算公式：(1 + 總報酬率) ^ (1 / 年數) - 1
    """
    higher_is_better = True
    target = 0.2
    description = '年化報酬率'
    
    def calculate(self, equity_curve: pd.Series, 
                  initial_capital: float,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算年化報酬率"""
        if equity_curve is None or equity_curve.empty:
            return {self.metric_name: np.nan}
        
        # 嘗試從 existing_stats 取得總報酬率，否則自己計算
        total_return = existing_stats.get(
            "策略_總報酬率",
            (equity_curve.iloc[-1] / initial_capital) - 1 if initial_capital else np.nan
        )
        
        if np.isnan(total_return):
            return {self.metric_name: np.nan}
        
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.0 if days > 0 else 1.0
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
        
        return {self.metric_name: annual_return}

