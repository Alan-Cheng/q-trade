import numpy as np
import pandas as pd
from typing import Dict
from .base_basic_metric import BaseBasicMetric

class FinalEquityMetric(BaseBasicMetric):
    """
    最終權益指標
    計算公式：權益曲線的最後一個值
    """
    
    def calculate(self, equity_curve: pd.Series, 
                  initial_capital: float,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """計算最終權益"""
        if equity_curve is None or equity_curve.empty:
            return {self.metric_name: np.nan}
        
        final_equity = equity_curve.iloc[-1]
        return {self.metric_name: final_equity}

