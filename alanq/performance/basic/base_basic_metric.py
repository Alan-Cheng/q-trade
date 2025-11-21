import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Optional

# =========================================================
# BaseBasicMetric：基本績效指標抽象基底類別
# =========================================================
class BaseBasicMetric(ABC):
    """
    所有「基本績效指標」的抽象基底類別（基於權益曲線）
    必須實作:
        - calculate()
    """
    
    def __init__(self, **kwargs):
        """
        Parameters:
        -----------
        **kwargs : dict
            指標參數
        """
        self.params = kwargs
        self.metric_name = self._get_metric_name()
    
    @abstractmethod
    def calculate(self, equity_curve: pd.Series, 
                  initial_capital: float,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """
        計算指標
        
        Parameters:
        -----------
        equity_curve : pd.Series
            權益曲線
        initial_capital : float
            初始資金
        existing_stats : dict
            已計算的其他指標（可用於依賴關係）
        
        Returns:
        --------
        dict : 包含指標名稱和值的字典，例如 {self.metric_name: value}
        """
        pass
    
    def _get_metric_name(self) -> str:
        """
        取得指標名稱（預設使用類別名稱，可覆寫）
        """
        # 將類別名稱轉換為中文鍵名
        # 例如：TotalReturnMetric -> 策略_總報酬率
        class_name = self.__class__.__name__
        
        # 如果類別名稱有對應的中文名稱，使用它
        name_mapping = {
            'TotalReturnMetric': '策略_總報酬率',
            'AnnualReturnMetric': '策略_年化報酬率',
            'VolatilityMetric': '策略_年化波動率',
            'SharpeMetric': '策略_Sharpe',
            'MaxDrawdownMetric': '策略_最大回撤',
            'FinalEquityMetric': '策略_最終權益',
        }
        
        return name_mapping.get(class_name, class_name)

