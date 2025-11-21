import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Optional

# =========================================================
# BaseDetailedMetric：詳細績效指標抽象基底類別
# =========================================================
class BaseDetailedMetric(ABC):
    """
    所有「詳細績效指標」的抽象基底類別（基於交易紀錄）
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
    
    @abstractmethod
    def calculate(self, trades: pd.DataFrame,
                  existing_stats: Dict[str, float]) -> Dict[str, float]:
        """
        計算指標
        
        Parameters:
        -----------
        trades : pd.DataFrame
            交易紀錄
        existing_stats : dict
            已計算的其他指標（可用於依賴關係）
        
        Returns:
        --------
        dict : 包含指標名稱和值的字典
        """
        pass

