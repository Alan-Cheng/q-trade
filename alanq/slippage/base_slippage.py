import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# =========================================================
# BaseSlippage：滑價模型抽象基底類別
# =========================================================

class BaseSlippage(ABC):
    """
    所有滑價模型的抽象基底類別。
    必須實現 set_current_data() 和 fit_price()。
    """
    def __init__(self, df: pd.DataFrame, **kwargs):
        # 整個 DataFrame (用於前置計算，如 ATR)
        self.df = df
        # 儲存所有參數，包括 action
        self.params = kwargs 
        # 產生唯一的 factor_name，用於在 Backtester 中儲存和識別
        action = kwargs.get('action', 'default')
        # 假設滑價模型也像因子一樣需要 _ensure_atr
        # self._ensure_atr() 
        self.factor_name = f"{self.__class__.__name__}_{action}_{kwargs}"
        
        # 每天迴圈中傳入的單行數據
        self.current_row = None 
        
        # 確保有 action 參數
        if 'action' not in kwargs or kwargs['action'] not in ['buy', 'sell']:
            raise ValueError("BaseSlippage 必須在參數中指定 action='buy' 或 action='sell'")

    # 新增：接收當天數據的方法
    def set_current_data(self, row: pd.Series):
        """在 Backtester 每日迴圈中被呼叫，傳入當天數據"""
        self.current_row = row

    @abstractmethod
    def fit_price(self) -> float:
        """
        回傳滑價後的成交價格。
        - 買入撤單回傳 np.inf
        - 賣出撤單回傳 np.inf (或 0 / -np.inf，取決於設計)
        """
        pass