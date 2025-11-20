import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# =========================================================
# 0. Base Buy / Sell Factor：介面
# =========================================================
class BaseBuyFactor(ABC):
    """
    所有「買入策略因子」的抽象基底類別
    必須實作:
        - reset()
        - generate()
    """

    def __init__(self, df, **kwargs):
        self.df = df
        self._create_atr()
        self._create_preClouse()
        self.params = kwargs
        self.factor_name = f"{self.__class__.__name__}_{kwargs}"
        self.reset()

    @abstractmethod
    def reset(self):
        """策略內部變數初始化"""
        pass

    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        回傳買入訊號陣列：
        - 1 = 當天買入
        - NaN = 無操作
        """
        pass
    
    def _create_atr(self):
        df = self.df
        if "atr14" not in df or "atr21" not in df:
            high_low = df["High"] - df["Low"]
            high_close = (df["High"] - df["Close"].shift(1)).abs()
            low_close = (df["Low"] - df["Close"].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            df["atr14"] = tr.rolling(14).mean().fillna(0)
            df["atr21"] = tr.rolling(21).mean().fillna(0)
            
    def _create_preClouse(self):
        df = self.df
        if "preClose" not in df:
            df["preClose"] = df["Close"].shift(1)


class BaseSellFactor(ABC):
    """
    所有「賣出策略因子」的抽象基底類別
    必須實作:
        - reset()
        - generate()
    """

    def __init__(self, df, **kwargs):
        self.df = df
        self._ensure_atr()
        self.params = kwargs
        self.factor_name = f"{self.__class__.__name__}_{kwargs}"
        self.reset()

    @abstractmethod
    def reset(self):
        """初始化策略狀態"""
        pass

    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        回傳賣出訊號陣列：
        - 0 = 當天賣出
        - NaN = 無操作
        """
        pass
    
    def _ensure_atr(self):
        df = self.df
        if "atr14" not in df or "atr21" not in df:
            high_low = df["High"] - df["Low"]
            high_close = (df["High"] - df["Close"].shift(1)).abs()
            low_close = (df["Low"] - df["Close"].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            df["atr14"] = tr.rolling(14).mean().fillna(0)
            df["atr21"] = tr.rolling(21).mean().fillna(0)