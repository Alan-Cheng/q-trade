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


class BaseSellFactor(ABC):
    """
    所有「賣出策略因子」的抽象基底類別
    必須實作:
        - reset()
        - generate()
    """

    def __init__(self, df, **kwargs):
        self.df = df
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