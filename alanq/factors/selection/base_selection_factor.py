from abc import ABC, abstractmethod

class StockPickerBase(ABC):
    """
    選股因子的抽象基類 (Abstract Base Class)。
    強制所有子類必須實作 fit_pick 方法。
    """
    def __init__(self, capital, benchmark, **kwargs):
        # 假設這些是回測系統中的核心物件
        self.capital = capital
        self.benchmark = benchmark
        self._init_self(**kwargs)

    # 普通方法，子類可選擇性覆蓋
    def _init_self(self, **kwargs):
        """子類初始化參數"""
        pass

    @abstractmethod
    def fit_pick(self, kl_pd, target_symbol):
        """
        核心選股邏輯：判斷是否應選中該股票。子類必須實作。
        
        參數:
            kl_pd (pd.DataFrame): 股票的 K 線資料。
            target_symbol (str): 股票代碼。
        回傳:
            bool: 如果滿足選股條件返回 True，否則返回 False。
        """
        pass