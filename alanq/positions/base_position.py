from abc import ABC, abstractmethod

# =========================================================
# BasePositionManager：倉位管理抽象基底類別
# =========================================================
class BasePositionManager(ABC):
    """
    倉位管理抽象基底類別
    必須實作:
        - calculate_position_size() 計算倉位大小
    """
    
    def __init__(self, **kwargs):
        self.params = kwargs
    
    @abstractmethod
    def calculate_position_size(self, 
                                current_price: float,
                                available_capital: float,
                                **kwargs) -> float:
        """
        計算應該買入的股數
        
        Parameters:
        -----------
        current_price : float
            當前價格
        available_capital : float
            可用資金
        **kwargs : dict
            其他參數（如 ATR、波動率等）
        
        Returns:
        --------
        float : 應該買入的股數（可以是小數）
        """
        pass