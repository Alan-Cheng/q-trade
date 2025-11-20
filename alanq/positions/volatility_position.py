import numpy as np
from .base_position import BasePositionManager

# =========================================================
# VolatilityBasedPositionManager：基於波動率的倉位管理
# =========================================================
class VolatilityBasedPositionManager(BasePositionManager):
    """
    基於波動率的倉位管理
    波動率越高，倉位越小
    """
    
    def __init__(self, base_position_ratio=0.5, volatility_window=20, max_position_ratio=1.0):
        """
        Parameters:
        -----------
        base_position_ratio : float
            基礎倉位比例
        volatility_window : int
            計算波動率的視窗期
        max_position_ratio : float
            最大持倉比例
        """
        super().__init__(base_position_ratio=base_position_ratio,
                        volatility_window=volatility_window,
                        max_position_ratio=max_position_ratio)
        self.base_position_ratio = base_position_ratio
        self.volatility_window = volatility_window
        self.max_position_ratio = max_position_ratio
    
    def calculate_position_size(self, current_price, available_capital, **kwargs):
        """
        計算應該買入的股數
        
        Parameters:
        -----------
        current_price : float
            當前價格
        available_capital : float
            可用資金
        **kwargs : dict
            其他參數，包含 returns（歷史報酬率序列）
        
        Returns:
        --------
        float : 應該買入的股數（可以是小數）
        """
        # 取得歷史報酬率（如果有的話）
        returns = kwargs.get('returns', None)
        
        if returns is not None and len(returns) >= self.volatility_window:
            # 計算波動率
            volatility = returns[-self.volatility_window:].std()
            # 標準化波動率（假設平均波動率為 0.02）
            normalized_vol = volatility / 0.02
            
            # 波動率越高，倉位越小
            position_ratio = self.base_position_ratio / max(normalized_vol, 0.5)
            position_ratio = min(position_ratio, self.max_position_ratio)
        else:
            position_ratio = self.base_position_ratio
        
        # 計算股數
        position_value = available_capital * position_ratio
        shares = position_value / current_price
        
        return shares

