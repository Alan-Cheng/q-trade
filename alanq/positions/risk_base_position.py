from .base_position import BasePositionManager

# =========================================================
# RiskBasePositionManager：基於風險的倉位管理
# =========================================================
class FixedRatioPositionManager(BasePositionManager):
    """
    固定比例倉位管理
    每次買入使用固定比例的可用資金
    """
    
    def __init__(self, position_ratio=0.5, max_position_ratio=1.0):
        """
        Parameters:
        -----------
        position_ratio : float
            每次買入使用的資金比例（0-1之間）
        max_position_ratio : float
            最大持倉比例（0-1之間），防止過度槓桿
        """
        super().__init__(position_ratio=position_ratio, 
                        max_position_ratio=max_position_ratio)
        self.position_ratio = position_ratio
        self.max_position_ratio = max_position_ratio
    
    def calculate_position_size(self, current_price, available_capital, **kwargs):
        # 計算可用於買入的資金
        position_value = available_capital * self.position_ratio
        
        # 計算股數
        shares = position_value / current_price
        
        return shares