from .base_position import BasePositionManager

# =========================================================
# RiskBasePositionManager：基於風險的倉位管理
# =========================================================
class RiskBasedPositionManager(BasePositionManager):
    """
    基於風險的倉位管理
    根據 ATR 和風險金額來計算倉位大小
    """
    
    def __init__(self, risk_per_trade=0.02, atr_multiplier=2.0, max_position_ratio=1.0):
        """
        Parameters:
        -----------
        risk_per_trade : float
            每筆交易願意承擔的風險比例（例如 0.02 表示 2%）
        atr_multiplier : float
            ATR 倍數，用於計算止損距離
        max_position_ratio : float
            最大持倉比例
        """
        super().__init__(risk_per_trade=risk_per_trade,
                        atr_multiplier=atr_multiplier,
                        max_position_ratio=max_position_ratio)
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.max_position_ratio = max_position_ratio
    
    def calculate_position_size(self, current_price, available_capital, **kwargs):
        # 取得 ATR（如果有的話）
        atr = kwargs.get('atr', current_price * 0.02)  # 預設 2% 作為止損距離
        
        # 計算風險金額
        total_capital = kwargs.get('total_capital', available_capital)
        risk_amount = total_capital * self.risk_per_trade
        
        # 計算止損距離
        stop_loss_distance = atr * self.atr_multiplier
        
        # 計算股數
        if stop_loss_distance > 0:
            shares = risk_amount / stop_loss_distance
        else:
            shares = 0
        
        # 檢查是否超過最大持倉限制
        position_value = shares * current_price
        max_position_value = total_capital * self.max_position_ratio
        if position_value > max_position_value:
            shares = max_position_value / current_price
        
        return shares