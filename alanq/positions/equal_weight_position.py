from .base_position import BasePositionManager

# =========================================================
# EqualWeightPositionManager：等權重倉位管理
# =========================================================
class EqualWeightPositionManager(BasePositionManager):
    """
    等權重倉位管理
    將資金平均分配給所有持倉股票
    """
    
    def __init__(self, max_stocks=10):
        """
        Parameters:
        -----------
        max_stocks : int
            最大同時持倉股票數量
        """
        super().__init__(max_stocks=max_stocks)
        self.max_stocks = max_stocks
    
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
            其他參數，包含 current_holdings（當前持倉股票數量）
        
        Returns:
        --------
        float : 應該買入的股數（可以是小數）
        """
        # 取得當前持倉股票數量
        current_holdings = kwargs.get('current_holdings', 0)
        
        # 計算目標持倉數量（包含這筆新交易）
        target_holdings = min(current_holdings + 1, self.max_stocks)
        
        if target_holdings == 0:
            return 0
        
        # 計算每檔股票應該分配的金額
        position_value_per_stock = available_capital / target_holdings
        
        # 計算股數
        shares = position_value_per_stock / current_price
        
        return shares

