import numpy as np
from .base_position import BasePositionManager

# =========================================================
# FixedKellyPositionManager：固定 Kelly 參數的倉位管理
# =========================================================
class FixedKellyPositionManager(BasePositionManager):
    """
    固定 Kelly 參數的倉位管理
    在初始化時設定固定的勝率(p)和盈虧比(r)，並計算一個固定的投入比例(f)。
    
    公式: f = p - (1 - p) / r
    """
    
    def __init__(self, win_rate: float, odds_ratio: float, full_kelly_ratio=1.0, max_position_ratio=1.0):
        """
        Parameters:
        -----------
        win_rate : float
            策略的固定勝率 p (0到1之間)。
        odds_ratio : float
            策略的固定盈虧比 r (r = 平均獲利 / 平均虧損)。
        full_kelly_ratio : float
            使用 Kelly 公式計算出來的比例乘上的係數 (例如 0.5 為半 Kelly)。
        max_position_ratio : float
            最大持倉比例（0-1之間）。
        """
        super().__init__(win_rate=win_rate, 
                         odds_ratio=odds_ratio,
                         full_kelly_ratio=full_kelly_ratio, 
                         max_position_ratio=max_position_ratio)
        
        self.win_rate = win_rate
        self.odds_ratio = odds_ratio
        self.full_kelly_ratio = full_kelly_ratio
        self.max_position_ratio = max_position_ratio
        
        # ⚠️ 在初始化時計算固定的 Kelly 投入比例 (f)
        self.kelly_ratio = self._calculate_fixed_kelly_ratio()
    
    def _calculate_fixed_kelly_ratio(self) -> float:
        """計算並返回固定的 Kelly 比例 f"""
        p = self.win_rate
        r = self.odds_ratio
        q = 1 - p  # 敗率
        
        if r <= 0 or r == np.inf:
            # 盈虧比無效，風險極高或為負期望，Kelly 比例應為 0
            return 0.0
            
        # 原始 Kelly 比例: f = p - q / r
        # 由於我們已經在前面檢查了 r > 0，這裡可以直接計算
        kelly_ratio = p - (q / r)
        
        # 應用用戶設定的 Kelly 係數
        kelly_ratio *= self.full_kelly_ratio
        
        # 限制 Kelly 比例：必須大於等於 0 (期望為負時投入 0)，且不高於最大限制
        kelly_ratio = max(0.0, min(kelly_ratio, self.max_position_ratio))
        
        # 打印信息供參考
        print(f"--- Fixed Kelly PM Initialized ---")
        print(f"Win Rate (p): {p:.4f}, Odds Ratio (r): {r:.4f}")
        print(f"Calculated Kelly Ratio (f): {kelly_ratio:.4f}")
        print(f"----------------------------------")
        
        return kelly_ratio

    def calculate_position_size(self, current_price: float, available_capital: float, **kwargs) -> float:
        """
        根據固定的 Kelly 比例來計算倉位大小
        
        Parameters:
        -----------
        current_price : float
            當前價格
        available_capital : float
            可用資金
        **kwargs : dict
            其他參數（此實作中不使用）
        
        Returns:
        --------
        float : 應該買入的股數（可以是小數）
        """
        # 投入比例就是初始化時計算好的固定比例
        position_ratio = self.kelly_ratio
        
        # 計算可用於買入的資金
        position_value = available_capital * position_ratio
        
        # 計算股數
        shares = position_value / current_price
        
        return shares

