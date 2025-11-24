import numpy as np
from functools import wraps
import statsmodels.api as sm 
from statsmodels import regression

# =========================================================
# RegressionUtil：計算標準化後的股價趨勢角度（可跨股票比較）。
# =========================================================
class RegressionUtil:
    """
    計算標準化後的股價趨勢角度（可跨股票比較）。
    """
    @staticmethod
    def calc_regress_deg(data, symbol=None, show=False): 
        """
        用線性回歸 + 標準化計算趨勢角度。
        """
        y_arr = data.values.astype(float)  
        n = len(y_arr)

        if n < 2:
            return 0.0

        # --- 1. X 與 Y 同時標準化（0~1 區間） ---
        x_raw = np.arange(n)

        # 檢查是否所有值都相同（避免除以零）
        x_range = x_raw.max() - x_raw.min()
        y_range = y_arr.max() - y_arr.min()
        
        if x_range == 0:
            x_norm = np.zeros_like(x_raw)
        else:
            x_norm = (x_raw - x_raw.min()) / x_range
            
        if y_range == 0:
            # 如果所有價格都相同，角度為 0（無趨勢）
            y_norm = np.zeros_like(y_arr)
        else:
            y_norm = (y_arr - y_arr.min()) / y_range

        # --- 2. OLS 回歸 ---
        X = sm.add_constant(x_norm)
        model = regression.linear_model.OLS(y_norm, X).fit()

        slope = model.params[1]

        # --- 3. 斜率轉角度：正確公式 angle = atan(slope) ---
        angle = np.degrees(np.arctan(slope))

        # --- 4. 畫圖（選擇性） ---
        if show:
            import matplotlib.pyplot as plt
            
            intercept = model.params[0]
            reg_y_fit = slope * x_norm + intercept

            plt.figure(figsize=(10, 5))
            plt.plot(x_norm, y_norm, label='Normalized Price', linewidth=2)
            plt.plot(x_norm, reg_y_fit, label='Regression Line', linestyle='--')

            symbol_str = f"{symbol} - " if symbol else ""
            plt.title(f"{symbol_str}Trend Angle (deg) = {angle:.2f}")
            plt.xlabel("Normalized Time")
            plt.ylabel("Normalized Price")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()

        return float(angle)

    
def reversed_result(func):
    """如果 self.reversed 為 True，則翻轉 fit_pick 的布林返回值"""
    @wraps(func)
    def wrapper(self, kl_pd, target_symbol):
        result = func(self, kl_pd, target_symbol)
        if hasattr(self, 'reversed') and self.reversed:
            return not result
        return result
    return wrapper