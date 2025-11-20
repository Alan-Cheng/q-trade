import numpy as np
import pandas as pd
from .base_timing_factor import BaseBuyFactor, BaseSellFactor

# =========================================================
# 實作1. BreakoutBuyFactor：N 日向上突破買入
# =========================================================

class BreakoutBuyFactor(BaseBuyFactor):
    """
    N 日向上突破：
    若今日收盤價 == 過去 xd 日最高價 → 發出買入訊號
    並且在之後 xd 天內忽略新的突破（skip_days）
    """

    def reset(self):
        self.skip_days = 0

    def generate(self):
        df = self.df
        xd = self.params["xd"]
        signal = np.full(len(df), np.nan)

        close = df["Close"].values

        for i in range(len(df)):
            if i < xd:
                continue

            # 忽略突破後的連續 N 天信號（防止瘋狂加碼）
            if self.skip_days > 0:
                self.skip_days -= 1
                continue

            rolling_high = close[i - xd + 1 : i + 1].max()

            # 收盤價創 xd 日新高 → 發出買入
            if close[i] == rolling_high:
                signal[i] = 1
                self.skip_days = xd

        return signal
    
# =========================================================
# 實作2. BreakdownSellFactor：N 日向下突破賣出
# =========================================================

class BreakdownSellFactor(BaseSellFactor):
    """
    N 日向下突破：
    若今日收盤價 == 過去 xd 日最低價 → 發出賣出訊號（清倉）
    """
    
    def reset(self):
        pass

    def generate(self):
        df = self.df
        xd = self.params["xd"]
        signal = np.full(len(df), np.nan)

        close = df["Close"].values

        for i in range(len(df)):
            if i < xd:
                continue

            rolling_low = close[i - xd + 1 : i + 1].min()

            # 收盤價創 xd 日新低 → 賣出
            if close[i] == rolling_low:
                signal[i] = 0

        return signal