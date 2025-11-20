import numpy as np
import pandas as pd
from .base_timing_factor import BaseSellFactor

# =========================================================
# CloseAtrStopSellFactor：Trailing ATR 止盈
# =========================================================
class CloseAtrStopSellFactor(BaseSellFactor):
    """
    Trailing ATR 止盈
    close_atr_n：觸發止盈的 ATR 倍數（如 1.2）
    """

    def reset(self):
        self.in_position = False
        self.entry_price = None
        self.max_close = None  # 持倉期間最高收盤價

    # def _ensure_atr(self):
    #     df = self.df
    #     if "atr21" not in df:
    #         high_low = df["High"] - df["Low"]
    #         high_close = (df["High"] - df["Close"].shift(1)).abs()
    #         low_close = (df["Low"] - df["Close"].shift(1)).abs()
    #         tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    #         df["atr21"] = tr.rolling(21).mean().fillna(0)

    def generate(self):
        df = self.df
        close_atr_n = self.params.get("close_atr_n", 1.2)

        # self._ensure_atr()

        close = df["Close"].values
        atr21 = df["atr21"].values

        signal = np.full(len(df), np.nan)

        # 找出所有買入訊號（你的邏輯：col 名稱中有 BuyFactor）
        buy_cols = [c for c in df.columns if "BuyFactor" in c]
        raw_buy = df[buy_cols].max(axis=1).fillna(0).values

        for i in range(len(df)):

            # ---------- 若當天觸發買入 ----------
            if raw_buy[i] == 1:
                self.in_position = True
                self.entry_price = close[i]
                self.max_close = close[i]
                continue

            # ---------- 若沒有持倉 ----------
            if not self.in_position:
                continue

            # 更新最大價格
            self.max_close = max(self.max_close, close[i])

            # Step 1：必須先獲利超過 1×ATR
            if self.max_close - self.entry_price <= atr21[i]:
                continue

            # Step 2：下降幅度超過 ATR 倍數 ⇒ 觸發止盈
            if (self.max_close - close[i]) > atr21[i] * close_atr_n:
                signal[i] = 0  # 賣出
                self.in_position = False
                self.entry_price = None
                self.max_close = None

        return signal