import numpy as np
import pandas as pd
import talib
from .base_timing_factor import BaseSellFactor

# =========================================================
# AtrStopSellFactor：ATR 止盈止損賣出因子
# =========================================================
class AtrStopSellFactor(BaseSellFactor):

    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        # 使用 TA-Lib 計算 ATR（此因子需要 atr14 和 atr21）
        if "atr14" not in df.columns:
            atr14_values = talib.ATR(df["High"].values, df["Low"].values, df["Close"].values, timeperiod=14)
            df["atr14"] = pd.Series(atr14_values, index=df.index)
        if "atr21" not in df.columns:
            atr21_values = talib.ATR(df["High"].values, df["Low"].values, df["Close"].values, timeperiod=21)
            df["atr21"] = pd.Series(atr21_values, index=df.index)

    def reset(self):
        self.in_position = False
        self.entry_price = None

    def generate(self):
        df = self.df

        n_loss = self.params.get("stop_loss_n")
        n_win = self.params.get("stop_win_n")

        signal = np.full(len(df), np.nan)

        close = df["Close"].values
        atr14 = df["atr14"].values
        atr21 = df["atr21"].values

        # 找出所有買入 raw signal（由你的 buy factors 產生）
        buy_cols = [c for c in df.columns if "BuyFactor" in c]
        raw_buy = df[buy_cols].max(axis=1).fillna(0).values

        for i in range(len(df)):

            # 若今天發生買入 → 記錄 entry_price
            if raw_buy[i] == 1:
                self.in_position = True
                self.entry_price = close[i]
                continue

            # 若沒有持倉 → 略過
            if not self.in_position:
                continue

            profit = close[i] - self.entry_price
            stop_base = atr14[i] + atr21[i]

            # ---- 止盈 ----
            if n_win is not None and profit > n_win * stop_base:
                signal[i] = 0
                self.in_position = False
                self.entry_price = None
                continue

            # ---- 止損 ----
            if n_loss is not None and profit < -n_loss * stop_base:
                signal[i] = 0
                self.in_position = False
                self.entry_price = None
                continue

        return signal