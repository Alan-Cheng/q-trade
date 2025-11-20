import numpy as np
import pandas as pd
from .base_timing_factor import BaseSellFactor

# =========================================================
# AtrStopSellFactor：ATR 止盈止損賣出因子
# =========================================================
class AtrStopSellFactor(BaseSellFactor):

    def reset(self):
        self.in_position = False
        self.entry_price = None

    # def _ensure_atr(self):
    #     df = self.df
    #     if "atr14" not in df or "atr21" not in df:
    #         high_low = df["High"] - df["Low"]
    #         high_close = (df["High"] - df["Close"].shift(1)).abs()
    #         low_close = (df["Low"] - df["Close"].shift(1)).abs()
    #         tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    #         df["atr14"] = tr.rolling(14).mean().fillna(0)
    #         df["atr21"] = tr.rolling(21).mean().fillna(0)

    def generate(self):
        df = self.df
        # self._ensure_atr()

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