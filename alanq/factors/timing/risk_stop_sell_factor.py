import numpy as np
import pandas as pd
import talib
from .base_timing_factor import BaseSellFactor

# =========================================================
# RiskStopSellFactor：風險控制賣出因子
# =========================================================
class RiskStopSellFactor(BaseSellFactor):
    """
    風險控制：
    若今天的跌幅（昨收 - 今收） > ATR21 * pre_atr_n
    → 強制止損賣出
    """

    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)
        # 使用 TA-Lib 計算 ATR（此因子需要 atr21）
        if "atr21" not in df.columns:
            atr21_values = talib.ATR(df["High"].values, df["Low"].values, df["Close"].values, timeperiod=21)
            df["atr21"] = pd.Series(atr21_values, index=df.index)

    def reset(self):
        pass

    def generate(self):
        df = self.df

        pre_atr_n = self.params.get("pre_atr_n", 1.5)  # 預設 1.5（與阿布一致）
        signal = np.full(len(df), np.nan)

        close = df["Close"].values
        preclose = df["Close"].shift(1).values
        atr21 = df["atr21"].values

        for i in range(1, len(df)):
            drop_amount = preclose[i] - close[i]

            # ✦ 今日跌幅超過 ATR21 × 倍數 → 強制賣出
            if drop_amount > atr21[i] * pre_atr_n:
                signal[i] = 0  # 賣出

        return signal