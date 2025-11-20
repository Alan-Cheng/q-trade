import numpy as np
import pandas as pd
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

    def reset(self):
        pass

    # -----------------------------------------------------
    # 如果 df 沒有 ATR 相關欄位，補上
    # -----------------------------------------------------
    # def _ensure_atr(self):
    #     df = self.df

    #     if "atr14" not in df.columns or "atr21" not in df.columns:
    #         high_low = df["High"] - df["Low"]
    #         high_close = (df["High"] - df["Close"].shift(1)).abs()
    #         low_close = (df["Low"] - df["Close"].shift(1)).abs()

    #         tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    #         df["atr14"] = tr.rolling(14).mean()
    #         df["atr21"] = tr.rolling(21).mean()

    #         df[["atr14", "atr21"]] = df[["atr14", "atr21"]].fillna(0)

    # -----------------------------------------------------
    # generate(): 回傳賣出訊號
    # -----------------------------------------------------
    def generate(self):
        df = self.df

        # 確保 ATR 欄位存在
        # self._ensure_atr()

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