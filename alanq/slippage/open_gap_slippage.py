import numpy as np
import pandas as pd
from base_slippage import BaseSlippage

# =========================================================
# SlippageOpenGap：開盤大跌滑價模型
# =========================================================
class SlippageOpenGap(BaseSlippage):
    
    # 這裡的 df 是整個價格資料，kwargs 包含 open_down_rate 和 action
    def __init__(self, df, open_down_rate=0.07, **kwargs): 
        # 確保 open_down_rate 參數被正確儲存
        kwargs['open_down_rate'] = open_down_rate
        super().__init__(df, **kwargs)
        self.open_down_rate = open_down_rate
        
        # (這裡可以放任何前置計算，例如計算波動率等)

    # set_current_data 會繼承自 BaseSlippage，負責更新 self.current_row

    def fit_price(self):
        # 確保當天數據已傳入
        if self.current_row is None:
            return np.inf # 或拋出錯誤

        row = self.current_row
        
        # 假設 preClose 是前一天的 Close 價 (通常需要 shift 取得)
        # 如果 df 中沒有 preClose 欄位，這部分邏輯需要調整。
        # 這裡我們假設 row["preClose"] 是正確的前收盤價。
        open_price = row["Open"]
        pre_close = row.get("preClose")
        high = row["High"]
        low = row["Low"]

        # 1. 開盤大跌 → 撤單
        if (
            pre_close > 0 
            and open_price / pre_close < (1 - self.open_down_rate)
        ):
            # 必須回傳 np.inf 才能讓 Backtester 判斷為撤單
            return np.inf 

        # 2. 均價成交
        price = (high + low) / 2
        # 不再需要 self.buy_price 屬性，直接回傳即可
        return price