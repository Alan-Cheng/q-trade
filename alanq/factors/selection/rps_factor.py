import numpy as np
import pandas as pd
from .base_selection_factor import StockPickerBase

# =========================================================
# RPSPicker：基於相對強弱 (Relative Price Strength) 進行篩選的選股因子。
# =========================================================
class RPSPicker(StockPickerBase):
    """
    基於相對強弱 (RPS) 進行篩選的選股因子。
    
    RPS = (個股總報酬率 / 基準總報酬率) - 1 (超額倍數)
    或
    RPS = 個股總報酬率 - 基準總報酬率 (超額報酬率)
    
    本實現採用 超額報酬率 方式。
    """
    
    def _init_self(self, **kwargs):
        """初始化 RPS 閥值、reversed 和 lookback 屬性"""
        
        # 參數 1: 最小超額報酬率
        self.min_excess_return = kwargs.get('min_excess_return', 0.0) 
        
        # 參數 2: 最大的超額報酬率
        self.max_excess_return = kwargs.get('max_excess_return', np.inf)
        
        self.reversed = kwargs.get('reversed', False)
        
        # ==========================================================
        # 【核心修正 A】從 kwargs 中獲取使用者傳遞的基準資料
        # 假設使用者在配置中傳入的參數名為 rps_benchmark_data
        # ==========================================================
        user_benchmark = kwargs.get('rps_benchmark_data')

        # 如果使用者在配置中傳入了基準，我們使用它並覆蓋 StockPickerBase 傳入的
        # 否則，我們沿用 StockPickerBase 傳入的 self.benchmark (可能是 None 或 Worker的預設值)
        if user_benchmark is not None:
            self.benchmark = user_benchmark
            
        # 儲存 RPS 值
        self.last_calculated_value = None
        
        # =================================================
        # 【核心修正 B】檢查 self.benchmark 是否最終為 None
        # =================================================
        if self.benchmark is None:
            raise ValueError(
                "RPSPicker 必須接收基準 (benchmark) 資料！"
                "請在配置參數時，使用 'rps_benchmark_data' 參數傳遞基準 Series/DataFrame。"
            )


    # @reversed_result 註解掉，因為 RPS 只需要正向篩選
    # 如果要反向篩選，可以透過調整 min/max 閾值來達成
    def fit_pick(self, kl_pd: pd.DataFrame, target_symbol: str) -> bool:
        """
        核心選股邏輯：計算股票相對於基準的超額報酬率。
        """
        if kl_pd.empty:
            self.last_calculated_value = None
            return False
        
        try:
            # 獲取個股的收盤價序列
            close_data = kl_pd['Close']
        except KeyError:
            try:
                close_data = kl_pd['close']
            except KeyError:
                self.last_calculated_value = None
                return False 
            
        # 1. 計算個股的總報酬率 (從回溯週期開始到結束)
        # (最後一筆 Close / 第一筆 Close) - 1
        # [優化] 檢查除數是否接近零 (使用 float 比較更安全)
        if close_data.iloc[0] < 1e-6: 
            self.last_calculated_value = None
            return False
            
        individual_ret = (close_data.iloc[-1] / close_data.iloc[0]) - 1.0

        # 2. 獲取基準 (Benchmark) 資料並對齊
        # 注意: 此處已假設 self.benchmark 必不為 None (已在 __init__ 檢查)
        
        # 嘗試從 benchmark 取得價格數據，並確保與 kl_pd 的時間範圍對齊
        try:
            # 優先假設 self.benchmark 是包含 'Close' 欄位的 DataFrame
            benchmark_close = self.benchmark['Close'].reindex(close_data.index)
        except (AttributeError, KeyError):
            # 其次假設 self.benchmark 已經是 Series
            benchmark_close = self.benchmark.reindex(close_data.index)
            
        # 刪除 NaN 值 (可能是因為日期沒有對齊)
        benchmark_close = benchmark_close.dropna()

        # 確保有足夠的基準數據
        if len(benchmark_close) < 2 or benchmark_close.iloc[0] < 1e-6:
            self.last_calculated_value = None
            return False
            
        # 3. 計算基準的總報酬率 (與個股相同的期間)
        benchmark_ret = (benchmark_close.iloc[-1] / benchmark_close.iloc[0]) - 1.0

        # 4. 計算相對強弱 (超額報酬率)
        rps = individual_ret - benchmark_ret
        
        # 儲存計算出的 RPS 值
        self.last_calculated_value = rps
        
        # 5. 根據閥值進行判斷
        pick_result = self.min_excess_return < rps < self.max_excess_return
        
        # 應用 reversed 邏輯
        if self.reversed:
            return not pick_result
            
        return pick_result