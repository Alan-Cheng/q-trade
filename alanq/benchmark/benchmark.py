import numpy as np
import pandas as pd

# =========================================================
# Benchmark：基準（單純 buy & hold）
# =========================================================

class Benchmark:
    """
    基準（單純 buy & hold）
    """
    @staticmethod
    def compute_log_ret(df: pd.DataFrame) -> pd.Series:
        """基準對數報酬率（持有一檔標的不交易）"""
        close = df["Close"]
        return np.log(close / close.shift(1))