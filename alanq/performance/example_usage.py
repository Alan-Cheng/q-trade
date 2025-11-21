"""
PerformanceMetrics 使用範例

展示如何使用類別化的指標系統
"""

from alanq.performance import (
    PerformanceMetrics,
    # 基本指標
    TotalReturnMetric,
    SharpeMetric,
    MaxDrawdownMetric,
    # 詳細指標
    BasicTradeStatsMetric,
    WinRateMetric,
)

# =========================================================
# 範例 1：使用預設指標組合（完全向後相容）
# =========================================================
def example_default_metrics():
    """
    使用預設指標組合，行為與之前完全相同
    """
    # 假設您已經有回測結果
    # results, trades, stats = backtester.run()
    
    # 使用預設指標（不傳入 basic_metrics 和 detailed_metrics）
    # metrics = PerformanceMetrics(
    #     trades=trades,
    #     equity_curve=results['equity_curve'],
    #     initial_capital=1_000_000
    # )
    
    # 或傳入 stats（跳過計算）
    # metrics = PerformanceMetrics(
    #     trades=trades,
    #     stats=stats,
    #     equity_curve=results['equity_curve'],
    #     initial_capital=1_000_000
    # )
    
    pass


# =========================================================
# 範例 2：使用自訂指標組合
# =========================================================
def example_custom_metrics():
    """
    只計算特定的指標
    """
    # 假設您已經有回測結果
    # results, trades, stats = backtester.run()
    
    # 只計算總報酬率、Sharpe 和最大回撤
    basic_metrics = [
        {"class": TotalReturnMetric},
        {"class": SharpeMetric},
        {"class": MaxDrawdownMetric},
    ]
    
    # 只計算基本交易統計和勝率
    detailed_metrics = [
        {"class": BasicTradeStatsMetric},
        {"class": WinRateMetric},
    ]
    
    # metrics = PerformanceMetrics(
    #     trades=trades,
    #     equity_curve=results['equity_curve'],
    #     initial_capital=1_000_000,
    #     basic_metrics=basic_metrics,
    #     detailed_metrics=detailed_metrics
    # )
    
    pass


# =========================================================
# 範例 3：創建自訂指標類別
# =========================================================
def example_custom_metric_class():
    """
    創建自訂指標類別
    """
    from alanq.performance.basic import BaseBasicMetric
    import numpy as np
    import pandas as pd
    from typing import Dict
    
    class SortinoRatioMetric(BaseBasicMetric):
        """
        自訂的 Sortino Ratio 指標（只考慮下行波動率）
        """
        
        def __init__(self, risk_free_rate=0.0, **kwargs):
            super().__init__(**kwargs)
            self.risk_free_rate = risk_free_rate
            # 覆寫指標名稱
            self.metric_name = "策略_Sortino"
        
        def calculate(self, equity_curve: pd.Series, 
                      initial_capital: float,
                      existing_stats: Dict[str, float]) -> Dict[str, float]:
            """計算 Sortino Ratio"""
            if equity_curve is None or equity_curve.empty:
                return {self.metric_name: np.nan}
            
            daily_returns = equity_curve.pct_change().dropna()
            downside_returns = daily_returns[daily_returns < 0]
            
            if len(downside_returns) > 1 and downside_returns.std() > 0:
                downside_vol = downside_returns.std() * np.sqrt(252)
                annual_return = existing_stats.get("策略_年化報酬率", np.nan)
                
                if not np.isnan(annual_return) and downside_vol > 0:
                    sortino = (annual_return - self.risk_free_rate) / downside_vol
                else:
                    sortino = np.nan
            else:
                sortino = np.nan
            
            return {self.metric_name: sortino}
    
    # 使用自訂指標
    # basic_metrics = [
    #     {"class": TotalReturnMetric},
    #     {"class": AnnualReturnMetric},
    #     {"class": SortinoRatioMetric, "risk_free_rate": 0.02},  # 可以傳入參數
    # ]
    # 
    # metrics = PerformanceMetrics(
    #     trades=trades,
    #     equity_curve=results['equity_curve'],
    #     initial_capital=1_000_000,
    #     basic_metrics=basic_metrics
    # )


if __name__ == "__main__":
    print("PerformanceMetrics 使用範例")
    print("=" * 60)
    print("\n1. 使用預設指標組合（完全向後相容）")
    print("2. 使用自訂指標組合")
    print("3. 創建自訂指標類別")
    print("\n詳細用法請參考函數註解")

