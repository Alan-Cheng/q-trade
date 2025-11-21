"""
優化器使用範例

展示如何使用 FactorOptimizer 和 MultiStockFactorOptimizer
包括如何指定權重和自訂指標組合
"""

from alanq.optimization import ParameterSpace, FactorOptimizer, MultiStockFactorOptimizer, AVAILABLE_METRICS
from alanq.factors.timing import BreakoutBuyFactor, BreakdownSellFactor, AtrStopSellFactor
from alanq.data import StockDataManager
from alanq.performance import (
    TotalReturnMetric, 
    SharpeMetric, 
    MaxDrawdownMetric, 
    WinRateMetric,
    AnnualReturnMetric,
    VolatilityMetric,
    ProfitLossRatioMetric,
    NetProfitMetric,
)

# =========================================================
# 範例 1：基本使用（使用預設指標和等權重）
# =========================================================
def example_basic_usage():
    """基本使用方式，使用預設指標組合和等權重"""
    
    # 1. 準備資料
    data_manager = StockDataManager(["TSLA"], start_date="2020-01-01")
    df = data_manager.get_kl_pd("TSLA")
    
    # 2. 定義參數空間
    param_space = ParameterSpace()
    param_space.add_buy_factor(BreakoutBuyFactor, {'xd': [20, 40, 60]})
    param_space.add_sell_factor(BreakdownSellFactor, {'xd': [10, 20]})
    
    # 3. 執行優化（使用預設指標和等權重）
    optimizer = FactorOptimizer(
        df=df,
        parameter_space=param_space,
        initial_capital=1_000_000,
        show_progress=True
        # 不傳入 metric_weights，使用等權重
        # 不傳入 basic_metrics 和 detailed_metrics，使用預設指標組合
    )
    
    best_config, results_df = optimizer.optimize()
    
    return best_config, results_df


# =========================================================
# 範例 2：使用指標類別指定權重（推薦方式）
# =========================================================
def example_metric_class_weights():
    """使用指標類別列表指定權重，會自動從類別中取得屬性"""
    
    # 1. 準備資料
    data_manager = StockDataManager(["TSLA"], start_date="2020-01-01")
    df = data_manager.get_kl_pd("TSLA")
    
    # 2. 定義參數空間
    param_space = ParameterSpace()
    param_space.add_buy_factor(BreakoutBuyFactor, {'xd': [20, 40, 60]})
    param_space.add_sell_factor(BreakdownSellFactor, {'xd': [10, 20]})
    
    # 3. 使用指標類別列表定義權重（推薦方式）
    # 會自動從指標類別中取得 metric_name, higher_is_better 等屬性
    # 並且會自動提取 basic_metrics 和 detailed_metrics，只計算這裡指定的指標
    # 注意：現在每個指標類別都是獨立的，不需要 key 參數
    metric_weights = [
        {"class": TotalReturnMetric, "weight": 3},      # 最重視總報酬率
        {"class": SharpeMetric, "weight": 2},            # 次重視 Sharpe
        {"class": MaxDrawdownMetric, "weight": 1},      # 也考慮最大回撤
        {"class": WinRateMetric, "weight": 1},           # 也考慮勝率
        {"class": ProfitLossRatioMetric, "weight": 1},  # 盈虧比（獨立的類別）
        {"class": NetProfitMetric, "weight": 1},        # 淨獲利（獨立的類別）
    ]
    # 注意：不需要再指定 basic_metrics 和 detailed_metrics，會自動從 metric_weights 提取
    
    # 4. 執行優化
    optimizer = FactorOptimizer(
        df=df,
        parameter_space=param_space,
        initial_capital=1_000_000,
        metric_weights=metric_weights,  # 傳入指標類別列表
        show_progress=True
    )
    
    best_config, results_df = optimizer.optimize()
    
    return best_config, results_df


# =========================================================
# 範例 2b：使用字典指定權重（向後相容）
# =========================================================
def example_dict_weights():
    """使用字典指定權重（向後相容的方式）"""
    
    # 1. 準備資料
    data_manager = StockDataManager(["TSLA"], start_date="2020-01-01")
    df = data_manager.get_kl_pd("TSLA")
    
    # 2. 定義參數空間
    param_space = ParameterSpace()
    param_space.add_buy_factor(BreakoutBuyFactor, {'xd': [20, 40, 60]})
    param_space.add_sell_factor(BreakdownSellFactor, {'xd': [10, 20]})
    
    # 3. 使用字典定義權重（向後相容）
    custom_weights = {
        '策略_總報酬率': 3,      # 最重視總報酬率（權重 3）
        '策略_Sharpe': 2,       # 次重視 Sharpe（權重 2）
        '策略_最大回撤': 1,     # 也考慮最大回撤（權重 1）
        '勝率': 1,              # 也考慮勝率（權重 1）
        '盈虧比': 0,            # 不考慮盈虧比（權重 0）
        # 其他未指定的指標權重自動為 0
    }
    
    # 4. 執行優化
    optimizer = FactorOptimizer(
        df=df,
        parameter_space=param_space,
        initial_capital=1_000_000,
        metric_weights=custom_weights,  # 傳入字典
        show_progress=True
    )
    
    best_config, results_df = optimizer.optimize()
    
    return best_config, results_df


# =========================================================
# 範例 3：只計算需要的指標（自動從 metric_weights 提取）
# =========================================================
def example_custom_metrics_auto_extract():
    """只計算需要的指標，會自動從 metric_weights 中提取"""
    
    # 1. 準備資料
    data_manager = StockDataManager(["TSLA"], start_date="2020-01-01")
    df = data_manager.get_kl_pd("TSLA")
    
    # 2. 定義參數空間
    param_space = ParameterSpace()
    param_space.add_buy_factor(BreakoutBuyFactor, {'xd': [20, 40, 60]})
    param_space.add_sell_factor(BreakdownSellFactor, {'xd': [10, 20]})
    
    # 3. 使用指標類別定義權重
    # 系統會自動從 metric_weights 中提取 basic_metrics 和 detailed_metrics
    # 只會計算這裡指定的指標，提高效率
    metric_weights = [
        {"class": TotalReturnMetric, "weight": 2},
        {"class": SharpeMetric, "weight": 1},
        {"class": MaxDrawdownMetric, "weight": 1},
        {"class": WinRateMetric, "weight": 1},
    ]
    
    # 4. 執行優化
    # 不需要指定 basic_metrics 和 detailed_metrics，會自動從 metric_weights 提取
    optimizer = FactorOptimizer(
        df=df,
        parameter_space=param_space,
        initial_capital=1_000_000,
        metric_weights=metric_weights,  # 只需要指定權重，指標會自動提取
        show_progress=True
    )
    
    best_config, results_df = optimizer.optimize()
    
    return best_config, results_df


# =========================================================
# 範例 4：多股票優化（使用指標類別權重）
# =========================================================
def example_multi_stock_optimization():
    """多股票優化，使用指標類別權重"""
    
    # 1. 準備多股票資料
    data_manager = StockDataManager(["AAPL", "TSLA", "MSFT"], start_date="2020-01-01")
    stock_data = {
        symbol: data_manager.get_kl_pd(symbol) 
        for symbol in ["AAPL", "TSLA", "MSFT"]
    }
    
    # 2. 定義參數空間
    param_space = ParameterSpace()
    param_space.add_buy_factor(BreakoutBuyFactor, {'xd': [20, 40, 60]})
    param_space.add_sell_factor(BreakdownSellFactor, {'xd': [10, 20]})
    
    # 3. 使用指標類別定義權重
    metric_weights = [
        {"class": TotalReturnMetric, "weight": 2},
        {"class": SharpeMetric, "weight": 3},        # 最重視 Sharpe
        {"class": MaxDrawdownMetric, "weight": 2},
        {"class": WinRateMetric, "weight": 1},
        {"class": ProfitLossRatioMetric, "weight": 1},  # 盈虧比
        {"class": NetProfitMetric, "weight": 1},        # 淨獲利
    ]
    
    # 4. 執行優化
    optimizer = MultiStockFactorOptimizer(
        stock_data=stock_data,
        parameter_space=param_space,
        initial_capital=1_000_000,
        metric_weights=metric_weights,  # 傳入指標類別列表
        show_progress=True,
        n_jobs=-1  # 使用所有 CPU 核心
    )
    
    best_config, results_df = optimizer.optimize()
    
    return best_config, results_df


# =========================================================
# 範例 5：查看指標類別的屬性
# =========================================================
def example_view_metric_attributes():
    """查看指標類別的屬性"""
    
    print("指標類別屬性範例：")
    print("=" * 60)
    
    # 創建指標實例以查看屬性
    metrics_to_check = [
        TotalReturnMetric(),
        SharpeMetric(),
        MaxDrawdownMetric(),
        WinRateMetric(),
    ]
    
    for metric in metrics_to_check:
        print(f"\n{metric.__class__.__name__}")
        print(f"  指標名稱 (metric_name): {metric.metric_name}")
        print(f"  越大越好 (higher_is_better): {metric.higher_is_better}")
        print(f"  目標值 (target): {metric.target}")
        print(f"  描述 (description): {metric.description}")
    
    print("\n" + "=" * 60)
    print("\n使用範例（推薦方式 - 使用指標類別）：")
    print("""
    from alanq.performance import (
        TotalReturnMetric, 
        SharpeMetric, 
        WinRateMetric,
        ProfitLossRatioMetric,
        NetProfitMetric
    )
    
    # 使用指標類別列表定義權重
    # 系統會自動：
    # 1. 從類別中取得 metric_name, higher_is_better 等屬性
    # 2. 自動提取 basic_metrics 和 detailed_metrics
    # 3. 只計算這裡指定的指標（提高效率）
    # 注意：現在每個指標類別都是獨立的，不需要 key 參數
    metric_weights = [
        {"class": TotalReturnMetric, "weight": 3},  # 會自動取得 metric_name = "策略_總報酬率"
        {"class": SharpeMetric, "weight": 2},      # 會自動取得 metric_name = "策略_Sharpe"
        {"class": WinRateMetric, "weight": 1},      # 會自動取得 metric_name = "勝率"
        {"class": ProfitLossRatioMetric, "weight": 1},  # 盈虧比（獨立的類別）
        {"class": NetProfitMetric, "weight": 1},        # 淨獲利（獨立的類別）
    ]
    
    optimizer = FactorOptimizer(
        df=df,
        parameter_space=param_space,
        metric_weights=metric_weights  # 只需要指定權重，指標會自動提取
        # 不需要指定 basic_metrics 和 detailed_metrics
    )
    """)
    
    print("\n使用範例（向後相容 - 使用字典）：")
    print("""
    # 仍然支援字典方式
    custom_weights = {
        '策略_總報酬率': 3,
        '策略_Sharpe': 2,
        '勝率': 1,
    }
    
    optimizer = FactorOptimizer(
        df=df,
        parameter_space=param_space,
        metric_weights=custom_weights  # 傳入字典
    )
    """)


if __name__ == "__main__":
    print("優化器使用範例")
    print("=" * 60)
    
    # 查看指標類別屬性
    example_view_metric_attributes()
    
    # 執行範例（取消註解以執行）
    # best_config, results_df = example_basic_usage()
    # best_config, results_df = example_metric_class_weights()  # 推薦方式
    # best_config, results_df = example_dict_weights()          # 向後相容
    # best_config, results_df = example_custom_metrics_with_class_weights()
    # best_config, results_df = example_multi_stock_optimization()

