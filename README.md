# AlanQ - 量化交易系統

AlanQ 是一個自行建立的量化交易回測框架，提供從資料管理、因子開發、回測執行到績效評估的完整功能。

## 關於

本專案參考 [阿布量化交易系統 (abu)](https://github.com/bbfamily/abu) 的教學書籍與開源框架，實作並學習量化交易相關概念與建立回測系統。

## 安裝

### 前置需求

- Python 3.8 或更高版本

### 安裝步驟

1. **Clone或下載專案**

```bash
git clone <repository-url>
cd q-trade
```

2. **建立虛擬環境（建議）**

```bash
python -m venv venv
```

3. **啟動虛擬環境**

- macOS/Linux:
```bash
source venv/bin/activate
```

- Windows:
```bash
venv\Scripts\activate
```

4. **安裝依賴套件**

```bash
pip install -r requirements.txt
```

### 依賴套件清單

專案所需的套件已列於 `requirements.txt`，包含：

- `pandas`: 資料處理
- `numpy`: 數值計算
- `matplotlib`: 圖表繪製
- `mplfinance`: 金融圖表繪製
- `yfinance`: 股票資料下載
- `scipy`: 科學計算
- `bokeh`: 互動式視覺化
- `seaborn`: 統計圖表
- `jupyter`: Jupyter Notebook 環境
- `ipykernel`: Jupyter 核心
- `ta-lib`: 技術分析指標庫
- `statsmodels`: 統計分析（用於回歸工具）
- `scikit-learn`: 機器學習工具
- `requests`: HTTP 請求庫

## 目錄結構

```
alanq/
├── backtest/          # 回測引擎
├── data/              # 資料管理
├── factors/           # 交易因子
│   ├── selection/     # 選股因子
│   └── timing/        # 時機因子（買入/賣出）
├── positions/         # 倉位管理
├── optimization/      # 因子優化
├── performance/       # 績效評估
│   ├── basic/         # 基本績效指標
│   └── detail/        # 詳細績效指標
├── slippage/          # 滑價模型
├── benchmark/         # 基準計算
└── util/              # 工具函數
```

---

## 模組說明

### 1. data - 資料管理模組

**功能**: 負責下載、儲存和管理股票 K 線資料

**主要類別**: `StockDataManager`

**輸入**:
- `symbols` (list): 股票代碼列表，例如 `["TSLA", "AAPL"]`
- `start_date` (str, optional): 資料起始日期，格式為 `"YYYY-MM-DD"`。如果為 `None`，則下載該股票所有可用的歷史資料（通常從股票上市日期或 yfinance 有資料的最早日期開始，約為 5-10 年前）
- `end_date` (str, optional): 資料結束日期，格式為 `"YYYY-MM-DD"`。如果為 `None`，則下載到最新日期

**輸出**:
- `stock_data` (dict): 股票資料字典，格式為 `{股票代號: DataFrame}`
- DataFrame 包含欄位: `Open`, `High`, `Low`, `Close`, `Volume`

**注意事項**:
- 如果 `start_date` 為 `None`，yfinance 會下載該股票所有可用的歷史資料，資料量可能很大，下載時間較長
- 建議明確指定 `start_date` 以控制資料範圍和提升下載效率

**主要方法**:
- `get_kl_pd(symbol)`: 取得指定股票的 K 線 DataFrame
- `get_stock_data()`: 取得所有股票資料字典
- `get_stock_data_by_symbol(symbol)`: 取得單一股票資料
- `get_stock_data_by_symbols(symbols)`: 取得多檔股票資料

---

### 2. factors - 交易因子模組

#### 2.1 selection - 選股因子

**功能**: 判斷股票是否符合選股條件

**範例**:使用股價趨勢角度選股因子： 
![選股因子角度示意圖](https://github.com/Alan-Cheng/q-trade/blob/master/data/img/CH8.2_selection_angle.png?raw=true)

**基類**: `StockPickerBase`

**輸入**:
- `kl_pd` (pd.DataFrame): 股票的 K 線資料
- `target_symbol` (str): 股票代碼

**輸出**:
- `bool`: `True` 表示選中，`False` 表示不選中

**實作範例**:
- `RegressAngleFactor`: 基於回歸角度的選股因子

#### 2.2 timing - 時機因子

**功能**: 產生買入或賣出訊號
![時機因子示意圖](https://github.com/Alan-Cheng/q-trade/blob/master/data/img/CH8.1_timing.png?raw=true)

**基類**:
- `BaseBuyFactor`: 買入因子基類
- `BaseSellFactor`: 賣出因子基類

**輸入**:
- `df` (pd.DataFrame): 價格資料，必須包含 `Open`, `High`, `Low`, `Close` 欄位
- 其他參數依因子而定（例如 `xd`, `atr_multiplier` 等）

**輸出**:
- `np.ndarray`: 訊號陣列
  - 買入因子: `1` = 買入訊號, `NaN` = 無操作
  - 賣出因子: `0` = 賣出訊號, `NaN` = 無操作

**實作範例**:
- `BreakoutFactor`: 突破買入因子
- `ATRStopSellFactor`: ATR 停損賣出因子
- `RiskStopSellFactor`: 風險停損賣出因子
- `CloseATRStopSellFactor`: 收盤價 ATR 停損賣出因子

---

### 3. backtest - 回測引擎模組

#### 3.1 selection_backtester.py

**功能**: 執行選股回測，篩選符合條件的股票

**主要類別**: `StockPickerWorker`

**輸入**:
- `data_manager`: `StockDataManager` 實例
- `stock_pickers` (list): 選股因子配置列表，格式：
  ```python
  [
      {"class": RegressAngleFactor, "angle_threshold": 30},
      ...
  ]
  ```

**輸出**:
- `choice_symbols` (list): 被選中的股票代號列表
- `factor_results_df` (pd.DataFrame): 詳細的因子篩選結果，包含：
  - 各因子的通過/失敗狀態
  - 最終選中結果

#### 3.2 ~~timing_backtester_single.py~~（已棄用）

**功能**: 單檔股票時機回測引擎

**⚠️ 注意**: 此模組已棄用，建議使用 `timing_backtester.py` 的 `MultiStockBacktester`，即使只回測單一股票也可以使用（只需傳入包含單一股票的字典）。

**主要類別**: `Backtester`

**輸入**:
- `df` (pd.DataFrame): 價格資料，index 為日期，至少包含 `Close` 欄位
- `buy_factors` (list): 買入因子配置列表
  ```python
  [{"class": BreakoutFactor, "xd": 60}, ...]
  ```
- `sell_factors` (list): 賣出因子配置列表
  ```python
  [{"class": ATRStopSellFactor, "atr_multiplier": 2.0}, ...]
  ```
- `initial_capital` (float): 初始資金，預設 1,000,000
- `slippage_factors` (list, optional): 滑價因子配置列表

**輸出**:
- `result` (pd.DataFrame): 回測結果，包含：
  - `strategy_equity`: 策略權益曲線
  - `position`: 持倉股數
  - 各因子訊號欄位
- `trades` (pd.DataFrame): 交易紀錄，包含：
  - `entry_date`, `exit_date`: 進出場日期
  - `entry_price`, `exit_price`: 進出場價格
  - `shares`: 股數
  - `return_pct`: 報酬率
  - `pnl`: 盈虧金額
- `stats` (dict): 績效統計，包含：
  - `策略_總報酬率`, `策略_年化報酬率`
  - `策略_Sharpe`, `策略_最大回撤` 等
- `canceled_trades` (pd.DataFrame, optional): 被滑價取消的交易紀錄（如果有滑價模型）

#### 3.3 timing_backtester.py

**功能**: 多股票時機回測引擎，支援倉位管理

**主要類別**: `MultiStockBacktester`

**輸入**:
- `stock_data` (dict): 股票資料字典，格式 `{股票代號: DataFrame}`
- `buy_factors` (list, optional): 全局買入因子配置
- `sell_factors` (list, optional): 全局賣出因子配置
- `strategy_config` (dict, optional): 客製化策略字典
  ```python
  {
      "TSLA": {
          "buy_factors": [...],
          "sell_factors": [...]
      }
  }
  ```
- `initial_capital` (float): 初始資金，預設 1,000,000
- `position_manager` (BasePositionManager, optional): 倉位管理器
- `slippage_factors` (list, optional): 滑價因子配置列表
- `enable_full_rate_factor` (bool): 是否啟用滿倉模擬（將收益放大為同全倉投入收益）

**輸出**:
- `stock_results` (dict): 回測結果字典，包含：
  - `equity_curve`: 策略權益曲線
  - `raw_equity_curve`: 實際資金曲線（未調整）
  - `cash_curve`: 現金曲線
  - `benchmark_equity`: 基準權益曲線
- `trades` (pd.DataFrame): 所有股票的交易紀錄
- `stats` (dict): 績效統計，包含策略和基準的各項指標
- `canceled_trades` (pd.DataFrame, optional): 被滑價取消的交易紀錄

---

### 4. positions - 倉位管理模組

**功能**: 計算每次交易應該買入的股數

**基類**: `BasePositionManager`

**輸入**:
- `current_price` (float): 當前價格
- `available_capital` (float): 可用資金
- `**kwargs`: 其他參數，可能包含：
  - `total_capital`: 總資金
  - `current_holdings`: 當前持倉股票數量
  - `atr`: ATR 值（用於風險基礎倉位）
  - `returns`: 歷史報酬率（用於波動率倉位）

**輸出**:
- `float`: 應該買入的股數（可以是小數）

**實作範例**:
- `EqualWeightPositionManager`: 等權重倉位管理
- `FixedRatioPositionManager`: 固定比例倉位管理
- `RiskBasePositionManager`: 風險基礎倉位管理
- `VolatilityPositionManager`: 波動率倉位管理
- `KellyPositionManager`: 凱利公式倉位管理

---

### 5. slippage - 滑價模型模組

**功能**: 模擬實際交易中的滑價和撤單情況

**基類**: `BaseSlippage`

**輸入**:
- `df` (pd.DataFrame): 整個價格資料（用於前置計算）
- `action` (str): `"buy"` 或 `"sell"`，指定應用於買入或賣出
- 其他參數依模型而定

**方法**:
- `set_current_data(row)`: 在回測每日迴圈中被呼叫，傳入當天數據
- `fit_price()`: 計算滑價後的成交價格

**輸出**:
- `float`: 滑價後的成交價格
  - 買入撤單: 回傳 `np.inf`
  - 賣出撤單: 回傳 `0` 或 `-np.inf`

**實作範例**:
- `SlippageOpenGap`: 開盤大跌滑價模型
  - 參數: `open_down_rate` (float): 開盤跌幅閾值，預設 0.07

---

### 6. performance - 績效評估模組

**功能**: 計算和顯示各種績效指標

**主要類別**: `PerformanceMetrics`

**輸入**:
- `trades` (pd.DataFrame): 交易紀錄 DataFrame
- `stats` (dict, optional): 基本統計字典
- `equity_curve` (pd.Series, optional): 權益曲線
- `initial_capital` (float, optional): 初始資金
- `basic_metrics` (list, optional): 基本指標配置列表
- `detailed_metrics` (list, optional): 詳細指標配置列表

**輸出**:
- `stats` (dict): 基本績效指標，包含：
  - `策略_總報酬率`, `策略_年化報酬率`
  - `策略_年化波動率`, `策略_Sharpe`
  - `策略_最大回撤`, `策略_最終權益`
- `detailed_stats` (dict): 詳細績效指標，包含：
  - `總交易次數`, `獲利交易次數`, `虧損交易次數`
  - `勝率`, `平均獲利`, `平均虧損`
  - `盈虧比`, `淨獲利`
  - `最大單筆獲利`, `最大單筆虧損`
  - `平均持倉天數`, `平均報酬率`
  - `最大連續獲利次數`, `最大連續虧損次數`

**主要方法**:
- `show_summary()`: 顯示績效摘要
- `show_by_symbol()`: 按股票分組顯示績效
- `plot_equity_curve()`: 繪製權益曲線圖
- `plot_drawdown()`: 繪製回撤曲線圖
- `plot_trade_distribution()`: 繪製交易盈虧分佈圖
- `plot_monthly_returns()`: 繪製月度報酬圖
- `get_metrics_dataframe()`: 取得所有指標的 DataFrame

#### 6.1 basic - 基本績效指標

**基類**: `BaseBasicMetric`

**實作範例**:
- `TotalReturnMetric`: 總報酬率
- `AnnualReturnMetric`: 年化報酬率
- `VolatilityMetric`: 年化波動率
- `SharpeMetric`: Sharpe 比率
- `MaxDrawdownMetric`: 最大回撤
- `FinalEquityMetric`: 最終權益

#### 6.2 detail - 詳細績效指標

**基類**: `BaseDetailedMetric`

**實作範例**:
- `TotalTradesMetric`: 總交易次數
- `WinRateMetric`: 勝率
- `WinningTradesMetric`: 獲利交易次數
- `LosingTradesMetric`: 虧損交易次數
- `ProfitLossRatioMetric`: 盈虧比
- `AvgProfitMetric`: 平均獲利
- `AvgLossMetric`: 平均虧損
- `NetProfitMetric`: 淨獲利
- `MaxSingleProfitMetric`: 最大單筆獲利
- `MaxSingleLossMetric`: 最大單筆虧損
- `AvgHoldingDaysMetric`: 平均持倉天數
- `AvgReturnMetric`: 平均報酬率
- `MaxConsecutiveWinsMetric`: 最大連續獲利次數
- `MaxConsecutiveLossesMetric`: 最大連續虧損次數

---

### 7. optimization - 因子優化模組

**功能**: 自動化測試不同因子組合，找出最佳參數配置

![參數優化示意圖](https://github.com/Alan-Cheng/q-trade/blob/master/data/img/CH9_param_opt.png?raw=true)

#### 7.1 factor_optimizer.py

**功能**: 單股票因子優化器

**主要類別**: `FactorOptimizer`

**輸入**:
- `df` (pd.DataFrame): 股票價格資料
- `parameter_space` (ParameterSpace): 參數空間定義
- `initial_capital` (float): 初始資金，預設 1,000,000
- `slippage_factors` (list, optional): 滑價因子列表
- `n_jobs` (int): 並行處理進程數，`-1` 表示使用所有 CPU 核心
- `metric_weights` (dict or list, optional): 績效指標權重配置
  - 可以是字典: `{'策略_總報酬率': 0.3, '策略_Sharpe': 0.2, ...}`
  - 或指標類別列表: `[{"class": TotalReturnMetric, "weight": 3}, ...]`
- `basic_metrics` (list, optional): 基本指標列表
- `detailed_metrics` (list, optional): 詳細指標列表

**輸出**:
- `best_config` (dict): 最佳配置字典，包含：
  - `buy_factors`: 最佳買入因子配置
  - `sell_factors`: 最佳賣出因子配置
  - `總得分`: 總得分
- `results_df` (pd.DataFrame): 所有組合的結果 DataFrame，包含：
  - `總得分`: 總得分（排序依據）
  - `買入因子`, `賣出因子`: 因子配置
  - 各項績效指標和單項得分

**主要方法**:
- `optimize()`: 執行優化
- `get_top_n(n)`: 取得前 N 名結果
- `save_results(filepath)`: 儲存結果到 CSV

#### 7.2 multi_stock_factor_optimizer.py

**功能**: 多股票因子優化器

**主要類別**: `MultiStockFactorOptimizer`

**輸入**:
- `stock_data` (dict): 股票資料字典
- `parameter_space` (ParameterSpace): 參數空間定義
- `initial_capital` (float): 初始資金
- `slippage_factors` (list, optional): 滑價因子列表
- `position_manager` (BasePositionManager, optional): 倉位管理器
- `enable_full_rate_factor` (bool): 是否啟用滿倉模擬
- `n_jobs` (int): 並行處理進程數
- `metric_weights` (dict or list, optional): 績效指標權重配置
- `basic_metrics` (list, optional): 基本指標列表
- `detailed_metrics` (list, optional): 詳細指標列表

**輸出**: 與 `FactorOptimizer` 相同

#### 7.3 parameter_space.py

**功能**: 定義參數空間，生成所有因子組合

**主要類別**: `ParameterSpace`

**輸入**: 因子配置列表，例如：
```python
ParameterSpace(
    buy_factors=[
        [{"class": BreakoutFactor, "xd": xd} for xd in [20, 40, 60]],
        ...
    ],
    sell_factors=[
        [{"class": ATRStopSellFactor, "atr_multiplier": mult} for mult in [1.5, 2.0, 2.5]],
        ...
    ]
)
```

**輸出**:
- `combinations` (list): 所有因子組合列表

#### 7.4 scorer.py

**功能**: 定義可用的績效指標和評分方法

**主要內容**:
- `AVAILABLE_METRICS` (dict): 可用指標字典，定義各指標的：
  - `key`: 指標鍵名
  - `source`: 資料來源（`stats` 或 `detailed_stats`）
  - `higher_is_better`: 是否越大越好

---

### 8. benchmark - 基準計算模組

**功能**: 計算基準績效（Buy & Hold 策略）

**主要類別**: `Benchmark`

**靜態方法**:

#### 8.1 單股票基準

- `compute_log_ret(df)`: 計算對數報酬率
  - **輸入**: `df` (pd.DataFrame) - 單股票價格資料
  - **輸出**: `pd.Series` - 對數報酬率序列

- `compute_equity_curve(df, initial_capital)`: 計算基準權益曲線
  - **輸入**: 
    - `df` (pd.DataFrame) - 單股票價格資料
    - `initial_capital` (float) - 初始資金
  - **輸出**: `pd.Series` - 基準權益曲線

- `compute_single_stock_benchmark(df, initial_capital)`: 計算完整基準績效
  - **輸入**: 同上
  - **輸出**: `dict` - 包含 `equity_curve`, `log_ret`, `stats`

#### 8.2 多股票基準

- `compute_multi_stock_equity_curve(stock_data, initial_capital)`: 計算多股票基準權益曲線（平均分散）
  - **輸入**:
    - `stock_data` (dict) - 股票資料字典
    - `initial_capital` (float) - 初始資金
  - **輸出**: `pd.Series` - 基準權益曲線

- `compute_multi_stock_benchmark(stock_data, initial_capital)`: 計算完整多股票基準績效
  - **輸入**: 同上
  - **輸出**: `dict` - 包含 `equity_curve`, `log_ret`, `stats`

#### 8.3 績效統計

- `compute_performance_stats(equity_curve, initial_capital, log_ret=None)`: 計算基準績效指標
  - **輸入**:
    - `equity_curve` (pd.Series) - 基準權益曲線
    - `initial_capital` (float) - 初始資金
    - `log_ret` (pd.Series, optional) - 對數報酬率
  - **輸出**: `dict` - 包含：
    - `總報酬率`, `年化報酬率`
    - `年化波動率`, `Sharpe`
    - `最大回撤`

---

### 9. util - 工具模組

#### 9.1 regression_util.py

**功能**: 計算標準化後的股價趨勢角度

**主要類別**: `RegressionUtil`

**靜態方法**:
- `calc_regress_deg(data, symbol=None, show=False)`: 計算趨勢角度
  - **輸入**:
    - `data` (pd.Series): 價格序列
    - `symbol` (str, optional): 股票代號（用於顯示）
    - `show` (bool): 是否顯示圖表
  - **輸出**: `float` - 趨勢角度（度數）

**裝飾器**:
- `reversed_result(func)`: 如果 `self.reversed` 為 `True`，則翻轉 `fit_pick` 的布林返回值

---

## 使用範例

### 基本回測流程

```python
from alanq.data.data_manager import StockDataManager
# ~~from alanq.backtest.timing_backtester_single import Backtester~~  # 已棄用
from alanq.backtest.timing_backtester import MultiStockBacktester  # 推薦使用
from alanq.factors.timing.breakout_factor import BreakoutFactor
from alanq.factors.timing.atr_stop_sell_factor import ATRStopSellFactor

# 1. 下載資料
data_manager = StockDataManager(
    symbols=["TSLA"],
    start_date="2020-01-01",
    end_date="2023-12-31"
)
stock_data = data_manager.get_stock_data()

# 2. 設定因子
buy_factors = [
    {"class": BreakoutFactor, "xd": 60}
]
sell_factors = [
    {"class": ATRStopSellFactor, "atr_multiplier": 2.0}
]

# 3. 執行回測（使用 MultiStockBacktester，即使只有單一股票）
backtester = MultiStockBacktester(
    stock_data=stock_data,  # 傳入字典格式
    buy_factors=buy_factors,
    sell_factors=sell_factors,
    initial_capital=1_000_000
)

stock_results, trades, stats = backtester.run(show_plot=True)

# 4. 查看結果
print(stats)
print(trades.head())
```

### 多股票回測

```python
from alanq.backtest.timing_backtester import MultiStockBacktester
from alanq.positions.equal_weight_position import EqualWeightPositionManager

# 1. 準備多股票資料
data_manager = StockDataManager(
    symbols=["TSLA", "AAPL", "MSFT"],
    start_date="2020-01-01",
    end_date="2023-12-31"
)
stock_data = data_manager.get_stock_data()

# 2. 設定倉位管理
position_manager = EqualWeightPositionManager(max_stocks=5)

# 3. 執行多股票回測
backtester = MultiStockBacktester(
    stock_data=stock_data,
    buy_factors=buy_factors,
    sell_factors=sell_factors,
    initial_capital=1_000_000,
    position_manager=position_manager
)

stock_results, trades, stats = backtester.run(show_plot=True)
```

### 因子優化

```python
from alanq.optimization.factor_optimizer import FactorOptimizer
from alanq.optimization.parameter_space import ParameterSpace

# 1. 定義參數空間
parameter_space = ParameterSpace(
    buy_factors=[
        [{"class": BreakoutFactor, "xd": xd} for xd in [20, 40, 60, 80]]
    ],
    sell_factors=[
        [{"class": ATRStopSellFactor, "atr_multiplier": mult} 
         for mult in [1.5, 2.0, 2.5, 3.0]]
    ]
)

# 2. 執行優化
optimizer = FactorOptimizer(
    df=df,
    parameter_space=parameter_space,
    initial_capital=1_000_000,
    n_jobs=-1  # 使用所有 CPU 核心
)

best_config, results_df = optimizer.optimize()

# 3. 查看最佳配置
print(f"最佳配置: {best_config}")
print(f"前 10 名結果:")
print(optimizer.get_top_n(10))
```

### 績效評估

```python
from alanq.performance.performance_metrics import PerformanceMetrics

# 計算績效指標
metrics = PerformanceMetrics(
    trades=trades,
    stats=stats,
    equity_curve=result['strategy_equity'],
    initial_capital=1_000_000
)

# 顯示摘要
metrics.show_summary()

# 繪製圖表
metrics.plot_equity_curve()
metrics.plot_drawdown()
metrics.plot_trade_distribution()
```

### 機器學習應用

本專案也包含機器學習相關的應用範例，詳見 `notebooks/CH10-監督與非監督機器學習.ipynb`。

![機器學習分類示意圖](https://github.com/Alan-Cheng/q-trade/blob/master/data/img/CH10_classification.png?raw=true)

![決策樹示意圖](https://github.com/Alan-Cheng/q-trade/blob/master/data/img/CH10_decision_tree.png?raw=true)

---

## 依賴套件

主要依賴套件：
- `pandas`: 資料處理
- `numpy`: 數值計算
- `matplotlib`: 圖表繪製
- `yfinance`: 股票資料下載
- `statsmodels`: 統計分析（用於回歸工具）

---

## 注意事項

1. **資料格式**: 所有價格資料的 DataFrame 必須包含 `Open`, `High`, `Low`, `Close` 欄位，且 index 為日期
2. **因子配置**: 因子配置必須包含 `"class"` 鍵，指定因子類別
3. **滑價模型**: 滑價模型必須指定 `action` 參數（`"buy"` 或 `"sell"`）
4. **並行處理**: 優化器支援多進程並行處理，可大幅提升效率
5. **滿倉模擬**: 多股票回測器支援 `enable_full_rate_factor` 選項，可將收益放大為同全倉投入收益，便於與基準比較

---

