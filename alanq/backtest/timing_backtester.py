import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..positions.base_position import BasePositionManager
from ..positions.fix_ratio_position import FixedRatioPositionManager
from ..benchmark.benchmark import Benchmark

# =========================================================
# MultiStockBacktester：多股票回測引擎
# =========================================================
class MultiStockBacktester:
    """
    多股票回測引擎，支援倉位管理
    """
    
    def __init__(self,
                 stock_data: dict,  # {股票代號: DataFrame}
                 # 將 buy/sell factors 設為可選，作為全局預設
                 buy_factors: list = None,
                 sell_factors: list = None,
                 # 新增 strategy_config 來接收每檔股票的客製化策略
                 strategy_config: dict = None, # {股票代號: {'buy_factors': list, 'sell_factors': list}}
                 initial_capital: float = 1_000_000,
                 position_manager: BasePositionManager = None,
                 slippage_factors: list = None):
        """
        Parameters:
        -----------
        stock_data : dict
            股票資料字典，格式：{股票代號: DataFrame}
            DataFrame 必須包含 'Close', 'High', 'Low', 'Open' 欄位
        buy_factors : list
            買入因子設定列表
        sell_factors : list
            賣出因子設定列表
        initial_capital : float
            初始資金
        position_manager : BasePositionManager
            倉位管理類別實例，如果為 None 則使用 FixedRatioPositionManager
        slippage_factors : list
            滑價因子設定列表（可選）
        strategy_config : dict
            客製化策略字典，格式：{股票代號: {'buy_factors': list, 'sell_factors': list}}
            如果某股票在此字典中，將使用其專屬策略，否則使用全局 buy_factors/sell_factors。
        """
        self.stock_data = stock_data
        # 將全局配置儲存起來
        self.global_buy_factors = buy_factors or []
        self.global_sell_factors = sell_factors or []
        
        # 儲存客製化配置
        self.strategy_config = strategy_config or {}
        
        self.initial_capital = initial_capital
        self.slippage_factors = slippage_factors or []
        self.buy_factors = buy_factors or []
        self.sell_factors = sell_factors or []
        self.initial_capital = initial_capital
        self.slippage_factors = slippage_factors or []
        
        # 倉位管理（預設使用固定比例）
        if position_manager is None:
            self.position_manager = FixedRatioPositionManager(position_ratio=0.5)
        else:
            self.position_manager = position_manager
        
        # 儲存結果
        self.stock_results = {}
        self.trades = None
        self.stats = None
        self.benchmark_stats = None  # 儲存 benchmark 統計資料
        self.benchmark_equity = None  # 儲存 benchmark 權益曲線
        
        # 內部變數
        self.stock_signals = {}  # 每個股票的訊號
        self.stock_positions = {}  # 每個股票的持倉
        
        # 滑價模型相關變數
        self.slippage_models = {}  # 儲存滑價模型實例的字典 (以 factor_name 為鍵)
        self.slippage_cols = []  # 記錄所有滑價因子欄位名稱的列表
        self.canceled_trades = None  # 儲存被滑價取消的交易紀錄
        
    def run(self, show_plot=False, show_trades_plot=False, plot_symbols=None, plot_start=None, plot_end=None):
        """
        執行回測
        
        Parameters:
        -----------
        show_plot : bool
            是否顯示權益曲線圖
        show_trades_plot : bool
            是否顯示交易點視覺化圖
        plot_symbols : list or None
            要繪製交易圖的股票代號列表，如果為 None 則繪製所有有交易的股票
        plot_start : str or datetime
            交易圖的起始日期（可選）
        plot_end : str or datetime
            交易圖的結束日期（可選）
        
        Returns:
        --------
        dict : 包含各股票結果的字典
        """
        # 1. 對每個股票產生訊號
        self._apply_factors_to_all_stocks()
        
        # 2. 應用滑價模型（必須在 _run_multi_stock_backtest 之前）
        self._apply_slippage_models()
        
        # 3. 建立統一的交易日曆
        self._create_trading_calendar()
        
        # 4. 執行多股票回測（含倉位管理）
        self._run_multi_stock_backtest()
        
        # 5. 計算績效
        self._compute_performance()
        
        # 6. 提取交易紀錄
        self._extract_all_trades()
        
        # 7. 提取被取消的交易
        self._extract_canceled_trades()
        
        if show_plot:
            self._plot_results()
        
        if show_trades_plot:
            self._plot_trades(plot_symbols=plot_symbols, plot_start=plot_start, plot_end=plot_end)
        
        # 根據是否有滑價，決定回傳數量
        base_results = (self.stock_results, self.trades, self.stats)
        
        if bool(self.slippage_factors):
            # 有滑價模型時，回傳 4 個值
            return (*base_results, self.canceled_trades)
        else:
            # 沒有滑價模型時，只回傳 3 個值
            return base_results
    
    def _apply_factors_to_all_stocks(self):
        """對所有股票應用買入/賣出因子 (支援客製化策略)"""
        for symbol, df in self.stock_data.items():
            # 根據股票代號決定使用的因子列表
            if symbol in self.strategy_config:
                # 使用該股票的客製化策略
                buy_factors = self.strategy_config[symbol].get('buy_factors', [])
                sell_factors = self.strategy_config[symbol].get('sell_factors', [])
            else:
                # 使用全局策略
                buy_factors = self.global_buy_factors
                sell_factors = self.global_sell_factors
            
            # 複製資料
            df_copy = df.copy()
            
            # 初始化訊號欄位
            df_copy["buy_factor_trigger"] = ""
            df_copy["sell_factor_trigger"] = ""
            df_copy["slippage_trigger"] = ""
            
            buy_cols = []
            sell_cols = []
            
            # 應用買入因子 (使用動態選取的 buy_factors)
            for f in buy_factors:
                FactorClass = f["class"]
                params = {k: v for k, v in f.items() if k != "class"}
                factor = FactorClass(df_copy, **params)
                
                col = factor.factor_name
                buy_cols.append(col)
                df_copy[col] = factor.generate()
                
                # 記錄觸發因子
                df_copy.loc[df_copy[col] == 1, "buy_factor_trigger"] += (col + ";")
            
            # 應用賣出因子 (使用動態選取的 sell_factors)
            for f in sell_factors:
                FactorClass = f["class"]
                params = {k: v for k, v in f.items() if k != "class"}
                factor = FactorClass(df_copy, **params)
                
                col = factor.factor_name
                sell_cols.append(col)
                df_copy[col] = factor.generate()
                
                # 記錄觸發因子
                df_copy.loc[df_copy[col] == 0, "sell_factor_trigger"] += (col + ";")
            
            # 儲存結果
            self.stock_signals[symbol] = {
                'df': df_copy,
                'buy_cols': buy_cols,
                'sell_cols': sell_cols
            }
    
    def _apply_slippage_models(self):
        """實例化所有滑價模型並儲存在 self.slippage_models 字典中"""
        if not self.slippage_factors:
            return
        
        # 為每個股票建立滑價模型
        for symbol, signals in self.stock_signals.items():
            df = signals['df']
            
            for sf in self.slippage_factors:
                SlippageClass = sf["class"]
                
                # 從設定中提取參數
                params = {k: v for k, v in sf.items() if k != "class"}
                
                # 實例化滑價類別
                model_instance = SlippageClass(df, **params)
                
                col = model_instance.factor_name
                if col not in self.slippage_cols:
                    self.slippage_cols.append(col)
                
                # 以 (symbol, factor_name) 為鍵儲存實例
                key = (symbol, col)
                self.slippage_models[key] = model_instance
    
    def _create_trading_calendar(self):
        """建立統一的交易日曆（所有股票的交集日期）"""
        all_dates = None
        
        for symbol, signals in self.stock_signals.items():
            df = signals['df']
            if all_dates is None:
                all_dates = set(df.index)
            else:
                all_dates = all_dates.intersection(set(df.index))
        
        # 轉換為排序後的列表
        self.trading_dates = sorted(list(all_dates))
    
    def _run_multi_stock_backtest(self):
        """
        執行多股票回測，包含倉位管理
        """
        # 初始化資金和持倉
        cash = self.initial_capital
        holdings = {}  # {股票代號: 股數}
        
        # 初始化結果記錄
        for symbol in self.stock_data.keys():
            self.stock_positions[symbol] = []
        
        equity_history = []
        cash_history = []
        
        # 逐日回測
        for date in self.trading_dates:
            # 計算當前總權益
            total_equity = cash
            
            # 更新每個股票的持倉市值
            for symbol in holdings.keys():
                if symbol in self.stock_signals:
                    df = self.stock_signals[symbol]['df']
                    if date in df.index:
                        current_price = df.loc[date, 'Close']
                        total_equity += holdings[symbol] * current_price
            
            # 處理每個股票的交易訊號
            for symbol, signals in self.stock_signals.items():
                df = signals['df']
                buy_cols = signals['buy_cols']
                sell_cols = signals['sell_cols']
                
                if date not in df.index:
                    # 如果該日期沒有資料，維持現狀
                    current_shares = holdings.get(symbol, 0)
                    self.stock_positions[symbol].append(current_shares)
                    continue
                
                row = df.loc[date]
                current_price = row['Close']
                current_shares = holdings.get(symbol, 0)
                
                # 初始化當天的滑價觸發記錄
                current_slippage_trigger = ""
                
                # 檢查買入訊號
                buy_signal = any(row[col] == 1 for col in buy_cols) if buy_cols else False
                
                # 檢查賣出訊號
                sell_signal = any(row[col] == 0 for col in sell_cols) if sell_cols else False
                
                # 處理賣出（優先）
                if sell_signal and current_shares > 0:
                    # 應用滑價模型
                    can_sell = True
                    final_sell_price = current_price  # 預設成交價
                    
                    # 檢查所有設定為 "sell" 的滑價模型
                    for col in self.slippage_cols:
                        key = (symbol, col)
                        if key in self.slippage_models:
                            model = self.slippage_models[key]
                            
                            if model.params.get("action") == "sell":
                                model.set_current_data(row)
                                adjusted_price = model.fit_price()
                                
                                # 如果滑價計算的價格為 0 或 -Inf，視為撤單（賣不出）
                                if adjusted_price == 0 or adjusted_price == -np.inf:
                                    can_sell = False
                                    current_slippage_trigger += (col + ";")
                                    break
                                else:
                                    # 取最差的價格 (即最低的賣價) 作為最終成交價
                                    final_sell_price = min(final_sell_price, adjusted_price)
                    
                    if can_sell:
                        # 全部賣出
                        cash += current_shares * final_sell_price
                        holdings[symbol] = 0
                        current_shares = 0
                    
                    # 記錄當天的滑價觸發事件
                    df.loc[date, "slippage_trigger"] = current_slippage_trigger
                
                # 處理買入
                elif buy_signal and current_shares == 0:
                    # 應用滑價模型
                    can_buy = True
                    final_buy_price = current_price  # 預設成交價
                    
                    # 檢查所有設定為 "buy" 的滑價模型
                    for col in self.slippage_cols:
                        key = (symbol, col)
                        if key in self.slippage_models:
                            model = self.slippage_models[key]
                            
                            if model.params.get("action") == "buy":
                                model.set_current_data(row)
                                adjusted_price = model.fit_price()
                                
                                # 如果滑價計算的價格為 Inf，視為撤單（買不到）
                                if adjusted_price == np.inf:
                                    can_buy = False
                                    current_slippage_trigger += (col + ";")
                                    break
                                else:
                                    # 取最差的價格 (即最高的買價) 作為最終成交價
                                    final_buy_price = max(final_buy_price, adjusted_price)
                    
                    if can_buy:
                        # 計算可用資金（考慮現有持倉）
                        available_capital = cash
                        
                        # 準備倉位計算的參數
                        position_kwargs = {
                            'total_capital': total_equity,
                            'current_holdings': len([s for s in holdings.values() if s > 0])
                        }
                        
                        # 如果有 ATR，加入參數
                        if 'atr14' in df.columns:
                            position_kwargs['atr'] = row.get('atr14', current_price * 0.02)
                        
                        # 如果有歷史報酬率，加入參數
                        if 'log_ret' in df.columns:
                            returns = df.loc[:date, 'log_ret'].dropna()
                            if len(returns) > 0:
                                position_kwargs['returns'] = returns
                        
                        # 計算應該買入的股數（使用滑價調整後的價格）
                        shares_to_buy = self.position_manager.calculate_position_size(
                            current_price=final_buy_price,
                            available_capital=available_capital,
                            **position_kwargs
                        )
                        
                        # 計算實際買入金額（使用滑價調整後的價格）
                        cost = shares_to_buy * final_buy_price
                        
                        # 檢查資金是否足夠
                        if cost <= cash and shares_to_buy > 0:
                            cash -= cost
                            holdings[symbol] = shares_to_buy
                            current_shares = shares_to_buy
                    
                    # 記錄當天的滑價觸發事件（已在買入/賣出邏輯中更新）
                    pass
                
                # 記錄當天的滑價觸發事件（確保所有日期都有記錄）
                df.loc[date, "slippage_trigger"] = current_slippage_trigger
                
                # 記錄持倉
                self.stock_positions[symbol].append(current_shares)
            
            # 重新計算總權益（處理完所有交易後）
            total_equity = cash
            for symbol in holdings.keys():
                if symbol in self.stock_signals:
                    df = self.stock_signals[symbol]['df']
                    if date in df.index:
                        current_price = df.loc[date, 'Close']
                        total_equity += holdings[symbol] * current_price
            
            # 記錄資金和權益
            equity_history.append(total_equity)
            cash_history.append(cash)
        
        # 儲存結果
        self.equity_history = pd.Series(equity_history, index=self.trading_dates)
        self.cash_history = pd.Series(cash_history, index=self.trading_dates)
        
        # 將持倉資訊加入每個股票的 DataFrame
        for symbol, positions in self.stock_positions.items():
            df = self.stock_signals[symbol]['df']
            df['position'] = pd.Series(positions, index=self.trading_dates)
            df['position'] = df['position'].fillna(0)
    
    def _compute_performance(self):
        """計算績效指標"""
        # 計算每日報酬率
        equity_series = self.equity_history
        daily_returns = equity_series.pct_change().dropna()
        
        # 總報酬率
        total_return = (equity_series.iloc[-1] / self.initial_capital)
        
        # 年化報酬率
        days = (equity_series.index[-1] - equity_series.index[0]).days
        years = days / 365.0 if days > 0 else 1.0
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
        
        # 波動率和 Sharpe
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else np.nan
        else:
            volatility = np.nan
            sharpe = np.nan
        
        # 最大回撤
        roll_max = equity_series.cummax()
        drawdown = equity_series / roll_max - 1.0
        max_drawdown = drawdown.min()
        
        # 計算 Benchmark 績效（多股票平均分散）
        benchmark_result = Benchmark.compute_multi_stock_benchmark(self.stock_data, self.initial_capital)
        self.benchmark_equity = benchmark_result['equity_curve']
        self.benchmark_stats = benchmark_result['stats']
        
        # 儲存統計資料
        self.stats = {
            "策略_總報酬率": total_return,
            "策略_年化報酬率": annual_return,
            "策略_年化波動率": volatility,
            "策略_Sharpe": sharpe,
            "策略_最大回撤": max_drawdown,
            "策略_最終權益": equity_series.iloc[-1],
            "策略_最終現金": self.cash_history.iloc[-1],
            "基準_總報酬率": self.benchmark_stats.get("總報酬率", np.nan),
            "基準_年化報酬率": self.benchmark_stats.get("年化報酬率", np.nan),
            "基準_年化波動率": self.benchmark_stats.get("年化波動率", np.nan),
            "基準_Sharpe": self.benchmark_stats.get("Sharpe", np.nan),
            "基準_最大回撤": self.benchmark_stats.get("最大回撤", np.nan),
        }
        
        # 儲存權益曲線
        self.stock_results['equity_curve'] = equity_series
        self.stock_results['cash_curve'] = self.cash_history
        self.stock_results['benchmark_equity'] = self.benchmark_equity
    
    def _extract_all_trades(self):
        """提取所有交易紀錄"""
        all_trades = []
        
        for symbol, signals in self.stock_signals.items():
            df = signals['df']
            pos = df['position']
            change = pos.diff()
            
            entries = change[change > 0].index
            exits = change[change < 0].index
            
            # 如果最後還有持倉，加入最後一個日期作為出場
            if len(entries) > len(exits):
                exits = list(exits) + [df.index[-1]]
            
            for entry_date, exit_date in zip(entries, exits):
                entry_price = df.loc[entry_date, 'Close']
                exit_price = df.loc[exit_date, 'Close']
                shares = df.loc[entry_date, 'position']
                
                all_trades.append({
                    "symbol": symbol,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "shares": shares,
                    "holding_days": (exit_date - entry_date).days,
                    "return_pct": (exit_price / entry_price) - 1,
                    "pnl": (exit_price - entry_price) * shares,
                    "buy_factor": df.loc[entry_date, "buy_factor_trigger"],
                    "sell_factor": df.loc[exit_date, "sell_factor_trigger"],
                })
        
        self.trades = pd.DataFrame(all_trades)
    
    def _extract_canceled_trades(self):
        """產生被滑價取消的交易紀錄"""
        if not self.slippage_factors:
            self.canceled_trades = pd.DataFrame()
            return
        
        all_canceled = []
        
        for symbol, signals in self.stock_signals.items():
            df = signals['df'].copy()
            
            # 為了判斷是否為「嘗試買入但失敗」或「嘗試賣出但失敗」，需要前一天的 position
            df["prev_position"] = df["position"].shift(1).fillna(0)
            
            # --- 1. 篩選出被取消的買入點 ---
            canceled_buy_mask = (
                (df["buy_factor_trigger"].str.len() > 0) &  # 有買入訊號
                (df["prev_position"] == 0) &                # 試圖買入前是空手
                (df["position"] == 0) &                     # 交易後仍是空手 (買入失敗)
                (df["slippage_trigger"].str.len() > 0)      # 失敗原因是滑價觸發
            )
            
            # --- 2. 篩選出被取消的賣出點 ---
            canceled_sell_mask = (
                (df["sell_factor_trigger"].str.len() > 0) & # 有賣出訊號
                (df["prev_position"] > 0) &                  # 試圖賣出前是持股
                (df["position"] > 0) &                     # 交易後仍是持股 (賣出失敗)
                (df["slippage_trigger"].str.len() > 0)      # 失敗原因是滑價觸發
            )
            
            canceled_df = df[canceled_buy_mask | canceled_sell_mask].copy()
            
            for date, row in canceled_df.iterrows():
                action = "Canceled Buy" if row["prev_position"] == 0 else "Canceled Sell"
                
                all_canceled.append({
                    "symbol": symbol,
                    "date": date,
                    "action": action,
                    "price": row["Close"],  # 紀錄當天收盤價
                    "signal_trigger": row["buy_factor_trigger"] if action == "Canceled Buy" else row["sell_factor_trigger"],
                    "slippage_factor": row["slippage_trigger"],
                    "current_holding": row["position"]
                })
        
        self.canceled_trades = pd.DataFrame(all_canceled)
    
    def _plot_results(self):
        """Plots the backtest results."""
        # Create two subplots: Equity Curve and Capital Allocation
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # --- Top Subplot: Equity Curve ---
        
        # Plot the strategy's total equity over time
        axes[0].plot(self.equity_history.index, self.equity_history.values, 
                    label='Strategy Equity', linewidth=2)
        
        # Plot benchmark equity if available
        if self.benchmark_equity is not None and not self.benchmark_equity.empty:
            # 對齊日期索引
            benchmark_aligned = self.benchmark_equity.reindex(self.equity_history.index, method='ffill')
            axes[0].plot(benchmark_aligned.index, benchmark_aligned.values, 
                        label='Benchmark (Buy & Hold)', linewidth=2, linestyle='--', alpha=0.7)
        
        # Draw a dashed line for the initial capital
        axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', 
                    label=f'Initial Capital ({self.initial_capital:,.0f})', alpha=0.5)
                    
        axes[0].set_title('Multi-Stock Backtest - Equity Curve', fontsize=14)
        axes[0].set_ylabel('Equity', fontsize=12) # Note: Adjust currency label as needed
        axes[0].legend()
        axes[0].grid(True)
        
        # --- Bottom Subplot: Capital Allocation ---
        
        # Calculate the current market value of the holdings (Position Value)
        position_value = self.equity_history - self.cash_history
        
        # Plot Cash and Position Value
        axes[1].plot(self.cash_history.index, self.cash_history.values, 
                    label='Cash', alpha=0.7)
        axes[1].plot(position_value.index, position_value.values, 
                    label='Position Value', alpha=0.7)
                    
        axes[1].set_title('Capital Allocation', fontsize=14)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Amount', fontsize=12) # Note: Adjust currency label as needed
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_trades(self, plot_symbols=None, plot_start=None, plot_end=None):
        """
        繪製交易點視覺化圖（類似 timing_backtester_single.py 的功能）
        
        Parameters:
        -----------
        plot_symbols : list or None
            要繪製的股票代號列表，如果為 None 則繪製所有有交易的股票
        plot_start : str or datetime
            起始日期（可選）
        plot_end : str or datetime
            結束日期（可選）
        """
        if self.trades is None or len(self.trades) == 0:
            print("沒有交易記錄可繪製")
            return
        
        # 決定要繪製哪些股票
        if plot_symbols is None:
            # 繪製所有有交易的股票
            symbols_to_plot = self.trades['symbol'].unique().tolist()
        else:
            # 只繪製指定的股票
            symbols_to_plot = [s for s in plot_symbols if s in self.trades['symbol'].values]
            if not symbols_to_plot:
                print(f"指定的股票代號 {plot_symbols} 沒有交易記錄")
                return
        
        # 為每個股票分別繪製
        for symbol in symbols_to_plot:
            # 取得該股票的資料
            if symbol not in self.stock_signals:
                continue
            
            df = self.stock_signals[symbol]['df'].copy()
            
            # 篩選該股票的交易記錄
            symbol_trades = self.trades[self.trades['symbol'] == symbol].copy()
            
            if len(symbol_trades) == 0:
                continue
            
            # 日期篩選
            if plot_start:
                start_date = pd.to_datetime(plot_start)
                df = df[df.index >= start_date]
                symbol_trades = symbol_trades[symbol_trades['entry_date'] >= start_date]
            
            if plot_end:
                end_date = pd.to_datetime(plot_end)
                df = df[df.index <= end_date]
                symbol_trades = symbol_trades[symbol_trades['exit_date'] <= end_date]
            
            if len(df) == 0 or len(symbol_trades) == 0:
                continue
            
            # 繪製圖表
            plt.figure(figsize=(18, 6))
            
            # ---- 黑色收盤線 ----
            plt.plot(df.index, df["Close"], color="black", label="Close", linewidth=1.5)
            
            # ---- 藍色底色 ----
            plt.fill_between(df.index, 0, df["Close"], color="blue", alpha=0.05)
            
            # ---- 逐筆畫出交易 ----
            is_first_trade = True
            for idx, t in symbol_trades.iterrows():
                buy = t["entry_date"]
                sell = t["exit_date"]
                buy_price = t["entry_price"]
                sell_price = t["exit_price"]
                pnl = sell_price - buy_price
                pnl_rate = t["return_pct"]
                
                color = "green" if pnl > 0 else "red"
                
                # 區間 mask
                mask = (df.index >= buy) & (df.index <= sell)
                
                if mask.sum() > 0:
                    # 區間背景
                    plt.fill_between(df.index[mask],
                                    0, df["Close"][mask],
                                    color=color, alpha=0.28)
                
                # Buy / Sell 散點
                if buy in df.index:
                    plt.scatter(buy, buy_price, color="blue", s=80, zorder=5, 
                              label="Buy" if is_first_trade else "")
                if sell in df.index:
                    plt.scatter(sell, sell_price, color="orange", s=80, zorder=5, 
                              label="Sell" if is_first_trade else "")
                
                # 盈虧數字
                if sell in df.index:
                    plt.text(sell, sell_price,
                            f"{pnl:+.2f} ({pnl_rate:+.2%})",
                            color=color, fontsize=9,
                            ha="left", va="bottom")
                
                is_first_trade = False
            
            plt.title(f"Trade Visualization - {symbol}", fontsize=14)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Price", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()