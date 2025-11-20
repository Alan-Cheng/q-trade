import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..benchmark.benchmark import Benchmark

# Backtester：單檔股票多因子回測引擎
# =========================================================
class Backtester:
    def __init__(self, 
                 df,
                 # 買入因子
                 buy_factors, 
                 # 賣出因子
                 sell_factors,
                 # 初始資金
                 initial_capital=1_000_000,
                 # 滑點模型
                 slippage_factors=None,
                 # 持股模型(待完成)
                 position_class=None):
        """
        df: 價格資料，index 為日期，至少包含 'Close'
        buy_factors: 買入因子設定 list
            e.g. [{"class": BreakoutBuyFactor, "xd": 60}, ...]
        sell_factors: 賣出因子設定 list
            e.g. [{"class": BreakdownSellFactor, "xd": 20}, ...]
        initial_capital: 初始資金（只用來算 equity，不做分批加減）
        """
        self.raw_df = df.copy()
        self.df = df.copy()
        self.buy_factors = buy_factors or []
        self.sell_factors = sell_factors or []
        self.initial_capital = initial_capital
        self.slippage_factors = slippage_factors or []

        self.buy_cols = []
        self.sell_cols = []
        self.result = None
        self.trades = None
        self.stats = None
        self.benchmark_stats = None  # 儲存 benchmark 統計資料
        # 新增：用於儲存滑價模型實例的字典 (以 factor_name 為鍵)
        self.slippage_models = {} 
        # 新增：用於記錄所有滑價因子欄位名稱的列表 (用於迭代)
        self.slippage_cols = []
        # 新增：用於儲存被滑價取消的交易紀錄
        self.canceled_trades = None

    # ---------------------
    # 主流程
    # ---------------------
    def run(self, show_plot=False, plot_start=None, plot_end=None):
        self._apply_factors()
        self._apply_slippage_models() # 必須在 _build_position 之前
        self._build_position()
        self._compute_equity()
        self._extract_trades()
        self._extract_canceled_trades() # 新增：提取被取消的交易
        self._compute_stats()

        # 如果使用者要求 → 畫圖
        if show_plot:
            self._plot_trades(plot_start, plot_end)

        # 根據是否有滑價，決定回傳數量
        base_results = (self.result, self.trades, self.stats)
        
        if bool(self.slippage_factors):
            # 有滑價模型時，回傳 4 個值
            return (*base_results, self.canceled_trades)
        else:
            # 沒有滑價模型時，只回傳 3 個值
            return base_results

    # ---------------------
    # 產生各因子訊號欄位
    # ---------------------
    def _apply_factors(self):
        df = self.df

        df["buy_factor_trigger"] = ""
        df["sell_factor_trigger"] = ""

        # 買入因子
        for f in self.buy_factors:
            FactorClass = f["class"]
            params = {k: v for k, v in f.items() if k != "class"}
            factor = FactorClass(df, **params)

            col = factor.factor_name
            self.buy_cols.append(col)

            df[col] = factor.generate()

            # 記錄：哪個因子產生買訊
            df.loc[df[col] == 1, "buy_factor_trigger"] += (col + ";")

        # 賣出因子
        for f in self.sell_factors:
            FactorClass = f["class"]
            params = {k: v for k, v in f.items() if k != "class"}
            factor = FactorClass(df, **params)

            col = factor.factor_name
            self.sell_cols.append(col)

            df[col] = factor.generate()

            # 記錄：哪個因子產生賣訊
            df.loc[df[col] == 0, "sell_factor_trigger"] += (col + ";")

    # # ---------------------
    # # 建立 signal & position（持股狀態）
    # # ---------------------
    # def _build_position(self):
    #     df = self.df

    #     # 買入：任一買入因子 == 1 就視為買入訊號
    #     if self.buy_cols:
    #         buy_signal = df[self.buy_cols].max(axis=1)
    #     else:
    #         buy_signal = pd.Series(np.nan, index=df.index)

    #     # 賣出：任一賣出因子 == 0 就視為賣出訊號
    #     if self.sell_cols:
    #         sell_signal = df[self.sell_cols].min(axis=1)
    #     else:
    #         sell_signal = pd.Series(np.nan, index=df.index)

    #     # 綜合 signal：優先考慮賣出，其次買入
    #     combined = pd.Series(np.nan, index=df.index)

    #     combined[buy_signal == 1] = 1   # 當天出現買入事件
    #     combined[sell_signal == 0] = 0  # 當天出現賣出事件（清倉）

    #     df["raw_signal"] = combined

    #     # position: 持股狀態
    #     # 規則：最後一次非 NaN 的 signal 決定目前狀態（1=持股, 0=空手）
    #     position = combined.ffill().fillna(0)

    #     # 為了避免「當天訊號吃到當天報酬」（偷看未來）
    #     # 我們用 position.shift(1) 來決定策略報酬
    #     df["position"] = position
    
    # ---------------------
    # 建立 position (持股狀態) (新邏輯)
    # ---------------------
    def _build_position(self):
        df = self.df

        position = []
        holding = 0
        
        # 準備記錄成交價格 (如果需要更精確的回測)
        # buy_price_list = []
        # sell_price_list = []

        for i, (date, row) in enumerate(df.iterrows()):

            buy_signal = any(row[col] == 1 for col in self.buy_cols)
            sell_signal = any(row[col] == 0 for col in self.sell_cols)
            
            current_slippage_trigger = "" # 重置當天滑價紀錄

            if buy_signal and holding == 0: 
                # -------------------
                # 買入處理
                # -------------------
                can_buy = True
                final_buy_price = row["Close"] # 預設成交價
                
                # 檢查所有設定為 "buy" 的滑價模型
                for name in self.slippage_cols:
                    model = self.slippage_models[name]
                    
                    # 假設滑價模型實例可以判斷其作用對象 (action)
                    # 假設模型實例的 params 字典中存有 "action" 鍵
                    if model.params.get("action") == "buy": 
                        
                        # 假設滑價模型有 set_current_data 來更新資料
                        model.set_current_data(row) 
                        adjusted_price = model.fit_price()
                        
                        # 如果滑價計算的價格為 Inf，視為撤單（買不到）
                        if adjusted_price == np.inf:
                            can_buy = False
                            current_slippage_trigger += (name + ";")
                            break # 只要有一個滑價模型導致撤單，就停止買入
                        else:
                            # 取最差的價格 (即最高的買價) 作為最終成交價
                            final_buy_price = max(final_buy_price, adjusted_price)
                
                if can_buy:
                    holding = 1
                    # buy_price_list.append(final_buy_price) # 儲存成交價
                # else:
                    # buy_price_list.append(np.nan)

            elif sell_signal and holding == 1: 
                # -------------------
                # 賣出處理
                # -------------------
                can_sell = True
                final_sell_price = row["Close"] # 預設成交價

                # 檢查所有設定為 "sell" 的滑價模型
                for name in self.slippage_cols:
                    model = self.slippage_models[name]
                    
                    if model.params.get("action") == "sell": 
                        
                        model.set_current_data(row)
                        adjusted_price = model.fit_price()
                        
                        # 如果滑價計算的價格為 0 (或 -Inf)，視為撤單（賣不出）
                        # 註：賣出通常假設不會被撤單，但以防萬一
                        if adjusted_price == 0 or adjusted_price == -np.inf:
                            can_sell = False
                            current_slippage_trigger += (name + ";")
                            break
                        else:
                            # 取最差的價格 (即最低的賣價) 作為最終成交價
                            final_sell_price = min(final_sell_price, adjusted_price)

                if can_sell:
                    holding = 0
                    # sell_price_list.append(final_sell_price) # 儲存成交價
                # else:
                    # sell_price_list.append(np.nan)
            
            # 記錄當天的滑價觸發事件
            df.loc[date, "slippage_trigger"] = current_slippage_trigger

            # 追加今天的持倉狀態
            position.append(holding)

        df["position"] = pd.Series(position, index=df.index)
        df["raw_signal"] = df["position"].diff()
        
        # 如果需要更精確的回測，需在 trades/equity 計算中使用儲存的 buy/sell price list


    # ---------------------
    # 計算基準與策略的 equity curve
    # ---------------------
    def _compute_equity(self):
        df = self.df

        # 使用 Benchmark 類別計算基準績效
        benchmark_result = Benchmark.compute_single_stock_benchmark(df, self.initial_capital)
        df["log_ret"] = benchmark_result['log_ret']
        df["benchmark_equity"] = benchmark_result['equity_curve']
        
        # 儲存 benchmark 統計資料供後續使用
        self.benchmark_stats = benchmark_result['stats']

        # 策略報酬：用「前一天」的持股狀態乘上今天的 log return
        df["strategy_log_ret"] = df["position"].shift(1).fillna(0) * df["log_ret"]

        # 累積報酬 → 換回金額
        df["strategy_equity"] = self.initial_capital * np.exp(df["strategy_log_ret"].cumsum())
        
        # 🔥 修正第一筆 NaN = 初始資金
        df.loc[df.index[0], "strategy_equity"] = self.initial_capital

        self.result = df

    # ---------------------
    # 產生交易紀錄（每筆進出場）
    # ---------------------
    def _extract_trades(self):
        df = self.result
        pos = df["position"]
        change = pos.diff()

        entries = change[change == 1].index
        exits = change[change == -1].index

        if len(entries) > len(exits):
            exits = pd.Index(list(exits) + [df.index[-1]])

        records = []
        for entry_date, exit_date in zip(entries, exits):
            entry_price = df.loc[entry_date, "Close"]
            exit_price = df.loc[exit_date, "Close"]
            
            # 計算可用資金（從 equity curve 變化推算，或使用 entry_date 前一天的 equity）
            if 'strategy_equity' in df.columns:
                # 使用 entry_date 前一天的 equity 作為可用資金
                prev_date_idx = df.index.get_loc(entry_date) - 1
                if prev_date_idx >= 0:
                    available_capital = df.iloc[prev_date_idx]['strategy_equity']
                else:
                    available_capital = self.initial_capital
            else:
                # 如果沒有 equity curve，使用初始資金
                available_capital = self.initial_capital
            
            # 計算股數（股票不可分割，必須是整數）
            # 向下取整，確保不超過可用資金
            shares = int(available_capital / entry_price) if entry_price > 0 else 0
            
            # 計算實際投入金額（基於整數股數）
            actual_cost = shares * entry_price
            
            # 計算剩餘現金（無法買一股的剩餘資金）
            remaining_cash = available_capital - actual_cost
            
            # 計算報酬率和盈虧
            return_pct = (exit_price / entry_price) - 1 if entry_price > 0 else 0
            pnl = (exit_price - entry_price) * shares
            
            records.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "actual_cost": actual_cost,  # 實際投入金額
                "remaining_cash": remaining_cash,  # 剩餘現金
                "holding_days": (exit_date - entry_date).days,
                "return_pct": return_pct,
                "pnl": pnl,
                "buy_factor": df.loc[entry_date, "buy_factor_trigger"],
                "sell_factor": df.loc[exit_date, "sell_factor_trigger"],
            })

        self.trades = pd.DataFrame(records)
        
    # ---------------------
    # 產生滑價模型實例
    # ---------------------
    def _apply_slippage_models(self):
        """實例化所有滑價模型並儲存在 self.slippage_models 字典中"""
        df = self.df
        
        # 預期 slippage_factors 結構：
        # [{"class": FixedSlippage, "action": "buy", "param": 0.001}, ...]
        
        df["slippage_trigger"] = "" # 新增一個欄位來記錄滑價事件
        
        for sf in self.slippage_factors:
            SlippageClass = sf["class"]
            
            # 從設定中提取參數
            params = {k: v for k, v in sf.items() if k != "class"}
            
            # 實例化滑價類別 (假設 BaseSlippage 也有 __init__(df, **kwargs) 和 factor_name 屬性)
            model_instance = SlippageClass(df, **params)
            
            col = model_instance.factor_name # 使用滑價模型自己的 factor_name
            self.slippage_cols.append(col)
            
            # 以 factor_name 為鍵儲存實例
            self.slippage_models[col] = model_instance
            
    # ---------------------
    # 產生被滑價取消的交易紀錄
    # ---------------------
    def _extract_canceled_trades(self):
        df = self.result.copy()
        
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
            (df["prev_position"] == 1) &                # 試圖賣出前是持股
            (df["position"] == 1) &                     # 交易後仍是持股 (賣出失敗)
            (df["slippage_trigger"].str.len() > 0)      # 失敗原因是滑價觸發
        )
        
        canceled_df = df[canceled_buy_mask | canceled_sell_mask].copy()

        records = []
        for date, row in canceled_df.iterrows():
            action = "Canceled Buy" if row["prev_position"] == 0 else "Canceled Sell"
            
            records.append({
                "date": date,
                "action": action,
                "price": row["Close"], # 紀錄當天收盤價
                "signal_trigger": row["buy_factor_trigger"] if action == "Canceled Buy" else row["sell_factor_trigger"],
                "slippage_factor": row["slippage_trigger"],
                "current_holding": row["position"]
            })

        self.canceled_trades = pd.DataFrame(records)

    # ---------------------
    # 計算一些基本績效指標
    # ---------------------
    def _compute_stats(self):
        df = self.result

        # ----------------------------
        # 策略績效（strategy）
        # ----------------------------
        total_ret = df["strategy_equity"].iloc[-1] / self.initial_capital

        days = (df.index[-1] - df.index[0]).days
        years = days / 365.0 if days > 0 else 1.0
        annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else total_ret

        daily_ret = df["strategy_log_ret"].dropna()
        if len(daily_ret) > 1 and daily_ret.std() > 0:
            vol = daily_ret.std() * np.sqrt(252)
            sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
        else:
            vol = np.nan
            sharpe = np.nan

        equity = df["strategy_equity"]
        roll_max = equity.cummax()
        dd = equity / roll_max - 1.0
        max_dd = dd.min()

        # ----------------------------
        # 基準績效（benchmark）- 使用 Benchmark 類別計算的結果
        # ----------------------------
        benchmark_stats = getattr(self, 'benchmark_stats', {})
        benchmark_total_ret = benchmark_stats.get("總報酬率", np.nan)
        benchmark_annual_ret = benchmark_stats.get("年化報酬率", np.nan)
        benchmark_vol = benchmark_stats.get("年化波動率", np.nan)
        benchmark_sharpe = benchmark_stats.get("Sharpe", np.nan)
        benchmark_max_dd = benchmark_stats.get("最大回撤", np.nan)

        # ----------------------------
        # 統整最終績效（中文）
        # ----------------------------
        self.stats = {
            "策略_總報酬率": total_ret,
            "策略_年化報酬率": annual_ret,
            "策略_年化波動率": vol,
            "策略_Sharpe": sharpe,
            "策略_最大回撤": max_dd,
            "策略_交易次數": 0 if self.trades is None else len(self.trades),

            "基準_總報酬率": benchmark_total_ret,
            "基準_年化報酬率": benchmark_annual_ret,
            "基準_年化波動率": benchmark_vol,
            "基準_Sharpe": benchmark_sharpe,
            "基準_最大回撤": benchmark_max_dd,
        }

        
    # =========================================================
    # ★ 交易視覺化
    # =========================================================
    def _plot_trades(self, start=None, end=None):
        df = self.result.copy()
        trades = self.trades

        if start:
            start = pd.to_datetime(start)
            df = df[df.index >= start]
        if end:
            end = pd.to_datetime(end)
            df = df[df.index <= end]

        plt.figure(figsize=(18, 6))

        # ---- 黑色收盤線 ----
        plt.plot(df.index, df["Close"], color="black", label="Close")

        # ---- 藍色底色 ----
        plt.fill_between(df.index, 0, df["Close"], color="blue", alpha=0.05)

        # ---- 逐筆畫出交易 ----
        for _, t in trades.iterrows():
            buy = t["entry_date"]
            sell = t["exit_date"]
            buy_price = t["entry_price"]
            sell_price = t["exit_price"]
            pnl = sell_price - buy_price
            pnl_rate = t["return_pct"]

            color = "green" if pnl > 0 else "red"

            # 區間 mask
            mask = (df.index >= buy) & (df.index <= sell)

            # 區間背景
            plt.fill_between(df.index[mask],
                            0, df["Close"][mask],
                            color=color, alpha=0.28)

            # Buy / Sell 散點
            plt.scatter(buy, buy_price, color="blue", s=80)
            plt.scatter(sell, sell_price, color="orange", s=80)

            # ⭐ 取得觸發策略名稱
            buy_label = df.loc[buy, "buy_factor_trigger"] if "buy_factor_trigger" in df.columns else ""
            sell_label = df.loc[sell, "sell_factor_trigger"] if "sell_factor_trigger" in df.columns else ""

            # 暫時註解掉文字不然太亂了
            # TODO： 到時候用參數選擇要關還開
            # # ⭐ 在圖上標文字
            # plt.text(buy, buy_price,
            #         f"{buy_label}",
            #         fontsize=9, color="blue",
            #         ha="right", va="bottom")

            # plt.text(sell, sell_price,
            #         f"{sell_label}",
            #         fontsize=9, color="orange",
            #         ha="left", va="top")

            # 盈虧數字
            plt.text(sell, sell_price,
                    f"{pnl:+.2f} ({pnl_rate:+.2f}%)",
                    color=color, fontsize=9,
                    ha="left", va="bottom")

        plt.title("Backtest Trade Visualization")
        plt.grid(True)
        plt.legend()
        plt.show()