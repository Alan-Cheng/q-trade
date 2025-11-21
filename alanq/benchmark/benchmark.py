import numpy as np
import pandas as pd

# =========================================================
# Benchmark：基準（單純 buy & hold）
# =========================================================

class Benchmark:
    """
    基準（單純 buy & hold）
    支援單股票和多股票（多股票使用平均分散）
    """
    
    @staticmethod
    def compute_log_ret(df: pd.DataFrame) -> pd.Series:
        """
        基準對數報酬率（持有一檔標的不交易）
        
        Parameters:
        -----------
        df : pd.DataFrame
            單股票價格資料，必須包含 'Close' 欄位
        
        Returns:
        --------
        pd.Series : 對數報酬率序列
        """
        close = df["Close"]
        return np.log(close / close.shift(1))
    
    @staticmethod
    def compute_equity_curve(df: pd.DataFrame, initial_capital: float) -> pd.Series:
        """
        計算單股票的基準權益曲線
        
        Parameters:
        -----------
        df : pd.DataFrame
            單股票價格資料，必須包含 'Close' 欄位
        initial_capital : float
            初始資金
        
        Returns:
        --------
        pd.Series : 基準權益曲線
        """
        log_ret = Benchmark.compute_log_ret(df)
        equity = initial_capital * np.exp(log_ret.cumsum())
        
        # 修正第一筆 NaN = 初始資金
        equity.iloc[0] = initial_capital
        
        return equity
    
    @staticmethod
    def compute_multi_stock_equity_curve(stock_data: dict, initial_capital: float) -> pd.Series:
        """
        計算多股票的基準權益曲線（平均分散到各持股）
        
        Parameters:
        -----------
        stock_data : dict
            股票資料字典，格式：{股票代號: DataFrame}
            DataFrame 必須包含 'Close' 欄位
        initial_capital : float
            初始資金
        
        Returns:
        --------
        pd.Series : 基準權益曲線
        
        Note:
        -----
        目前使用平均分散到各持股的方法（buy and hold）
        未來可能會調整為其他分配方式
        """
        if not stock_data:
            return pd.Series(dtype=float)
        
        # 取得所有股票的共同交易日
        all_dates = None
        for symbol, df in stock_data.items():
            if all_dates is None:
                all_dates = set(df.index)
            else:
                all_dates = all_dates.intersection(set(df.index))
        
        if not all_dates:
            return pd.Series(dtype=float)
        
        trading_dates = sorted(list(all_dates))
        num_stocks = len(stock_data)
        
        # 每檔股票分配相同資金
        capital_per_stock = initial_capital / num_stocks
        
        # 計算每檔股票的權益曲線
        stock_equities = {}
        for symbol, df in stock_data.items():
            stock_equities[symbol] = Benchmark.compute_equity_curve(df, capital_per_stock)
        
        # 合併所有股票的權益（只取共同交易日）
        total_equity = pd.Series(0.0, index=trading_dates)
        for symbol, equity in stock_equities.items():
            # 只取共同交易日
            equity_aligned = equity.reindex(trading_dates, method='ffill')
            total_equity += equity_aligned.fillna(0)
        
        return total_equity
    
    @staticmethod
    def compute_performance_stats(equity_curve: pd.Series, initial_capital: float, 
                                  log_ret: pd.Series = None) -> dict:
        """
        計算基準績效指標
        
        Parameters:
        -----------
        equity_curve : pd.Series
            基準權益曲線
        initial_capital : float
            初始資金
        log_ret : pd.Series, optional
            對數報酬率序列（如果提供則使用，否則從 equity_curve 計算）
        
        Returns:
        --------
        dict : 包含各項績效指標的字典
        """
        if equity_curve.empty:
            return {
                "總報酬率": np.nan,
                "年化報酬率": np.nan,
                "年化波動率": np.nan,
                "Sharpe": np.nan,
                "最大回撤": np.nan,
            }
        
        # 總報酬率（倍數形式，與策略保持一致）
        total_ret = equity_curve.iloc[-1] / initial_capital - 1
        
        # 年化報酬率（與策略計算方式保持一致）
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.0 if days > 0 else 1.0
        annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else total_ret
        
        # 計算每日報酬率
        if log_ret is not None:
            daily_ret = log_ret.dropna()
        else:
            # 從權益曲線計算每日報酬率
            daily_ret = equity_curve.pct_change().dropna()
        
        # 波動率和 Sharpe
        if len(daily_ret) > 1 and daily_ret.std() > 0:
            vol = daily_ret.std() * np.sqrt(252)
            sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else np.nan
        else:
            vol = np.nan
            sharpe = np.nan
        
        # 最大回撤
        roll_max = equity_curve.cummax()
        drawdown = equity_curve / roll_max - 1.0
        max_dd = drawdown.min()
        
        return {
            "總報酬率": total_ret,
            "年化報酬率": annual_ret,
            "年化波動率": vol,
            "Sharpe": sharpe,
            "最大回撤": max_dd,
        }
    
    @staticmethod
    def compute_single_stock_benchmark(df: pd.DataFrame, initial_capital: float) -> dict:
        """
        計算單股票的完整基準績效（包含權益曲線和績效指標）
        
        Parameters:
        -----------
        df : pd.DataFrame
            單股票價格資料，必須包含 'Close' 欄位
        initial_capital : float
            初始資金
        
        Returns:
        --------
        dict : 包含 'equity_curve' 和 'stats' 的字典
        """
        log_ret = Benchmark.compute_log_ret(df)
        equity_curve = Benchmark.compute_equity_curve(df, initial_capital)
        stats = Benchmark.compute_performance_stats(equity_curve, initial_capital, log_ret)
        
        return {
            'equity_curve': equity_curve,
            'log_ret': log_ret,
            'stats': stats
        }
    
    @staticmethod
    def compute_multi_stock_benchmark(stock_data: dict, initial_capital: float) -> dict:
        """
        計算多股票的完整基準績效（包含權益曲線和績效指標）
        
        Parameters:
        -----------
        stock_data : dict
            股票資料字典，格式：{股票代號: DataFrame}
        initial_capital : float
            初始資金
        
        Returns:
        --------
        dict : 包含 'equity_curve' 和 'stats' 的字典
        """
        equity_curve = Benchmark.compute_multi_stock_equity_curve(stock_data, initial_capital)
        
        # 計算平均對數報酬率（用於統計）
        if equity_curve.empty:
            log_ret = pd.Series(dtype=float)
        else:
            # 從權益曲線計算每日報酬率
            log_ret = np.log(equity_curve / equity_curve.shift(1))
        
        stats = Benchmark.compute_performance_stats(equity_curve, initial_capital, log_ret)
        
        return {
            'equity_curve': equity_curve,
            'log_ret': log_ret,
            'stats': stats
        }
