import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any

# 導入指標類別
from .basic import (
    BaseBasicMetric,
    TotalReturnMetric,
    AnnualReturnMetric,
    VolatilityMetric,
    SharpeMetric,
    MaxDrawdownMetric,
    FinalEquityMetric,
)
from .detail import (
    BaseDetailedMetric,
    WinRateMetric,
    TotalTradesMetric,
    WinningTradesMetric,
    LosingTradesMetric,
    ProfitLossRatioMetric,
    AvgProfitMetric,
    AvgLossMetric,
    NetProfitMetric,
    MaxSingleProfitMetric,
    MaxSingleLossMetric,
    AvgHoldingDaysMetric,
    AvgReturnMetric,
    MaxConsecutiveWinsMetric,
    MaxConsecutiveLossesMetric,
    # 保留舊的作為向後相容
    BasicTradeStatsMetric,
    ProfitLossStatsMetric,
    ExtremeTradesMetric,
    HoldingStatsMetric,
    ConsecutiveTradesMetric,
)

# =========================================================
# Performance Metrics Display Class
# =========================================================
class PerformanceMetrics:
    """
    Trading Performance Metrics Display Class
    Used to calculate and display various performance metrics from backtest results
    """
    
    def __init__(self, trades: pd.DataFrame, stats: dict = None, 
                 equity_curve: pd.Series = None, 
                 initial_capital: float = None,
                 basic_metrics: Optional[List[Dict[str, Any]]] = None,
                 detailed_metrics: Optional[List[Dict[str, Any]]] = None):
        """
        Parameters:
        -----------
        trades : pd.DataFrame
            Trade records DataFrame, should contain:
            - symbol: stock symbol
            - entry_date: entry date
            - exit_date: exit date
            - entry_price: entry price
            - exit_price: exit price
            - shares: number of shares
            - return_pct: return percentage
            - pnl: profit and loss amount
        stats : dict, optional
            Basic statistics dictionary. If not provided, will be calculated from equity_curve
        equity_curve : pd.Series, optional
            Equity curve time series. Required if stats is not provided
        initial_capital : float, optional
            Initial capital. Required if stats is not provided
        basic_metrics : list, optional
            基本指標列表，格式：[{"class": TotalReturnMetric}, {"class": SharpeMetric}, ...]
            如果為 None，則使用預設指標組合
        detailed_metrics : list, optional
            詳細指標列表，格式：[{"class": BasicTradeStatsMetric}, {"class": WinRateMetric}, ...]
            如果為 None，則使用預設指標組合
        """
        self.trades = trades.copy()
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital or (stats.get('initial_capital', None) if stats else None)
        self.basic_metrics_config = basic_metrics
        self.detailed_metrics_config = detailed_metrics
        
        # If stats is not provided, calculate it from equity_curve
        if stats is None:
            if equity_curve is None or initial_capital is None:
                raise ValueError("If stats is not provided, both equity_curve and initial_capital must be provided")
            self.stats = self._calculate_basic_stats()
        else:
            self.stats = stats.copy()
        
        # Calculate detailed metrics
        self._calculate_detailed_metrics()
    
    def _calculate_detailed_metrics(self):
        """Calculate detailed performance metrics using metric classes"""
        if len(self.trades) == 0:
            self.detailed_stats = {}
            return
        
        # 決定使用哪些指標
        if self.detailed_metrics_config is None:
            metrics_config = self._get_default_detailed_metrics()
        else:
            metrics_config = self.detailed_metrics_config
        
        self.detailed_stats = {}
        
        # 按順序計算每個指標
        for metric_config in metrics_config:
            MetricClass = metric_config["class"]
            params = {k: v for k, v in metric_config.items() if k != "class"}
            
            # 實例化指標類別
            metric = MetricClass(**params)
            
            try:
                # 計算指標
                result = metric.calculate(
                    self.trades,
                    self.detailed_stats  # 傳入已計算的指標，允許依賴
                )
                self.detailed_stats.update(result)
            except Exception as e:
                # 如果計算失敗，記錄警告但繼續
                import warnings
                warnings.warn(f"計算詳細指標 '{metric.__class__.__name__}' 時發生錯誤: {e}")
    
    
    def _get_default_basic_metrics(self):
        """取得預設的基本指標組合"""
        return [
            {"class": TotalReturnMetric},
            {"class": AnnualReturnMetric},
            {"class": VolatilityMetric},
            {"class": SharpeMetric},
            {"class": MaxDrawdownMetric},
            {"class": FinalEquityMetric},
        ]
    
    def _get_default_detailed_metrics(self):
        """取得預設的詳細指標組合"""
        return [
            {"class": TotalTradesMetric},          # 先計算總交易次數（其他指標可能依賴）
            {"class": WinningTradesMetric},        # 計算獲利交易次數（WinRateMetric 依賴）
            {"class": WinRateMetric},              # 計算勝率（依賴 winning_trades 和 total_trades）
            {"class": ProfitLossRatioMetric},
            {"class": AvgProfitMetric},
            {"class": AvgLossMetric},
            {"class": NetProfitMetric},
            {"class": MaxSingleProfitMetric},
            {"class": MaxSingleLossMetric},
            {"class": AvgHoldingDaysMetric},
            {"class": AvgReturnMetric},
            {"class": MaxConsecutiveWinsMetric},
            {"class": MaxConsecutiveLossesMetric},
        ]
    
    def _calculate_basic_stats(self):
        """
        Calculate basic performance statistics from equity curve using metric classes
        
        Returns:
        --------
        dict : Dictionary containing basic performance metrics
        """
        if self.equity_curve is None or self.initial_capital is None:
            return {}
        
        if self.equity_curve.empty:
            return {
                "策略_總報酬率": np.nan,
                "策略_年化報酬率": np.nan,
                "策略_年化波動率": np.nan,
                "策略_Sharpe": np.nan,
                "策略_最大回撤": np.nan,
                "策略_最終權益": np.nan,
            }
        
        # 決定使用哪些指標
        if self.basic_metrics_config is None:
            metrics_config = self._get_default_basic_metrics()
        else:
            metrics_config = self.basic_metrics_config
        
        stats = {}
        
        # 按順序計算每個指標
        for metric_config in metrics_config:
            MetricClass = metric_config["class"]
            params = {k: v for k, v in metric_config.items() if k != "class"}
            
            # 實例化指標類別
            metric = MetricClass(**params)
            
            try:
                # 計算指標
                result = metric.calculate(
                    self.equity_curve,
                    self.initial_capital,
                    stats  # 傳入已計算的指標，允許依賴
                )
                stats.update(result)
            except Exception as e:
                # 如果計算失敗，記錄警告但繼續
                import warnings
                warnings.warn(f"計算指標 '{metric.metric_name}' 時發生錯誤: {e}")
        
        return stats
    
    def show_summary(self, format_numbers=True):
        """
        Display performance summary
        
        Parameters:
        -----------
        format_numbers : bool
            Whether to format number display
        """
        print("=" * 80)
        print("Performance Summary")
        print("=" * 80)
        
        # Basic performance metrics
        print("\n[Basic Performance Metrics]")
        basic_metrics = [
            'total_return', 'annual_return', 'annual_volatility', 
            'Sharpe', 'max_drawdown', 'final_equity'
        ]
        
        # Map English keys to Chinese keys (support both with and without prefix)
        key_mapping = {
            'total_return': ['總報酬率', '策略_總報酬率'],
            'annual_return': ['年化報酬率', '策略_年化報酬率'],
            'annual_volatility': ['年化波動率', '策略_年化波動率'],
            'Sharpe': ['Sharpe', '策略_Sharpe'],
            'max_drawdown': ['最大回撤', '策略_最大回撤'],
            'final_equity': ['最終權益', '策略_最終權益']
        }
        
        display_names = {
            'total_return': 'Total Return',
            'annual_return': 'Annual Return',
            'annual_volatility': 'Annual Volatility',
            'Sharpe': 'Sharpe Ratio',
            'max_drawdown': 'Max Drawdown',
            'final_equity': 'Final Equity'
        }
        
        for key in basic_metrics:
            chinese_keys = key_mapping.get(key, [key])
            # Try to find the key in stats (support both with and without prefix)
            value = None
            for chinese_key in chinese_keys:
                if chinese_key in self.stats:
                    value = self.stats[chinese_key]
                    break
            
            if value is not None:
                display_name = display_names.get(key, key)
                if format_numbers and isinstance(value, float):
                    if key in ['total_return', 'annual_return', 'annual_volatility', 'Sharpe', 'max_drawdown']:
                        print(f"  {display_name:20s}: {value:>10.4f} ({value*100:>6.2f}%)")
                    else:
                        print(f"  {display_name:20s}: {value:>10,.2f}")
                else:
                    print(f"  {display_name:20s}: {value}")
        
        # Detailed trade statistics
        print("\n[Trade Statistics]")
        display_mapping = {
            'total_trades': 'Total Trades',
            'winning_trades': 'Winning Trades',
            'losing_trades': 'Losing Trades',
            'breakeven_trades': 'Breakeven Trades',
            'win_rate': 'Win Rate',
            'avg_profit': 'Avg Profit',
            'avg_loss': 'Avg Loss',
            'profit_loss_ratio': 'Profit/Loss Ratio',
            'total_profit': 'Total Profit',
            'total_loss': 'Total Loss',
            'net_profit': 'Net Profit',
            'max_single_profit': 'Max Single Profit',
            'max_single_loss': 'Max Single Loss',
            'avg_holding_days': 'Avg Holding Days',
            'avg_return': 'Avg Return',
            'max_consecutive_wins': 'Max Consecutive Wins',
            'max_consecutive_losses': 'Max Consecutive Losses',
        }
        
        for key, value in self.detailed_stats.items():
            display_name = display_mapping.get(key, key)
            if isinstance(value, float):
                if key in ['win_rate', 'avg_return']:
                    print(f"  {display_name:20s}: {value:>10.4f} ({value*100:>6.2f}%)")
                elif key in ['profit_loss_ratio']:
                    print(f"  {display_name:20s}: {value:>10.2f}")
                elif key in ['avg_holding_days']:
                    print(f"  {display_name:20s}: {value:>10.1f} days")
                else:
                    print(f"  {display_name:20s}: {value:>10,.2f}")
            else:
                print(f"  {display_name:20s}: {value:>10}")
        
        print("=" * 80)
    
    def show_by_symbol(self):
        """Display performance grouped by stock symbol"""
        if 'symbol' not in self.trades.columns:
            print("No symbol information in trade records")
            return
        
        print("=" * 80)
        print("Performance by Symbol")
        print("=" * 80)
        
        symbol_stats = []
        
        for symbol in self.trades['symbol'].unique():
            symbol_trades = self.trades[self.trades['symbol'] == symbol]
            
            total_trades = len(symbol_trades)
            winning = len(symbol_trades[symbol_trades['pnl'] > 0])
            losing = len(symbol_trades[symbol_trades['pnl'] < 0])
            win_rate = winning / total_trades if total_trades > 0 else 0
            
            total_pnl = symbol_trades['pnl'].sum()
            avg_return = symbol_trades['return_pct'].mean()
            avg_holding = symbol_trades['holding_days'].mean() if 'holding_days' in symbol_trades.columns else 0
            
            symbol_stats.append({
                'symbol': symbol,
                'total_trades': total_trades,
                'winning': winning,
                'losing': losing,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_return': avg_return,
                'avg_holding_days': avg_holding
            })
        
        stats_df = pd.DataFrame(symbol_stats)
        
        # Format display
        print("\n")
        for idx, row in stats_df.iterrows():
            print(f"[{row['symbol']}]")
            print(f"  Total Trades: {int(row['total_trades'])}")
            print(f"  Winning: {int(row['winning'])}, Losing: {int(row['losing'])}")
            print(f"  Win Rate: {row['win_rate']:.2%}")
            print(f"  Total P&L: {row['total_pnl']:,.2f}")
            print(f"  Avg Return: {row['avg_return']:.2%}")
            print(f"  Avg Holding Days: {row['avg_holding_days']:.1f} days")
            print()
        
        print("=" * 80)
        
        return stats_df
    
    def plot_equity_curve(self, figsize=(14, 6)):
        """Plot equity curve"""
        if self.equity_curve is None:
            print("No equity curve data")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(self.equity_curve.index, self.equity_curve.values, 
                linewidth=2, label='Strategy Equity')
        
        if self.initial_capital:
            ax.axhline(y=self.initial_capital, color='r', linestyle='--', 
                      alpha=0.7, label=f'Initial Capital ({self.initial_capital:,.0f})')
        
        # Mark maximum drawdown
        max_dd_key = None
        for key in ['策略_最大回撤', '最大回撤']:
            if key in self.stats:
                max_dd_key = key
                break
        
        if max_dd_key is not None:
            roll_max = self.equity_curve.cummax()
            drawdown = self.equity_curve / roll_max - 1.0
            max_dd_idx = drawdown.idxmin()
            max_dd_value = self.equity_curve.loc[max_dd_idx]
            
            ax.plot(max_dd_idx, max_dd_value, 'ro', markersize=10, 
                   label=f'Max Drawdown Point ({self.stats[max_dd_key]:.2%})')
        
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown(self, figsize=(14, 6)):
        """Plot drawdown curve"""
        if self.equity_curve is None:
            print("No equity curve data")
            return
        
        roll_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve / roll_max - 1.0) * 100
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.fill_between(drawdown.index, drawdown.values, 0, 
                        alpha=0.3, color='red', label='Drawdown')
        ax.plot(drawdown.index, drawdown.values, 
               linewidth=1.5, color='darkred')
        
        ax.set_title('Drawdown Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_trade_distribution(self, figsize=(14, 5)):
        """Plot trade P&L distribution"""
        if len(self.trades) == 0:
            print("No trade records")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Left plot: P&L distribution histogram
        axes[0].hist(self.trades['pnl'], bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('P&L Amount', fontsize=10)
        axes[0].set_ylabel('Number of Trades', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Right plot: Return distribution histogram
        axes[1].hist(self.trades['return_pct'] * 100, bins=30, 
                    edgecolor='black', alpha=0.7, color='green')
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_title('Trade Return Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Return (%)', fontsize=10)
        axes[1].set_ylabel('Number of Trades', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_returns(self, figsize=(14, 6)):
        """Plot monthly returns"""
        if self.equity_curve is None:
            print("No equity curve data")
            return
        
        # Calculate monthly returns
        monthly_equity = self.equity_curve.resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna() * 100
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['green' if x > 0 else 'red' for x in monthly_returns.values]
        ax.bar(monthly_returns.index, monthly_returns.values, 
              color=colors, alpha=0.7, edgecolor='black')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title('Monthly Returns', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def get_metrics_dataframe(self):
        """Convert all metrics to DataFrame"""
        all_metrics = {}
        
        # Mapping for detailed stats keys to Chinese
        detailed_stats_mapping = {
            'total_trades': '總交易次數',
            'winning_trades': '獲利交易次數',
            'losing_trades': '虧損交易次數',
            'breakeven_trades': '持平交易次數',
            'win_rate': '勝率',
            'avg_profit': '平均獲利',
            'avg_loss': '平均虧損',
            'profit_loss_ratio': '盈虧比',
            'total_profit': '總獲利',
            'total_loss': '總虧損',
            'net_profit': '淨獲利',
            'max_single_profit': '最大單筆獲利',
            'max_single_loss': '最大單筆虧損',
            'avg_holding_days': '平均持倉天數',
            'avg_return': '平均報酬率',
            'max_consecutive_wins': '最大連續獲利次數',
            'max_consecutive_losses': '最大連續虧損次數',
        }
        
        # Basic metrics (already in Chinese from stats)
        for key, value in self.stats.items():
            all_metrics[key] = value
        
        # Detailed metrics (convert English keys to Chinese)
        for key, value in self.detailed_stats.items():
            chinese_key = detailed_stats_mapping.get(key, key)
            all_metrics[chinese_key] = value
        
        return pd.DataFrame([all_metrics]).T.rename(columns={0: '數值'})
    
    def show_all(self):
        """Display all performance metrics and charts"""
        self.show_summary()
        print("\n")
        self.show_by_symbol()
        print("\n")
        self.plot_equity_curve()
        self.plot_drawdown()
        self.plot_trade_distribution()
        self.plot_monthly_returns()
