import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# =========================================================
# Performance Metrics Display Class
# =========================================================
class PerformanceMetrics:
    """
    Trading Performance Metrics Display Class
    Used to calculate and display various performance metrics from backtest results
    """
    
    def __init__(self, trades: pd.DataFrame, stats: dict, 
                 equity_curve: pd.Series = None, 
                 initial_capital: float = None):
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
        stats : dict
            Basic statistics dictionary
        equity_curve : pd.Series
            Equity curve time series (optional)
        initial_capital : float
            Initial capital (optional)
        """
        self.trades = trades.copy()
        self.stats = stats.copy()
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital or stats.get('initial_capital', None)
        
        # Calculate detailed metrics
        self._calculate_detailed_metrics()
    
    def _calculate_detailed_metrics(self):
        """Calculate detailed performance metrics"""
        if len(self.trades) == 0:
            self.detailed_stats = {}
            return
        
        # Basic trade statistics
        total_trades = len(self.trades)
        winning_trades = len(self.trades[self.trades['pnl'] > 0])
        losing_trades = len(self.trades[self.trades['pnl'] < 0])
        breakeven_trades = len(self.trades[self.trades['pnl'] == 0])
        
        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Average profit/loss
        avg_win = self.trades[self.trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(self.trades[self.trades['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        # Profit/Loss ratio
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf
        
        # Total profit/loss
        total_profit = self.trades[self.trades['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        total_loss = abs(self.trades[self.trades['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
        
        # Max single trade profit/loss
        max_profit = self.trades['pnl'].max()
        max_loss = self.trades['pnl'].min()
        
        # Average holding days
        avg_holding_days = self.trades['holding_days'].mean() if 'holding_days' in self.trades.columns else 0
        
        # Average return
        avg_return = self.trades['return_pct'].mean()
        
        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades()
        
        # Store detailed statistics
        self.detailed_stats = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'breakeven_trades': breakeven_trades,
            'win_rate': win_rate,
            'avg_profit': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': total_profit - total_loss,
            'max_single_profit': max_profit,
            'max_single_loss': max_loss,
            'avg_holding_days': avg_holding_days,
            'avg_return': avg_return,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
        }
    
    def _calculate_consecutive_trades(self):
        """Calculate maximum consecutive wins and losses"""
        if len(self.trades) == 0:
            return 0, 0
        
        # Sort by date
        trades_sorted = self.trades.sort_values('entry_date')
        
        # Determine if each trade is profit or loss
        is_profit = (trades_sorted['pnl'] > 0).astype(int)
        is_loss = (trades_sorted['pnl'] < 0).astype(int)
        
        # Calculate consecutive wins
        max_consecutive_wins = 0
        current_wins = 0
        for profit in is_profit:
            if profit == 1:
                current_wins += 1
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_wins = 0
        
        # Calculate consecutive losses
        max_consecutive_losses = 0
        current_losses = 0
        for loss in is_loss:
            if loss == 1:
                current_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_losses = 0
        
        return max_consecutive_wins, max_consecutive_losses
    
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
        
        # Map Chinese keys to English keys
        key_mapping = {
            'total_return': '總報酬率',
            'annual_return': '年化報酬率',
            'annual_volatility': '年化波動率',
            'Sharpe': 'Sharpe',
            'max_drawdown': '最大回撤',
            'final_equity': '最終權益'
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
            chinese_key = key_mapping.get(key, key)
            if chinese_key in self.stats:
                value = self.stats[chinese_key]
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
        max_dd_key = '最大回撤'
        if max_dd_key in self.stats:
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
