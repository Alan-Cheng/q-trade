import numpy as np
from typing import Dict

class Scorer:
    """
    根據 PerformanceMetrics 計算得分
    每個指標等權重 100 分
    """
    
    def __init__(self, metrics: 'PerformanceMetrics'):
        """
        Parameters:
        -----------
        metrics : PerformanceMetrics
            績效指標實例
        """
        self.metrics = metrics
        
    def calculate_score(self) -> Dict[str, float]:
        """
        計算總得分和各指標得分
        
        Returns:
        --------
        dict : 包含各指標得分和總得分
        """
        scores = {}
        total_score = 0
        
        # 1. 總報酬率得分（100分）
        total_return = self.metrics.stats.get('策略_總報酬率', 0)
        if isinstance(total_return, (int, float)) and not np.isnan(total_return):
            scores['總報酬率得分'] = self._normalize_score(total_return, target=1.0) * 100
        else:
            scores['總報酬率得分'] = 0
        total_score += scores['總報酬率得分']
        
        # 2. 年化報酬率得分（100分）
        annual_return = self.metrics.stats.get('策略_年化報酬率', 0)
        if isinstance(annual_return, (int, float)) and not np.isnan(annual_return):
            scores['年化報酬率得分'] = self._normalize_score(annual_return, target=0.2) * 100  # 目標20%
        else:
            scores['年化報酬率得分'] = 0
        total_score += scores['年化報酬率得分']
        
        # 3. Sharpe Ratio 得分（100分）
        sharpe = self.metrics.stats.get('策略_Sharpe', 0)
        if isinstance(sharpe, (int, float)) and not np.isnan(sharpe):
            scores['Sharpe得分'] = self._normalize_score(sharpe, target=2.0) * 100  # 目標Sharpe=2
        else:
            scores['Sharpe得分'] = 0
        total_score += scores['Sharpe得分']
        
        # 4. 最大回撤得分（100分，回撤越小越好）
        max_dd = self.metrics.stats.get('策略_最大回撤', 0)
        if isinstance(max_dd, (int, float)) and not np.isnan(max_dd):
            max_dd_abs = abs(max_dd)
            scores['最大回撤得分'] = (1 - min(max_dd_abs, 1.0)) * 100  # 回撤越小分數越高
        else:
            scores['最大回撤得分'] = 0
        total_score += scores['最大回撤得分']
        
        # 5. 勝率得分（100分）
        win_rate = self.metrics.detailed_stats.get('win_rate', 0)
        if isinstance(win_rate, (int, float)) and not np.isnan(win_rate):
            scores['勝率得分'] = win_rate * 100
        else:
            scores['勝率得分'] = 0
        total_score += scores['勝率得分']
        
        # 6. 盈虧比得分（100分）
        profit_loss_ratio = self.metrics.detailed_stats.get('profit_loss_ratio', 0)
        if isinstance(profit_loss_ratio, (int, float)) and not np.isnan(profit_loss_ratio) and profit_loss_ratio != np.inf:
            scores['盈虧比得分'] = self._normalize_score(profit_loss_ratio, target=2.0) * 100
        else:
            scores['盈虧比得分'] = 0
        total_score += scores['盈虧比得分']
        
        # 7. 總交易次數得分（100分，適中的交易次數較好）
        total_trades = self.metrics.detailed_stats.get('total_trades', 0)
        if isinstance(total_trades, (int, float)):
            # 假設理想交易次數在 20-100 之間
            if 20 <= total_trades <= 100:
                scores['交易次數得分'] = 100
            elif total_trades < 20:
                scores['交易次數得分'] = (total_trades / 20) * 100
            else:
                scores['交易次數得分'] = max(0, 100 - (total_trades - 100) * 0.5)
        else:
            scores['交易次數得分'] = 0
        total_score += scores['交易次數得分']
        
        # 8. 淨獲利得分（100分）
        net_profit = self.metrics.detailed_stats.get('net_profit', 0)
        if isinstance(net_profit, (int, float)) and not np.isnan(net_profit) and self.metrics.initial_capital:
            net_profit_ratio = net_profit / self.metrics.initial_capital
            scores['淨獲利得分'] = self._normalize_score(net_profit_ratio, target=0.5) * 100
        else:
            scores['淨獲利得分'] = 0
        total_score += scores['淨獲利得分']
        
        scores['總得分'] = total_score
        
        return scores
    
    def _normalize_score(self, value: float, target: float) -> float:
        """
        將指標值標準化到 0-1 區間
        
        Parameters:
        -----------
        value : float
            指標值
        target : float
            目標值（達到此值時得分為 1.0）
        
        Returns:
        --------
        float : 標準化後的分數（0-1）
        """
        if target == 0:
            return 0.0
        
        # 使用 sigmoid 函數進行標準化
        normalized = value / target
        # 限制在合理範圍內
        normalized = max(0, min(normalized, 2.0))  # 最多是目標值的2倍
        
        # 使用平滑的轉換函數
        score = normalized / (1 + abs(normalized - 1))
        
        return score

