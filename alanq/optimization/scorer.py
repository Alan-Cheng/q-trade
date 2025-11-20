import numpy as np
from typing import Dict, Optional

# 定義可用的績效指標列表（來自 PerformanceMetrics）
AVAILABLE_METRICS = {
    # 來自 stats 的指標
    '策略_總報酬率': {
        'source': 'stats',
        'key': '策略_總報酬率',
        'target': 1.0,
        'higher_is_better': True,
        'description': '總報酬率'
    },
    '策略_年化報酬率': {
        'source': 'stats',
        'key': '策略_年化報酬率',
        'target': 0.2,
        'higher_is_better': True,
        'description': '年化報酬率'
    },
    '策略_Sharpe': {
        'source': 'stats',
        'key': '策略_Sharpe',
        'target': 2.0,
        'higher_is_better': True,
        'description': 'Sharpe Ratio'
    },
    '策略_最大回撤': {
        'source': 'stats',
        'key': '策略_最大回撤',
        'target': 0.0,
        'higher_is_better': False,  # 回撤越小越好
        'description': '最大回撤'
    },
    '策略_年化波動率': {
        'source': 'stats',
        'key': '策略_年化波動率',
        'target': 0.15,
        'higher_is_better': False,  # 波動率越小越好
        'description': '年化波動率'
    },
    # 來自 detailed_stats 的指標
    '勝率': {
        'source': 'detailed_stats',
        'key': 'win_rate',
        'target': 1.0,
        'higher_is_better': True,
        'description': '勝率'
    },
    '盈虧比': {
        'source': 'detailed_stats',
        'key': 'profit_loss_ratio',
        'target': 2.0,
        'higher_is_better': True,
        'description': '盈虧比'
    },
}


class Scorer:
    """
    根據 PerformanceMetrics 計算得分
    支援自定義權重
    """
    
    def __init__(self, metrics: 'PerformanceMetrics', weights: Optional[Dict[str, float]] = None):
        """
        Parameters:
        -----------
        metrics : PerformanceMetrics
            績效指標實例
        weights : dict, optional
            各指標的權重字典，例如：
            {
                '策略_總報酬率': 0.2,
                '策略_Sharpe': 0.3,
                '勝率': 0.15,
                ...
            }
            如果為 None，則使用等權重（每個指標權重相同，總和為 1）
            權重會自動歸一化，所以總和不需要等於 1，系統會自動歸一化
        """
        self.metrics = metrics
        
        # 驗證並設置權重
        if weights is None:
            # 預設等權重：每個指標權重相同，總和為 1
            num_metrics = len(AVAILABLE_METRICS)
            self.weights = {metric: 1.0 / num_metrics for metric in AVAILABLE_METRICS.keys()}
        else:
            # 驗證權重中的指標是否都在可用列表中
            invalid_metrics = set(weights.keys()) - set(AVAILABLE_METRICS.keys())
            if invalid_metrics:
                raise ValueError(
                    f"無效的績效指標: {invalid_metrics}. "
                    f"可用指標: {list(AVAILABLE_METRICS.keys())}"
                )
            # 設置權重，未指定的指標權重為 0
            self.weights = {metric: weights.get(metric, 0.0) for metric in AVAILABLE_METRICS.keys()}
        
        # 歸一化權重（確保總和為 1）
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        else:
            # 如果所有權重都是 0，使用等權重
            num_metrics = len(AVAILABLE_METRICS)
            self.weights = {metric: 1.0 / num_metrics for metric in AVAILABLE_METRICS.keys()}
        
    def calculate_score(self) -> Dict[str, float]:
        """
        計算總得分和各指標得分
        
        Returns:
        --------
        dict : 包含各指標得分和總得分
        """
        scores = {}
        total_score = 0
        
        # 遍歷所有可用指標
        for metric_name, metric_config in AVAILABLE_METRICS.items():
            # 跳過權重為 0 的指標
            if self.weights.get(metric_name, 0) == 0:
                scores[f'{metric_name}_得分'] = 0
                continue
            
            # 獲取指標值
            if metric_config['source'] == 'stats':
                value = self.metrics.stats.get(metric_config['key'], None)
            else:  # detailed_stats
                value = self.metrics.detailed_stats.get(metric_config['key'], None)
            
            # 計算該指標的得分（0-100）
            if value is None or (isinstance(value, float) and np.isnan(value)):
                metric_score = 0
            else:
                metric_score = self._calculate_metric_score(
                    value, 
                    metric_config, 
                    metric_name
                )
            
            # 應用權重（權重總和為 1）
            weighted_score = metric_score * self.weights[metric_name]
            scores[f'{metric_name}_得分'] = weighted_score
            total_score += weighted_score
        
        scores['總得分'] = total_score
        
        return scores
    
    def _calculate_metric_score(self, value: float, metric_config: Dict, metric_name: str) -> float:
        """
        計算單個指標的得分（0-100）
        
        Parameters:
        -----------
        value : float
            指標值
        metric_config : dict
            指標配置
        metric_name : str
            指標名稱
        
        Returns:
        --------
        float : 得分（0-100）
        """
        higher_is_better = metric_config['higher_is_better']
        target = metric_config['target']
        
        # 特殊處理：盈虧比（可能為 inf）
        if metric_name == '盈虧比':
            if value == np.inf or np.isnan(value):
                return 0
            return self._normalize_score(value, metric_config['target']) * 100
        
        # 處理最大回撤（越小越好）
        if metric_name == '策略_最大回撤':
            max_dd_abs = abs(value)
            return (1 - min(max_dd_abs, 1.0)) * 100
        
        # 處理年化波動率（越小越好）
        if metric_name == '策略_年化波動率':
            target_val = metric_config['target']
            return (1 - min(value / target_val, 1.0)) * 100 if target_val > 0 else 0
        
        # 處理勝率（已經是 0-1 的比例）
        if metric_name == '勝率':
            return value * 100
        
        # 一般處理：使用 normalize_score
        if target is None or target == 0:
            return 0
        
        normalized = self._normalize_score(value, target)
        return normalized * 100
    
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
        float : 標準化後的分數（0-1），但允許超過 1.0 以區分超過目標值的情況
        """
        if target == 0:
            return 0.0
        
        # 計算比例
        ratio = value / target
        
        # 使用分段映射：
        # - ratio < 0: 得分為 0
        # - 0 <= ratio <= 1: 線性映射到 0-1
        # - ratio > 1: 使用對數縮放，允許超過 1.0 但有限制
        if ratio < 0:
            return 0.0
        elif ratio <= 1.0:
            # 線性映射：0 -> 0, 1 -> 1
            return ratio
        else:
            # 超過目標值時，使用對數縮放來區分不同水平
            # ratio = 1.0 -> score = 1.0 (100分)
            # ratio = 1.5 -> score ≈ 1.2 (120分)
            # ratio = 2.0 -> score ≈ 1.35 (135分)
            # ratio = 3.0 -> score ≈ 1.45 (145分)
            import math
            excess = ratio - 1.0
            # 使用對數縮放：log(1 + excess) / log(2) 映射到 [0, 0.5]
            if excess <= 0:
                bonus = 0
            else:
                # 使用對數縮放，讓超過目標值的部分逐漸飽和
                log_bonus = math.log(1 + excess) / math.log(2)  # 對數縮放
                bonus = min(log_bonus * 0.3, 0.5)  # 限制最高 bonus 為 0.5
            
            return 1.0 + bonus

