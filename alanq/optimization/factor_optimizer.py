import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from ..backtest.timing_backtester_single import Backtester
from ..performance.performance_metrics import PerformanceMetrics
from .parameter_space import ParameterSpace
from .scorer import Scorer


def _evaluate_combination_worker_raw(combo_num: int, combo: Dict, df: pd.DataFrame, 
                                     initial_capital: float, slippage_factors: List) -> Optional[Dict]:
    """
    並行處理的 worker 函數（只收集原始指標值，不計算得分）
    
    Parameters:
    -----------
    combo_num : int
        組合編號
    combo : dict
        因子組合配置
    df : pd.DataFrame
        股票價格資料
    initial_capital : float
        初始資金
    slippage_factors : list
        滑價因子列表
    
    Returns:
    --------
    dict : 結果記錄字典，如果失敗則返回 None
    """
    try:
        # 執行回測
        backtester = Backtester(
            df=df.copy(),
            buy_factors=combo['buy_factors'],
            sell_factors=combo['sell_factors'],
            initial_capital=initial_capital,
            slippage_factors=slippage_factors
        )
        
        result, trades, stats = backtester.run()
        
        # 計算績效指標
        equity_curve = result['strategy_equity'] if 'strategy_equity' in result.columns else None
        metrics = PerformanceMetrics(
            trades=trades,
            stats=stats,
            equity_curve=equity_curve,
            initial_capital=initial_capital
        )
        
        # 只收集原始指標值，不計算得分
        # 注意: 這裡保留 '組合編號' 方便 worker 識別，但在 _calculate_statistical_scores 中會被移除。
        result_record = {
            '組合編號': combo_num,
            '買入因子': str(combo['buy_factors']),
            '賣出因子': str(combo['sell_factors']),
            **stats,
            **metrics.detailed_stats
        }
        
        return result_record
        
    except Exception as e:
        # 在並行處理中，錯誤會被捕獲並返回 None
        return None

class FactorOptimizer:
    """
    因子組合優化引擎
    使用 Backtester 和 PerformanceMetrics 來找出最佳因子組合
    """
    
    def __init__(self,
                 df: pd.DataFrame,
                 parameter_space: ParameterSpace,
                 initial_capital: float = 1_000_000,
                 slippage_factors: Optional[List] = None,
                 show_progress: bool = True,
                 n_jobs: int = 1,
                 metric_weights: Optional[Dict[str, float]] = None):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            股票價格資料
        parameter_space : ParameterSpace
            參數空間定義
        initial_capital : float
            初始資金
        slippage_factors : list, optional
            滑價因子（可選）
        show_progress : bool
            是否顯示進度條（需要 tqdm）
        n_jobs : int
            並行處理的進程數，預設為 1（順序執行）
            -1 表示使用所有 CPU 核心
        metric_weights : dict, optional
            各績效指標的權重字典，例如：
            {
                '策略_總報酬率': 0.2,
                '策略_Sharpe': 0.3,
                '勝率': 0.15,
                ...
            }
            如果為 None，則使用等權重
            可用的指標請參考 scorer.AVAILABLE_METRICS
        """
        self.df = df
        self.parameter_space = parameter_space
        self.initial_capital = initial_capital
        self.slippage_factors = slippage_factors or []
        self.show_progress = show_progress
        self.metric_weights = metric_weights
        
        # 計算實際使用的進程數
        if n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        elif n_jobs > 0:
            self.n_jobs = min(n_jobs, mp.cpu_count())
        else:
            self.n_jobs = 1
        
        self.results = []
        
    def optimize(self) -> Tuple[Dict, pd.DataFrame]:
        """
        執行優化
        
        Returns:
        --------
        tuple : (最佳配置字典, 結果 DataFrame)
        """
        # 生成所有組合
        combinations = self.parameter_space.generate_combinations()
        total_combinations = len(combinations)
        
        print(f"總共需要測試 {total_combinations} 種組合...")
        if self.n_jobs > 1:
            print(f"使用 {self.n_jobs} 個進程進行並行處理...")
        
        # 第一階段：執行所有回測，收集原始指標值（不計算得分）
        raw_results = self._collect_raw_results(combinations)
        
        # 轉換為 DataFrame
        if len(raw_results) == 0:
            print("警告: 沒有成功執行任何組合")
            return None, pd.DataFrame()
        
        raw_results_df = pd.DataFrame(raw_results)
        
        # 第二階段：計算統計分佈並使用統計方法計算得分
        print("\n計算統計分佈並標準化得分...")
        self.results = self._calculate_statistical_scores(raw_results_df)
        results_df = pd.DataFrame(self.results)
        
        # 找出最佳組合
        if len(results_df) > 0 and '總得分' in results_df.columns:
            best_idx = results_df['總得分'].idxmax()
            best_row = results_df.loc[best_idx]
            
            # 安全地解析因子配置
            try:
                import ast
                # 注意：這裡使用 '買入因子' 和 '賣出因子' 欄位，它們是字串。
                buy_factors = ast.literal_eval(best_row['買入因子'])
                sell_factors = ast.literal_eval(best_row['賣出因子'])
            except:
                # 如果解析失敗，使用字符串
                buy_factors = best_row['買入因子']
                sell_factors = best_row['賣出因子']
            
            best_config = {
                'buy_factors': buy_factors,
                'sell_factors': sell_factors,
                '總得分': best_row['總得分']
            }
        else:
            best_config = None
        
        return best_config, results_df
    
    def _collect_raw_results(self, combinations: List[Dict]) -> List[Dict]:
        """收集所有組合的原始指標值（支援並行和順序執行）"""
        if self.n_jobs > 1:
            return self._collect_raw_results_parallel(combinations)
        else:
            return self._collect_raw_results_sequential(combinations)
    
    def _collect_raw_results_sequential(self, combinations: List[Dict]) -> List[Dict]:
        """順序收集原始結果"""
        results = []
        
        # 遍歷每個組合
        if self.show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(combinations, desc="收集指標數據")
            except ImportError:
                print("提示: 安裝 tqdm 可以顯示進度條 (pip install tqdm)")
                iterator = combinations
        else:
            iterator = combinations
        
        for i, combo in enumerate(iterator):
            try:
                result_record = self._evaluate_single_combination_raw(i + 1, combo)
                if result_record:
                    results.append(result_record)
            except Exception as e:
                print(f"\n組合 {i+1} 執行失敗: {e}")
                continue
        
        return results
    
    def _collect_raw_results_parallel(self, combinations: List[Dict]) -> List[Dict]:
        """並行收集原始結果"""
        results = []
        
        # 準備進度條
        if self.show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(combinations), desc="收集指標數據")
            except ImportError:
                pbar = None
        else:
            pbar = None
        
        # 使用進程池並行執行
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # 提交所有任務（不傳遞 weights，因為我們不計算得分）
            future_to_combo = {
                executor.submit(
                    _evaluate_combination_worker_raw,
                    i + 1,
                    combo,
                    self.df.copy(),
                    self.initial_capital,
                    self.slippage_factors
                ): (i + 1, combo)
                for i, combo in enumerate(combinations)
            }
            
            # 收集結果
            for future in as_completed(future_to_combo):
                combo_num, combo = future_to_combo[future]
                try:
                    result_record = future.result()
                    if result_record:
                        results.append(result_record)
                except Exception as e:
                    print(f"\n組合 {combo_num} 執行失敗: {e}")
                
                if pbar:
                    pbar.update(1)
        
        if pbar:
            pbar.close()
        
        return results
    
    def _optimize_sequential(self, combinations: List[Dict]) -> List[Dict]:
        """順序執行優化（已廢棄，改用 _collect_raw_results + _calculate_statistical_scores）"""
        # 這個方法保留用於向後兼容，但實際上不會被調用
        return self._collect_raw_results(combinations)
    
    def _optimize_parallel(self, combinations: List[Dict]) -> List[Dict]:
        """並行執行優化（已廢棄，改用 _collect_raw_results_parallel + _calculate_statistical_scores）"""
        # 這個方法保留用於向後兼容，但實際上不會被調用
        return self._collect_raw_results_parallel(combinations)
    
    def _evaluate_single_combination_raw(self, combo_num: int, combo: Dict) -> Optional[Dict]:
        """評估單個組合，只收集原始指標值（不計算得分）"""
        # 執行回測
        backtester = Backtester(
            df=self.df.copy(),
            buy_factors=combo['buy_factors'],
            sell_factors=combo['sell_factors'],
            initial_capital=self.initial_capital,
            slippage_factors=self.slippage_factors
        )
        
        result, trades, stats = backtester.run()
        
        # 計算績效指標
        equity_curve = result['strategy_equity'] if 'strategy_equity' in result.columns else None
        metrics = PerformanceMetrics(
            trades=trades,
            stats=stats,
            equity_curve=equity_curve,
            initial_capital=self.initial_capital
        )
        
        # 只收集原始指標值，不計算得分
        result_record = {
            '組合編號': combo_num,
            '買入因子': str(combo['buy_factors']),
            '賣出因子': str(combo['sell_factors']),
            **stats,
            **metrics.detailed_stats
        }
        
        return result_record
    
    def _evaluate_single_combination(self, combo_num: int, combo: Dict, weights: Optional[Dict[str, float]] = None) -> Optional[Dict]:
        """評估單個組合（內部方法，已廢棄，改用 _evaluate_single_combination_raw）"""
        # 這個方法保留用於向後兼容
        return self._evaluate_single_combination_raw(combo_num, combo)
    
    def _calculate_statistical_scores(self, raw_results_df: pd.DataFrame) -> List[Dict]:
        """使用 Min-Max 縮放方法計算得分，並確保權重已歸一化"""
        from .scorer import AVAILABLE_METRICS
        
        results = []

        # 1. 處理並歸一化權重
        if self.metric_weights:
            # 檢查並清理權重
            valid_weights = {
                metric: self.metric_weights.get(metric, 0.0) 
                for metric in AVAILABLE_METRICS.keys()
            }
        else:
            # 預設等權重
            valid_weights = {metric: 1.0 for metric in AVAILABLE_METRICS.keys()}

        total_weight = sum(valid_weights.values())
        if total_weight > 0:
            # 歸一化權重，確保總和為 1
            normalized_weights = {k: v / total_weight for k, v in valid_weights.items()}
        else:
            # 如果所有權重為 0，則使用等權重
            num_metrics = len(AVAILABLE_METRICS)
            normalized_weights = {metric: 1.0 / num_metrics for metric in AVAILABLE_METRICS.keys()}
        
        # 輸出使用的權重供檢查
        print(f"使用的歸一化權重（總和為 1.0）:")
        for k, v in normalized_weights.items():
            if v > 1e-6:
                 print(f"  - {k}: {v:.4f}")

        # 計算每個指標的統計分佈
        metric_stats = {}
        for metric_name, metric_config in AVAILABLE_METRICS.items():
            # 獲取該指標的所有值
            if metric_config['source'] == 'stats':
                col_name = metric_config['key']
            else:  # detailed_stats
                col_name = metric_config['key']
            
            if col_name not in raw_results_df.columns:
                continue
            
            values = raw_results_df[col_name].dropna()
            if len(values) == 0:
                continue
            
            # 計算統計資訊
            metric_stats[metric_name] = {
                'values': values,
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median(),
                'q25': values.quantile(0.25),
                'q75': values.quantile(0.75),
                'higher_is_better': metric_config['higher_is_better']
            }
        
        # 為每個組合計算得分
        for idx, row in raw_results_df.iterrows():
            scores = {}
            total_score = 0
            
            for metric_name, metric_config in AVAILABLE_METRICS.items():
                
                weight = normalized_weights.get(metric_name, 0.0)
                
                # 跳過權重為 0 的指標
                if weight == 0:
                    scores[f'{metric_name}_得分'] = 0
                    continue
                
                # 獲取指標值
                if metric_config['source'] == 'stats':
                    col_name = metric_config['key']
                else:  # detailed_stats
                    col_name = metric_config['key']
                
                if col_name not in row.index or metric_name not in metric_stats:
                    scores[f'{metric_name}_得分'] = 0
                    continue
                
                value = row[col_name]
                if pd.isna(value):
                    scores[f'{metric_name}_得分'] = 0
                    continue
                
                # 使用 Min-Max 縮放方法計算得分 (0-100)
                stats_info = metric_stats[metric_name]
                metric_score_100 = self._calculate_statistical_score(value, stats_info, metric_name)
                
                # 應用歸一化權重
                weighted_score = metric_score_100 * weight
                scores[f'{metric_name}_得分'] = weighted_score
                total_score += weighted_score
            
            scores['總得分'] = total_score
            
            # 組合原始數據和得分 (修改處)
            # 按照您的要求：
            # 1. 總得分 (Total Score) 放在最前面
            # 2. 因子配置 (買入/賣出因子) 緊隨其後
            # 3. 其他得分和原始績效指標 (不包括 '組合編號')
            
            # 排除的欄位
            excluded_keys = ['組合編號'] 

            # 創建結果字典，並按照需要的順序排列
            result_record = {
                '總得分': scores['總得分'],
                '買入因子': row['買入因子'],
                '賣出因子': row['賣出因子'],
                **{k: v for k, v in scores.items() if k != '總得分'}, # 其他單項得分
                **{k: v for k, v in row.items() if k not in excluded_keys and k not in ['買入因子', '賣出因子']} # 原始數據
            }
            
            results.append(result_record)
        
        return results
    
    def _calculate_statistical_score(self, value: float, stats_info: Dict, metric_name: str) -> float:
        """
        使用 Min-Max 縮放方法計算單個指標的得分（0-100）。
        
        Parameters:
        -----------
        value : float
            當前組合的指標值
        stats_info : dict
            所有組合該指標的統計資訊 (min, max, higher_is_better)
        
        Returns:
        --------
        float : 得分（0-100），反映其在所有組合中的相對表現
        """
        
        higher_is_better = stats_info['higher_is_better']
        
        # 獲取 Min 和 Max 值
        data_min = stats_info['min']
        data_max = stats_info['max']
        
        # 處理極端情況：如果 Min == Max，表示所有組合的該指標值都一樣
        if abs(data_max - data_min) < 1e-9:
            # 如果所有值都一樣
            # 如果該值為正向指標 (越大越好)，且值 > 0，則給予高分 (100)
            if higher_is_better is True and value > 0:
                return 100.0
            # 如果該值為負向指標 (越小越好)，且值 < 0，則給予高分 (100)
            if higher_is_better is False and value < 0:
                 return 100.0
            # 否則給予中等分數
            return 50.0

        # 計算 Min-Max 縮放
        if higher_is_better is True:
            # 越大越好：(Value - Min) / (Max - Min)
            scaled_value = (value - data_min) / (data_max - data_min)
            score = scaled_value * 100
            
        elif higher_is_better is False:
            # 越小越好：(Max - Value) / (Max - Min)
            scaled_value = (data_max - value) / (data_max - data_min)
            score = scaled_value * 100
            
        else:
            # 適中最好：如果你的 AVAILABLE_METRICS 有這種指標，則可以保留原來的百分位數邏輯
            print("警告: 遇到 higher_is_better 為 None 的指標，使用 Min-Max 縮放。")
            scaled_value = (value - data_min) / (data_max - data_min)
            score = scaled_value * 100 # 此處可能需要更精確的適中最好計分法
        
        # 限制得分在 0-100 之間
        return max(0, min(score, 100))
    
    def get_top_n(self, n: int = 10) -> pd.DataFrame:
        """
        獲取前 N 名結果
        
        Parameters:
        -----------
        n : int
            返回前 N 名
        
        Returns:
        --------
        pd.DataFrame : 排序後的前 N 名結果
        """
        if len(self.results) == 0:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(self.results)
        if '總得分' not in results_df.columns:
            return results_df
        
        return results_df.nlargest(n, '總得分')
    
    def save_results(self, filepath: str):
        """
        儲存結果到 CSV 檔案
        
        Parameters:
        -----------
        filepath : str
            儲存路徑
        """
        if len(self.results) == 0:
            print("沒有結果可儲存")
            return
        
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"結果已儲存至: {filepath}")