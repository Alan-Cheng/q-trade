import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from ..backtest.timing_backtester_single import Backtester
from ..performance.performance_metrics import PerformanceMetrics
from .parameter_space import ParameterSpace
from .scorer import Scorer


def _evaluate_combination_worker(combo_num: int, combo: Dict, df: pd.DataFrame, 
                                  initial_capital: float, slippage_factors: List) -> Optional[Dict]:
    """
    並行處理的 worker 函數（必須在模組層級定義）
    
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
        
        # 計算得分
        scorer = Scorer(metrics)
        scores = scorer.calculate_score()
        
        # 記錄結果
        result_record = {
            '組合編號': combo_num,
            '買入因子': str(combo['buy_factors']),
            '賣出因子': str(combo['sell_factors']),
            '總得分': scores['總得分'],
            **scores,
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
                 n_jobs: int = 1):
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
        """
        self.df = df
        self.parameter_space = parameter_space
        self.initial_capital = initial_capital
        self.slippage_factors = slippage_factors or []
        self.show_progress = show_progress
        
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
        
        # 根據 n_jobs 決定使用並行還是順序執行
        if self.n_jobs > 1:
            self.results = self._optimize_parallel(combinations)
        else:
            self.results = self._optimize_sequential(combinations)
        
        # 轉換為 DataFrame
        if len(self.results) == 0:
            print("警告: 沒有成功執行任何組合")
            return None, pd.DataFrame()
        
        results_df = pd.DataFrame(self.results)
        
        # 找出最佳組合
        if len(results_df) > 0 and '總得分' in results_df.columns:
            best_idx = results_df['總得分'].idxmax()
            best_row = results_df.loc[best_idx]
            
            # 安全地解析因子配置
            try:
                import ast
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
    
    def _optimize_sequential(self, combinations: List[Dict]) -> List[Dict]:
        """順序執行優化"""
        results = []
        
        # 遍歷每個組合
        if self.show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(combinations, desc="優化進度")
            except ImportError:
                print("提示: 安裝 tqdm 可以顯示進度條 (pip install tqdm)")
                iterator = combinations
        else:
            iterator = combinations
        
        for i, combo in enumerate(iterator):
            try:
                result_record = self._evaluate_single_combination(i + 1, combo)
                if result_record:
                    results.append(result_record)
            except Exception as e:
                print(f"\n組合 {i+1} 執行失敗: {e}")
                continue
        
        return results
    
    def _optimize_parallel(self, combinations: List[Dict]) -> List[Dict]:
        """並行執行優化"""
        results = []
        
        # 準備進度條
        if self.show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(combinations), desc="優化進度")
            except ImportError:
                pbar = None
        else:
            pbar = None
        
        # 使用進程池並行執行
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # 提交所有任務
            future_to_combo = {
                executor.submit(
                    _evaluate_combination_worker,
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
    
    def _evaluate_single_combination(self, combo_num: int, combo: Dict) -> Optional[Dict]:
        """評估單個組合（內部方法）"""
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
        
        # 計算得分
        scorer = Scorer(metrics)
        scores = scorer.calculate_score()
        
        # 記錄結果
        result_record = {
            '組合編號': combo_num,
            '買入因子': str(combo['buy_factors']),
            '賣出因子': str(combo['sell_factors']),
            '總得分': scores['總得分'],
            **scores,
            **stats,
            **metrics.detailed_stats
        }
        
        return result_record
    
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

