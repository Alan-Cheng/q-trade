import itertools
from typing import List, Dict, Any

class ParameterSpace:
    """
    定義因子參數的搜索空間，並生成所有可能的組合
    
    支援選擇性使用因子：
    - 買入因子和賣出因子各自可以選擇 1 個或多個因子（每個因子只能出現一次）
    - 買入因子和賣出因子都至少要有一個（不允許空集）
    - 例如：如果有 2 個賣出因子，可以只選第 1 個、只選第 2 個、或同時選兩個
    """
    
    def __init__(self):
        self.buy_factor_configs = []
        self.sell_factor_configs = []
        
    def add_buy_factor(self, factor_class, param_ranges: Dict[str, List]):
        """
        添加買入因子及其參數範圍
        
        Parameters:
        -----------
        factor_class : class
            因子類別（如 BreakoutBuyFactor）
        param_ranges : dict
            參數範圍字典，例如 {'xd': [20, 30, 40, 50, 60], 'skip_days': [10, 15, 20]}
        """
        if not param_ranges:
            raise ValueError(f"{factor_class.__name__} 必須至少有 1 個參數")

        for k, v in param_ranges.items():
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError(
                    f"{factor_class.__name__} 中參數 '{k}' 的取值不得為空"
                )
        
        # 驗證完成後，只添加一次因子配置
        self.buy_factor_configs.append({
            'class': factor_class,
            'param_ranges': param_ranges
        })
    
    def add_sell_factor(self, factor_class, param_ranges: Dict[str, List]):
        """
        添加賣出因子及其參數範圍
        
        Parameters:
        -----------
        factor_class : class
            因子類別（如 BreakdownSellFactor）
        param_ranges : dict
            參數範圍字典，例如 {'xd': [10, 15, 20, 25]}
        """
        if not param_ranges:
            raise ValueError(f"{factor_class.__name__} 必須至少有 1 個參數")

        for k, v in param_ranges.items():
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError(
                    f"{factor_class.__name__} 中參數 '{k}' 的取值不得為空"
                )
        
        # 驗證完成後，只添加一次因子配置
        self.sell_factor_configs.append({
            'class': factor_class,
            'param_ranges': param_ranges
        })
    
    def generate_combinations(self) -> List[Dict]:
        """
        生成所有可能的因子組合
        
        Returns:
        --------
        list : 每個元素是一個配置字典
            {
                'buy_factors': [...],
                'sell_factors': [...]
            }
        """
        all_combinations = []
        
        # 生成買入因子組合
        buy_combinations = self._generate_factor_combinations(self.buy_factor_configs)
        
        # 生成賣出因子組合
        sell_combinations = self._generate_factor_combinations(self.sell_factor_configs)
        
        # 買入和賣出因子的笛卡爾積
        for buy_combo in buy_combinations:
            for sell_combo in sell_combinations:
                all_combinations.append({
                    'buy_factors': buy_combo,
                    'sell_factors': sell_combo
                })
        
        return all_combinations
    
    def _generate_factor_combinations(self, factor_configs: List[Dict]) -> List[List]:
        """
        為單一因子類型生成所有參數組合
        支援選擇性使用因子：可以選擇 1 個或多個因子（每個因子只能出現一次）
        但至少需要選擇一個因子（不允許空集）
        
        Parameters:
        -----------
        factor_configs : list
            因子配置列表
        
        Returns:
        --------
        list : 因子組合列表，每個元素是一個因子配置列表
        """
        if not factor_configs:
            return []
        
        # 為每個因子生成所有參數組合
        factor_combos = []
        for config in factor_configs:
            factor_class = config['class']
            param_ranges = config['param_ranges']
            
            # 生成該因子的所有參數組合
            param_names = list(param_ranges.keys())
            param_values = [param_ranges[name] for name in param_names]
            
            factor_param_combos = []
            for param_combo in itertools.product(*param_values):
                param_dict = dict(zip(param_names, param_combo))
                factor_param_combos.append({
                    'class': factor_class,
                    **param_dict
                })
            
            factor_combos.append(factor_param_combos)
        
        # 生成所有可能的非空子集組合
        # 對於每個子集大小（1 到 len(factor_combos)），生成所有可能的組合
        result = []
        n_factors = len(factor_combos)
        
        # 遍歷所有可能的子集大小（從 1 到 n_factors）
        for subset_size in range(1, n_factors + 1):
            # 生成所有大小為 subset_size 的子集
            for subset_indices in itertools.combinations(range(n_factors), subset_size):
                # 對於這個子集，生成所有參數組合的笛卡爾積
                selected_factor_combos = [factor_combos[i] for i in subset_indices]
                for combo in itertools.product(*selected_factor_combos):
                    result.append(list(combo))
        
        return result
    
    def get_total_combinations(self) -> int:
        """
        計算總組合數量（不實際生成，用於預估）
        
        Returns:
        --------
        int : 總組合數量
        """
        buy_count = self._count_combinations(self.buy_factor_configs)
        sell_count = self._count_combinations(self.sell_factor_configs)
        return buy_count * sell_count
    
    def _count_combinations(self, factor_configs: List[Dict]) -> int:
        """
        計算單一因子類型的組合數量
        支援選擇性使用因子：可以選擇 1 個或多個因子（每個因子只能出現一次）
        但至少需要選擇一個因子（不允許空集）
        """
        if not factor_configs:
            return 0
        
        # 計算每個因子的參數組合數
        factor_counts = []
        for config in factor_configs:
            param_ranges = config['param_ranges']
            count = 1
            for param_values in param_ranges.values():
                count *= len(param_values)
            factor_counts.append(count)
        
        # 計算所有非空子集的組合數
        # 對於每個子集大小 k，組合數 = C(n, k) * (子集內所有因子參數組合數的乘積)
        # 但這樣計算比較複雜，我們用另一種方式：
        # 總組合數 = 所有可能的非空子集的組合數總和
        total = 0
        n_factors = len(factor_counts)
        
        # 遍歷所有可能的子集大小（從 1 到 n_factors）
        for subset_size in range(1, n_factors + 1):
            # 生成所有大小為 subset_size 的子集
            for subset_indices in itertools.combinations(range(n_factors), subset_size):
                # 計算這個子集的組合數（子集內所有因子參數組合數的乘積）
                subset_count = 1
                for i in subset_indices:
                    subset_count *= factor_counts[i]
                total += subset_count
        
        return total