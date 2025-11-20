import itertools
from typing import List, Dict, Any

class ParameterSpace:
    """
    定義因子參數的搜索空間，並生成所有可能的組合
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
        
        Parameters:
        -----------
        factor_configs : list
            因子配置列表
        
        Returns:
        --------
        list : 因子組合列表，每個元素是一個因子配置列表
        """
        if not factor_configs:
            return [[]]
        
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
        
        # 生成因子之間的組合（使用所有因子的所有參數組合）
        # 如果需要選擇性使用因子，可以在這裡修改邏輯
        result = []
        for combo in itertools.product(*factor_combos):
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
        """計算單一因子類型的組合數量"""
        if not factor_configs:
            return 1
        
        total = 1
        for config in factor_configs:
            param_ranges = config['param_ranges']
            count = 1
            for param_values in param_ranges.values():
                count *= len(param_values)
            total *= count
        
        return total

