import copy
import pandas as pd

# =========================================================
# StockPickerWorker：模擬回測系統中的選股核心，負責遍歷股票並應用選股因子。
# =========================================================
class StockPickerWorker:
    """模擬回測系統中的選股核心，負責遍歷股票並應用選股因子。"""
    def __init__(self, data_manager, stock_pickers):
        self.data_manager = data_manager
        self.stock_pickers = stock_pickers
        self.picker_instances = self._init_pickers()
        self.choice_symbols = []
        # 新增屬性：用於儲存詳細的因子篩選結果 DataFrame
        self.factor_results_df = None 
        # 新增屬性：用於儲存所有因子的名稱 (用於 DataFrame 欄位)
        self.factor_names = [f"{p['class'].__name__}_{i}" for i, p in enumerate(stock_pickers)]


    def _init_pickers(self):
        """根據配置實例化選股因子"""
        picker_list = []
        for config in self.stock_pickers:
            picker_config = copy.deepcopy(config)
            picker_class = picker_config['class']
            del picker_config['class']
            picker = picker_class(None, None, **picker_config)
            picker_list.append(picker)
        return picker_list

    def fit(self):
        """執行選股過程，並將結果儲存到 self.factor_results_df"""
        print("\n--- 正在執行選股過程並記錄詳細因子結果 ---")
        
        results_list = []
        
        for symbol in self.data_manager.stock_data.keys():
            kl_pd = self.data_manager.get_kl_pd(symbol)
            is_picked_overall = True
            
            # 初始化該股票的結果字典
            result_row = {'symbol': symbol}
            
            # --- 1. 遍歷並紀錄每個因子的結果 ---
            for i, picker in enumerate(self.picker_instances):
                factor_passed = picker.fit_pick(kl_pd, symbol)
                factor_name = self.factor_names[i]
                
                result_row[factor_name] = factor_passed
                
                # 記錄因子的計算值（如果因子有儲存計算值）
                factor_value_name = f"{factor_name}_值"
                if hasattr(picker, 'last_calculated_value'):
                    result_row[factor_value_name] = picker.last_calculated_value
                else:
                    result_row[factor_value_name] = None
                
                if not factor_passed:
                    is_picked_overall = False
            
            # --- 2. 紀錄關鍵指標和最終篩選結果 ---
            # current_angle = RegressionUtil.calc_regress_deg(kl_pd['Close'])
            
            # result_row['趨勢角度'] = current_angle
            result_row['最終選中'] = is_picked_overall
            
            results_list.append(result_row)
            
            # print(f"  > {symbol}: 趨勢角度={current_angle:.2f}°, 最終結果={is_picked_overall}")
            
            if is_picked_overall:
                self.choice_symbols.append(symbol)

        # --- 3. 轉換為 DataFrame 並儲存 ---
        self.factor_results_df = pd.DataFrame(results_list)
        # 將 symbol 設為索引，更方便檢視
        self.factor_results_df.set_index('symbol', inplace=True)
        
        print("\n--- 詳細因子結果已儲存至 self.factor_results_df ---")