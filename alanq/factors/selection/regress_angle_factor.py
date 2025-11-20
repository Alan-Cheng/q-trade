import numpy as np
from .base_selection_factor import StockPickerBase
from ...util.regression_util import RegressionUtil, reversed_result

# =========================================================
# RegressAnglePicker：基於股票 K 線收盤價的線性迴歸趨勢角度進行篩選的選股因子。
# =========================================================
class RegressAnglePicker(StockPickerBase):
    """
    基於股票 K 線收盤價的線性迴歸趨勢角度進行篩選的選股因子。
    """
    def _init_self(self, **kwargs):
        """初始化角度閥值、reversed 和 show 屬性"""
        
        self.threshold_ang_min = kwargs.get('threshold_ang_min', -np.inf)
        self.threshold_ang_max = kwargs.get('threshold_ang_max', np.inf)
        
        self.reversed = kwargs.get('reversed', False)
        
        # *** 新增：接收並儲存 show 參數 ***
        # 預設為 False，除非使用者明確設定為 True
        self.show_plot = kwargs.get('show', False) 

    @reversed_result
    def fit_pick(self, kl_pd, target_symbol):
        if kl_pd.empty:
            return False
            
        # ... (close_data 獲取邏輯不變)
        try:
            close_data = kl_pd['Close']
        except KeyError:
            try:
                close_data = kl_pd['close']
            except KeyError:
                return False 
            
        ang = RegressionUtil.calc_regress_deg(
            close_data, 
            symbol=target_symbol,
            show=self.show_plot
        )
        
        # 根據參數進行角度條件判斷
        return self.threshold_ang_min < ang < self.threshold_ang_max