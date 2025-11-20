import yfinance as yf
import pandas as pd
import numpy as np

# =========================================================
# StockDataManager：負責使用 yfinance 下載、儲存和提供股票 K 線資料的類別。
# =========================================================
class StockDataManager:
    """
    負責使用 yfinance 下載、儲存和提供股票 K 線資料的類別。
    """
    def __init__(self, symbols, start_date=None, end_date=None):
        """
        初始化管理器，並下載指定股票的資料。
        
        參數:
            symbols (list): 要下載的股票代碼列表 (e.g., ["TSLA", "AAPL"])。
            start_date (str): 資料下載的起始日期。
            end_date (str, optional): 資料下載的結束日期。
        """
        self.all_symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = {}
        
        self._download_data()

    def _download_data(self):
        """
        實際執行 yfinance 下載資料的內部方法。
        """
        print(f"正在使用 yfinance 下載 {len(self.all_symbols)} 檔股票資料...")
        
        for symbol in self.all_symbols:
            try:
                # progress=False 避免輸出過多進度條
                df = yf.download(symbol, 
                                 start=self.start_date, 
                                 end=self.end_date, 
                                 progress=False, 
                                 auto_adjust=True)
                
                if df.empty:
                    print(f"警告: {symbol} 未能下載到資料，跳過。")
                    continue
                
                # 自動處理 MultiIndex columns（yfinance 下載單一股票時可能會有 MultiIndex）
                if isinstance(df.columns, pd.MultiIndex):
                    if df.columns.nlevels > 1:
                        df.columns = df.columns.droplevel(1)
                
                # 儲存 K 線資料
                self.stock_data[symbol] = df
                
            except Exception as e:
                print(f"下載 {symbol} 時發生錯誤: {e}")

        # 打印下載摘要
        print("---" * 10)
        print(f"已成功下載 {len(self.stock_data)} 檔股票資料")
        for symbol, df in self.stock_data.items():
            print(f"  - {symbol}: {len(df)} 筆資料，日期範圍 {df.index[0].date()} 至 {df.index[-1].date()}")
        print("---" * 10)

    def get_kl_pd(self, symbol):
        """
        提供給選股因子 fit_pick 方法使用的 K 線 DataFrame。
        
        參數:
            symbol (str): 股票代碼。

        回傳:
            pd.DataFrame: 股票的 K 線資料，如果資料不存在，回傳空 DataFrame。
        """
        df = self.stock_data.get(symbol, pd.DataFrame())
        
        # 確保返回的資料沒有 MultiIndex columns（雙重保險）
        if not df.empty and isinstance(df.columns, pd.MultiIndex):
            if df.columns.nlevels > 1:
                df = df.copy()
                df.columns = df.columns.droplevel(1)
        
        return df
    
    def get_stock_data(self):
        return self.stock_data
    
    def get_stock_data_by_symbol(self, symbol: str) -> pd.DataFrame:
        return self.stock_data.get(symbol, pd.DataFrame())
    
    def get_stock_data_by_symbols(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        return {symbol: self.stock_data.get(symbol, pd.DataFrame()) for symbol in symbols}