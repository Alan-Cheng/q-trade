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
        return self.stock_data.get(symbol, pd.DataFrame())
    
    
# =========================================================
print("="*60)
print("StockDataManager 使用範例")
print("="*60)
# =========================================================

# 使用範例    
# 初始化 StockDataManager
# 1. 定義要下載的股票代碼和時間範圍
symbols_to_download = ["TSLA", "AAPL", "MSFT", "BABA"]
start_date = "2023-01-01"
# 結束日期如果為 None，則預設下載到最新日期

print("## 步驟 1: 實例化 StockDataManager 並下載資料")
data_manager = StockDataManager(symbols_to_download, start_date=start_date)

# 2. 存取單一股票的 K 線資料
target_symbol = "AAPL"
aapl_df = data_manager.get_kl_pd(target_symbol)

print(f"\n## 步驟 2: 存取 {target_symbol} 的 K 線資料")
if not aapl_df.empty:
    print(f"{target_symbol} 資料前 5 行:")
    print(aapl_df.head())
    
    # 存取收盤價序列
    close_series = aapl_df['Close']
    print(f"\n{target_symbol} 收盤價序列長度: {len(close_series)}")
else:
    print(f"{target_symbol} 的資料不存在或下載失敗。")


# 3. 遍歷所有已下載的股票
print("\n## 步驟 3: 遍歷所有股票並檢查資料大小")
for symbol in symbols_to_download:
    df = data_manager.get_kl_pd(symbol)
    if not df.empty:
        print(f"  - {symbol} 的資料筆數: {len(df)}")
    else:
        print(f"  - {symbol} 的資料為空。")