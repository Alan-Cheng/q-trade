import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json

# =========================================================
# StockDataManager：負責使用 yfinance 下載、儲存和提供股票 K 線資料的類別。
# =========================================================
class StockDataManager:
    """
    負責使用 yfinance 下載、儲存和提供股票 K 線資料的類別。
    """
    def __init__(self, symbols=None, start_date=None, end_date=None, country_code=None):
        """
        初始化管理器，並下載指定股票的資料。
        
        參數:
            symbols (str or list, optional): 要下載的股票代碼，可以是：
                - 單個股票代碼字串 (e.g., "AAPL")
                - 股票代碼列表 (e.g., ["TSLA", "AAPL"])
                如果提供了 country_code，則此參數會被忽略。
            start_date (str): 資料下載的起始日期。
            end_date (str, optional): 資料下載的結束日期。
            country_code (str, optional): 國碼，目前支援 "TW"（台灣）。
                當提供此參數時，會自動抓取該國家的所有股票代號。
        """
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = {}
        
        # 如果提供了國碼，則從 API 抓取該國家的所有股票代號
        if country_code:
            self.all_symbols = self._fetch_stock_symbols_by_country(country_code)
        elif symbols:
            # 支援單個字串或列表
            if isinstance(symbols, str):
                self.all_symbols = [symbols]
            elif isinstance(symbols, list):
                self.all_symbols = symbols
            else:
                raise ValueError(f"symbols 參數必須是字串或列表，收到: {type(symbols)}")
        else:
            raise ValueError("必須提供 symbols 或 country_code 參數")
        
        self._download_data()

    def _fetch_stock_symbols_by_country(self, country_code):
        """
        根據國碼抓取該國家的所有股票代號。
        
        參數:
            country_code (str): 國碼，目前支援 "TW"。
        
        回傳:
            list: 股票代號列表，已轉換為 yfinance 可用的格式。
        """
        if country_code == "TW":
            return self._fetch_tw_stock_symbols()
        else:
            raise ValueError(f"目前不支援國碼: {country_code}。目前支援的國碼: TW")
    
    def _fetch_tw_stock_symbols(self):
        """
        從台灣證交所 API 抓取所有股票代號。
        
        回傳:
            list: 股票代號列表，格式為 ["0050.TW", "2330.TW", ...]。
        """
        url = 'https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL'
        
        try:
            print("正在從台灣證交所 API 抓取股票代號...")
            res = requests.get(url)
            res.raise_for_status()
            
            jsondata = json.loads(res.text)
            
            # 從 JSON 陣列中提取所有 Code，並轉換為 yfinance 格式 (Code + ".TW")
            symbols = [item['Code'] + '.TW' for item in jsondata if 'Code' in item]
            
            print(f"成功抓取 {len(symbols)} 檔台灣股票代號")
            return symbols
            
        except requests.RequestException as e:
            print(f"抓取台灣股票代號時發生網路錯誤: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"解析 API 回應時發生錯誤: {e}")
            raise
        except Exception as e:
            print(f"抓取台灣股票代號時發生未知錯誤: {e}")
            raise

    def _download_data(self):
        """
        實際執行 yfinance 下載資料的內部方法。
        使用批量下載以提升速度。
        """
        print(f"正在使用 yfinance 下載 {len(self.all_symbols)} 檔股票資料...")
        
        # 如果股票數量較少，直接批量下載
        if len(self.all_symbols) <= 100:
            self._download_batch(self.all_symbols)
        else:
            # 如果股票數量很多，分批下載以避免 API 限制
            batch_size = 100
            for i in range(0, len(self.all_symbols), batch_size):
                batch = self.all_symbols[i:i + batch_size]
                print(f"正在下載第 {i//batch_size + 1} 批（共 {len(batch)} 檔）...")
                self._download_batch(batch)

        # 打印下載摘要
        print("---" * 10)
        print(f"已成功下載 {len(self.stock_data)} 檔股票資料")
        if len(self.stock_data) > 0:
            # 只顯示前 5 檔的詳細資訊，避免輸出過多
            for i, (symbol, df) in enumerate(list(self.stock_data.items())[:5]):
                print(f"  - {symbol}: {len(df)} 筆資料，日期範圍 {df.index[0].date()} 至 {df.index[-1].date()}")
            if len(self.stock_data) > 5:
                print(f"  ... 還有 {len(self.stock_data) - 5} 檔股票")
        print("---" * 10)
    
    def _download_batch(self, symbols):
        """
        批量下載股票資料。
        
        參數:
            symbols (list): 要下載的股票代號列表。
        """
        try:
            # 使用批量下載，yfinance 會自動並行處理
            df_all = yf.download(
                symbols,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )
            
            if df_all.empty:
                print(f"警告: 批量下載失敗或無資料")
                return
            
            # 處理批量下載的結果
            if len(symbols) == 1:
                # 單一股票時，直接處理
                symbol = symbols[0]
                df = df_all.copy()
                
                # 處理 MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    if df.columns.nlevels > 1:
                        df.columns = df.columns.droplevel(1)
                
                if not df.empty:
                    self.stock_data[symbol] = df
            else:
                # 多個股票時，yfinance 返回的 DataFrame columns 是 MultiIndex
                # 第一層是價格類型（Open, High, Low, Close, Volume, Adj Close）
                # 第二層是股票代號
                if isinstance(df_all.columns, pd.MultiIndex) and df_all.columns.nlevels == 2:
                    # 按股票代號拆分
                    for symbol in symbols:
                        try:
                            # 提取該股票的所有價格欄位
                            symbol_cols = df_all.columns[df_all.columns.get_level_values(1) == symbol]
                            
                            if len(symbol_cols) == 0:
                                continue
                            
                            # 提取該股票的資料
                            df = df_all[symbol_cols].copy()
                            
                            # 移除第二層（股票代號），只保留價格類型
                            df.columns = df.columns.droplevel(1)
                            
                            if df.empty:
                                continue
                            
                            self.stock_data[symbol] = df
                        except (KeyError, IndexError):
                            # 該股票在結果中不存在（可能已下市或無資料）
                            continue
                else:
                    # 如果沒有 MultiIndex 或格式不符合預期，回退到逐個下載
                    raise ValueError("批量下載返回格式不符合預期，回退到逐個下載")
                    
        except Exception as e:
            # 如果批量下載失敗，回退到逐個下載
            print(f"批量下載失敗，改為逐個下載: {e}")
            for symbol in symbols:
                try:
                    df = yf.download(
                        symbol,
                        start=self.start_date,
                        end=self.end_date,
                        progress=False,
                        auto_adjust=True
                    )
                    
                    if df.empty:
                        continue
                    
                    if isinstance(df.columns, pd.MultiIndex):
                        if df.columns.nlevels > 1:
                            df.columns = df.columns.droplevel(1)
                    
                    self.stock_data[symbol] = df
                except Exception as e2:
                    # 靜默處理單個股票下載失敗，避免輸出過多錯誤訊息
                    pass

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