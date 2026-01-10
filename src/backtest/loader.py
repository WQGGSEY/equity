import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from .cache import CacheManager # [NEW] Import CacheManager

class MarketData:
    """
    [Big Matrix Loader with Cache]
    """
    def __init__(self, platinum_dir):
        self.platinum_dir = Path(platinum_dir)
        self.prices = {}   
        self.features = {} 
        self.tickers = []
        self.dates = []
        self.cache_manager = CacheManager() # 캐시 매니저 인스턴스
        
    def load_all(self):
        # 1. 캐시 시도
        cache_name = "market_data_matrix"
        cached_data = self.cache_manager.load(cache_name, expiration_hours=24)
        
        if cached_data:
            # 캐시가 있으면 내부 상태 복원
            self.prices = cached_data['prices']
            self.features = cached_data['features']
            self.tickers = cached_data['tickers']
            self.dates = cached_data['dates']
            print("  ✅ Big Matrix Loaded from Cache.")
            return

        # 2. 캐시 없으면 원본 로딩 (Heavy Task)
        print("🚀 [Loader] Building Big Matrix from Platinum (No Cache Found)...")
        files = list(self.platinum_dir.glob("*.parquet"))
        
        if not files:
            raise FileNotFoundError("No platinum files found.")

        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap(_load_single_parquet, files), 
                              total=len(files), desc="  Reading Parquets"))
        
        results = [r for r in results if r is not None]
        if not results: raise ValueError("No valid data loaded.")

        # 메타데이터 설정
        all_dates = set()
        for _, df in results: all_dates.update(df.index)
        self.dates = sorted(list(all_dates))
        self.tickers = [t for t, _ in results]
        
        print(f"  📊 Matrix Shape: {len(self.dates)} dates x {len(self.tickers)} tickers")
        
        # 딕셔너리 매핑
        ticker_map = {t: df for t, df in results}
        
        # Price Matrix
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in price_cols:
            data_dict = {t: ticker_map[t].get(col) for t in self.tickers}
            self.prices[col] = pd.DataFrame(data_dict).reindex(self.dates)
            
        # Feature Matrix
        sample_cols = results[0][1].columns
        feature_cols = [c for c in sample_cols if c not in price_cols and not c.startswith('FD_')]
        
        for col in feature_cols:
            data_dict = {t: ticker_map[t].get(col) for t in self.tickers}
            self.features[col] = pd.DataFrame(data_dict).reindex(self.dates)
            
        # Amount Calculation
        avg_price = (self.prices['Open'] + self.prices['High'] + 
                     self.prices['Low'] + self.prices['Close']) / 4
        self.prices['Amount'] = avg_price * self.prices['Volume']
        
        print("  ✅ Big Matrix Ready.")
        
        # 3. 캐시 저장
        save_data = {
            'prices': self.prices,
            'features': self.features,
            'tickers': self.tickers,
            'dates': self.dates
        }
        self.cache_manager.save(save_data, cache_name)

def _load_single_parquet(path):
    try:
        df = pd.read_parquet(path)
        return (path.stem, df)
    except:
        return None