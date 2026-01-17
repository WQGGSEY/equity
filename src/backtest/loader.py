import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
from .cache import CacheManager

class MarketData:
    """
    [Memory-Safe Matrix Loader]
    Original Architecture Restored:
    1. Automatic Column Discovery (No manual 'required_features' needed)
    2. Ticker-First Reading -> Feature-Matrix Pivoting
    3. Smart Universe Cutting (Top 500) to survive 8GB RAM
    """
    def __init__(self, platinum_dir="data/platinum"):
        self.platinum_dir = Path(platinum_dir)
        self.prices = {}   
        self.features = {} 
        self.tickers = []
        self.dates = []
        self.cache_manager = CacheManager()
        
    def load_all(self): # 서명(Signature)을 기존과 동일하게 복구
        # 1. 캐시 확인
        # ---------------------------------------------------------
        # (메모리 보호를 위해 기본적으로 Top 500 캐시를 사용한다고 가정)
        cache_name = "market_data_matrix_optimized"
        cached_data = self.cache_manager.load(cache_name, expiration_hours=12)
        
        if cached_data:
            print("🚀 [Loader] Cache Hit! Loading from disk cache...")
            self.prices = cached_data['prices']
            self.features = cached_data['features']
            self.tickers = cached_data['tickers']
            self.dates = cached_data['dates']
            print(f"  ✅ Loaded {len(self.tickers)} tickers, {len(self.features)} features.")
            return

        # 2. 원본 로딩 (No Cache)
        # ---------------------------------------------------------
        print("🚀 [Loader] Building Matrix from Platinum (Original Logic + Safe Mode)...")
        files = list(self.platinum_dir.glob("*.parquet"))
        
        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.platinum_dir}")

        # [Step 1] 스키마 발견 (Schema Discovery)
        # 첫 번째 파일을 열어서 "어떤 컬럼(피처)들이 있는지" 자동으로 알아냅니다.
        # 기존 코드의 results[0].columns 로직을 계승합니다.
        sample_df = pd.read_parquet(files[0])
        all_columns = sample_df.columns.tolist()
        
        # 가격 컬럼과 피처 컬럼 분류
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [c for c in all_columns if c not in price_cols and c != 'Date']
        
        print(f"  🔍 Discovered Features: {feature_cols}")
        
        # [Step 2] 유니버스 선정 (Universe Selection)
        # 3,000개를 다 읽으면 터지니까, '거래대금'만 먼저 훑어서 상위 500개를 정합니다.
        print("  ✂️ Selecting Top 500 Universe (to save RAM)...")
        temp_amounts = {}
        
        # 가벼운 스캔 (Close, Volume만 읽기)
        for p in tqdm(files, desc="  Scanning Liquidity"):
            try:
                # 필요한 컬럼만 읽어서 메모리 절약
                df = pd.read_parquet(p, columns=['Close', 'Volume'])
                amt = (df['Close'] * df['Volume']).iloc[-20:] # 최근 20일 평균만 봄
                mean_amt = amt.mean()
                if pd.notna(mean_amt):
                    temp_amounts[p.stem] = mean_amt
            except:
                continue
        
        # 상위 500개 파일 확정
        top_tickers = sorted(temp_amounts, key=temp_amounts.get, reverse=True)[:500]
        self.tickers = top_tickers
        target_files = [self.platinum_dir / f"{t}.parquet" for t in self.tickers]
        
        print(f"  ✅ Universe set to {len(self.tickers)} tickers.")
        del temp_amounts
        gc.collect()

        # [Step 3] 데이터 로드 & 매트릭스 변환 (Main Loop)
        # 기존 로직: ticker_map = {t: df for ...} -> Matrix 변환
        # 최적화 로직: 파일을 하나씩 읽으면서 바로바로 각 매트릭스(딕셔너리)에 꽂아 넣음
        
        # 1. 저장소 초기화
        # prices['Close'] = {ticker: series, ...}
        # features['FD_Close'] = {ticker: series, ...}
        data_store = {col: {} for col in all_columns}
        
        # 2. 파일 순회 (직렬 처리)
        for p in tqdm(target_files, desc="  Loading Data"):
            try:
                t = p.stem
                df = pd.read_parquet(p) # 상위 500개라 전체 로드해도 안전함
                
                # float32 최적화 (기존 로직 계승)
                float_cols = df.select_dtypes(include=['float64']).columns
                if len(float_cols) > 0:
                    df[float_cols] = df[float_cols].astype('float32')
                
                # 각 컬럼별로 쪼개서 저장
                for col in df.columns:
                    data_store[col][t] = df[col]
                    
            except Exception as e:
                print(f"  ⚠️ Failed to load {p.stem}: {e}")
                continue

        # [Step 4] DataFrame 매트릭스 생성 (Pivot)
        print("  🧩 Pivoting to Matrix...")
        
        # 날짜 인덱스 통합
        if 'Close' in data_store and len(data_store['Close']) > 0:
            first_ticker = list(data_store['Close'].keys())[0]
            self.dates = data_store['Close'][first_ticker].index
        
        # Prices 완성
        for col in price_cols:
            if col in data_store:
                self.prices[col] = pd.DataFrame(data_store[col]).reindex(self.dates)
                del data_store[col] # 메모리 해제
        
        # Amount 자동 생성 (기존 로직 계승)
        if 'Amount' not in self.prices and 'Close' in self.prices:
            avg = (self.prices['Open'] + self.prices['Close']) / 2
            self.prices['Amount'] = avg * self.prices['Volume']

        # Features 완성
        for col in feature_cols:
            if col in data_store and data_store[col]:
                self.features[col] = pd.DataFrame(data_store[col]).reindex(self.dates)
                del data_store[col] # 메모리 해제

        print("  ✅ Matrix Build Complete.")
        
        # 3. 캐시 저장
        save_data = {
            'prices': self.prices,
            'features': self.features,
            'tickers': self.tickers,
            'dates': self.dates
        }
        self.cache_manager.save(save_data, cache_name)
        gc.collect()