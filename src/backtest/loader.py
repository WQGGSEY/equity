import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
from .cache import CacheManager
import pyarrow.parquet as pq

class MarketData:
    """
    [Robust Matrix Loader - Final Fixed Version]
    - Bias-Free: 전체 유니버스 로드
    - Auto-Universe: 거래대금 기준 Dynamic Universe Mask 자동 생성
    - Strict Alignment: (Dates x Tickers) 형태 보장
    """
    def __init__(self, platinum_dir="data/platinum"):
        self.platinum_dir = Path(platinum_dir)
        self.prices = {}   
        self.features = {} 
        self.tickers = []
        self.dates = []
        self.cache_manager = CacheManager()
        
    def load_all(self, required_features=None):
        # ---------------------------------------------------------
        # 1. Base Data (Prices) 로딩 - 캐시 우선
        # ---------------------------------------------------------
        # 캐시 이름에 'v2'를 붙여서 기존 캐시(유니버스 마스크 없는 버전)와 충돌 방지
        base_cache_name = "market_data_base_full_universe" 
        base_data = self.cache_manager.load(base_cache_name, expiration_hours=24)
        
        if base_data:
            print(f"🚀 [Loader] Base Cache Hit! Using {len(base_data['tickers'])} tickers.")
            self.prices = base_data['prices']
            self.tickers = base_data['tickers']
            self.dates = base_data['dates']
        else:
            print("🚀 [Loader] Building Base Matrix (Full Universe)...")
            files = list(self.platinum_dir.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No parquet files in {self.platinum_dir}")

            # [Step 1] 전체 유니버스 스캔
            print("  🧩 Scanning All Files (No Limit)...")
            all_dates = set()
            all_tickers = []
            
            for p in tqdm(files, desc="  Indexing"):
                try:
                    pf = pq.ParquetFile(p)
                    if 'Close' in pf.schema.names:
                        df = pd.read_parquet(p, columns=['Close'])
                        all_dates.update(df.index)
                        all_tickers.append(p.stem)
                except:
                    continue
            
            self.dates = sorted(list(all_dates))
            self.tickers = sorted(all_tickers)
            print(f"  ✅ Universe Locked: {len(self.tickers)} tickers, {len(self.dates)} days")

            # [Step 2] 가격 데이터 로드 & 정렬
            price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            data_store = {c: {} for c in price_cols}
            
            print(f"  📥 Loading Prices for {len(self.tickers)} tickers...")
            for t in tqdm(self.tickers, desc="  Reading Prices"):
                p = self.platinum_dir / f"{t}.parquet"
                try:
                    df = pd.read_parquet(p, columns=price_cols)
                    for c in price_cols:
                        if c in df.columns:
                            data_store[c][t] = df[c].astype('float32')
                except:
                    continue
            
            print("  📐 Aligning Base Matrices...")
            for c in price_cols:
                if data_store[c]:
                    df = pd.DataFrame(data_store[c])
                    # Index=Dates, Columns=Tickers
                    df = df.reindex(index=self.dates, columns=self.tickers)
                    self.prices[c] = df.astype('float32')
                del data_store[c]

            # Amount 생성
            if 'Close' in self.prices and 'Volume' in self.prices:
                open_p = self.prices.get('Open', self.prices['Close'])
                self.prices['Amount'] = ((open_p + self.prices['Close']) / 2.0 * self.prices['Volume']).astype('float32')

            # [Step 3] Dynamic Universe Mask 생성 (여기가 핵심!)
            # ---------------------------------------------------------
            if 'Amount' in self.prices:
                print("  🌌 Generating Dynamic Universe Mask (Top 500)...")
                amt = self.prices['Amount']
                
                # 1. 최근 20일 평균 거래대금 (Time-series Rolling)
                # axis=0 (기본값)이 index(날짜) 방향 rolling
                rolling_amt = amt.rolling(window=20, min_periods=1).mean()
                
                # 2. 일별 랭킹 산출 (Cross-sectional Rank)
                # axis=1 (컬럼=종목) 방향 랭킹. 큰 게 1등(ascending=False)
                # method='min' -> 동점자 처리
                daily_rank = rolling_amt.rank(axis=1, ascending=False, method='min')
                
                # 3. 마스크 생성 (Top 500은 1.0, 나머지는 NaN)
                # universe에 포함되지 않는 종목을 NaN으로 만들면 
                # 전략에서 곱하기 연산 시 자동으로 신호가 죽음(NaN)
                universe_mask = daily_rank.where(daily_rank <= 500, np.nan)
                universe_mask = universe_mask.where(universe_mask.isna(), 1.0)
                
                self.prices['universe'] = universe_mask.astype('float32')
            else:
                print("  ⚠️ Warning: Could not calculate Amount, skipping Universe generation.")

            # Base 캐시 저장
            base_save = {
                'prices': self.prices,
                'tickers': self.tickers,
                'dates': self.dates
            }
            self.cache_manager.save(base_save, base_cache_name)
            gc.collect()

        # ---------------------------------------------------------
        # 2. Feature Data 로딩 (On-Demand)
        # ---------------------------------------------------------
        if required_features:
            print(f"  📥 Loading Features: {required_features}")
            feat_store = {f: {} for f in required_features}
            target_paths = [self.platinum_dir / f"{t}.parquet" for t in self.tickers]
            
            for p in tqdm(target_paths, desc="  Reading Features"):
                if not p.exists(): continue
                t = p.stem
                try:
                    pf = pq.ParquetFile(p)
                    file_cols = set(pf.schema.names)
                    col_map = {c.lower(): c for c in file_cols}
                    
                    read_map = {}
                    for req in required_features:
                        if req in file_cols:
                            read_map[req] = req
                        elif req.lower() in col_map:
                            read_map[col_map[req.lower()]] = req
                            
                    if not read_map: continue
                    
                    df = pd.read_parquet(p, columns=list(read_map.keys()))
                    df.rename(columns=read_map, inplace=True)
                    
                    for req in required_features:
                        if req in df.columns:
                            feat_store[req][t] = df[req].astype('float32')
                except:
                    continue
            
            for f in required_features:
                if feat_store[f]:
                    df = pd.DataFrame(feat_store[f])
                    df = df.reindex(index=self.dates, columns=self.tickers)
                    self.features[f] = df.astype('float32')
                else:
                    print(f"  ⚠️ Feature '{f}' not found. Creating NaN matrix.")
                    self.features[f] = pd.DataFrame(np.nan, index=self.dates, columns=self.tickers).astype('float32')
                
                del feat_store[f]
            
            gc.collect()
            
        print("  ✅ Loading Complete.")