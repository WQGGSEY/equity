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
    - Bias-Free: 인위적인 종목 수 제한(3000개) 없이 전체 유니버스를 로드합니다.
    - Efficient Caching: 변하지 않는 '가격(Base)'과 변하는 '피처(Feature)'를 분리하여 처리합니다.
    - Strict Alignment: 모든 행렬이 (Dates x Tickers) 형태를 갖도록 강제하여 연산 오류를 방지합니다.
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
        # 피처가 바뀌어도 가격 데이터 캐시는 그대로 씁니다. (비효율 제거)
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

            # [Step 1] 전체 유니버스 스캔 (Bias-Free)
            print("  🧩 Scanning All Files (No Limit)...")
            all_dates = set()
            all_tickers = []
            
            # 13,000개 파일 스캔 (날짜축 확정용)
            for p in tqdm(files, desc="  Indexing"):
                try:
                    pf = pq.ParquetFile(p)
                    # Close 컬럼이 있는 파일만 유효한 종목으로 인정
                    if 'Close' in pf.schema.names:
                        # 날짜 인덱스만 빠르게 추출
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
                    # 필요한 컬럼만 읽기
                    df = pd.read_parquet(p, columns=price_cols)
                    for c in price_cols:
                        if c in df.columns:
                            # float32로 변환하여 메모리 절약 (13,000개 로드 필수 조건)
                            data_store[c][t] = df[c].astype('float32')
                except:
                    continue
            
            # 매트릭스 변환 (Strict Alignment)
            print("  📐 Aligning Base Matrices...")
            for c in price_cols:
                if data_store[c]:
                    df = pd.DataFrame(data_store[c])
                    # [핵심] 기준 Date와 Ticker로 강제 재정렬 (빈 곳은 NaN)
                    # Engine은 (Date x Ticker)를 원하므로 Transpose 안 함 (BacktestEngine.run 로직 기준)
                    # 만약 Engine이 Transpose를 원하면 self.prices[c] = df.reindex(...).T 로 변경해야 함
                    # 여기서는 사용자 코드가 pandas DataFrame(Date index)를 원한다고 가정
                    df = df.reindex(index=self.dates, columns=self.tickers)
                    self.prices[c] = df.astype('float32')
                del data_store[c]

            # Amount 생성
            if 'Close' in self.prices and 'Volume' in self.prices:
                open_p = self.prices.get('Open', self.prices['Close'])
                self.prices['Amount'] = ((open_p + self.prices['Close']) / 2.0 * self.prices['Volume']).astype('float32')

            # Base 캐시 저장
            base_save = {
                'prices': self.prices,
                'tickers': self.tickers,
                'dates': self.dates
            }
            self.cache_manager.save(base_save, base_cache_name)
            gc.collect()

        # ---------------------------------------------------------
        # 2. Feature Data 로딩 (On-Demand from Disk)
        # ---------------------------------------------------------
        # 피처는 캐시하지 않고, 이미 확보된 self.tickers를 이용해 필요한 것만 빠르게 읽습니다.
        # 이렇게 하면 "피처 바뀔 때마다 캐시 다시 만드는" 문제가 해결됩니다.
        
        if required_features:
            print(f"  📥 Loading Features: {required_features}")
            
            # 피처별 임시 저장소
            feat_store = {f: {} for f in required_features}
            
            # 이미 확보된 유니버스(self.tickers)에 대해서만 파일을 엽니다.
            # (전체 디렉토리 스캔 X -> 속도 향상)
            target_paths = [self.platinum_dir / f"{t}.parquet" for t in self.tickers]
            
            for p in tqdm(target_paths, desc="  Reading Features"):
                if not p.exists(): continue
                t = p.stem
                
                try:
                    # 스키마 확인 (대소문자 보정 및 존재 여부 확인)
                    pf = pq.ParquetFile(p)
                    file_cols = set(pf.schema.names)
                    col_map = {c.lower(): c for c in file_cols}
                    
                    read_map = {} # {실제이름: 요청이름}
                    for req in required_features:
                        if req in file_cols:
                            read_map[req] = req
                        elif req.lower() in col_map: # Fuzzy Match
                            read_map[col_map[req.lower()]] = req
                            
                    if not read_map: continue
                    
                    # 읽기
                    df = pd.read_parquet(p, columns=list(read_map.keys()))
                    df.rename(columns=read_map, inplace=True)
                    
                    for req in required_features:
                        if req in df.columns:
                            feat_store[req][t] = df[req].astype('float32')
                            
                except:
                    continue
            
            # 매트릭스 변환 및 정렬
            for f in required_features:
                if feat_store[f]:
                    df = pd.DataFrame(feat_store[f])
                    # Base와 동일한 Shape 강제
                    df = df.reindex(index=self.dates, columns=self.tickers)
                    self.features[f] = df.astype('float32')
                else:
                    print(f"  ⚠️ Feature '{f}' not found. Creating NaN matrix.")
                    self.features[f] = pd.DataFrame(np.nan, index=self.dates, columns=self.tickers).astype('float32')
                
                del feat_store[f]
            
            gc.collect()
            
        print("  ✅ Loading Complete.")