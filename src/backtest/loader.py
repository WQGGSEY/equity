import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
import pyarrow.parquet as pq
# [FIX] CACHE_DIR 직접 Import (data/cache 경로 보장)
from .cache import CacheManager, CACHE_DIR 

class MarketData:
    """
    [Robust Matrix Loader - Clean Cache Version]
    - Data Path: 'data/platinum/features' (Parquet Read)
    - Cache Path: 'data/cache' (Unified Cache Storage) -> Platinum 폴더 오염 방지
    """
    def __init__(self, platinum_dir="data/platinum/features"):
        self.platinum_dir = Path(platinum_dir)
        
        # [FIX] Platinum 경로와 무관하게 고정된 data/cache 경로 사용
        self.root_cache_dir = CACHE_DIR
        self.feature_cache_dir = self.root_cache_dir / "features"
        
        # 폴더 생성 (data/cache/features)
        self.feature_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.prices = {}   
        self.features = {} 
        self.tickers = []
        self.dates = []
        
        # CacheManager도 data/cache 사용
        self.cache_manager = CacheManager(cache_dir=self.root_cache_dir)
        
    def load_all(self, required_features=None):
        # ---------------------------------------------------------
        # 1. Base Data (Prices) 로딩
        # ---------------------------------------------------------
        base_cache_name = "market_data_base" 
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
                # features 폴더가 아닐 경우를 대비해 상위 폴더도 검색 (호환성)
                files = list(self.platinum_dir.parent.glob("features/*.parquet"))
                if not files:
                    raise FileNotFoundError(f"No parquet files found in {self.platinum_dir}")

            # [Step 1] 전체 유니버스 인덱싱
            print("  🧩 Scanning All Files (Indexing)...")
            all_dates = set()
            all_tickers = []
            for p in tqdm(files, desc="  Indexing"):
                try:
                    pf = pq.ParquetFile(p)
                    # 스키마에 Close가 있는지 확인
                    if 'Close' in pf.schema.names:
                        df = pd.read_parquet(p, columns=['Close'])
                        all_dates.update(df.index)
                        all_tickers.append(p.stem)
                except: continue
            
            self.dates = sorted(list(all_dates))
            self.tickers = sorted(all_tickers)
            
            # [Step 2] 가격 데이터 로드
            price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            data_store = {c: {} for c in price_cols}
            
            print(f"  📥 Loading Prices for {len(self.tickers)} tickers...")
            for t in tqdm(self.tickers, desc="  Reading Prices"):
                # 파일 경로 찾기
                p = self.platinum_dir / f"{t}.parquet"
                if not p.exists(): continue
                
                try:
                    df = pd.read_parquet(p, columns=price_cols)
                    for c in price_cols:
                        if c in df.columns: data_store[c][t] = df[c].astype('float32')
                except: continue
            
            # 정렬 및 결측치 보간
            print("  📐 Aligning Base Matrices...")
            for c in price_cols:
                if data_store[c]:
                    df = pd.DataFrame(data_store[c])
                    df = df.reindex(index=self.dates, columns=self.tickers)
                    self.prices[c] = df.astype('float32').ffill() # ffill 적용
                del data_store[c]

            # Amount & Universe 생성
            if 'Close' in self.prices and 'Volume' in self.prices:
                open_p = self.prices.get('Open', self.prices['Close'])
                self.prices['Amount'] = ((open_p + self.prices['Close']) / 2.0 * self.prices['Volume']).astype('float32')
                
                # Universe Mask (동전주 필터 적용)
                print("  🌌 Generating Dynamic Universe Mask (Top 500, Price > $1)...")
                amt = self.prices['Amount']
                close = self.prices['Close']
                valid_price_mask = (close > 1.0)
                rolling_amt = amt.rolling(window=20, min_periods=1).mean()
                filtered_amt = rolling_amt.where(valid_price_mask, 0.0)
                daily_rank = filtered_amt.rank(axis=1, ascending=False, method='min')
                universe_mask = daily_rank.where(daily_rank <= 500, np.nan)
                self.prices['universe'] = universe_mask.where(universe_mask.isna(), 1.0).astype('float32')

            # Base 캐시 저장 (data/cache/...)
            base_save = {'prices': self.prices, 'tickers': self.tickers, 'dates': self.dates}
            self.cache_manager.save(base_save, base_cache_name)
            gc.collect()

        # ---------------------------------------------------------
        # 2. Modular Feature Loading (개별 캐싱 적용)
        # ---------------------------------------------------------
        if required_features:
            self._load_features_modular(required_features)
            
        print("  ✅ Loading Complete.")

    def _load_features_modular(self, required_features):
        """
        필요한 피처만 골라서 로드하고, 없는 것만 파일에서 추출하여 'data/cache/features'에 저장함.
        """
        missing_features = []
        
        # 1. 기존 캐시 확인 및 로드
        print(f"  🔍 Checking Feature Caches: {required_features}")
        for feat in required_features:
            # [FIX] Feature 캐시도 data/cache/features 에서 찾음
            cache_path = self.feature_cache_dir / f"{feat}.parquet"
            if cache_path.exists():
                try:
                    self.features[feat] = pd.read_parquet(cache_path)
                    # 인덱스 정합성 체크
                    if not self.features[feat].index.equals(pd.Index(self.dates)):
                        print(f"    ⚠️ Cache mismatch for {feat}. Re-queuing.")
                        missing_features.append(feat)
                except:
                    missing_features.append(feat)
            else:
                missing_features.append(feat)
                
        if not missing_features:
            print("    -> All features loaded from cache!")
            return

        # 2. 없는 피처(Missing)만 파일에서 추출
        print(f"  📥 Extracting Missing Features: {missing_features}")
        feat_store = {f: {} for f in missing_features}
        
        # 파일 스캔
        for t in tqdm(self.tickers, desc="  Scanning Files"):
            p = self.platinum_dir / f"{t}.parquet"
            if not p.exists(): continue
            try:
                pf = pq.ParquetFile(p)
                file_cols = set(pf.schema.names)
                
                read_map = {}
                col_map_lower = {c.lower(): c for c in file_cols}
                
                for req in missing_features:
                    if req in file_cols:
                        read_map[req] = req
                    elif req.lower() in col_map_lower:
                        read_map[col_map_lower[req.lower()]] = req
                
                if not read_map: continue
                
                df = pd.read_parquet(p, columns=list(read_map.keys()))
                df.rename(columns=read_map, inplace=True)
                
                for req in missing_features:
                    if req in df.columns:
                        feat_store[req][t] = df[req].astype('float32')
            except:
                continue
        
        # 3. DataFrame 변환 및 저장
        print("  💾 Caching New Features to data/cache/features...")
        for f in missing_features:
            if feat_store[f]:
                df = pd.DataFrame(feat_store[f])
                df = df.reindex(index=self.dates, columns=self.tickers).astype('float32')
                
                self.features[f] = df
                
                # [FIX] data/cache/features 에 저장
                save_path = self.feature_cache_dir / f"{f}.parquet"
                df.to_parquet(save_path)
            else:
                print(f"    ⚠️ Feature '{f}' not found in any file. Filling with NaN.")
                df = pd.DataFrame(np.nan, index=self.dates, columns=self.tickers).astype('float32')
                self.features[f] = df
                df.to_parquet(self.feature_cache_dir / f"{f}.parquet")
            
            del feat_store[f]
        
        gc.collect()