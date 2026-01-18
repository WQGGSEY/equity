import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from pathlib import Path
from .base import GlobalFeature
from ..config import DATA_DIR

# 경로 구조 변경: interim/features/{FeatureName}/{Ticker}.parquet
INTERIM_FEATURE_DIR = DATA_DIR / "interim" / "features"

class CrossSectionalBase(GlobalFeature):
    """GlobalFeature 공통 로직 (Split File Read 방식)"""
    
    def get_my_file_path(self, ticker):
        # Class Name 폴더 아래 Ticker 파일 (예: interim/features/SectorGroup/AAPL.parquet)
        return INTERIM_FEATURE_DIR / self.__class__.__name__ / f"{ticker}.parquet"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """[Local] 잘게 쪼개진 전역 파일에서 내 정보만 로드"""
        if 'ticker' not in df.columns:
            return df
        
        ticker = df['ticker'].iloc[0]
        file_path = self.get_my_file_path(ticker)
        
        if not file_path.exists():
            return df

        try:
            # 1. 작은 파일 로드 (속도 매우 빠름, 수 KB 수준)
            series_df = pd.read_parquet(file_path)
            
            # 2. 컬럼명 설정 및 병합
            col_name = self.params.get('output_name', self._default_name())
            
            if not series_df.empty:
                # 저장된 파일의 첫 번째 컬럼을 가져옴
                series = series_df.iloc[:, 0]
                series.name = col_name
                
                # 3. 병합 (Left Join)
                df = df.join(series, how='left')
                
                # 4. 결측치 처리 (-1)
                df[col_name] = df[col_name].fillna(-1).astype(np.int8)
        except Exception:
            pass
            
        return df

    def _default_name(self):
        return "grp_" + self.__class__.__name__.replace("Group", "").lower()

class SectorGroup(CrossSectionalBase):
    """ETF 상관계수 기반 섹터 분류"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.window = self.params.get('window', 126)
        self.SECTOR_MAP = {
            'XLK': 0, 'XLF': 1, 'XLV': 2, 'XLY': 3, 'XLP': 4, 'XLE': 5,
            'XLC': 6, 'XLI': 7, 'XLB': 8, 'XLRE': 9, 'XLU': 10
        }
        self.MARKET_TICKER = 'SPY'

    def _download_benchmarks(self, current_cols):
        needed = list(self.SECTOR_MAP.keys()) + [self.MARKET_TICKER]
        missing = [t for t in needed if t not in current_cols]
        if missing:
            try:
                data = yf.download(missing, period="max", progress=False, threads=True)['Close']
                if hasattr(data, 'tz_localize') and data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                elif isinstance(data, pd.DataFrame) and data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                return data
            except: pass
        return pd.DataFrame()

    def compute_global(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
        print(f"  🏗️ [SectorGroup] Calculating Correlations (Window={self.window})...")
        bench_df = self._download_benchmarks(prices.columns)
        if not bench_df.empty:
            prices = prices.combine_first(bench_df)
            
        returns = prices.pct_change(fill_method=None)
        
        group_ids = pd.DataFrame(-1, index=prices.index, columns=prices.columns, dtype=np.int8)
        max_corrs = pd.DataFrame(-1.0, index=prices.index, columns=prices.columns, dtype=np.float32)
        
        for etf, idx in tqdm(self.SECTOR_MAP.items(), desc="    Correlating"):
            if etf not in returns.columns: continue
            
            y = returns[etf]
            # Vectorized Rolling Correlation
            xy = returns.multiply(y, axis=0)
            
            xy_mean = xy.rolling(self.window, min_periods=60).mean()
            x_mean = returns.rolling(self.window, min_periods=60).mean()
            y_mean = y.rolling(self.window, min_periods=60).mean()
            
            cov = xy_mean - x_mean.multiply(y_mean, axis=0)
            x_std = returns.rolling(self.window, min_periods=60).std()
            y_std = y.rolling(self.window, min_periods=60).std()
            
            corr = cov.div(x_std.multiply(y_std, axis=0)).fillna(-1.0)
            corr = corr.astype(np.float32)
            
            mask = (corr > max_corrs)
            np.putmask(group_ids.values, mask.values, idx)
            np.putmask(max_corrs.values, mask.values, corr.values)
            
        return group_ids

class LiquidityGroup(CrossSectionalBase):
    """거래대금 유동성 순위"""
    def compute_global(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
        print("  🏗️ [LiquidityGroup] Calculating Ranks...")
        window = self.params.get('window', 20)
        n_bins = self.params.get('bins', 10)
        
        amt = prices * volumes
        amt_ma = amt.rolling(window).mean()
        ranks = amt_ma.rank(axis=1, pct=True)
        
        groups = np.floor(ranks * n_bins).clip(0, n_bins - 1).fillna(-1).astype(np.int8)
        return groups