import pandas as pd
import numpy as np
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import MASTER_PATH, SILVER_DIR

def main():
    print(">>> [Script 03] Silver Layer 생성 (NAN Handler 포함)")
    
    if not MASTER_PATH.exists(): return

    # 1. 초기화
    if SILVER_DIR.exists(): shutil.rmtree(SILVER_DIR)
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 장부 로드 (NAN 보존)
    df_master = pd.read_csv(MASTER_PATH, dtype={'ticker': str}, keep_default_na=False)
    
    # count > 0 인 것만
    df_master['count'] = pd.to_numeric(df_master['count'], errors='coerce').fillna(0)
    targets = df_master[df_master['count'] > 0]
    
    print(f"  🔨 변환 대상: {len(targets)} 개")
    success_cnt = 0

    for _, row in tqdm(targets.iterrows(), total=len(targets), desc="Standardizing"):
        ticker = str(row['ticker']).strip()
        if ticker.lower() == 'nan': ticker = 'NAN'

        if pd.isna(row['file_path']): continue
        full_path = BASE_DIR / str(row['file_path'])
        
        if not full_path.exists(): continue

        try:
            df = pd.read_parquet(full_path)
            if df.empty: continue

            # =================================================
            # [Logic Sync] NAN 티커 강제 매핑 (Pipeline과 동일)
            # =================================================
            if ticker == 'NAN':
                if len(df.columns) == 6:
                    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                elif len(df.columns) == 5:
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            # 컬럼 표준화
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            
            new_cols = []
            for c in df.columns:
                col_str = str(c).replace("'", "").replace(")", "").split(", ")[-1]
                new_cols.append(col_str.strip().lower())
            df.columns = new_cols

            rename_map = {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'adj close':'Adj Close', 'volume':'Volume', 'date':'Date'}
            df.rename(columns=rename_map, inplace=True)
            
            if 'Adj Close' in df.columns: df['Close'] = df['Adj Close']

            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required): continue

            df = df[required]
            for c in required: df[c] = df[c].astype('float32')

            # 인덱스 정리
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
            
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)

            safe_ticker = ticker.replace(".", "-").upper()
            save_path = SILVER_DIR / f"{safe_ticker}.parquet"
            df.to_parquet(save_path)
            success_cnt += 1

        except: pass

    print(f"  ✅ Silver 생성 완료: {success_cnt} 개")

if __name__ == "__main__":
    main()