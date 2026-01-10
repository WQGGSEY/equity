import pandas as pd
import numpy as np
from tqdm import tqdm
from src.config import *

def transform_silver():
    print(">>> [Phase 3] Silver Transformer (Standardization)")
    
    if not MASTER_PATH.exists(): return
    
    # NAN 문자열 보호
    df_master = pd.read_csv(MASTER_PATH, dtype={'ticker': str}, keep_default_na=False)
    targets = df_master[df_master['count'] > 0]
    
    print(f"  🔨 변환 대상: {len(targets)} 개")
    success_cnt = 0

    if SILVER_DIR.exists():
        pass # 전체 재생성을 원하면 여기서 삭제 로직 추가 가능

    for _, row in tqdm(targets.iterrows(), total=len(targets), desc="Standardizing"):
        ticker = str(row['ticker']).strip()
        if ticker.lower() == 'nan': ticker = 'NAN'

        if pd.isna(row['file_path']): continue
        bronze_path = BASE_DIR / str(row['file_path'])
        
        if not bronze_path.exists(): continue

        try:
            df = pd.read_parquet(bronze_path)
            if df.empty: continue

            # [NAN 처리]
            if ticker == 'NAN':
                if len(df.columns) == 6:
                    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                elif len(df.columns) == 5:
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            # [스키마 표준화]
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

            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
            
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)

            safe_ticker = ticker.replace(".", "-").upper()
            save_path = SILVER_DIR / f"{safe_ticker}.parquet"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(save_path)
            success_cnt += 1

        except: pass

    print(f"  ✅ Silver 변환 완료: {success_cnt} 개")

if __name__ == "__main__":
    transform_silver()