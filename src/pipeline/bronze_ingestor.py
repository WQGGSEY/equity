import pandas as pd
import yfinance as yf
import time
import random
import sys
import os
from contextlib import contextmanager
from datetime import datetime
from tqdm import tqdm
from src.config import *

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def is_junk_ticker(ticker):
    """[강화됨] 스크립트 02번과 동일한 강력한 필터"""
    t = str(ticker).upper()
    # 1. 명시적 워런트
    if "-WT" in t or "WARRANT" in t: return True
    # 2. 접미사 패턴 (5글자 이상 & 끝자리가 W, R, P, U)
    if len(t) >= 5 and t[-1] in ['W', 'R', 'P', 'U']: return True
    return False

def ingest_bronze():
    print(">>> [Pipeline 03] Bronze 데이터 수집 (Junk Filter Enhanced)")
    
    if not MASTER_PATH.exists(): return

    df = pd.read_csv(MASTER_PATH, dtype={'ticker': str}, keep_default_na=False)
    today_str = datetime.now().strftime(DATE_FORMAT)
    
    if 'fail_count' not in df.columns: df['fail_count'] = 0
    df['last_updated'] = df['last_updated'].fillna('')
    
    mask_needed = ((df['count'] == 0) | (df['last_updated'] != today_str))
    mask_valid = (
        (df['is_active'] == True) & 
        (pd.to_numeric(df['fail_count'], errors='coerce').fillna(0) < 5)
    )
    
    targets = df[mask_needed & mask_valid]['ticker'].tolist()
    
    # [수정] 함수 기반 필터링으로 교체
    clean_targets = [t for t in targets if not is_junk_ticker(t)]
    
    if not clean_targets:
        print("  ✅ 모든 데이터가 최신입니다.")
        return

    print(f"  🎯 업데이트 대상: {len(clean_targets)} 개")
    
    chunks = [clean_targets[i:i + BATCH_SIZE] for i in range(0, len(clean_targets), BATCH_SIZE)]
    success_cnt = 0

    for chunk in tqdm(chunks, desc="Updating"):
        try:
            real_tickers = [t for t in chunk if "_OLD" not in t]
            if not real_tickers: continue

            with suppress_stdout_stderr():
                data = yf.download(
                    real_tickers, period="max", auto_adjust=True, 
                    group_by='ticker', progress=False, threads=USE_THREADS
                )
            
            if data is None or data.empty: continue

            for t in real_tickers:
                try:
                    sub_df = pd.DataFrame()
                    if len(real_tickers) == 1: sub_df = data
                    elif t in data.columns.levels[0]: sub_df = data[t].copy()
                    
                    sub_df.dropna(how='all', inplace=True)
                    
                    if not sub_df.empty:
                        safe_ticker = str(t).replace(".", "-").upper()
                        save_dir = BRONZE_DIR / f"ticker={safe_ticker}"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        save_path = save_dir / "price.parquet"
                        sub_df.to_parquet(save_path)
                        
                        idx = df[df['ticker'] == t].index
                        df.loc[idx, 'count'] = len(sub_df)
                        df.loc[idx, 'start_date'] = sub_df.index[0].strftime(DATE_FORMAT)
                        df.loc[idx, 'end_date'] = sub_df.index[-1].strftime(DATE_FORMAT)
                        df.loc[idx, 'last_updated'] = today_str
                        df.loc[idx, 'fail_count'] = 0
                        success_cnt += 1
                    else:
                        idx = df[df['ticker'] == t].index
                        df.loc[idx, 'fail_count'] = df.loc[idx, 'fail_count'].fillna(0) + 1
                except: pass
            time.sleep(random.uniform(1.0, 2.0))
        except Exception: pass

    df.to_csv(MASTER_PATH, index=False)
    print(f"  ✅ 업데이트 완료 (성공: {success_cnt})")

if __name__ == "__main__":
    ingest_bronze()