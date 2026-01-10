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

# (소음 억제기 코드는 그대로 유지)
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

def ingest_bronze():
    print(">>> [Pipeline 02] Bronze 데이터 수집 (Incremental Update)")
    
    if not MASTER_PATH.exists(): return

    df = pd.read_csv(MASTER_PATH, dtype={'ticker': str}, keep_default_na=False)
    today_str = datetime.now().strftime(DATE_FORMAT)
    
    # 필수 컬럼 초기화
    if 'fail_count' not in df.columns: df['fail_count'] = 0
    
    # [핵심 수정] 다운로드 타겟 선정 로직 강화
    # 1. 파일이 없거나 (count == 0)
    # 2. 마지막 업데이트가 오늘이 아닌 경우 (last_updated != today)
    # 3. 단, 상장폐지(is_active=False) 되거나 실패가 너무 많은 건 제외
    
    # last_updated가 NaN이면 다운로드 대상이 됨
    df['last_updated'] = df['last_updated'].fillna('')
    
    mask_needed = (
        (df['count'] == 0) | (df['last_updated'] != today_str)
    )
    mask_valid = (
        (df['is_active'] == True) & 
        (pd.to_numeric(df['fail_count'], errors='coerce').fillna(0) < 5)
    )
    
    targets = df[mask_needed & mask_valid]['ticker'].tolist()
    
    # 정크 필터링
    clean_targets = [t for t in targets if not ("-WT" in str(t).upper() or "WARRANT" in str(t).upper())]
    
    if not clean_targets:
        print("  ✅ 모든 데이터가 최신입니다.")
        return

    print(f"  🎯 업데이트 대상: {len(clean_targets)} 개 (신규 + 구형 데이터)")
    
    # 배치 다운로드 (기존 로직 유지)
    chunks = [clean_targets[i:i + BATCH_SIZE] for i in range(0, len(clean_targets), BATCH_SIZE)]
    success_cnt = 0

    for chunk in tqdm(chunks, desc="Updating"):
        try:
            real_tickers = [t for t in chunk if "_OLD" not in t]
            if not real_tickers: continue

            # 'max'로 받아서 덮어쓰기 (가장 안전하고 확실한 동기화)
            # 데이터 양이 많으면 '1mo' 등으로 줄여서 append 로직을 짤 수도 있음
            with suppress_stdout_stderr():
                data = yf.download(
                    real_tickers, period="max", auto_adjust=True, 
                    group_by='ticker', progress=False, threads=USE_THREADS
                )
            
            if data is None or data.empty:
                # 실패 처리 로직...
                continue

            for t in real_tickers:
                try:
                    sub_df = pd.DataFrame()
                    if len(real_tickers) == 1: sub_df = data
                    elif t in data.columns.levels[0]: sub_df = data[t].copy()
                    
                    sub_df.dropna(how='all', inplace=True)
                    
                    if not sub_df.empty:
                        # 저장
                        safe_ticker = str(t).replace(".", "-").upper()
                        save_dir = BRONZE_DIR / f"ticker={safe_ticker}"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        save_path = save_dir / "price.parquet"
                        sub_df.to_parquet(save_path)
                        
                        # 장부 갱신 (여기서 last_updated를 오늘로 찍음!)
                        idx = df[df['ticker'] == t].index
                        df.loc[idx, 'count'] = len(sub_df)
                        df.loc[idx, 'start_date'] = sub_df.index[0].strftime(DATE_FORMAT)
                        df.loc[idx, 'end_date'] = sub_df.index[-1].strftime(DATE_FORMAT)
                        df.loc[idx, 'last_updated'] = today_str  # <--- [중요] 여기서만 갱신!
                        df.loc[idx, 'fail_count'] = 0
                        success_cnt += 1
                    else:
                        # 실패 카운트 증가
                        idx = df[df['ticker'] == t].index
                        current_fail = df.loc[idx, 'fail_count'].fillna(0).astype(int)
                        df.loc[idx, 'fail_count'] = current_fail + 1
                except: pass
            
            time.sleep(random.uniform(1.0, 2.0))
            
        except Exception: pass

    df.to_csv(MASTER_PATH, index=False)
    print(f"  ✅ 업데이트 완료 (성공: {success_cnt})")

if __name__ == "__main__":
    ingest_bronze()