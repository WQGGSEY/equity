import pandas as pd
import yfinance as yf
import time
import random
import sys
import os
from contextlib import contextmanager
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# 프로젝트 루트 경로 설정 (필요시)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

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
    """[강화됨] 워런트 및 잡주 필터"""
    t = str(ticker).upper()
    if "-WT" in t or "WARRANT" in t: return True
    if len(t) >= 5 and t[-1] in ['W', 'R', 'P', 'U']: return True
    return False

def ingest_bronze():
    print(">>> [Pipeline 03] Bronze 데이터 수집 (File Existence Check Applied)")
    
    if not MASTER_PATH.exists(): 
        print("❌ 장부 파일(Master Ticker List)이 없습니다.")
        return

    df = pd.read_csv(MASTER_PATH, dtype={'ticker': str}, keep_default_na=False)
    today_str = datetime.now().strftime(DATE_FORMAT)
    
    # 필수 컬럼 보장
    if 'fail_count' not in df.columns: df['fail_count'] = 0
    if 'last_updated' not in df.columns: df['last_updated'] = ''
    
    # 데이터 타입 변환
    df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0)
    df['fail_count'] = pd.to_numeric(df['fail_count'], errors='coerce').fillna(0)
    
    # [수정] 업데이트 대상 선정 로직 (메타데이터 + 파일 실존 여부)
    targets = []
    print("  🔍 업데이트 및 복구 대상 분석 중...")
    
    for idx, row in df.iterrows():
        # 1. 비활성 또는 실패 과다 종목 제외
        if not row['is_active']: continue
        if row['fail_count'] >= 5: continue
        
        ticker = row['ticker']
        
        # 2. 업데이트 필요 조건 확인
        # (A) 오늘 업데이트 안 됨
        needs_update = str(row['last_updated']) != today_str
        
        # (B) 파일이 실제로 없음 (메타데이터와 무관하게 강제 다운로드)
        safe_ticker = str(ticker).replace(".", "-").upper()
        expected_path = BRONZE_DIR / f"ticker={safe_ticker}" / "price.parquet"
        is_missing = not expected_path.exists()
        
        if needs_update or is_missing:
            targets.append(ticker)

    # 정크 필터링
    clean_targets = [t for t in targets if not is_junk_ticker(t)]
    
    if not clean_targets:
        print("  ✅ 모든 데이터가 최신이며 파일이 존재합니다.")
        return

    print(f"  🎯 수집 대상: {len(clean_targets)} 개 (결측 파일 포함)")
    
    chunks = [clean_targets[i:i + BATCH_SIZE] for i in range(0, len(clean_targets), BATCH_SIZE)]
    success_cnt = 0

    for chunk in tqdm(chunks, desc="Ingesting"):
        try:
            real_tickers = [t for t in chunk if "_OLD" not in t]
            if not real_tickers: continue

            # yfinance 다운로드 (출력 억제)
            with suppress_stdout_stderr():
                data = yf.download(
                    real_tickers, period="max", auto_adjust=True, 
                    group_by='ticker', progress=False, threads=USE_THREADS
                )
            
            if data is None or data.empty:
                # 다운로드 실패 처리
                mask = df['ticker'].isin(real_tickers)
                df.loc[mask, 'fail_count'] += 1
                continue

            for t in real_tickers:
                try:
                    sub_df = pd.DataFrame()
                    if len(real_tickers) == 1: sub_df = data
                    elif t in data.columns.levels[0]: sub_df = data[t].copy()
                    
                    sub_df.dropna(how='all', inplace=True)
                    
                    if not sub_df.empty:
                        # 파일 저장
                        safe_ticker = str(t).replace(".", "-").upper()
                        save_dir = BRONZE_DIR / f"ticker={safe_ticker}"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        save_path = save_dir / "price.parquet"
                        sub_df.to_parquet(save_path)
                        
                        # 메타데이터 업데이트
                        idx = df[df['ticker'] == t].index
                        df.loc[idx, 'count'] = len(sub_df)
                        df.loc[idx, 'start_date'] = sub_df.index[0].strftime(DATE_FORMAT)
                        df.loc[idx, 'end_date'] = sub_df.index[-1].strftime(DATE_FORMAT)
                        df.loc[idx, 'last_updated'] = today_str
                        df.loc[idx, 'fail_count'] = 0
                        success_cnt += 1
                    else:
                        # 빈 데이터 -> 실패 카운트 증가
                        idx = df[df['ticker'] == t].index
                        df.loc[idx, 'fail_count'] += 1
                except:
                    pass
            
            # 차단 방지 딜레이
            time.sleep(random.uniform(1.0, 2.0))
            
        except Exception:
            pass

    df.to_csv(MASTER_PATH, index=False)
    print(f"  ✅ Bronze 수집 완료 (성공: {success_cnt})")

if __name__ == "__main__":
    ingest_bronze()