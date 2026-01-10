import pandas as pd
import yfinance as yf
import time
import random
from datetime import datetime
from tqdm import tqdm
from src.config import *

def is_junk_ticker(ticker):
    t = str(ticker).upper()
    if "-WT" in t or "WARRANT" in t: return True
    if len(t) >= 5 and t[-1] in ['W', 'R', 'P', 'U']: return True
    return False

def ingest_bronze():
    print(">>> [Phase 2] Bronze Ingestor (Safe Download)")
    
    if not MASTER_PATH.exists(): return

    df = pd.read_csv(MASTER_PATH, dtype={'ticker': str}, keep_default_na=False)
    
    # 다운로드 대상: count가 0인 종목 (파일 없음)
    df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0)
    targets = df[df['count'] == 0]['ticker'].tolist()
    
    # 필터링
    clean_targets = [t for t in targets if not is_junk_ticker(t)]
    
    if not clean_targets:
        print("  ✅ 다운로드할 신규 종목이 없습니다.")
        return

    print(f"  🎯 다운로드 대상: {len(clean_targets)} 개")
    
    # 배치 다운로드
    chunks = [clean_targets[i:i + BATCH_SIZE] for i in range(0, len(clean_targets), BATCH_SIZE)]
    success_cnt = 0
    today_str = datetime.now().strftime(DATE_FORMAT)

    for chunk in tqdm(chunks, desc="Downloading"):
        try:
            # GM_OLD 등 가상 티커 제외
            real_tickers = [t for t in chunk if "_OLD" not in t]
            if not real_tickers: continue

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
                    
                    if sub_df.empty or sub_df.isnull().all().all(): continue
                    sub_df.dropna(how='all', inplace=True)
                    
                    # 인덱스/타임존 정리
                    if not isinstance(sub_df.index, pd.DatetimeIndex):
                        sub_df.reset_index(inplace=True)
                        if 'Date' in sub_df.columns:
                            sub_df['Date'] = pd.to_datetime(sub_df['Date'])
                            sub_df.set_index('Date', inplace=True)
                    if sub_df.index.tz is not None:
                        sub_df.index = sub_df.index.tz_localize(None)
                    sub_df.sort_index(inplace=True)

                    # 저장
                    safe_ticker = str(t).replace(".", "-").upper()
                    save_dir = BRONZE_DIR / f"ticker={safe_ticker}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / "price.parquet"
                    sub_df.to_parquet(save_path)
                    
                    # 장부 갱신 (메모리)
                    idx = df[df['ticker'] == t].index
                    df.loc[idx, 'count'] = len(sub_df)
                    df.loc[idx, 'start_date'] = sub_df.index[0].strftime(DATE_FORMAT)
                    df.loc[idx, 'end_date'] = sub_df.index[-1].strftime(DATE_FORMAT)
                    df.loc[idx, 'last_updated'] = today_str
                    df.loc[idx, 'file_path'] = str(save_path.relative_to(BASE_DIR))
                    success_cnt += 1
                except: pass
            
            time.sleep(random.uniform(1.0, 2.0))
            
        except Exception as e:
            print(f"  ⚠️ Batch Error: {e}")

    df.to_csv(MASTER_PATH, index=False)
    print(f"  ✅ 수집 완료 (성공: {success_cnt})")

if __name__ == "__main__":
    ingest_bronze()