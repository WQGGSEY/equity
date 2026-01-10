import pandas as pd
import yfinance as yf
import time
import random
import sys
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Config 로드
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import MASTER_PATH, BRONZE_DIR, BATCH_SIZE, USE_THREADS, DATE_FORMAT

def is_junk_ticker(ticker):
    t = str(ticker).upper()
    if "-WT" in t or "WARRANT" in t: return True
    if len(t) >= 5 and t[-1] in ['W', 'R', 'P', 'U']: return True
    return False

def save_single_ticker(df, ticker):
    try:
        safe_ticker = str(ticker).replace(".", "-").upper()
        save_dir = BRONZE_DIR / f"ticker={safe_ticker}"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "price.parquet"
        df.to_parquet(save_path)
        return str(save_path.relative_to(BASE_DIR))
    except:
        return None

def main():
    print(">>> [Script 02] 데이터 다운로드 (Safe Mode & Fail Count 적용)")
    
    if not MASTER_PATH.exists():
        print("❌ 장부 파일이 없습니다.")
        return

    df = pd.read_csv(MASTER_PATH)
    
    # fail_count 컬럼 보장
    if 'fail_count' not in df.columns: df['fail_count'] = 0
    if 'last_failed_date' not in df.columns: df['last_failed_date'] = None
    
    # 다운로드 대상: 데이터가 없거나(count=0) 업데이트가 필요한 경우
    # 여기서는 초기화를 위해 count=0 이고 fail_count < 5 인 것만
    df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0)
    targets = df[
        (df['count'] == 0) & 
        (df['fail_count'] < 5) &
        (df['is_active'] == True)
    ]['ticker'].tolist()
    
    clean_targets = [t for t in targets if not is_junk_ticker(t)]
    print(f"  🎯 다운로드 대상: {len(clean_targets)} 개 (Junk 제외됨)")

    chunks = [clean_targets[i:i + BATCH_SIZE] for i in range(0, len(clean_targets), BATCH_SIZE)]
    today_str = datetime.now().strftime(DATE_FORMAT)
    
    success_cnt = 0
    
    for chunk in tqdm(chunks, desc="Downloading"):
        try:
            # GM_OLD 제외
            real_tickers = [t for t in chunk if "_OLD" not in t]
            if not real_tickers: continue

            # yfinance 다운로드
            data = yf.download(
                real_tickers, period="max", auto_adjust=True, 
                group_by='ticker', progress=False, threads=USE_THREADS
            )
            
            if data is None or data.empty:
                # 전체 실패 처리
                for t in real_tickers:
                    mask = df['ticker'] == t
                    df.loc[mask, 'fail_count'] += 1
                    df.loc[mask, 'last_failed_date'] = today_str
                continue

            for t in real_tickers:
                mask = df['ticker'] == t
                try:
                    sub_df = pd.DataFrame()
                    if len(real_tickers) == 1: sub_df = data
                    elif t in data.columns.levels[0]: sub_df = data[t].copy()
                    
                    sub_df.dropna(how='all', inplace=True)
                    
                    if not sub_df.empty:
                        # 저장 및 메타 갱신
                        rel_path = save_single_ticker(sub_df, t)
                        if rel_path:
                            df.loc[mask, 'count'] = len(sub_df)
                            df.loc[mask, 'start_date'] = sub_df.index[0].strftime(DATE_FORMAT)
                            df.loc[mask, 'end_date'] = sub_df.index[-1].strftime(DATE_FORMAT)
                            df.loc[mask, 'file_path'] = rel_path
                            df.loc[mask, 'last_updated'] = today_str
                            df.loc[mask, 'fail_count'] = 0 # 성공 시 초기화
                            df.loc[mask, 'note'] = 'Downloaded (Script)'
                            success_cnt += 1
                    else:
                        # 빈 데이터 -> 실패 처리
                        df.loc[mask, 'fail_count'] += 1
                        df.loc[mask, 'last_failed_date'] = today_str
                        
                except Exception:
                    df.loc[mask, 'fail_count'] += 1

            # 차단 방지 딜레이
            time.sleep(random.uniform(1.0, 2.0))
            
        except Exception as e:
            print(f"Batch Error: {e}")

    df.to_csv(MASTER_PATH, index=False)
    print(f"  ✅ 다운로드 완료 (성공: {success_cnt})")

if __name__ == "__main__":
    main()