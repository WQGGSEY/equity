import pandas as pd
import yfinance as yf
import time
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ==========================================
# [Phase 2] Smart Download (Safe Mode & Filtered)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_PATH = BASE_DIR / "data" / "bronze" / "master_ticker_list.csv"
YAHOO_DATA_DIR = BASE_DIR / "data" / "bronze" / "yahoo_price_data"

# [안전 설정] 속도를 줄여서 차단을 방지함
BATCH_SIZE = 20
USE_THREADS = False  # True면 빠르지만 차단됨 -> False로 안전하게

def normalize_ticker_for_download(ticker):
    return str(ticker).replace(".", "-").upper()

def is_junk_ticker(ticker):
    """
    분석 가치가 없는 워런트(W), 권리(R), 유닛(U), 우선주(P) 등을 필터링
    예: JOBY-WT, AGFSW, HYAC-U
    """
    t = str(ticker).upper()
    
    # 명백한 워런트 표기
    if "-WT" in t or "WARRANT" in t: return True
    
    # 5글자 이상인데 끝자리가 파생상품 코드인 경우
    # (NASDAQ 데이터에서 흔함)
    if len(t) >= 5:
        suffix = t[-1]
        if suffix in ['W', 'R', 'P', 'U', 'Z']: # W:Warrant, R:Right, P:Preferred, U:Unit
            return True
            
    return False

def main():
    print(">>> [Phase 2] 안전모드 다운로드 (Junk Filter + Anti-Ban)")
    
    if not MASTER_PATH.exists():
        print("❌ 장부 파일이 없습니다.")
        return

    # 1. 장부 로드
    df = pd.read_csv(MASTER_PATH)
    
    # 2. 타겟 선정 (Count=0)
    df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0)
    raw_targets = df[df['count'] == 0]['ticker'].tolist()
    
    print(f"  📖 원본 대상: {len(raw_targets)} 개")
    
    # [필터링] 쓰레기 티커 제거
    clean_targets = []
    skipped_junk = 0
    
    for t in raw_targets:
        if is_junk_ticker(t):
            skipped_junk += 1
            # 장부에는 'N/A' 등으로 표시해두면 좋지만, 일단은 건너뜀
        else:
            clean_targets.append(t)
            
    print(f"  🗑️ 파생상품(W/R/U) 제외: {skipped_junk} 개")
    print(f"  🎯 최종 다운로드 대상: {len(clean_targets)} 개")
    
    if not clean_targets:
        print("✅ 다운로드할 종목이 없습니다.")
        return

    # 3. 배치 다운로드
    chunks = [clean_targets[i:i + BATCH_SIZE] for i in range(0, len(clean_targets), BATCH_SIZE)]
    
    success_cnt = 0
    fail_cnt = 0
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    print(f"  🚀 다운로드 시작 (배치: {BATCH_SIZE}, 스레드: {USE_THREADS})")
    
    for chunk in tqdm(chunks, desc="Downloading"):
        try:
            yahoo_tickers = [normalize_ticker_for_download(t) for t in chunk]
            
            # [요청] 에러나면 잠시 대기 후 재시도
            try:
                data = yf.download(
                    yahoo_tickers, 
                    period="max", 
                    auto_adjust=True, 
                    group_by='ticker', 
                    progress=False, 
                    threads=USE_THREADS  # 안전모드
                )
            except Exception as e:
                print(f"  ⚠️ 네트워크/API 에러, 10초 대기... ({e})")
                time.sleep(10)
                continue
            
            if data is None or data.empty:
                fail_cnt += len(chunk)
                continue

            # 결과 처리
            for t_raw, t_yahoo in zip(chunk, yahoo_tickers):
                try:
                    if len(yahoo_tickers) == 1:
                        sub_df = data
                    else:
                        if t_yahoo not in data.columns.levels[0]:
                            fail_cnt += 1
                            continue
                        sub_df = data[t_yahoo].copy()
                    
                    # 유효성 검사
                    if sub_df.isnull().all().all():
                        fail_cnt += 1
                        continue
                    
                    sub_df.dropna(how='all', inplace=True)
                    if sub_df.empty:
                        fail_cnt += 1
                        continue

                    # 인덱스 정리
                    if not isinstance(sub_df.index, pd.DatetimeIndex):
                        sub_df.reset_index(inplace=True)
                        if 'Date' in sub_df.columns:
                            sub_df['Date'] = pd.to_datetime(sub_df['Date'])
                            sub_df.set_index('Date', inplace=True)
                    
                    if sub_df.index.tz is not None:
                        sub_df.index = sub_df.index.tz_localize(None)
                    
                    sub_df.sort_index(inplace=True)

                    # 저장
                    save_dir = YAHOO_DATA_DIR / f"ticker={t_raw}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / "price.parquet"
                    sub_df.to_parquet(save_path)
                    
                    # 장부 업데이트
                    idx = df[df['ticker'] == t_raw].index
                    if not idx.empty:
                        df.loc[idx, 'start_date'] = sub_df.index[0].strftime("%Y-%m-%d")
                        df.loc[idx, 'end_date'] = sub_df.index[-1].strftime("%Y-%m-%d")
                        df.loc[idx, 'count'] = len(sub_df)
                        df.loc[idx, 'file_path'] = str(save_path.relative_to(BASE_DIR))
                        df.loc[idx, 'last_updated'] = today_str
                        df.loc[idx, 'source'] = 'yahoo_new'
                        df.loc[idx, 'is_active'] = True
                    
                    success_cnt += 1
                    
                except Exception:
                    fail_cnt += 1
            
            # Rate Limit 방지를 위한 충분한 휴식
            time.sleep(random.uniform(2.0, 4.0))
            
        except Exception as e:
            print(f"Batch Error: {e}")
            fail_cnt += len(chunk)
            df.to_csv(MASTER_PATH, index=False) # 중간 저장

    # 4. 최종 저장
    df.to_csv(MASTER_PATH, index=False)
    
    print("\n" + "="*40)
    print("  ✅ 완료")
    print(f"  - 성공: {success_cnt}")
    print(f"  - 실패/없음: {fail_cnt}")
    print(f"  - 제외된 Junk: {skipped_junk}")
    print(f"  📂 {MASTER_PATH}")

if __name__ == "__main__":
    main()