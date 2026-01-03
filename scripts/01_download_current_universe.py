import os
import sys
import time
import random
import contextlib
import pandas as pd
import yfinance as yf
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

# ==========================================
# [설정]
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DATA_DIR = BASE_DIR / "data" / "temp_reference"
MASTER_LIST_PATH = BASE_DIR / "data" / "bronze" / "master_ticker_list.csv"

N_JOBS = 4  
MAX_RETRIES = 3 

# [NEW] 소음기(Silencer) 정의
# yfinance가 내부적으로 뱉는 프린트문과 에러 로그를 억제합니다.
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def get_nasdaq_traded_tickers():
    print(">>> NASDAQ Traded 리스트 다운로드 중...")
    try:
        url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt"
        df = pd.read_csv(url, sep="|")
        df = df[df['Test Issue'] == 'N']
        tickers = df['Symbol'].dropna().astype(str).tolist()
        print(f"  - 확보된 전체 거래 종목 수: {len(tickers)}개")
        return tickers
    except Exception as e:
        print(f"⚠️ NASDAQ 리스트 다운로드 실패: {e}")
        return []

def get_kaggle_tickers():
    if MASTER_LIST_PATH.exists():
        try:
            df = pd.read_csv(MASTER_LIST_PATH)
            return df['ticker'].dropna().astype(str).tolist()
        except Exception as e:
            print(f"⚠️ Kaggle 리스트 로드 실패: {e}")
            return []
    return []

def download_with_retry(ticker):
    """안전장치 + 음소거 적용된 Yahoo Downloader"""
    save_dir = TEMP_DATA_DIR / f"ticker={ticker}"
    save_path = save_dir / "price.parquet"
    
    if save_path.exists():
        return "skipped"

    for attempt in range(MAX_RETRIES):
        try:
            # [NEW] 여기서 소음기를 켭니다.
            # 이 블록 안에서 발생하는 yfinance의 지저분한 에러 출력은 모두 무시됩니다.
            with suppress_output():
                df = yf.download(ticker, period="max", auto_adjust=True, progress=False, threads=False)
            
            if df.empty:
                return "empty"
            
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df.xs(ticker, level=1, axis=1, drop_level=True)
                except:
                    pass
            
            save_dir.mkdir(parents=True, exist_ok=True)
            df.reset_index(inplace=True)
            df.to_parquet(save_path, index=False, compression='snappy')
            return "success"
            
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                # 우리가 잡은 에러는 리턴값으로 조용히 처리
                return "error"
            time.sleep((0.1 * (2 ** attempt)) + random.uniform(0.0, 0.2))
            
    return "failed"

def main():
    print(">>> [Phase 1: Revised v2.2] Quiet Reference Download")
    
    targets = set()
    
    # 리스트 확보
    kaggle_list = get_kaggle_tickers()
    targets.update(kaggle_list)
    
    nasdaq_list = get_nasdaq_traded_tickers()
    targets.update(nasdaq_list)
    
    # 정제
    cleaned_targets = []
    for t in targets:
        s = str(t).strip().upper()
        if s and s != 'NAN' and '$' not in s:
            cleaned_targets.append(s)
            
    target_list = sorted(list(set(cleaned_targets)))
    
    print(f"  - 총 다운로드 대상: {len(target_list)}개")
    print(f"  - 저장 위치: {TEMP_DATA_DIR}")
    print("  - (다운로드 중 발생하는 yfinance 에러 메시지는 숨겨집니다)")
    
    # 병렬 실행
    results = Parallel(n_jobs=N_JOBS)(
        delayed(download_with_retry)(ticker) 
        for ticker in tqdm(target_list, desc="Downloading Universe")
    )
    
    # 결과 집계
    success = results.count("success")
    skipped = results.count("skipped")
    empty = results.count("empty")
    errors = results.count("error") + results.count("failed")
    
    print("\n>>> 요약")
    print(f"  ✅ 저장됨: {success}")
    print(f"  ⏭️ 스킵됨: {skipped}")
    print(f"  📭 데이터 없음: {empty} (상폐/티커변경 등)")
    print(f"  ❌ 에러: {errors}")
    print("완료되면 '02_detect_and_map.py'로 넘어갑니다.")

if __name__ == "__main__":
    main()