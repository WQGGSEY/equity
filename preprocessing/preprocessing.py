import pandas as pd
import os
import time
from dotenv import load_dotenv
import numpy as np
import shutil  # 임시 폴더 삭제용
import traceback  # 오류 스택 트레이스 출력용

# --- 멀티프로세싱 및 Dask 관련 ---
from multiprocessing import Pool, cpu_count
import dask.dataframe as dd
import pyarrow.parquet as pq
import pyarrow as pa

# --- 시각화 및 진행률 라이브러리 ---
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- [수정] 수학 라이브러리 ---
import numpy.linalg # 조건수 계산용

# --- indicators.py의 함수들을 임포트 (파일이 존재한다고 가정) ---
# (테스트용 더미 함수)
def log_return(df): return np.log(df['close'] / df['close'].shift(1))
def compute_rsi(df, period): return pd.Series(np.random.rand(len(df)), index=df.index)
def compute_mfi(df, period): return pd.Series(np.random.rand(len(df)), index=df.index)
def compute_bayesian_trend(df, window):
    return pd.DataFrame({
        f'b_slope_{window}': np.random.rand(len(df)),
        f'b_uncert_{window}': np.random.rand(len(df))
    }, index=df.index)
def compute_bb_width(df, window): return pd.Series(np.random.rand(len(df)), index=df.index)
def compute_skewness(s, window): return s.rolling(window).skew()
def compute_kurtosis(s, window): return s.rolling(window).kurt()
def compute_rolling_autocorr(s, window): return s.rolling(window).corr(s.shift(1))
def compute_amihud_indicator(df, window): return pd.Series(np.random.rand(len(df)), index=df.index)
# ---

load_dotenv()
DB_DIR = os.environ.get("DIR_PATH", "./db")

# --- Parquet 파일 경로 설정 ---
LIST_PATH = os.path.join(DB_DIR, "russell_3000_list.parquet")
OHLCV_PATH = os.path.join(DB_DIR, "ohlcv.parquet")
TEMP_CACHE_DIR = os.path.join(DB_DIR, "temp_indicator_cache")
OUTPUT_DIR = os.path.join(DB_DIR, "indicators_partitioned")
HEATMAP_PATH = os.path.join(DB_DIR, "feature_correlation_heatmap_sampled.png")
# ---

def get_ticker_list() -> list[str]:
    """Russell 3000 티커 리스트를 Parquet 파일에서 가져옵니다."""
    if not os.path.exists(LIST_PATH):
        print(f"[CRITICAL_ERROR] Ticker list file not found at: {LIST_PATH}")
        return []
    try:
        df = pd.read_parquet(LIST_PATH)
        tickers = df['ticker'].dropna().unique().tolist()
        if not tickers:
            print(f"[CRITICAL_ERROR] Ticker list file is empty or contains no valid tickers.")
            return []
        return tickers
    except Exception as e:
        print(f"[CRITICAL_ERROR] Failed to read ticker list from {LIST_PATH}: {e}")
        print(traceback.format_exc())
        return []

def compute_indicators_by_tickers(price_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """단일 티커 DataFrame을 받아 모든 지표를 계산합니다. (오류 처리 강화)"""
    
    if 'date' not in price_df.columns or 'close' not in price_df.columns:
        print(f"[ERROR:{ticker}] DataFrame is missing 'date' or 'close' column. Skipping.")
        return pd.DataFrame() 

    if not pd.api.types.is_datetime64_any_dtype(price_df['date']):
        price_df['date'] = pd.to_datetime(price_df['date'])
    price_df = price_df.sort_values(by='date').reset_index(drop=True)

    try:
        price_df['year'] = price_df['date'].dt.year
    except Exception as e:
        print(f"[ERROR:{ticker}] Failed to create 'year' column: {e}")
        price_df['year'] = np.nan # 실패 시 NaN으로 채움
    
    try:
        price_df['log_return'] = log_return(price_df)
    except Exception as e:
        print(f"[ERROR:{ticker}] Failed 'log_return': {e}")
        price_df['log_return'] = np.nan 

    # (...이하 다른 지표 계산 함수들은 이전과 동일하게 try-except로 래핑...)
    try: price_df['rsi_20'] = compute_rsi(price_df, period=20)
    except Exception as e: price_df['rsi_20'] = np.nan; print(f"[ERROR:{ticker}] Failed 'rsi_20': {e}")
    try: price_df['rsi_120'] = compute_rsi(price_df, period=120)
    except Exception as e: price_df['rsi_120'] = np.nan; print(f"[ERROR:{ticker}] Failed 'rsi_120': {e}")
    try: price_df['mfi_14'] = compute_mfi(price_df, period=14)
    except Exception as e: price_df['mfi_14'] = np.nan; print(f"[ERROR:{ticker}] Failed 'mfi_14': {e}")
    try: price_df = price_df.join(compute_bayesian_trend(price_df, window=20))
    except Exception as e: price_df['b_slope_20'] = np.nan; price_df['b_uncert_20'] = np.nan; print(f"[ERROR:{ticker}] Failed 'bayesian_trend_20': {e}")
    try: price_df = price_df.join(compute_bayesian_trend(price_df, window=240))
    except Exception as e: price_df['b_slope_240'] = np.nan; price_df['b_uncert_240'] = np.nan; print(f"[ERROR:{ticker}] Failed 'bayesian_trend_240': {e}")
    try: price_df['bb_width_20'] = compute_bb_width(price_df, window=20)
    except Exception as e: price_df['bb_width_20'] = np.nan; print(f"[ERROR:{ticker}] Failed 'bb_width_20': {e}")
    try: price_df['bb_width_120'] = compute_bb_width(price_df, window=120)
    except Exception as e: price_df['bb_width_120'] = np.nan; print(f"[ERROR:{ticker}] Failed 'bb_width_120': {e}")

    if 'log_return' in price_df.columns:
        log_ret_filled = price_df['log_return'].ffill().bfill() 
        if not log_ret_filled.isnull().all():
            try: price_df['skewness_20'] = compute_skewness(log_ret_filled, window=20)
            except Exception as e: price_df['skewness_20'] = np.nan; print(f"[ERROR:{ticker}] Failed 'skewness_20': {e}")
            try: price_df['skewness_120'] = compute_skewness(log_ret_filled, window=120)
            except Exception as e: price_df['skewness_120'] = np.nan; print(f"[ERROR:{ticker}] Failed 'skewness_120': {e}")
            try: price_df['kurtosis_20'] = compute_kurtosis(log_ret_filled, window=20)
            except Exception as e: price_df['kurtosis_20'] = np.nan; print(f"[ERROR:{ticker}] Failed 'kurtosis_20': {e}")
            try: price_df['kurtosis_120'] = compute_kurtosis(log_ret_filled, window=120)
            except Exception as e: price_df['kurtosis_120'] = np.nan; print(f"[ERROR:{ticker}] Failed 'kurtosis_120': {e}")
            try: price_df['auto_corr_20'] = compute_rolling_autocorr(log_ret_filled, window=20)
            except Exception as e: price_df['auto_corr_20'] = np.nan; print(f"[ERROR:{ticker}] Failed 'auto_corr_20': {e}")
            try: price_df['auto_corr_120'] = compute_rolling_autocorr(log_ret_filled, window=120)
            except Exception as e: price_df['auto_corr_120'] = np.nan; print(f"[ERROR:{ticker}] Failed 'auto_corr_120': {e}")
        else:
            print(f"[WARNING:{ticker}] 'log_return' is all NaN after fill. Skipping rolling stats.")
    else:
         print(f"[WARNING:{ticker}] 'log_return' column not found. Skipping rolling stats.")

    try: price_df['amihud_60'] = compute_amihud_indicator(price_df, window=60)
    except Exception as e: price_df['amihud_60'] = np.nan; print(f"[ERROR:{ticker}] Failed 'amihud_60': {e}")

    return price_df

def process_one_ticker(ticker: str) -> bool:
    """[워커 함수] Parquet 필터링, 지표 계산, 임시 파일 저장"""
    try:
        price_df = pd.read_parquet(
            OHLCV_PATH,
            filters=[('ticker', '==', ticker)],
            engine='pyarrow'
        )
        if price_df.empty:
            return False

        indicator_df = compute_indicators_by_tickers(price_df, ticker)
        if indicator_df.empty:
             return False

        output_file = os.path.join(TEMP_CACHE_DIR, f"{ticker}.parquet")
        indicator_df.to_parquet(output_file, engine='pyarrow', index=False)
        return True
    except Exception as e:
        print(f"\n--- [CRITICAL_WORKER_ERROR] Ticker: {ticker} ---")
        print(f"Error: {e}")
        print(traceback.format_exc())
        print("---------------------------------------------------\n")
        return False

def compute_indicators_parallel(ticker_list: list[str]) -> int:
    """[매니저 함수] 멀티프로세싱 Pool 실행"""
    num_processes = max(1, cpu_count() - 2 if cpu_count() else 4)

    success_count = 0
    with Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(process_one_ticker, ticker_list)
        for success in tqdm(results_iterator, total=len(ticker_list), desc="Calculating & Saving Temp Files"):
            if success:
                success_count += 1
    return success_count


def main():
    start_time_main = time.time()

    # --- 1. 임시 캐시 폴더 생성/비우기 ---
    try:
        if os.path.exists(TEMP_CACHE_DIR):
            shutil.rmtree(TEMP_CACHE_DIR)
        os.makedirs(TEMP_CACHE_DIR)
    except Exception as e:
        print(f"[CRITICAL_ERROR] Failed to create or clear temp directory: {e}")
        print(traceback.format_exc())
        exit() 

    # --- 2. 티커 리스트 로드 ---
    tickers = get_ticker_list()
    if not tickers:
        print("티커 목록을 가져올 수 없습니다. 스크립트를 종료합니다.")
        exit()

    # --- 3. 병렬 계산 함수 호출 ---
    successful_tickers = compute_indicators_parallel(tickers)

    # --- 4. Dask 통합 및 파티셔닝 저장 ---
    if successful_tickers > 0:
        try:
            temp_files_pattern = os.path.join(TEMP_CACHE_DIR, "*.parquet")
            ddf = dd.read_parquet(temp_files_pattern, engine='pyarrow')

            if 'date' in ddf.columns:
                ddf['date'] = dd.to_datetime(ddf['date'])
            else:
                 raise ValueError("'date' column not found in temporary files. Cannot partition.")
            
            if 'year' in ddf.columns:
                 # 파티셔닝 컬럼은 정수형(integer)이 안정적입니다.
                 ddf['year'] = ddf['year'].astype(int)
            else:
                 raise ValueError("'year' column not found. Cannot partition by year.")

            start_save = time.time()
            ddf.to_parquet(
                OUTPUT_DIR,
                engine='pyarrow',
                partition_on=['year'], 
                overwrite=True,
                schema='infer'
            )

        except Exception as dask_e:
            print(f"\n[CRITICAL_ERROR] Dask 처리 또는 최종 저장 중 오류 발생: {dask_e}")
            print(traceback.format_exc())
        finally:
            # --- 6. 임시 캐시 폴더 삭제 ---
            try:
                if os.path.exists(TEMP_CACHE_DIR):
                    shutil.rmtree(TEMP_CACHE_DIR)
            except Exception as e:
                print(f"[WARNING] Failed to clean up temp directory: {e}")
    else:
        if os.path.exists(TEMP_CACHE_DIR):
            try:
                shutil.rmtree(TEMP_CACHE_DIR)
            except Exception as e:
                 print(f"[WARNING] Failed to clean up empty temp directory: {e}")


# --- 메인 실행 로직 (성공 출력 제거) ---
if __name__ == "__main__":
    main()