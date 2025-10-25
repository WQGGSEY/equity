import pandas as pd
import sqlite3
import os
import time
from dotenv import load_dotenv
import numpy as np
from indicators import * # indicators.py 파일의 모든 함수를 가져옴

# 시각화 및 진행률 라이브러리
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- [신규] 멀티프로세싱 라이브러리 ---
from multiprocessing import Pool
# import os # os는 이미 임포트되어 있음

# .env 파일에서 환경 변수 로드
load_dotenv()

DB_DIR = os.environ.get("DIR_PATH")
DB_PATH = f"{DB_DIR}/equity.db"

def get_ticker_list() -> list[str]:
    """DB에서 Russell 3000 티커 리스트를 가져옵니다."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT ticker FROM history_russell_3000_list"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df['ticker'].tolist()

# --- [신규] 병렬 처리를 위한 워커(Worker) 함수 ---
def process_one_ticker(ticker: str) -> pd.DataFrame | None:
    """
    [워커 함수] 단 하나의 티커를 DB에서 읽어와 모든 지표를 계산합니다.
    멀티프로세싱 Pool의 각 프로세스에서 이 함수가 실행됩니다.
    
    중요: 각 프로세스는 자신만의 DB 연결(conn)을 가져야 합니다.
    """
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT ticker, date, open, high, low, close, volume
    FROM daily_prices
    WHERE ticker = ?
    """
    try:
        # DB에서 데이터 조회
        price_df = pd.read_sql_query(query, conn, params=(ticker,))
        
        if price_df.empty:
            return None
            
        # 지표 계산 함수 호출
        indicator_df = compute_indicators_by_tickers(price_df)
        return indicator_df

    except Exception as e:
        # 오류가 발생해도 전체 작업이 멈추지 않도록 처리
        print(f"\n[오류] 티커 {ticker} 처리 중 문제 발생: {e}")
        return None
    finally:
        # 성공하든 실패하든, 이 프로세스의 DB 연결은 닫습니다.
        conn.close()

# --- [수정] compute_indicators를 병렬 처리 '매니저'로 변경 ---
def compute_indicators(ticker_list: list[str]) -> pd.DataFrame:
    """
    [매니저 함수] 멀티프로세싱 Pool을 사용해 ticker_list를
    여러 코어에 분배하여 병렬로 처리합니다.
    """
    
    # 사용할 CPU 코어 수 (시스템 환경에 맞게 조절)
    # (예: 전체 코어 수 - 2)
    num_processes = max(1, os.cpu_count() - 2 if os.cpu_count() else 4)
    
    print(f"{len(ticker_list)}개 티커의 지표를 {num_processes}개 코어로 병렬 계산합니다...")

    all_indicator_dfs = [] # 결과를 저장할 리스트

    # 멀티프로세싱 풀(Pool) 생성
    with Pool(processes=num_processes) as pool:
        
        # pool.imap: ticker_list의 항목들을 process_one_ticker 함수에 하나씩 보냄
        # tqdm: pool.imap의 진행 상황을 모니터링
        results_iterator = pool.imap(process_one_ticker, ticker_list)
        
        for df in tqdm(results_iterator, total=len(ticker_list), desc="Calculating Indicators"):
            # 워커가 성공적으로 반환한(None이 아닌) DataFrame만 추가
            if df is not None:
                all_indicator_dfs.append(df)

    if not all_indicator_dfs:
        print("\nWarning: No data was processed.")
        return pd.DataFrame()
        
    # 모든 DataFrame을 하나로 합치기
    print("\n모든 티커 데이터 취합 중...")
    return pd.concat(all_indicator_dfs)

def compute_indicators_by_tickers(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    [변경 없음] 단일 티커 DataFrame을 받아 모든 지표를 계산하고 추가합니다.
    """
    price_df['log_return'] = log_return(price_df)

    price_df['rsi_20'] = compute_rsi(price_df, period=20)
    price_df['rsi_120'] = compute_rsi(price_df, period=120)

    price_df['mfi_14'] = compute_mfi(price_df, period=14)

    bayesian_trend_20 = compute_bayesian_trend(price_df, window=20)
    price_df = price_df.join(bayesian_trend_20)

    bayesian_trend_240 = compute_bayesian_trend(price_df, window=240)
    price_df = price_df.join(bayesian_trend_240)

    price_df['bb_width_20'] = compute_bb_width(price_df, window=20)
    price_df['bb_width_120'] = compute_bb_width(price_df, window=120)

    price_df['skewness_20'] = compute_skewness(price_df['log_return'], window=20)
    price_df['skewness_120'] = compute_skewness(price_df['log_return'], window=120)

    price_df['kurtosis_20'] = compute_kurtosis(price_df['log_return'], window=20)
    price_df['kurtosis_120'] = compute_kurtosis(price_df['log_return'], window=120)

    price_df['auto_corr_20'] = compute_rolling_autocorr(price_df['log_return'], window=20)
    price_df['auto_corr_120'] = compute_rolling_autocorr(price_df['log_return'], window=120)

    # (주의: indicators.py에 compute_amihud_indicator 함수가 정의되어 있어야 함)
    price_df['amihud_60'] = compute_amihud_indicator(price_df, window=60) 

    return price_df

def test_correlation_analysis(indicators_df: pd.DataFrame, feature_cols: list[str]):
    """
    [변경 없음] 상관관계 분석 및 히트맵 생성
    """
    clean_df = indicators_df[feature_cols].dropna()
        
    if clean_df.empty:
        print("NaN 값을 제거한 후 분석할 데이터가 없습니다. (윈도우 기간이 너무 길거나 샘플이 작을 수 있습니다)")
    else:
        print(f"\n--- 직교성 분석 (총 {len(clean_df)}개 데이터 포인트) ---")
        
        corr_matrix = clean_df.corr()

        plt.figure(figsize=(24, 20)) 
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            vmin=-1, vmax=1
        )
        plt.title("Feature Correlation Heatmap (Orthogonality Check)", fontsize=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        heatmap_path = f"./feature_correlation_heatmap.png"
        try:
            plt.savefig(heatmap_path)
            print(f"\n[성공] 히트맵이 {heatmap_path} 에 저장되었습니다.")
        except Exception as e:
            print(f"\n[실패] 히트맵 저장 중 오류 발생: {e}")

def test_main():
    """
    [변경 없음] 메인 실행 로직
    """
    tickers = get_ticker_list()
    print(f"총 {len(tickers)}개의 티커를 데이터베이스에서 불러왔습니다.")
    
    
    # 이 함수는 이제 내부적으로 병렬 처리됩니다.
    all_indicators_df = compute_indicators(tickers)

    all_indicators_df.to_parquet(f"{DB_DIR}/indicators.parquet", engine='pyarrow', index=False)

    if all_indicators_df.empty:
        print("처리할 데이터가 없습니다. 종료합니다.")
    else:
        features_to_analyze = [
            'log_return', 'rsi_20', 'rsi_120', 'mfi_14',
            'b_slope_20', 'b_uncert_20', 'b_slope_240', 'b_uncert_240',
            'bb_width_20', 'bb_width_120',
            'skewness_20', 'skewness_120', 'kurtosis_20', 'kurtosis_120',
            'auto_corr_20', 'auto_corr_120', 'amihud_60'
        ]
        test_correlation_analysis(all_indicators_df, features_to_analyze)
        
if __name__ == "__main__":
    # 멀티프로세싱은 __name__ == "__main__": 블록 안에서
    # 실행되어야 안전합니다. (현재 구조가 이미 그렇게 되어 있음)
    test_main()