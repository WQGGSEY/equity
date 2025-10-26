import pandas as pd
import numpy as np
import os
import sqlite3
import yfinance as yf
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime, timedelta

# DTW 클러스터링을 위한 tslearn 임포트
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.clustering import TimeSeriesKMeans

# .env 파일에서 환경 변수 로드
load_dotenv()

DB_DIR = os.environ.get("DIR_PATH")
DB_PATH = f"{DB_DIR}/equity.db"
# 1단계(preprocessing.py)에서 생성된 피처맵 경로
FEATURE_PATH = f"{DB_DIR}/indicators.parquet"

def setup_database():
    """
    S&P500 데이터 및 도메인 레이블을 저장할 모든 테이블을 생성합니다.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. S&P 500 시계열 테이블
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sp500 (
        date TEXT PRIMARY KEY UNIQUE,
        open NUMERIC(10, 4),
        high NUMERIC(10, 4),
        low NUMERIC(10, 4),
        close NUMERIC(10, 4),
        volume NUMERIC(32, 0)
    )
    """)

    # 2. 날짜 기반 도메인 레이블 (예: 변동성 체제)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS date_domain_labels (
        date TEXT PRIMARY KEY UNIQUE,
        volatility_domain INTEGER,
        FOREIGN KEY(date) REFERENCES sp500(date)
    )
    """)

    # 3. 주식 기반 도메인 레이블 (예: DTW 클러스터)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_domain_labels (
        ticker TEXT PRIMARY KEY UNIQUE,
        dtw_domain INTEGER,
        FOREIGN KEY(ticker) REFERENCES history_russell_3000_list(ticker)
    )
    """)

    conn.commit()
    conn.close()

def download_and_insert_sp500():
    """S&P 500 데이터를 다운로드하여 DB에 삽입합니다."""
    print("Downloading ^GSPC data...")
    sp500_df = yf.download("^GSPC", period='max')

    if sp500_df.empty:
        print("Failed to download ^GSPC data.")
        return

    sp500_df.reset_index(inplace=True)
    sp500_df['Date'] = sp500_df['Date'].dt.strftime('%Y-%m-%d')
    data_tuples = list(sp500_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].itertuples(index=False, name=None))

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executemany("""
    INSERT OR IGNORE INTO sp500 (date, open, high, low, close, volume)
    VALUES (?, ?, ?, ?, ?, ?)
    """, data_tuples)
    conn.commit()
    conn.close()
    print("S&P 500 data updated.")

def generate_volatility_domains(window=60, low_q=0.25, high_q=0.75):
    """
    2.1단계 (시간적 분할): S&P 500 변동성을 기준으로
    'date_domain_labels' 테이블을 생성/업데이트합니다.
    -1: Low Vol, 0: Mid Vol, 1: High Vol
    """
    print("Generating volatility domain labels...")
    conn = sqlite3.connect(DB_PATH)

    # 1. DB에서 S&P 500 종가 로드
    df = pd.read_sql_query("SELECT date, close FROM sp500 ORDER BY date",
                           conn, parse_dates=['date'])
    if df.empty:
        print("Error: No S&P 500 data found in DB.")
        conn.close()
        return

    df = df.set_index('date')

    # 2. 롤링 변동성 계산
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_return'].rolling(window=window).std()

    # 3. 분위수 계산 (NaN 제외하고 계산)
    valid_volatility = df['volatility'].dropna()
    if valid_volatility.empty:
        print("Error: Volatility calculation resulted in all NaNs.")
        conn.close()
        return

    q_low = valid_volatility.quantile(low_q)
    q_high = valid_volatility.quantile(high_q)

    # 4. 레이블 할당
    conditions = [
        df['volatility'] <= q_low,
        df['volatility'] >= q_high
    ]
    choices = [-1, 1] # Low, High
    df['volatility_domain'] = np.select(conditions, choices, default=0) # Mid

    # 롤링 윈도우로 인해 초기에 NaN이 발생한 구간은 'Mid'(0)으로 채움
    df['volatility_domain'] = df['volatility_domain'].fillna(0).astype(int)

    # 5. DB에 저장
    data_to_insert = list(df.reset_index()[['date', 'volatility_domain']].itertuples(index=False, name=None))
    data_to_insert = [(date.strftime('%Y-%m-%d'), label) for date, label in data_to_insert]

    cursor = conn.cursor()
    cursor.executemany("""
    INSERT OR REPLACE INTO date_domain_labels (date, volatility_domain)
    VALUES (?, ?)
    """, data_to_insert)
    conn.commit()
    conn.close()
    print(f"Volatility domains generated and saved. (Low: -1, Mid: 0, High: 1)")
    # 레이블 분포 출력 (선택적)
    # print("DEBUG: Volatility Domain Label Distribution:")
    # print(df['volatility_domain'].value_counts())


def generate_dtw_domains(n_clusters=8, period_days=252):
    """
    2.2단계 (교차자산 분할): 모든 주식의 최근 1년(252일) 가격 패턴을
    DTW K-Means로 클러스터링하여 'stock_domain_labels' 테이블을 생성합니다.
    (최종 NaN 처리 추가됨)
    """
    print("Generating DTW domain labels (this may take several minutes)...")

    # 1. Parquet 피처맵에서 종가 데이터 로드
    try:
        df_long = pd.read_parquet(FEATURE_PATH, columns=['date', 'ticker', 'close'])
    except FileNotFoundError:
        print(f"Error: '{FEATURE_PATH}' not found.")
        print("Please run 'preprocessing.py' first.")
        return

    df_long['date'] = pd.to_datetime(df_long['date'])

    # 2. 최근 데이터 선택 위한 날짜 계산
    latest_date = df_long['date'].max()
    start_date_approx = latest_date - pd.Timedelta(days=int(period_days * 1.5))
    df_period = df_long[df_long['date'] >= start_date_approx].copy()

    # 3. (날짜 x 티커) 매트릭스로 피벗
    df_pivot = df_period.pivot_table(index='date', columns='ticker', values='close')

    # 4. 기간 내 데이터 부족 주식 제외 (thresh)
    df_pivot = df_pivot.dropna(axis=1, thresh=period_days)

    # 5. 정확히 마지막 period_days 일치 데이터만 선택
    if len(df_pivot) < period_days:
        print(f"Error: Not enough days ({len(df_pivot)}) available after initial filtering. Need {period_days}.")
        return
    df_pivot = df_pivot.iloc[-period_days:]

    if df_pivot.empty or df_pivot.shape[1] == 0:
        print("Error: No stocks left after filtering for sufficient data.")
        return

    # --- [핵심 수정] 최종 윈도우 내 NaN 처리 ---
    print(f"DEBUG: Filling potential NaNs in the final {period_days}-day window (Shape before fill: {df_pivot.shape})...")
    # Forward fill: 이전 값으로 채우기
    df_pivot_filled = df_pivot.fillna(method='ffill')
    # Backward fill: 맨 처음 NaN이 있을 경우 다음 값으로 채우기
    df_pivot_filled = df_pivot_filled.fillna(method='bfill')

    # NaN 처리 후에도 여전히 NaN이 남아있는 컬럼(주식)이 있는지 최종 확인 및 제외
    nan_cols_after_fill = df_pivot_filled.columns[df_pivot_filled.isnull().any()].tolist()
    if nan_cols_after_fill:
        print(f"Warning: {len(nan_cols_after_fill)} stocks still contain NaNs after ffill/bfill. Removing them.")
        df_pivot_filled = df_pivot_filled.drop(columns=nan_cols_after_fill)
        if df_pivot_filled.empty or df_pivot_filled.shape[1] == 0:
            print("Error: No stocks left after final NaN removal.")
            return
    print(f"DEBUG: Shape after NaN fill and final check: {df_pivot_filled.shape}")
    # --- [NaN 처리 완료] ---

    # 이제 df_pivot_filled는 NaN이 없는 상태
    tickers = df_pivot_filled.columns.tolist()

    # 6. tslearn 입력 형식 (n_stocks, n_timesteps)으로 변환 (.T 사용)
    stock_series_np = df_pivot_filled.to_numpy().T

    # 7. 정규화 (MinMax Scaling)
    print(f"Normalizing {stock_series_np.shape[0]} stocks...")
    scaler = TimeSeriesScalerMinMax()
    stock_series_norm = scaler.fit_transform(stock_series_np)
    if stock_series_norm.ndim == 3:
        stock_series_norm = stock_series_norm.squeeze()

    # (이론상 이제 거의 없어야 함) 정규화 후 NaN 발생 확인 (예: 상수 가격 문제)
    nan_mask_after_norm = np.isnan(stock_series_norm).any(axis=1)
    if nan_mask_after_norm.any():
        nan_stock_indices_after_norm = np.where(nan_mask_after_norm)[0]
        num_nan_stocks_after_norm = len(nan_stock_indices_after_norm)
        print(f"Warning: NaNs found after normalization in {num_nan_stocks_after_norm} stocks (likely constant price issue). Removing them.")

        # 정규화 후 NaN 발생 시 해당 주식 제외
        valid_indices_after_norm = ~nan_mask_after_norm
        stock_series_norm = stock_series_norm[valid_indices_after_norm]
        tickers = [tickers[i] for i in np.where(valid_indices_after_norm)[0]] # 티커 리스트 최종 업데이트
        print(f"DEBUG: Proceeding with {len(tickers)} valid stocks after removing normalization NaNs.")
        if not tickers:
            print("Error: No valid stocks left after removing normalization NaNs.")
            return

    # 8. TimeSeriesKMeans (DTW) 실행
    print(f"Running TimeSeriesKMeans (DTW) with k={n_clusters} on {len(tickers)} stocks...")
    dtw_kmeans = TimeSeriesKMeans(n_clusters=n_clusters,
                                  metric='dtw',
                                  n_jobs=-1, # 모든 CPU 코어 사용
                                  random_state=42,
                                  verbose=0) # verbose=1 로 하면 진행 상황 더 자세히 보임

    cluster_labels = dtw_kmeans.fit_predict(stock_series_norm)

    print(f"Clustering finished.")

    # 9. (티커, 클러스터_레이블) 튜플 생성 (최종 업데이트된 tickers 사용)
    # change the cluster_labels to int. Note: if we use astype(int), it will be np.int64, not int
    cluster_labels = [int(label) for label in cluster_labels]
    data_to_insert = list(zip(tickers, cluster_labels))

    print("\nDEBUG: First 5 tuples for stock_domain_labels insertion:")
    for item in data_to_insert[:5]:
        print(f"  Data: {item}, Types: ({type(item[0])}, {type(item[1])})")

    # 10. DB에 저장
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 기존 데이터 삭제 (매번 실행 시 최신 클러스터링 결과로 덮어쓰기)
    print("Clearing previous DTW domain labels...")
    cursor.execute("DELETE FROM stock_domain_labels")
    # 새 데이터 삽입
    print("Inserting new DTW domain labels...")
    cursor.executemany("""
    INSERT INTO stock_domain_labels (ticker, dtw_domain)
    VALUES (?, ?)
    """, data_to_insert)
    conn.commit()
    conn.close()

    print(f"DTW domains generated and saved for {len(data_to_insert)} stocks.")
    print("DEBUG: DTW Domain Label Distribution:")
    print(pd.Series(cluster_labels).value_counts().sort_index())


def main():
    if not os.path.exists(DB_DIR):
        print(f"Creating database directory: {DB_DIR}")
        os.makedirs(DB_DIR)

    # 1. 모든 테이블 스키마 설정
    print("Setting up database tables...")
    setup_database()

    # 2. S&P 500 데이터 다운로드 및 저장
    download_and_insert_sp500()

    # 3. (2.1단계) 변동성 도메인 생성
    generate_volatility_domains()

    # 4. (2.2단계) DTW 클러스터링 도메인 생성
    # (파라미터: 클러스터 개수, 사용할 시계열 길이(거래일))
    generate_dtw_domains(n_clusters=8, period_days=252)
    

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Delete existing stock_domain_labels table if exists
    cursor.execute("DROP TABLE IF EXISTS stock_domain_labels")
    main()