import pandas as pd
import sqlite3
import yfinance as yf
from tqdm import trange
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()
DB_DIR = os.environ.get("DIR_PATH")
DB_PATH = f"{DB_DIR}/equity.db"

def get_stock_list() -> list[str]:
    """DB에서 Russell 3000 티커 리스트를 가져옵니다."""
    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT ticker FROM history_russell_3000_list"
        df = pd.read_sql_query(query, conn)
        return df['ticker'].tolist()
    finally:
        conn.close()

def construct_price_db():
    """daily_prices 테이블을 생성합니다."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # PRIMARY KEY (ticker, date)는 자동으로 인덱스를 생성하며,
    # (ticker, date) 조합의 중복 삽입을 방지합니다.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_prices (
        ticker VARCHAR(10),
        date DATE,
        open NUMERIC(20, 4),
        high NUMERIC(20, 4),
        low NUMERIC(20, 4),
        close NUMERIC(20, 4),
        volume NUMERIC(32, 4),
        PRIMARY KEY (ticker, date)
        FOREIGN KEY (ticker) REFERENCES history_russell_3000_list(ticker)
    )
    """)
    conn.commit()
    conn.close()

def get_ohlcv_data(ticker_list: list[str], chuck_size=100) -> pd.DataFrame:
    """
    yfinance에서 OHLCV 데이터를 다운로드합니다.
    (Metric, Ticker) 구조의 MultiIndex DataFrame을 반환합니다.
    """
    all_ohlcv_chunks = []  # DataFrame 조각을 저장할 리스트
    
    for i in trange(0, len(ticker_list), chuck_size, desc="Downloading OHLCV data"):
        tickers = ticker_list[i:i + chuck_size]
        
        try:
            temp_ohlcv = yf.download(
                tickers,
                period="max",        # 항상 전체 기간 다운로드
                interval="1d",
                auto_adjust=True,  # 수정종가 자동 반영
                threads=True,
                progress=False
                # group_by='ticker' 제거 -> 기본값 (Metric, Ticker) 사용
            )
            
            if not temp_ohlcv.empty:
                all_ohlcv_chunks.append(temp_ohlcv)
        except Exception as e:
            print(f"Error downloading batch {tickers[0]}...{tickers[-1]}: {e}")

    if not all_ohlcv_chunks:
        print("Warning: No data was downloaded.")
        return pd.DataFrame()
        
    # 루프가 끝난 후, 리스트에 모인 DataFrame들을 한 번에 concat
    print("Concatenating all downloaded data...")
    ohlcv = pd.concat(all_ohlcv_chunks, axis=1)
    return ohlcv

def insert_ohlcv_to_db(ohlcv_df: pd.DataFrame):
    """
    (Metric, Ticker) 구조의 DataFrame을 
    (ticker, date, open, ...)의 긴 형식(Long format)으로 변환하여
    DB에 단 한 번의 executemany로 삽입합니다.
    """
    
    if ohlcv_df.empty:
        print("No data to insert.")
        return

    # 1. (Metric, Ticker) 컬럼 -> (Date, Ticker) 인덱스로 변환 (Unpivot)
    #    stack(level=1)은 컬럼의 1레벨(Ticker)을 인덱스로 내립니다.
    print("Transforming data (unpivoting)...")
    df_long = ohlcv_df.stack(level=1).reset_index()
    print(df_long.info())
    
    # 2. 컬럼 이름 정리 (DB 스키마와 일치)
    df_long.rename(columns={
        'Date': 'date', 
        'Ticker': 'ticker',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    # 3. 날짜 형식을 'YYYY-MM-DD' 문자열로 변환 (SQLite 호환)
    df_long['date'] = df_long['date'].dt.strftime('%Y-%m-%d')
    
    # 4. DB에 삽입할 튜플 리스트 생성 (NaN 값이 있는 행은 제외)
    data_tuples = list(df_long[
        ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
    ].dropna().itertuples(index=False, name=None))

    if not data_tuples:
        print("No new valid data to insert.")
        return

    # 5. DB 연결 및 단 한 번의 삽입
    print(f"Connecting to DB and inserting {len(data_tuples):,} total rows...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # INSERT OR IGNORE: PRIMARY KEY(ticker, date)가 중복되면 무시합니다.
        cursor.executemany("""
        INSERT OR IGNORE INTO daily_prices 
            (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, data_tuples)
        
        conn.commit()
        print(f"Insertion complete. {cursor.rowcount:,} new rows added.")
    except Exception as e:
        print(f"Error during bulk insert: {e}")
        conn.rollback()
    finally:
        conn.close()

def check_table_exists() -> bool:
    """daily_prices 테이블이 존재하는지 확인합니다."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        SELECT name FROM sqlite_master WHERE type='table' AND name='daily_prices';
        """)
        result = cursor.fetchone()
        return result is not None
    finally:
        conn.close()
    
def main():
    if not DB_DIR or not os.path.exists(DB_DIR):
        raise Exception("Database directory does not exist or DB_DIR env var is not set.")
    
    # 1. 대상 티커 리스트 가져오기
    stock_list = get_stock_list()
    if not stock_list:
        raise Exception("Failed to get stock list from DB.")

    # 2. 테이블이 없으면 생성
    if not check_table_exists():
        print("daily_prices table not found. Creating table...")
        construct_price_db()
    else:
        print("daily_prices table already exists.")

    # 3. 항상 전체 기간 데이터 다운로드
    # (INSERT OR IGNORE가 중복/최신 데이터를 알아서 처리)
    print(f"Downloading max OHLCV data for {len(stock_list)} tickers...")
    ohlcv = get_ohlcv_data(stock_list)

    if ohlcv.empty:
        print("No data was downloaded. Exiting.")
        return

    # 4. 마지막 행에 NaN이 있다면 (오늘 자 데이터가 아직 집계 중) 제거
    if ohlcv.isnull().iloc[-1].any():
        print("Dropping last row due to NaN values (likely today's incomplete data).")
        ohlcv = ohlcv.iloc[:-1]
    
    # 5. 최적화된 DB 삽입 함수 호출
    insert_ohlcv_to_db(ohlcv)

if __name__ == "__main__":
    main()