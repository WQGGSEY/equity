import pandas as pd
import yfinance as yf
from tqdm import trange
import os
from dotenv import load_dotenv
import traceback

# .env 파일에서 환경 변수 로드
load_dotenv()
DB_DIR = os.environ.get("DIR_PATH")
LIST_PATH = f"{DB_DIR}/russell_3000_list.parquet"
FILE_PATH = f"{DB_DIR}/ohlcv.parquet"
# Atomic 연산을 위한 임시 파일 경로
TEMP_PATH = f"{DB_DIR}/ohlcv.parquet.tmp"

def get_stock_list() -> list[str]:
    """
    Russell 3000 종목 리스트를 Parquet 파일에서 로드합니다.
    """
    if not os.path.exists(LIST_PATH):
        print(f"Error: Stock list file does not exist at {LIST_PATH}")
        return []
    
    df = pd.read_parquet(LIST_PATH)
    return df['ticker'].tolist()

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
            )
            
            if not temp_ohlcv.empty:
                all_ohlcv_chunks.append(temp_ohlcv)
        except Exception as e:
            print(f"Error downloading batch {tickers[0]}...{tickers[-1]}: {e}")

    if not all_ohlcv_chunks:
        print("Warning: No data was downloaded.")
        return pd.DataFrame()
        
    print("Concatenating all downloaded data...")
    ohlcv = pd.concat(all_ohlcv_chunks, axis=1)
    return ohlcv
    
def main():
    if not DB_DIR or not os.path.exists(DB_DIR):
        raise Exception("Database directory does not exist or DB_DIR env var is not set.")
    
    try:
        stock_list = get_stock_list()
        if not stock_list:
            raise Exception(f"Failed to get stock list from {LIST_PATH}")

        print(f"Downloading max OHLCV data for {len(stock_list)} tickers...")
        ohlcv = get_ohlcv_data(stock_list) # (Metric, Ticker) Wide Format

        if ohlcv.empty:
            print("No data was downloaded. Exiting without changes.")
            return

        if ohlcv.isnull().iloc[-1].any():
            print("Dropping last row due to NaN values (likely today's incomplete data).")
            ohlcv = ohlcv.iloc[:-1]
        
        # --- [핵심 수정] ---
        # "Wide" 포맷을 "Long" 포맷으로 변환합니다.
        # (이 로직은 사용자의 원본 SQLite 스크립트에 있었습니다.)
        print("Transforming data (unpivoting to Long Format)...")
        
        # 1. (Metric, Ticker) 컬럼 -> (Date, Ticker) 인덱스로 변환
        df_long = ohlcv.stack(level=1).reset_index()
        
        # 2. 컬럼 이름 정리 (preprocessing.py가 사용할 이름과 일치)
        df_long.rename(columns={
            'Date': 'date', 
            'Ticker': 'ticker',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        # 3. 날짜 형식 변환 (Parquet 호환성)
        df_long['date'] = pd.to_datetime(df_long['date'])
        
        # 4. DB에 삽입할 컬럼만 선택
        df_to_save = df_long[
            ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        ]
        # --- [수정 완료] ---

        # --- Atomic Write 시작 (이제 df_long을 저장) ---
        print(f"Writing {len(df_to_save):,} rows to temporary file: {TEMP_PATH}...")
        
        # [수정] ohlcv 대신 df_to_save를 저장, index=False로 변경
        df_to_save.to_parquet(TEMP_PATH, index=False, engine='pyarrow')
        
        print(f"Atomically replacing {FILE_PATH}...")
        os.replace(TEMP_PATH, FILE_PATH)
        
        # print(f"Successfully updated {FILE_PATH}.") # 성공 출력 제거

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"Error during operation: {e}")
        print(traceback.format_exc())
        print("Operation rolled back. Original file (if any) is preserved.")
        print("---------------")
    
    finally:
        if os.path.exists(TEMP_PATH):
            # print(f"Cleaning up temporary file: {TEMP_PATH}") # 성공 출력 제거
            os.remove(TEMP_PATH)

if __name__ == "__main__":
    main()