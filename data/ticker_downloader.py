import requests
import pandas as pd
from io import StringIO
import os
import re
from dotenv import load_dotenv

load_dotenv()

ishares_russell_3000_url = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"

# (참고) DB_DIR 경로는 실제 환경에 맞게 설정되어 있어야 합니다.
# Colab의 경우: "/content/drive/MyDrive/equity/db"
# 로컬의 경우: "/Users/seongje/Desktop/project/domain shift lab/equity/db"
DB_DIR = os.environ.get("DIR_PATH", "/Users/seongje/Desktop/project/domain shift lab/equity/db")
FILE_PATH = f"{DB_DIR}/russell_3000_list.parquet"
# Atomic 연산을 위한 임시 파일 경로
TEMP_PATH = f"{DB_DIR}/russell_3000_list.parquet.tmp"

def download_ishares_russell_3000() -> pd.DataFrame:
    response = requests.get(ishares_russell_3000_url)
    data = response.content.decode('utf-8')
    with StringIO(data) as f:
        line = f.readlines()
        is_started = False
        data_tuple = []
        for l in line:
            parsed_line = re.findall(r'"(.*?)"', l)
            # print(len(parsed_line), parsed_line)
            if len(parsed_line) == 0 or parsed_line[0] == '-': 
                is_started = True
            
            if is_started:
                try:
                    ticker = parsed_line[0]
                    sector = parsed_line[2]
                    asset_type = parsed_line[3]
                    exchange = parsed_line[10] if len(parsed_line) > 10 else 'N/A'
                    # print(f"Parsed: {ticker}, {sector}, {asset_type}, {exchange}")
                    if asset_type == 'Equity':
                        data_tuple.append((ticker, sector, exchange))
                except IndexError:
                    pass
    return pd.DataFrame(data_tuple, columns=['ticker', 'sector', 'exchange'])

def main():
    if not os.path.exists(DB_DIR):
        print(f"Creating directory: {DB_DIR}")
        os.makedirs(DB_DIR)

    try:
        new_data = download_ishares_russell_3000()
        if new_data.empty:
            print("No new data downloaded. Operation aborted.")
            return
        if os.path.exists(FILE_PATH):
            print(f"Loading old data from {FILE_PATH}...")
            old_data = pd.read_parquet(FILE_PATH)
            print(f"Loaded {len(old_data)} old records.")
            
            # 1. 새 데이터를 먼저 배치하여 결합
            # 2. 'ticker' 기준 중복 제거, 'first' (즉, new_data)를 유지
            combined_data = pd.concat([new_data, old_data]).drop_duplicates(
                subset=['ticker'],
                keep='first'
            )
            print(f"Combined data. Total unique tickers: {len(combined_data)}")
        else:
            print("No old data file found. Using new data.")
            combined_data = new_data

        # --- Atomic Write 시작 ---
        
        # 1. 임시 파일에 먼저 저장
        print(f"Writing data to temporary file: {TEMP_PATH}...")
        combined_data.to_parquet(TEMP_PATH, index=False, engine='pyarrow')

        # 2. 임시 파일 쓰기 성공 시, 원자적(atomic)으로 원본 파일 덮어쓰기
        print(f"Atomically replacing {FILE_PATH}...")
        os.replace(TEMP_PATH, FILE_PATH)
        
        print(f"Successfully updated {FILE_PATH}.")

    except Exception as e:
        print(f"Error during operation: {e}")
        print("Operation rolled back. Original file (if any) is preserved.")
    
    finally:
        # 3. 성공/실패 여부와 관계없이 임시 파일이 남아있다면 삭제
        if os.path.exists(TEMP_PATH):
            print(f"Cleaning up temporary file: {TEMP_PATH}")
            os.remove(TEMP_PATH)

if __name__ == "__main__":
    main()