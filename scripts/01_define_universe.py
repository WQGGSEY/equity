import pandas as pd
import requests
import shutil
import sys
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# 프로젝트 루트 경로 설정 (src.config 임포트용)
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import MASTER_PATH, BRONZE_DIR

# 저장소 위치 정의 (Script 전용)
# Yahoo 폴더는 Config의 BRONZE_DIR과 동일
KAGGLE_DIR = BASE_DIR / "data" / "bronze" / "daily_prices" # Kaggle legacy

# SEC 및 NASDAQ 소스
SEC_HEADERS = {'User-Agent': 'Individual_Researcher my_email@example.com'}
SEC_URL = "https://www.sec.gov/files/company_tickers.json"
NASDAQ_URL = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt"

def normalize_ticker(ticker):
    return str(ticker).strip().upper().replace(".", "-")

def get_new_tickers_from_web():
    """웹에서 최신 종목 리스트 수집"""
    found_tickers = set()
    print("  📡 신규 종목 리스트 수집 중 (SEC/NASDAQ)...")
    try:
        resp = requests.get(SEC_URL, headers=SEC_HEADERS, timeout=5)
        if resp.status_code == 200:
            df = pd.DataFrame.from_dict(resp.json(), orient='index')
            found_tickers.update(df['ticker'].apply(normalize_ticker).tolist())
    except: pass

    try:
        df = pd.read_csv(NASDAQ_URL, sep="|")
        ts = df[df['Test Issue'] == 'N']['Symbol'].dropna().apply(normalize_ticker).tolist()
        found_tickers.update(ts)
    except: pass
        
    return found_tickers

def scan_and_resolve_conflicts():
    """Yahoo(신규)와 Kaggle(구형) 데이터 충돌 자동 해결"""
    inventory = {} 
    
    # 1. Yahoo 폴더 스캔 (우선순위 1위)
    if BRONZE_DIR.exists():
        for p in tqdm(list(BRONZE_DIR.glob("ticker=*")), desc="Scanning Yahoo"):
            if not p.is_dir(): continue
            ticker = p.name.split("=")[-1]
            ticker = normalize_ticker(ticker)
            
            file_path = p / "price.parquet"
            if file_path.exists():
                inventory[ticker] = {
                    'source': 'yahoo',
                    'file_path': str(file_path.relative_to(BASE_DIR)),
                    'is_active': True
                }

    # 2. Kaggle 폴더 스캔 (우선순위 2위)
    if KAGGLE_DIR.exists():
        for p in tqdm(list(KAGGLE_DIR.glob("ticker=*")), desc="Scanning Kaggle"):
            if not p.is_dir(): continue
            original_ticker = p.name.split("=")[-1]
            original_ticker = normalize_ticker(original_ticker)
            
            file_path = p / "price.parquet"
            if not file_path.exists(): continue

            if original_ticker in inventory:
                new_ticker_name = f"{original_ticker}_OLD"
                inventory[new_ticker_name] = {
                    'source': 'kaggle_legacy',
                    'file_path': str(file_path.relative_to(BASE_DIR)),
                    'is_active': True
                }
            else:
                inventory[original_ticker] = {
                    'source': 'kaggle',
                    'file_path': str(file_path.relative_to(BASE_DIR)),
                    'is_active': True
                }
                
    return inventory

def main():
    print(">>> [Script 01] Universe 정의 및 초기화 (Pipeline 호환)")
    
    # 1. 로컬 스캔
    local_data = scan_and_resolve_conflicts()
    print(f"  ✅ 로컬 파일 식별: {len(local_data)} 개")

    # 2. 기존 장부 백업
    old_meta = {}
    if MASTER_PATH.exists():
        backup = MASTER_PATH.parent / "master_ticker_list_backup.csv"
        shutil.copy2(MASTER_PATH, backup)
        df_old = pd.read_csv(MASTER_PATH)
        for _, row in df_old.iterrows():
            old_meta[row['ticker']] = row.to_dict()

    # 3. 데이터 통합
    final_rows = []
    today = datetime.now().strftime("%Y-%m-%d")
    
    # [A] 로컬 파일
    for ticker, info in local_data.items():
        row = {
            'ticker': ticker,
            'source': info['source'],
            'is_active': info['is_active'],
            'file_path': info['file_path'],
            'count': 0,
            'start_date': None,
            'end_date': None,
            'last_updated': today,
            # [Pipeline 호환 필드 추가]
            'fail_count': 0,
            'last_failed_date': None,
            'note': 'Initialized from Local'
        }
        
        # 메타데이터 복원
        if ticker in old_meta:
            prev = old_meta[ticker]
            if str(prev.get('file_path')) == str(info['file_path']):
                row.update({k: v for k, v in prev.items() if k in row})

        final_rows.append(row)

    # [B] 웹 신규
    web_tickers = get_new_tickers_from_web()
    existing = set(local_data.keys())
    
    cnt_new = 0
    for t in web_tickers:
        if t not in existing and f"{t}_OLD" not in existing:
            if "$" in t: continue
            final_rows.append({
                'ticker': t,
                'source': 'new_ipo',
                'is_active': True,
                'file_path': f"data/bronze/yahoo_price_data/ticker={t}/price.parquet",
                'count': 0,
                'start_date': None,
                'end_date': None,
                'last_updated': None, # 다운로드 대상 마킹
                'fail_count': 0,
                'last_failed_date': None,
                'note': 'Discovered from Web'
            })
            cnt_new += 1
            
    print(f"  🔍 웹 신규 추가: {cnt_new} 개")

    # 4. 저장 및 파일 메타데이터 갱신
    df_final = pd.DataFrame(final_rows)
    
    print("  📝 메타데이터(행 개수) 갱신 중...")
    for idx, row in tqdm(df_final.iterrows(), total=len(df_final)):
        if row['count'] == 0 and row['source'] != 'new_ipo':
            full_path = BASE_DIR / str(row['file_path'])
            if full_path.exists():
                try:
                    meta = pd.read_parquet(full_path, columns=['Close'])
                    df_final.at[idx, 'count'] = len(meta)
                    df_final.at[idx, 'start_date'] = meta.index[0].strftime("%Y-%m-%d")
                    df_final.at[idx, 'end_date'] = meta.index[-1].strftime("%Y-%m-%d")
                except: pass

    df_final.to_csv(MASTER_PATH, index=False)
    print("  ✅ 초기화 완료. (Master List Saved)")

if __name__ == "__main__":
    main()