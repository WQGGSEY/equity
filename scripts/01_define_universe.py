import pandas as pd
import requests
import shutil
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ==========================================
# [Phase 1] Universe Definition + Auto Conflict Resolution
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_PATH = BASE_DIR / "data" / "bronze" / "master_ticker_list.csv"
BACKUP_PATH = BASE_DIR / "data" / "bronze" / "master_ticker_list_backup.csv"

# 저장소 위치 정의
YAHOO_DIR = BASE_DIR / "data" / "bronze" / "yahoo_price_data"
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
    """
    [핵심 로직] 하드디스크를 스캔하여 Yahoo와 Kaggle의 충돌을 자동 해결
    """
    inventory = {} # { 'TICKER': {Info} }
    
    # 1. Yahoo 폴더 스캔 (우선순위 1위 - 정본)
    # Yahoo에 있는 건 무조건 그 이름 그대로 가져감 (예: GM -> GM)
    if YAHOO_DIR.exists():
        for p in tqdm(list(YAHOO_DIR.glob("ticker=*")), desc="Scanning Yahoo (High Priority)"):
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

    # 2. Kaggle 폴더 스캔 (우선순위 2위 - 충돌 시 이름 변경)
    if KAGGLE_DIR.exists():
        for p in tqdm(list(KAGGLE_DIR.glob("ticker=*")), desc="Scanning Kaggle (Legacy Check)"):
            if not p.is_dir(): continue
            original_ticker = p.name.split("=")[-1]
            original_ticker = normalize_ticker(original_ticker)
            
            file_path = p / "price.parquet"
            if not file_path.exists(): continue

            # [자동 충돌 해결]
            if original_ticker in inventory:
                # 이미 Yahoo에서 등록된 티커라면? -> "_OLD"를 붙여서 별도 등록
                # 예: Yahoo(GM)이 있으므로, Kaggle(GM) -> GM_OLD로 장부 등록
                new_ticker_name = f"{original_ticker}_OLD"
                
                # 로그가 너무 많으면 주석 처리 하세요
                # print(f"  ⚡️ 충돌 감지: {original_ticker} -> {new_ticker_name} (자동 변경)")
                
                inventory[new_ticker_name] = {
                    'source': 'kaggle_legacy',
                    'file_path': str(file_path.relative_to(BASE_DIR)),
                    'is_active': True # 데이터 살림
                }
            else:
                # Yahoo에는 없는 경우 (상폐주 등) -> 원래 이름 그대로 등록
                inventory[original_ticker] = {
                    'source': 'kaggle',
                    'file_path': str(file_path.relative_to(BASE_DIR)),
                    'is_active': True
                }
                
    return inventory

def main():
    print(">>> [Phase 1] Universe 정의 및 자동 충돌 해결 (Auto-Resolve)")
    
    # 1. 파일 시스템 스캔 (자동 매핑 수행)
    print("  🕵️ 로컬 파일 전수 조사 중...")
    local_data = scan_and_resolve_conflicts()
    print(f"  ✅ 로컬 파일 스캔 완료: {len(local_data)} 개 종목 식별됨")

    # 2. 기존 장부 메타데이터 백업 (count, date 등 유지용)
    old_meta = {}
    if MASTER_PATH.exists():
        shutil.copy2(MASTER_PATH, BACKUP_PATH)
        df_old = pd.read_csv(MASTER_PATH)
        for _, row in df_old.iterrows():
            old_meta[row['ticker']] = row.to_dict()

    # 3. 최종 리스트 병합
    final_rows = []
    
    # [A] 로컬 파일 등록
    for ticker, info in local_data.items():
        row = {
            'ticker': ticker,
            'source': info['source'],
            'is_active': info['is_active'],
            'file_path': info['file_path'],
            'count': 0, 
            'start_date': None, 
            'end_date': None,
            'last_updated': datetime.now().strftime("%Y-%m-%d")
        }
        
        # 기존 메타데이터 복구 (파일 경로가 같을 때만)
        if ticker in old_meta:
            prev = old_meta[ticker]
            if str(prev.get('file_path')) == str(info['file_path']):
                row['count'] = prev.get('count', 0)
                row['start_date'] = prev.get('start_date')
                row['end_date'] = prev.get('end_date')
        
        final_rows.append(row)

    # [B] 웹 신규 종목 추가 (로컬에 없는 것만)
    web_tickers = get_new_tickers_from_web()
    existing_keys = set(local_data.keys())
    
    new_candidates = []
    for t in web_tickers:
        # GM이 있든 GM_OLD가 있든 하나라도 있으면 신규 아님
        if t not in existing_keys and f"{t}_OLD" not in existing_keys:
             new_candidates.append(t)
    
    print(f"  🔍 웹 신규 종목 추가: {len(new_candidates)} 개")
    
    for t in new_candidates:
        if "$" in t: continue 
        row = {
            'ticker': t,
            'source': 'new_ipo',
            'is_active': True,
            'file_path': f"data/bronze/yahoo_price_data/ticker={t}/price.parquet",
            'count': 0,
            'start_date': None,
            'end_date': None,
            'last_updated': datetime.now().strftime("%Y-%m-%d")
        }
        final_rows.append(row)

    # 4. 저장 및 메타데이터 갱신
    df_final = pd.DataFrame(final_rows)
    
    print("  📝 메타데이터(행 개수 등) 갱신 중...")
    # 속도를 위해 count가 0인 것만 실제 파일 열어서 확인
    updates = 0
    for idx, row in tqdm(df_final.iterrows(), total=len(df_final)):
        if row['source'] != 'new_ipo' and (pd.isna(row['count']) or row['count'] == 0):
            full_path = BASE_DIR / str(row['file_path'])
            if full_path.exists():
                try:
                    # 헤더만 읽어서 빠르게 처리
                    meta = pd.read_parquet(full_path, columns=['Close'])
                    df_final.at[idx, 'count'] = len(meta)
                    df_final.at[idx, 'start_date'] = meta.index[0].strftime("%Y-%m-%d")
                    df_final.at[idx, 'end_date'] = meta.index[-1].strftime("%Y-%m-%d")
                    updates += 1
                except:
                    pass

    df_final.to_csv(MASTER_PATH, index=False)
    
    print("\n" + "="*40)
    print("  ✅ 장부 생성 완료")
    print(f"  - 총 종목: {len(df_final)}")
    print(f"  - Yahoo(신규): {len(df_final[df_final['source']=='yahoo'])}")
    print(f"  - Kaggle(구형/OLD): {len(df_final[df_final['source']=='kaggle_legacy'])}")
    print("="*40)

if __name__ == "__main__":
    main()