import pandas as pd
import requests
from datetime import datetime
from src.config import *

# SEC 및 NASDAQ 소스
SEC_HEADERS = {'User-Agent': 'Individual_Researcher my_email@example.com'}
SEC_URL = "https://www.sec.gov/files/company_tickers.json"
NASDAQ_URL = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt"

def get_market_tickers():
    """웹에서 실시간 거래 종목 수집"""
    found_tickers = set()
    print("  📡 시장 현황 파악 중 (SEC/NASDAQ)...")
    
    # 1. SEC
    try:
        resp = requests.get(SEC_URL, headers=SEC_HEADERS, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame.from_dict(data, orient='index')
            ts = df['ticker'].astype(str).str.strip().str.upper().str.replace(".", "-")
            found_tickers.update(ts.tolist())
    except: pass

    # 2. NASDAQ
    try:
        df = pd.read_csv(NASDAQ_URL, sep="|")
        ts = df[df['Test Issue'] == 'N']['Symbol'].dropna().astype(str).str.strip().str.upper().str.replace(".", "-")
        found_tickers.update(ts.tolist())
    except: pass
        
    return found_tickers

def update_universe():
    print(">>> [Pipeline 01] Universe Updater (IPO Scanning)")
    
    if not MASTER_PATH.exists():
        print("❌ 장부 파일이 없습니다. 초기화가 필요합니다.")
        return

    # 1. 장부 로드
    df_master = pd.read_csv(MASTER_PATH, dtype={'ticker': str}, keep_default_na=False)
    known_tickers = set(df_master['ticker'].unique())
    
    # 2. 시장 데이터 수집
    market_tickers = get_market_tickers()
    
    if not market_tickers:
        print("  ⚠️ 네트워크 오류로 시장 데이터를 가져오지 못했습니다.")
        return

    # 3. 신규 상장(IPO) 감지
    # GM_OLD 같은 로컬 전용 티커는 제외하고 비교
    new_ipos = []
    for t in market_tickers:
        if t not in known_tickers and f"{t}_OLD" not in known_tickers:
            if "$" in t: continue # 특수문자 제외
            new_ipos.append(t)
            
    # 4. 장부 업데이트
    if new_ipos:
        new_ipos.sort()
        print(f"  ✨ 신규 상장 종목 발견: {len(new_ipos)} 개")
        
        new_rows = []
        today_str = datetime.now().strftime(DATE_FORMAT)
        
        for t in new_ipos:
            new_rows.append({
                'ticker': t,
                'source': 'new_ipo',
                'is_active': True,
                'file_path': f"data/bronze/yahoo_price_data/ticker={t}/price.parquet",
                'count': 0,
                'start_date': None,
                'end_date': None,
                'last_updated': None, # 다운로드 트리거
                'fail_count': 0,      # [중요] Safe Mode 호환
                'last_failed_date': None,
                'note': f'IPO Detected {today_str}'
            })
            
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df_master = pd.concat([df_master, df_new], ignore_index=True)
            df_master.to_csv(MASTER_PATH, index=False)
            print("  ✅ 장부 업데이트 완료.")
    else:
        print("  ✅ 신규 상장 종목 없음.")

if __name__ == "__main__":
    update_universe()