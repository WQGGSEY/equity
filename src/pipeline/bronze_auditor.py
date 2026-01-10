import pandas as pd
import requests
import shutil
from datetime import datetime
from src.config import *

# 소스 정의
SEC_HEADERS = {'User-Agent': 'Individual_Researcher my_email@example.com'}
SEC_URL = "https://www.sec.gov/files/company_tickers.json"
NASDAQ_URL = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt"

def get_new_tickers_from_web():
    """웹에서 최신 종목 리스트 수집"""
    found_tickers = set()
    print("  📡 신규 종목 리스트 확인 중 (SEC/NASDAQ)...")
    
    try:
        resp = requests.get(SEC_URL, headers=SEC_HEADERS, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame.from_dict(data, orient='index')
            ts = df['ticker'].astype(str).str.strip().str.upper().str.replace(".", "-")
            found_tickers.update(ts.tolist())
    except: pass

    try:
        df = pd.read_csv(NASDAQ_URL, sep="|")
        ts = df[df['Test Issue'] == 'N']['Symbol'].dropna().astype(str).str.strip().str.upper().str.replace(".", "-")
        found_tickers.update(ts.tolist())
    except: pass
        
    return found_tickers

def run_audit():
    print(">>> [Phase 1] Bronze Auditor (Backup & Ledger Check)")
    
    # [Safety] 장부 백업
    if MASTER_PATH.exists():
        BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        backup_path = BACKUP_ROOT / f"master_ticker_list_{timestamp}.csv"
        try:
            shutil.copy2(MASTER_PATH, backup_path)
            print(f"  🛡️ 장부 백업 완료: {backup_path.name}")
        except Exception as e:
            print(f"  ⚠️ 백업 실패: {e}")

    # 장부 로드 및 초기화
    if not MASTER_PATH.exists():
        print("❌ 장부 파일이 없습니다. (scripts/01_define_universe.py 실행 권장)")
        return

    df = pd.read_csv(MASTER_PATH, dtype={'ticker': str}, keep_default_na=False)
    
    # Audit: 파일 실존 여부 확인
    print(f"  🕵️ 등록된 {len(df)}개 종목 상태 점검...")
    updates = 0
    today_str = datetime.now().strftime(DATE_FORMAT)
    
    for idx, row in df.iterrows():
        if pd.isna(row['file_path']): continue
        full_path = BASE_DIR / str(row['file_path'])
        
        if full_path.exists():
            # 메타데이터 갱신 (오늘 미확인 건만)
            if str(row['last_updated']) != today_str or row['count'] == 0:
                try:
                    meta = pd.read_parquet(full_path, columns=['Close'])
                    df.at[idx, 'count'] = len(meta)
                    df.at[idx, 'start_date'] = meta.index[0].strftime(DATE_FORMAT)
                    df.at[idx, 'end_date'] = meta.index[-1].strftime(DATE_FORMAT)
                    df.at[idx, 'last_updated'] = today_str
                    updates += 1
                except:
                    df.at[idx, 'count'] = 0 # 파일 깨짐
        else:
            if row['count'] > 0:
                df.at[idx, 'count'] = 0 # 유실됨 -> 재다운로드 대상

    # 신규 종목 추가
    current = set(df['ticker'].unique())
    web = get_new_tickers_from_web()
    new_candidates = sorted(list(web - current))
    
    if new_candidates:
        print(f"  ✨ 신규 상장 발견: {len(new_candidates)} 개")
        new_rows = []
        for t in new_candidates:
            if "$" in t: continue
            new_rows.append({
                'ticker': t,
                'source': 'new_ipo',
                'is_active': True,
                'file_path': f"data/bronze/yahoo_price_data/ticker={t}/price.parquet",
                'count': 0,
                'start_date': None,
                'end_date': None,
                'last_updated': today_str
            })
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    df.to_csv(MASTER_PATH, index=False)
    print(f"  ✅ Audit 완료 (갱신: {updates}, 총: {len(df)})")

if __name__ == "__main__":
    run_audit()