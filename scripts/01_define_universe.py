import pandas as pd
import requests
import shutil
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ==========================================
# [Phase 1] Universe Definition + Physical Audit (All-in-One)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_PATH = BASE_DIR / "data" / "bronze" / "master_ticker_list.csv"
BACKUP_PATH = BASE_DIR / "data" / "bronze" / "master_ticker_list_backup.csv"

# 저장소 위치 정의
YAHOO_DIR = BASE_DIR / "data" / "bronze" / "yahoo_price_data"
KAGGLE_DIR = BASE_DIR / "data" / "bronze" / "daily_prices"

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

def check_physical_file(ticker):
    """
    해당 티커의 파일이 실제로 존재하는지 확인하고 메타데이터 반환
    """
    # 1. Yahoo 폴더 확인 (우선순위)
    y_ticker = normalize_ticker(ticker)
    y_path = YAHOO_DIR / f"ticker={y_ticker}" / "price.parquet"
    
    if y_path.exists():
        return y_path, "yahoo"

    # 2. Kaggle 폴더 확인 (Legacy 포함)
    # Kaggle은 원본 티커명을 그대로 폴더명으로 씀
    k_path = KAGGLE_DIR / f"ticker={ticker}" / "price.parquet"
    if k_path.exists():
        return k_path, "kaggle"
        
    return None, None

def main():
    print(">>> [Phase 1] 리스트 정의 및 파일 전수 조사 (통합본)")
    
    # 1. 기존 파일 로드
    if not MASTER_PATH.exists():
        print("❌ 기존 리스트가 없습니다. (최초 생성 모드로 진행)")
        df = pd.DataFrame(columns=['ticker', 'source', 'is_active', 'count', 'file_path', 'start_date', 'end_date', 'last_updated'])
        existing_tickers = set()
        original_cols = df.columns.tolist()
    else:
        print(f"  📖 장부 로드: {MASTER_PATH.name}")
        shutil.copy2(MASTER_PATH, BACKUP_PATH) # 백업
        df = pd.read_csv(MASTER_PATH)
        existing_tickers = set(df['ticker'].apply(normalize_ticker).tolist())
        original_cols = df.columns.tolist()

    # 2. 신규 종목 추가 (Web)
    web_tickers = get_new_tickers_from_web()
    new_candidates = sorted(list(web_tickers - existing_tickers))
    new_candidates = [t for t in new_candidates if "$" not in t]
    
    if new_candidates:
        print(f"  🔍 신규 종목 추가: {len(new_candidates)} 개")
        new_rows = []
        today = datetime.now().strftime("%Y-%m-%d")
        for t in new_candidates:
            row = {col: None for col in original_cols}
            row['ticker'] = t
            row['source'] = 'new_ipo' # 일단 표시
            row['is_active'] = True
            row['count'] = 0
            # 예상 경로 (실제 파일 확인 전 임시)
            row['file_path'] = f"data/bronze/yahoo_price_data/ticker={t}/price.parquet"
            row['last_updated'] = today
            new_rows.append(row)
        
        df_new = pd.DataFrame(new_rows)
        # 컬럼 매칭
        for col in original_cols:
            if col not in df_new.columns: df_new[col] = None
            
        df = pd.concat([df, df_new[original_cols]], ignore_index=True)
    else:
        print("  ✅ 신규 추가될 종목이 없습니다.")

    # =========================================================
    # [핵심] 리스트에 있는 모든 종목에 대해 "파일이 진짜 있는지" 확인
    # =========================================================
    print(f"  🕵️ 전체 종목 실물 전수 조사 (Audit)... 총 {len(df)}개")
    
    audit_updated = 0
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # tqdm으로 진행 상황 표시
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Auditing"):
        ticker = row['ticker']
        
        # 파일 찾기
        found_path, source_type = check_physical_file(ticker)
        
        if found_path:
            # [파일 있음] -> 장부 업데이트
            try:
                # 헤더만 읽어서 정보 갱신
                meta = pd.read_parquet(found_path, columns=['Close'])
                
                df.at[idx, 'count'] = len(meta)
                df.at[idx, 'start_date'] = meta.index[0].strftime("%Y-%m-%d")
                df.at[idx, 'end_date'] = meta.index[-1].strftime("%Y-%m-%d")
                df.at[idx, 'file_path'] = str(found_path.relative_to(BASE_DIR))
                df.at[idx, 'last_updated'] = today_str
                
                # source 정보가 없거나 new_ipo라면 실제 소스로 변경
                if pd.isna(row.get('source')) or row.get('source') == 'new_ipo':
                    df.at[idx, 'source'] = source_type
                
                audit_updated += 1
            except:
                # 파일 깨짐 -> 없음 처리
                df.at[idx, 'count'] = 0
        else:
            # [파일 없음] -> 0 처리 (Phase 2 다운로드 대상)
            df.at[idx, 'count'] = 0
            
            # 경로가 비어있으면 예상 경로라도 채워둠
            if pd.isna(row.get('file_path')):
                df.at[idx, 'file_path'] = f"data/bronze/yahoo_price_data/ticker={normalize_ticker(ticker)}/price.parquet"

    # 4. 저장
    df.to_csv(MASTER_PATH, index=False)
    
    # 결과 요약
    need_download = len(df[df['count'] == 0])
    has_data = len(df[df['count'] > 0])
    
    print("\n" + "="*40)
    print("  ✅ Phase 1 (정의 + 감사) 완료")
    print("="*40)
    print(f"  - 파일 보유 확인됨: {has_data} 개 (Safe)")
    print(f"  - 다운로드 필요(0): {need_download} 개")
    print(f"  📂 저장 완료: {MASTER_PATH}")
    print("-" * 40)
    print("👉 이제 Phase 2를 실행하면 '다운로드 필요' 개수만큼만 요청합니다.")

if __name__ == "__main__":
    main()