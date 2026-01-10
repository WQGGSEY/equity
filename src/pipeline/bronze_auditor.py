import pandas as pd
from datetime import datetime
from src.config import *

def run_audit():
    print(">>> [Pipeline 02] Bronze Auditor (File Integrity Check)")
    
    # 1. 장부 로드
    if not MASTER_PATH.exists():
        print("❌ 장부 파일이 없습니다.")
        return

    df = pd.read_csv(MASTER_PATH, dtype={'ticker': str}, keep_default_na=False)
    
    # 2. 파일 실존 여부 검사
    print(f"  🕵️ 로컬 파일 정합성 검사 중 ({len(df)} 개)...")
    
    updates = 0
    today_str = datetime.now().strftime(DATE_FORMAT)
    
    for idx, row in df.iterrows():
        if pd.isna(row['file_path']): continue
        
        full_path = BASE_DIR / str(row['file_path'])
        
        if full_path.exists():
            # 메타데이터 갱신 (오늘 체크 안 된 것만)
            if str(row['last_updated']) != today_str or row['count'] == 0:
                try:
                    meta = pd.read_parquet(full_path, columns=['Close'])
                    df.at[idx, 'count'] = len(meta)
                    df.at[idx, 'start_date'] = meta.index[0].strftime(DATE_FORMAT)
                    df.at[idx, 'end_date'] = meta.index[-1].strftime(DATE_FORMAT)
                    df.at[idx, 'last_updated'] = today_str
                    updates += 1
                except:
                    # 파일 깨짐 -> 0 처리 (재수집 유도)
                    df.at[idx, 'count'] = 0 
        else:
            # 장부엔 있는데 파일이 없음 -> 재수집 대상
            if row['count'] > 0:
                df.at[idx, 'count'] = 0
                # fail_count 초기화하여 ingestor가 다시 시도하게 함
                if 'fail_count' in df.columns:
                    df.at[idx, 'fail_count'] = 0

    # 3. 저장 (웹 크롤링 로직은 universe_updater로 이관됨)
    df.to_csv(MASTER_PATH, index=False)
    print(f"  ✅ Audit 완료 (메타데이터 갱신: {updates} 건)")

if __name__ == "__main__":
    run_audit()