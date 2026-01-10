import pandas as pd
from datetime import datetime
from src.config import *

def run_audit():
    print(">>> [Pipeline 02] Bronze Auditor (Integrity Check Only)")
    
    if not MASTER_PATH.exists():
        print("❌ 장부 파일이 없습니다.")
        return

    df = pd.read_csv(MASTER_PATH, dtype={'ticker': str}, keep_default_na=False)
    
    print(f"  🕵️ 로컬 파일 정합성 검사 중 ({len(df)} 개)...")
    
    updates = 0
    # today_str = datetime.now().strftime(DATE_FORMAT)  <-- 삭제 불필요
    
    for idx, row in df.iterrows():
        if pd.isna(row['file_path']): continue
        
        full_path = BASE_DIR / str(row['file_path'])
        
        if full_path.exists():
            # [수정] 파일이 있으면 개수(count)와 날짜 범위(start/end)만 갱신
            # last_updated는 건드리지 않음! (Ingestor가 판단하도록)
            if row['count'] == 0:  # 혹은 주기적으로 체크
                try:
                    meta = pd.read_parquet(full_path, columns=['Close'])
                    df.at[idx, 'count'] = len(meta)
                    df.at[idx, 'start_date'] = meta.index[0].strftime(DATE_FORMAT)
                    df.at[idx, 'end_date'] = meta.index[-1].strftime(DATE_FORMAT)
                    # df.at[idx, 'last_updated'] = today_str  <-- [삭제] 이거 절대 금지!
                    updates += 1
                except:
                    df.at[idx, 'count'] = 0 
        else:
            if row['count'] > 0:
                df.at[idx, 'count'] = 0
                if 'fail_count' in df.columns:
                    df.at[idx, 'fail_count'] = 0

    df.to_csv(MASTER_PATH, index=False)
    print(f"  ✅ Audit 완료 (메타데이터 갱신: {updates} 건)")

if __name__ == "__main__":
    run_audit()