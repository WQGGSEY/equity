import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# ==========================================
# [Phase 3] Silver Layer Generation (Standardization)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_PATH = BASE_DIR / "data" / "bronze" / "master_ticker_list.csv"
SILVER_DIR = BASE_DIR / "data" / "silver" / "daily_prices"

def optimize_dtypes(df):
    """
    데이터 타입을 최적화하여 용량을 줄이고 ML 호환성을 높임
    Price -> float32
    Volume -> float32 (NaN 처리를 위해 float 선호)
    """
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype('float32')
    return df

def standardize_columns(df):
    """
    컬럼명을 통일하고 불필요한 컬럼 제거
    목표: Open, High, Low, Close, Volume
    """
    # 1. 컬럼명 소문자 변환 후 맵핑 준비
    df.columns = [c.strip().lower() for c in df.columns]
    
    rename_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'adj close': 'Adj Close', # 임시
        'volume': 'Volume'
    }
    
    # 존재하는 컬럼만 변경
    curr_cols = {c: rename_map[c] for c in df.columns if c in rename_map}
    df.rename(columns=curr_cols, inplace=True)
    
    # 2. 수정주가(Adj Close) 처리
    # Yahoo(auto_adjust=True)는 이미 Close가 수정주가임.
    # Kaggle 데이터 등에서 Adj Close가 별도로 있다면, 이를 Close로 덮어쓰는 것이 분석에 유리함.
    if 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
        
    # 3. 필수 컬럼 확인 및 순서 보장
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # 없는 컬럼은 NaN으로 채움 (극히 드문 케이스)
    for c in required:
        if c not in df.columns:
            df[c] = np.nan
            
    return df[required] # 순서 강제 및 기타 컬럼(Dividends 등) 제거

def main():
    print(">>> [Phase 3] Silver Layer 생성 (Bronze -> Silver)")
    
    if not MASTER_PATH.exists():
        print("❌ 장부 파일이 없습니다. Phase 1, 2를 먼저 완료하세요.")
        return

    # 1. 초기화 (기존 Silver 폴더 삭제 후 재생성)
    if SILVER_DIR.exists():
        shutil.rmtree(SILVER_DIR)
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"  📂 저장소 초기화됨: {SILVER_DIR}")

    # 2. 장부 로드
    df_master = pd.read_csv(MASTER_PATH)
    
    # 데이터가 있는 것만 대상 (Count > 0)
    df_master['count'] = pd.to_numeric(df_master['count'], errors='coerce').fillna(0)
    target_rows = df_master[df_master['count'] > 0]
    
    print(f"  📖 변환 대상: {len(target_rows)} 개 종목")
    
    success_count = 0
    fail_count = 0
    
    # 3. 변환 루프
    for _, row in tqdm(target_rows.iterrows(), total=len(target_rows), desc="Standardizing"):
        ticker = row['ticker']
        rel_path = row['file_path']
        
        # 경로 유효성 확인
        if pd.isna(rel_path):
            fail_count += 1
            continue
            
        full_path = BASE_DIR / rel_path
        
        if not full_path.exists():
            # 장부에는 있는데 파일이 없는 경우 (Sync 문제)
            fail_count += 1
            continue
            
        try:
            # 로드
            df = pd.read_parquet(full_path)
            
            if df.empty:
                fail_count += 1
                continue
                
            # 인덱스 처리 (Date)
            if not isinstance(df.index, pd.DatetimeIndex):
                # Date 컬럼이 있는지 확인
                cols_lower = [c.lower() for c in df.columns]
                if 'date' in cols_lower:
                    # 컬럼명 찾기
                    date_col = df.columns[cols_lower.index('date')]
                    df['Date'] = pd.to_datetime(df[date_col])
                    df.set_index('Date', inplace=True)
                else:
                    # 날짜가 없으면 스킵
                    fail_count += 1
                    continue
            
            # Timezone 제거
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # 정제 및 표준화
            df = standardize_columns(df)
            df = optimize_dtypes(df)
            
            # 중복 날짜 제거 및 정렬
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)
            
            # 저장 (파일명: 티커.parquet)
            # 윈도우 파일명 호환을 위해 특수문자 처리는 유지하되, 여기선 단순화
            safe_name = str(ticker).replace(".", "-").upper()
            save_path = SILVER_DIR / f"{safe_name}.parquet"
            
            df.to_parquet(save_path)
            success_count += 1
            
        except Exception as e:
            # 개별 파일 에러는 무시하고 진행 (로그만 남길 수도 있음)
            fail_count += 1

    print("\n" + "="*40)
    print("  ✅ Phase 3 완료")
    print("="*40)
    print(f"  - 성공(생성됨): {success_count} 개")
    print(f"  - 실패(스킵됨): {fail_count} 개")
    print(f"  📂 Silver 경로: {SILVER_DIR}")
    print("-" * 40)
    print("👉 이제 모든 데이터가 '동일한 포맷'으로 Silver 폴더에 모였습니다.")
    print("👉 다음 단계인 Feature Engineering(Gold)으로 넘어갈 수 있습니다.")

if __name__ == "__main__":
    main()