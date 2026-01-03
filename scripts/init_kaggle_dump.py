import os
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import time

# ==========================================
# [설정] 경로 및 옵션
# ==========================================
# 현재 스크립트 위치(scripts/)의 부모 폴더가 프로젝트 루트라고 가정
BASE_DIR = Path(__file__).resolve().parent.parent 

# 소스 데이터 위치 (사용자 트리에 맞춤)
SOURCE_DIR = BASE_DIR / "data" / "bronze" / "kaggle_dump"

# 결과 저장 위치
DEST_DIR = BASE_DIR / "data" / "bronze" / "daily_prices"
MASTER_LIST_PATH = BASE_DIR / "data" / "bronze" / "master_ticker_list.csv"

# 병렬 처리 개수 (-1: 모든 CPU 코어 사용, M2 성능 극대화)
N_JOBS = -1 

def process_file(file_path):
    """
    개별 txt 파일을 읽어 Parquet로 변환 및 메타데이터 추출
    파일명 예시: 'aapl.us.txt' -> Ticker: 'AAPL'
    """
    try:
        # 1. 파일명에서 티커 추출 (예: aapl.us.txt -> AAPL)
        file_name = file_path.name
        if not file_name.endswith('.txt'):
            return None
            
        # '.us.txt' 제거 후 대문자 변환
        ticker = file_name.replace('.us.txt', '').replace('.txt', '').upper()
        
        # 2. 데이터 로드 (Kaggle 데이터셋은 헤더가 없거나 소문자일 수 있음, 확인 필요)
        # 보통 이 데이터셋은: Date,Open,High,Low,Close,Volume,OpenInt 형식을 가짐
        df = pd.read_csv(file_path)
        
        # 컬럼명 표준화 (공백 제거 및 Capitalize)
        df.columns = [c.strip().capitalize() for c in df.columns]
        
        # 필수 컬럼 확인
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return None # 필수 컬럼 없으면 스킵
            
        # 필요한 컬럼만 선택
        df = df[required_cols]
        
        # 3. 데이터 타입 변환
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # 데이터가 비어있으면 스킵
        if df.empty:
            return None

        # 4. Parquet 저장 (Hive Partitioning 스타일: ticker=AAPL)
        save_dir = DEST_DIR / f"ticker={ticker}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / "price.parquet"
        # Snappy 압축 사용 (속도/용량 밸런스 좋음)
        df.to_parquet(save_path, index=False, compression='snappy')
        
        # 5. 메타데이터 반환 (Master List 생성을 위해)
        return {
            'ticker': ticker,
            'source': 'kaggle_dump',
            'last_updated': pd.Timestamp.now(),
            'start_date': df['Date'].min(),
            'end_date': df['Date'].max(),
            'count': len(df),
            'is_active': False, # 2017년 데이터이므로 일단 비활성 처리 (추후 Yahoo 확인 시 활성)
            'file_path': str(save_path)
        }
        
    except Exception as e:
        # 에러 발생 시 로그만 찍고 계속 진행
        # print(f"Error converting {file_path.name}: {e}")
        return None

def main():
    print(f"Directory Setup:")
    print(f"  - Source: {SOURCE_DIR}")
    print(f"  - Dest  : {DEST_DIR}")
    
    # 1. 파일 리스트 수집 (Stocks와 ETFs 폴더 모두 탐색)
    print(">>> 파일 검색 중...")
    files = list(SOURCE_DIR.rglob("*.txt"))
    
    if not files:
        print("❌ 오류: .txt 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    print(f"  - 발견된 파일 수: {len(files)}개")
    print(">>> 변환 작업 시작 (M2 코어 가동, 약 5~10분 소요)...")
    
    # 2. 병렬 처리 실행
    start_time = time.time()
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_file)(f) for f in tqdm(files, desc="Converting")
    )
    end_time = time.time()
    
    # 3. 결과 집계 및 Master List 저장
    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        master_df = pd.DataFrame(valid_results)
        master_df.sort_values('ticker', inplace=True)
        master_df.to_csv(MASTER_LIST_PATH, index=False)
        
        print(f"\n>>> ✅ 초기화 완료!")
        print(f"  - 성공: {len(valid_results)}개")
        print(f"  - 실패/스킵: {len(files) - len(valid_results)}개")
        print(f"  - 소요 시간: {end_time - start_time:.2f}초")
        print(f"  - Master List 저장됨: {MASTER_LIST_PATH}")
    else:
        print("\n❌ 변환된 데이터가 없습니다.")

if __name__ == "__main__":
    main()