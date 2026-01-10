import pandas as pd
import sys
import random
from pathlib import Path

# 프로젝트 루트 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import PLATINUM_FEATURES_DIR

def inspect_platinum_data():
    print(f"🔍 Platinum 데이터 검사를 시작합니다...")
    print(f"📂 대상 경로: {PLATINUM_FEATURES_DIR}")

    # 1. 파일 목록 확인
    if not PLATINUM_FEATURES_DIR.exists():
        print("❌ [Error] Platinum 디렉토리가 존재하지 않습니다. 먼저 06_create_platinum.py를 실행하세요.")
        return

    files = list(PLATINUM_FEATURES_DIR.glob("*.parquet"))
    if not files:
        print("❌ [Error] Platinum 데이터 파일(.parquet)이 없습니다.")
        return

    print(f"✅ 총 {len(files)}개의 Platinum 파일을 발견했습니다.")

    # 2. 샘플 파일 로드 (첫 번째 파일 또는 랜덤 선택)
    # 특정 종목을 확인하고 싶다면 아래 코드를 수정하세요 (예: target_ticker = '005930')
    target_file = files[3753] 
    # target_file = random.choice(files) # 랜덤 확인 시 주석 해제

    print(f"\n========================================================")
    print(f"📊 분석 대상 파일: {target_file.name}")
    print(f"========================================================")

    try:
        df = pd.read_parquet(target_file)
        
        # 3. 기본 정보 출력
        print(f"\n[1] 데이터 Shape (행, 열): {df.shape}")
        print(f"   - Index Type: {df.index.dtype}")
        
        # 4. 컬럼 목록 확인 (우리가 만든 Feature들이 잘 들어갔는지)
        print(f"\n[2] 컬럼 목록:")
        print(df.columns.tolist())

        # 5. 데이터 샘플 (Head & Tail)
        print(f"\n[3] 상위 5개 데이터 (Head):")
        print(df.head())

        print(f"\n[4] 하위 5개 데이터 (Tail):")
        print(df.tail())

        # 6. 결측치(NaN) 점검
        # Alignment 과정에서 ffill이 잘 되었는지, 혹은 앞부분에 NaN이 남았는지 확인
        nan_sum = df.isna().sum()
        nan_cols = nan_sum[nan_sum > 0]
        
        print(f"\n[5] 결측치(NaN) 보유 컬럼 현황:")
        if nan_cols.empty:
            print("   ✅ 결측치 없음 (Clean Data)")
        else:
            print(nan_cols)
            print("   ⚠️ 상장 초기 데이터 부재 혹은 지표 계산 Window로 인한 NaN일 수 있음.")

        # 7. 통계 요약 (Feature들의 값 범위 확인)
        print(f"\n[6] 기술 통계 (Describe) - 일부 컬럼:")
        # 너무 많으면 보기 힘드므로 새로 생성된 FD_ 컬럼 위주로 확인
        fd_cols = [c for c in df.columns if c.startswith('FD_') or 'PCA' in c]
        if fd_cols:
            print(df[fd_cols].describe())
        else:
            print(df.describe())

    except Exception as e:
        print(f"❌ 파일 로드 중 에러 발생: {e}")

if __name__ == "__main__":
    inspect_platinum_data()