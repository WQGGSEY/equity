import pandas as pd
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 프로젝트 루트 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import PLATINUM_FEATURES_DIR

def diagnose_fd_issues():
    print(f"🔍 [FD_ 변수 정밀 진단] Platinum 데이터를 검사합니다...")
    print(f"📂 경로: {PLATINUM_FEATURES_DIR}\n")

    if not PLATINUM_FEATURES_DIR.exists():
        print("❌ Platinum 디렉토리가 없습니다.")
        return

    files = list(PLATINUM_FEATURES_DIR.glob("*.parquet"))
    if not files:
        print("❌ 데이터 파일이 없습니다.")
        return

    # 리포트용 컨테이너
    missing_fd_files = []   # FD 컬럼이 아예 없는 파일
    all_nan_files = []      # FD 컬럼이 전부 NaN인 파일
    high_nan_files = []     # FD 컬럼의 NaN 비율이 너무 높은 파일 (>50%)
    
    print(f"✅ 총 {len(files)}개 파일 스캔 시작 (FD_Close 위주 점검)...\n")
    
    for file_path in tqdm(files, desc="Scanning FD_"):
        try:
            # 전체를 읽지 않고 컬럼 확인을 위해 가볍게 로드 시도
            # (PyArrow 엔진 사용 시 메타데이터만 읽을 수도 있으나, 여기선 그냥 로드)
            df = pd.read_parquet(file_path)
            
            # 1. FD_Close 컬럼 존재 확인
            fd_cols = [c for c in df.columns if c.startswith('FD_')]
            
            if not fd_cols:
                missing_fd_files.append(file_path.stem)
                continue
                
            # 'FD_Close'가 있다면 그걸 기준으로, 없으면 첫 번째 FD 컬럼 기준
            target_col = 'FD_Close' if 'FD_Close' in fd_cols else fd_cols[0]
            
            # 2. NaN 비율 확인
            total_len = len(df)
            nan_count = df[target_col].isna().sum()
            nan_ratio = nan_count / total_len if total_len > 0 else 0
            
            if nan_count == total_len:
                all_nan_files.append(file_path.stem)
            elif nan_ratio > 0.5: # 50% 이상이 결측이면 문제 의심 (Window가 너무 크거나 데이터가 짧음)
                high_nan_files.append({
                    'ticker': file_path.stem,
                    'ratio': f"{nan_ratio*100:.1f}%",
                    'len': total_len
                })
                
        except Exception as e:
            print(f"❌ 읽기 에러 ({file_path.name}): {e}")

    # === 진단 리포트 ===
    print("\n" + "="*50)
    print("📊 [FD_ 변수 진단 결과]")
    print("="*50)

    # 1. FD 컬럼 미보유
    if missing_fd_files:
        print(f"\n🚨 [Critical] 'FD_' 컬럼이 없는 종목: {len(missing_fd_files)}개")
        print(f"   - 예: {missing_fd_files[:10]}...")
    else:
        print("\n✅ 모든 파일에 'FD_' 컬럼이 존재합니다.")

    # 2. 전부 NaN인 경우 (계산 실패)
    if all_nan_files:
        print(f"\n💀 [Fatal] 'FD_' 값이 전부 NaN인 종목: {len(all_nan_files)}개")
        print(f"   - 예: {all_nan_files[:10]}...")
        print("   -> 06_create_platinum.py의 FD 계산 로직이나 d값 설정을 확인해야 합니다.")
    else:
        print("\n✅ 'FD_' 값이 전부 NaN인 '죽은 파일'은 없습니다.")

    # 3. 결측 비율 과다
    if high_nan_files:
        print(f"\n⚠️ [Warning] NaN 비율이 50%를 넘는 종목: {len(high_nan_files)}개")
        print("   (데이터 길이가 짧아서 FD 윈도우만큼 날아가고 남은 게 별로 없는 경우일 수 있음)")
        for item in high_nan_files[:5]:
            print(f"   - {item['ticker']}: NaN {item['ratio']} (Total: {item['len']})")
    else:
        print("\n✅ 대다수 종목의 유효 데이터 비율이 양호합니다.")

    print("\n" + "="*50)
    
    # 샘플 데이터 출력 (첫 번째 정상 파일)
    if files and not missing_fd_files:
        print("\n🔍 [Sample Data Check]")
        sample_path = files[0]
        df_sample = pd.read_parquet(sample_path)
        fd_c = [c for c in df_sample.columns if c.startswith('FD_')][0]
        print(f"File: {sample_path.name}")
        print(df_sample[[fd_c]].head(10))
        print("...\n(Head 부분은 NaN이 정상입니다. d=0.4 등의 차분 과정에서 소실됨)")

if __name__ == "__main__":
    diagnose_fd_issues()