import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

# 프로젝트 루트 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import PLATINUM_FEATURES_DIR

# 점검할 주요 대장주 리스트 (Top 20 Mega Caps)
TARGET_GIANTS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "LLY", "V", 
    "TSM", "JPM", "WMT", "XOM", "UNH", "MA", "PG", "JNJ", "HD", "COST"
]

def diagnose_dataset():
    print(f"🔍 [데이터셋 정밀 진단] Platinum 데이터를 검사합니다...")
    print(f"📂 경로: {PLATINUM_FEATURES_DIR}\n")

    if not PLATINUM_FEATURES_DIR.exists():
        print("❌ Platinum 디렉토리가 없습니다.")
        return

    # 1. 파일 목록 로드
    files = list(PLATINUM_FEATURES_DIR.glob("*.parquet"))
    existing_tickers = set(f.stem for f in files)
    
    print(f"✅ 총 발견된 파일 수: {len(files)}개")

    # ==========================================
    # [진단 1] 주요 대장주(Giants) 존재 여부 확인
    # ==========================================
    print("\n" + "="*50)
    print("💎 [Step 1] 주요 대장주 존재 여부 체크")
    print("="*50)
    
    missing_giants = []
    found_giants = []
    
    for t in TARGET_GIANTS:
        if t in existing_tickers:
            found_giants.append(t)
        else:
            missing_giants.append(t)
            
    if found_giants:
        print(f"✅ 발견됨 ({len(found_giants)}개): {found_giants}")
    
    if missing_giants:
        print(f"🚨 [CRITICAL] 누락됨 ({len(missing_giants)}개): {missing_giants}")
        print("   -> 01_define_universe.py 또는 02_data_download.py에서 누락되었습니다.")
        print("   -> 'scripts/force_download_giants.py'를 실행하여 긴급 복구가 필요합니다.")
    else:
        print("🎉 모든 대장주가 정상적으로 존재합니다.")

    # ==========================================
    # [진단 2] 잡주(XYZ) 및 이상 종목 확인
    # ==========================================
    print("\n" + "="*50)
    print("🗑️ [Step 2] 테스트용 잡주(XYZ) 확인")
    print("="*50)
    
    suspicious = ["XYZ", "ABC", "TEST"]
    found_suspicious = [t for t in suspicious if t in existing_tickers]
    
    if found_suspicious:
        print(f"⚠️ [WARNING] 테스트용 데이터 발견: {found_suspicious}")
        print("   -> 백테스트 결과를 왜곡할 수 있으므로 삭제를 권장합니다.")
    else:
        print("✅ 이상한 종목(XYZ 등)은 발견되지 않았습니다.")

    # ==========================================
    # [진단 3] FD 변수 건강 상태 체크 (기존 로직)
    # ==========================================
    print("\n" + "="*50)
    print("🏥 [Step 3] FD 변수 결측(NaN) 진단")
    print("="*50)
    
    missing_fd_files = []
    all_nan_files = []
    
    # 너무 많으면 오래 걸리므로 Giants 위주로 먼저 샘플링하거나 전체 수행
    # 여기서는 발견된 Giants + 랜덤 100개만 체크
    check_targets = list(found_giants) + list(existing_tickers)[:100]
    check_targets = list(set(check_targets)) # 중복 제거
    
    for t in tqdm(check_targets, desc="Checking FD Columns"):
        file_path = PLATINUM_FEATURES_DIR / f"{t}.parquet"
        try:
            df = pd.read_parquet(file_path)
            fd_cols = [c for c in df.columns if c.startswith('FD_')]
            
            if not fd_cols:
                missing_fd_files.append(t)
                continue
            
            # 첫 번째 FD 컬럼 기준 검사
            target_col = 'FD_Close' if 'FD_Close' in fd_cols else fd_cols[0]
            if df[target_col].isna().all():
                all_nan_files.append(t)
                
        except Exception:
            pass

    if missing_fd_files:
        print(f"\n❌ FD 컬럼 없음: {len(missing_fd_files)}개 ({missing_fd_files[:5]}...)")
    if all_nan_files:
        print(f"💀 FD 전부 NaN (계산실패): {len(all_nan_files)}개 ({all_nan_files[:5]}...)")
    
    if not missing_fd_files and not all_nan_files:
        print("✅ 체크한 파일들의 데이터 상태는 양호합니다.")

if __name__ == "__main__":
    diagnose_dataset()