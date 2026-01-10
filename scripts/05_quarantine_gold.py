import sys
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# -------------------------------------------------------------------------
# [Setup] Project Root Path 설정 (ImportError 해결 핵심)
# -------------------------------------------------------------------------
FILE_PATH = Path(__file__).resolve()
PROJECT_DIR = FILE_PATH.parent.parent  # equity/
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# 이제 절대 경로로 안전하게 임포트합니다.
from src.config import GOLD_DIR, QUARANTINE_DIR

def run_quarantine():
    """
    [Gold Validator]
    Gold 데이터의 무결성을 검증하고, 불량 데이터(특히 0.0 가격)를 격리합니다.
    """
    print(f"🚀 Running Gold Data Quarantine...")
    print(f"📂 Project Root: {PROJECT_DIR}")
    
    # 격리 폴더 생성
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    
    gold_files = list(GOLD_DIR.glob("*.parquet"))
    if not gold_files:
        print("⚠️ 검사할 Gold 데이터 파일이 없습니다.")
        return

    moved_count = 0
    valid_count = 0
    
    print(f"🔍 총 {len(gold_files)}개 파일 검사 시작...")

    for f in tqdm(gold_files, desc="Inspecting"):
        try:
            is_valid = True
            reason = ""
            
            # 1. 파일 읽기
            try:
                df = pd.read_parquet(f)
            except Exception as e:
                is_valid = False
                reason = f"Read Error: {e}"

            if is_valid:
                # 2. 빈 데이터 확인
                if df.empty:
                    is_valid = False
                    reason = "Empty DataFrame"
                
                # 3. 데이터 길이 확인
                elif len(df) < 50:
                    is_valid = False
                    reason = f"Too Short (Rows={len(df)} < 50)"

                # 4. [Critical] 0.0 또는 음수 가격 확인
                else:
                    price_cols = ['Open', 'High', 'Low', 'Close']
                    cols_to_check = [c for c in price_cols if c in df.columns]
                    
                    if cols_to_check:
                        # 0 이하인 값이 하나라도 있으면 불량
                        if (df[cols_to_check] <= 0).any().any():
                            is_valid = False
                            reason = "Zero or Negative Prices Found"

            # 5. 격리 조치
            if not is_valid:
                dest = QUARANTINE_DIR / f.name
                shutil.move(str(f), str(dest))
                # print(f"  🚫 [Quarantine] {f.stem}: {reason}")
                moved_count += 1
            else:
                valid_count += 1
                
        except Exception as e:
            print(f"❌ Error processing {f.name}: {e}")

    print("\n" + "="*50)
    print(f"✅ 검사 완료 Report")
    print(f"  - 정상 파일: {valid_count} 개")
    print(f"  - 격리 파일: {moved_count} 개 (불량 데이터)")
    print(f"  - 격리 위치: {QUARANTINE_DIR}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_quarantine()