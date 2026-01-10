import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm
from ..config import GOLD_DIR, QUARANTINE_DIR

def run_quarantine():
    """
    [Gold Validator]
    Gold 데이터의 무결성을 검증하고, 불량 데이터는 Quarantine 폴더로 격리합니다.
    
    [검증 기준]
    1. 데이터가 비어있는가?
    2. 필수 컬럼(OHLCV)이 존재하는가?
    3. [NEW] 가격(OHLC)에 0.0 또는 음수가 포함되어 있는가? (Fatal Error)
    4. 데이터의 길이가 너무 짧은가? (e.g., < 30 days)
    """
    print(">>> [Phase 5] Gold Data Quarantine (Validator)")
    
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    gold_files = list(GOLD_DIR.glob("*.parquet"))
    
    moved_count = 0
    valid_count = 0
    
    for f in tqdm(gold_files, desc="Inspecting Gold Data"):
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
                
                # 3. 데이터 길이 확인 (너무 짧으면 ML 불가)
                elif len(df) < 50:
                    is_valid = False
                    reason = f"Too Short (Rows={len(df)} < 50)"

                # 4. [핵심] 0.0 또는 음수 가격 확인 (Logical Corruption)
                else:
                    price_cols = ['Open', 'High', 'Low', 'Close']
                    # 존재하는 컬럼만 체크
                    cols_to_check = [c for c in price_cols if c in df.columns]
                    
                    if cols_to_check:
                        # (df <= 0) 조건이 하나라도 True면 불량
                        if (df[cols_to_check] <= 0).any().any():
                            is_valid = False
                            # 0이 있는 컬럼과 개수 파악
                            zeros = (df[cols_to_check] <= 0).sum()
                            zeros = zeros[zeros > 0].to_dict()
                            reason = f"Zero/Negative Prices Found: {zeros}"

            # 5. 격리 조치
            if not is_valid:
                # Quarantine으로 이동
                shutil.move(str(f), str(QUARANTINE_DIR / f.name))
                # 로그 남기기 (선택 사항)
                # print(f"🚫 Quarantine: {f.stem} -> {reason}")
                moved_count += 1
            else:
                valid_count += 1
                
        except Exception as e:
            print(f"❌ Error processing {f.name}: {e}")

    print(f"  ✅ 검증 완료: 정상 {valid_count}개 / 격리 {moved_count}개")
    print(f"  🗑️ 격리된 파일 위치: {QUARANTINE_DIR}")

if __name__ == "__main__":
    run_quarantine()