import sys
import os
import shutil
import pandas as pd
from pathlib import Path

# 프로젝트 루트 경로 설정 (scripts 폴더에서 실행하든 루트에서 실행하든 동작하게)
# 현재 파일 위치: project/equity/scripts/05_quarantine_gold.py
# 루트 위치: project/equity
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import GOLD_DIR, MASTER_PATH

def quarantine_bad_files():
    print("=" * 50)
    print(" 🧹 Gold Data Quarantine Script")
    print("=" * 50)
    
    # 1. 리포트 확인
    # 리포트는 Gold 폴더 상위(data/gold/..) 혹은 data/audit_report.csv 등에 저장됨
    # gold_auditor.py가 저장한 위치: GOLD_DIR.parent / "audit_report.csv"
    report_path = GOLD_DIR.parent / "audit_report.csv"
    
    if not report_path.exists():
        print(f"❌ 에러 리포트가 없습니다: {report_path}")
        print("   먼저 'python -m src.pipeline.gold_auditor'를 실행하세요.")
        return

    df_error = pd.read_csv(report_path)
    
    # 2. 격리 폴더 생성 (data/quarantine)
    quarantine_dir = GOLD_DIR.parent / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  📂 격리 대상: {len(df_error)} 개 종목")
    print(f"  🗑️ 이동 경로: {quarantine_dir}")

    # 3. 장부 로드
    df_master = None
    if MASTER_PATH.exists():
        df_master = pd.read_csv(MASTER_PATH)

    moved_count = 0
    
    # 4. 이동 처리
    for ticker in df_error['ticker']:
        src_path = GOLD_DIR / f"{ticker}.parquet"
        dst_path = quarantine_dir / f"{ticker}.parquet"
        
        if src_path.exists():
            try:
                shutil.move(str(src_path), str(dst_path))
                moved_count += 1
                
                # 장부 업데이트 (비활성화)
                if df_master is not None:
                    mask = df_master['ticker'] == ticker
                    df_master.loc[mask, 'is_active'] = False
                    df_master.loc[mask, 'note'] = 'Quarantined: Integrity Fail'
            except Exception as e:
                print(f"    ⚠️ 이동 실패 ({ticker}): {e}")
    
    # 5. 장부 저장
    if df_master is not None:
        df_master.to_csv(MASTER_PATH, index=False)
        print("  📝 Master List 업데이트 완료 (is_active=False)")

    print("-" * 50)
    print(f"  ✅ 격리 완료: {moved_count} 개 파일 이동됨.")
    print(f"  ✨ 남은 Gold 파일: {len(list(GOLD_DIR.glob('*.parquet')))} 개")

if __name__ == "__main__":
    quarantine_bad_files()