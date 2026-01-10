import sys
import time
from pathlib import Path

# =========================================================
# [System Setup] 경로 설정 (Import보다 우선)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# =========================================================
# [Import Modules]
# =========================================================
try:
    from src.pipeline import (
        backup_manager as backup,
        universe_updater as universe,
        bronze_ingestor as bronze,
        silver_transformer as silver,
        gold_processor as gold,
        gold_auditor as gold_audit,
        gold_quarantine as quarantine  # [NEW] 격리 모듈 추가
    )
except ImportError as e:
    print(f"❌ [Critical] 모듈 임포트 실패: {e}")
    sys.exit(1)

def main():
    print("=" * 60)
    print(" 🚀 Quant Data Pipeline: Production Mode")
    print(f" 📂 Project Root: {BASE_DIR}")
    print("=" * 60)
    start_time = time.time()

    # ---------------------------------------------------------
    # Step 0: 안전 백업
    # ---------------------------------------------------------
    try:
        backup.run_backup()
    except Exception as e:
        print(f"❌ [CRITICAL] 백업 실패! ({e})")
        user_input = input("⚠️ 백업 없이 진행하시겠습니까? (y/n): ")
        if user_input.lower() != 'y': sys.exit(1)

    print("-" * 60)

    # ---------------------------------------------------------
    # Step 1: 유니버스 갱신 (선택사항, 필요시 주석 해제)
    # ---------------------------------------------------------
    # universe.update_universe()

    # ---------------------------------------------------------
    # Step 2: Bronze 데이터 수집
    # ---------------------------------------------------------
    try:
        bronze.ingest_bronze()
    except Exception as e:
        print(f"❌ Bronze 업데이트 실패: {e}")

    print("-" * 60)

    # ---------------------------------------------------------
    # Step 3: Silver 표준화
    # ---------------------------------------------------------
    try:
        silver.transform_silver()
    except Exception as e:
        print(f"❌ Silver 변환 실패: {e}")

    print("-" * 60)

    # ---------------------------------------------------------
    # Step 4: Gold 동기화
    # ---------------------------------------------------------
    try:
        gold.process_gold()
    except Exception as e:
        print(f"❌ Gold 동기화 실패: {e}")

    print("-" * 60)

    # ---------------------------------------------------------
    # Step 5: Final Audit & Quarantine
    # ---------------------------------------------------------
    try:
        # 1. 검사
        gold_audit.run_audit()
        # 2. 격리 (문제 파일 자동 이동)
        quarantine.run_quarantine()
    except Exception as e:
        print(f"❌ Final Audit/Quarantine 실패: {e}")

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f" ✅ All Sequences Completed in {elapsed:.2f} sec.")
    print("=" * 60)

if __name__ == "__main__":
    main()