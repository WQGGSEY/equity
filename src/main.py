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
        backup_manager as backup,  # [NEW] 백업 매니저 추가
        bronze_auditor,
        bronze_ingestor,
        silver_transformer,
        gold_processor,
        gold_auditor
    )
except ImportError as e:
    print(f"❌ [Critical] 모듈 임포트 실패: {e}")
    sys.exit(1)

def main():
    start_time = time.time()
    print("=" * 60)
    print(" 🚀 Quant Data Pipeline: Production Mode")
    print(f" 📂 Project Root: {BASE_DIR}")
    print("=" * 60)

    # ---------------------------------------------------------
    # Step 0: 안전 백업 (Safety First)
    # ---------------------------------------------------------
    # 작업 시작 전, 현재 상태를 통째로 백업합니다.
    try:
        backup.run_backup()
    except Exception as e:
        print(f"❌ [CRITICAL] 백업 실패! ({e})")
        # 백업 실패 시 진행할지 말지 결정해야 함. 
        # 안전을 위해 여기서 멈추는 것을 권장.
        user_input = input("⚠️ 백업 없이 진행하시겠습니까? (y/n): ")
        if user_input.lower() != 'y':
            sys.exit(1)

    print("-" * 60)

    # ---------------------------------------------------------
    # Phase 1: Bronze Auditor
    # ---------------------------------------------------------
    try:
        bronze_auditor.run_audit()
    except Exception as e:
        print(f"❌ [Phase 1 Failed] Auditor Error: {e}")
        sys.exit(1)

    print("-" * 60)

    # ---------------------------------------------------------
    # Phase 2: Bronze Ingestor
    # ---------------------------------------------------------
    try:
        bronze_ingestor.ingest_bronze()
    except Exception as e:
        print(f"❌ [Phase 2 Failed] Ingestor Error: {e}")

    print("-" * 60)

    # ---------------------------------------------------------
    # Phase 3: Silver Transformer
    # ---------------------------------------------------------
    try:
        silver_transformer.transform_silver()
    except Exception as e:
        print(f"❌ [Phase 3 Failed] Transformer Error: {e}")

    print("-" * 60)

    # ---------------------------------------------------------
    # Phase 4: Gold Processor
    # ---------------------------------------------------------
    try:
        gold_processor.process_gold()
    except Exception as e:
        print(f"❌ [Phase 4 Failed] Processor Error: {e}")

    print("-" * 60)

    # ---------------------------------------------------------
    # Final Audit
    # ---------------------------------------------------------
    try:
        gold_auditor.run_audit()
    except Exception as e:
        print(f"❌ [Final Audit Failed] Error: {e}")

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f" ✅ All Sequences Completed in {elapsed:.2f} sec.")
    print("=" * 60)

if __name__ == "__main__":
    main()