import sys
import time
from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.pipeline import (
        backup_manager as backup,
        universe_updater as universe,  # [활성화]
        bronze_auditor as auditor,     # [명칭변경] bronze_auditor
        bronze_ingestor as ingestor,
        silver_transformer as silver,
        gold_processor as gold,
        gold_auditor as gold_audit,
        gold_quarantine as quarantine
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

    # Step 0: 안전 백업
    try:
        backup.run_backup()
    except Exception as e:
        print(f"❌ 백업 실패: {e}")
        if input("진행하시겠습니까? (y/n): ").lower() != 'y': sys.exit(1)
    print("-" * 60)

    # Step 1: 유니버스 갱신 (신규 상장 추가)
    try:
        universe.update_universe()
    except Exception as e:
        print(f"❌ 유니버스 갱신 실패: {e}")
    print("-" * 60)

    # Step 2: Bronze Auditor (파일 정합성 체크)
    try:
        auditor.run_audit()
    except Exception as e:
        print(f"❌ Auditor 실패: {e}")
    print("-" * 60)

    # Step 3: Bronze Ingestor (다운로드)
    try:
        ingestor.ingest_bronze()
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
    print("-" * 60)

    # Step 4: Silver Transformer
    try:
        silver.transform_silver()
    except Exception as e:
        print(f"❌ Silver 변환 실패: {e}")
    print("-" * 60)

    # Step 5: Gold Processor
    try:
        gold.process_gold()
    except Exception as e:
        print(f"❌ Gold 처리 실패: {e}")
    print("-" * 60)

    # Step 6: Final Audit & Quarantine
    try:
        gold_audit.run_audit()
        quarantine.run_quarantine()
    except Exception as e:
        print(f"❌ 최종 감사 실패: {e}")

    print("=" * 60)
    print(f" ✅ All Completed in {time.time() - start_time:.2f} sec")

if __name__ == "__main__":
    main()