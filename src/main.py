import sys
import time
from pathlib import Path

# =========================================================
# [System Setup]
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# =========================================================
# [Import Modules]
# =========================================================
try:
    from src.pipeline import (
        backup_manager as backup,      # Step 0: 백업 (필수)
        universe_updater as universe,  # Step 1: 유니버스 갱신
        bronze_auditor as auditor,     # Step 2: 파일 검사 (이전 universe_updater와 분리됨)
        bronze_ingestor as ingestor,   # Step 3: 다운로드
        silver_transformer as silver,  # Step 4: NAN 처리 및 표준화
        gold_processor as gold,        # Step 5: 병합 및 로직 처리
        gold_auditor as gold_audit,    # Step 6: 검사
        gold_quarantine as quarantine,  # Step 6: 격리
        platinum_processor as platinum # Step 7: feature 생성
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
    # Step 0: 안전 백업 (Safety First)
    # ---------------------------------------------------------
    try:
        backup.run_backup()
    except Exception as e:
        print(f"❌ 백업 실패: {e}")
        # 백업 실패는 치명적일 수 있으므로 사용자 확인
        if input("⚠️ 백업 없이 진행하시겠습니까? (y/n): ").lower() != 'y': sys.exit(1)

    print("-" * 60)

    # ---------------------------------------------------------
    # Step 1: 유니버스 갱신 (IPO 감지)
    # ---------------------------------------------------------
    try:
        universe.update_universe()
    except Exception as e:
        print(f"❌ 유니버스 갱신 실패: {e}")

    print("-" * 60)

    # ---------------------------------------------------------
    # Step 2: Bronze Auditor (파일 정합성 체크)
    # ---------------------------------------------------------
    try:
        auditor.run_audit()
    except Exception as e:
        print(f"❌ Auditor 실패: {e}")

    print("-" * 60)

    # ---------------------------------------------------------
    # Step 3: Bronze Ingestor (데이터 다운로드)
    # ---------------------------------------------------------
    try:
        ingestor.ingest_bronze()
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")

    print("-" * 60)

    # ---------------------------------------------------------
    # Step 4: Silver Transformer (표준화 & NAN 처리)
    # ---------------------------------------------------------
    try:
        silver.transform_silver()
    except Exception as e:
        print(f"❌ Silver 변환 실패: {e}")

    print("-" * 60)

    # ---------------------------------------------------------
    # Step 5: Gold Processor (통합 및 병합)
    # ---------------------------------------------------------
    try:
        gold.process_gold()
    except Exception as e:
        print(f"❌ Gold 처리 실패: {e}")

    print("-" * 60)

    # ---------------------------------------------------------
    # Step 6: Final Audit & Quarantine (최종 점검)
    # ---------------------------------------------------------
    try:
        gold_audit.run_audit()
        quarantine.run_quarantine()
    except Exception as e:
        print(f"❌ 최종 감사 실패: {e}")

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f" ✅ All Completed in {elapsed:.2f} sec")
    print("=" * 60)

    # ---------------------------------------------------------
    # Step 7: feature 생성
    # ---------------------------------------------------------
    try:
        platinum.process_features()
    except Exception as e:
        print(f"❌ Feature 생성 실패: {e}")

if __name__ == "__main__":
    main()