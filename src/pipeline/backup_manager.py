import shutil
import os
from datetime import datetime, timedelta
from pathlib import Path
from src.config import BASE_DIR, DATA_DIR, MASTER_PATH

# 백업 저장소 위치 (프로젝트 루트 / backups)
BACKUP_ROOT = BASE_DIR / "backups"

def clean_old_backups(keep_days=3):
    """
    오래된 백업 자동 삭제 (디스크 용량 관리)
    keep_days: 며칠 치 백업을 남길지 설정 (기본 3일)
    """
    print(f"  🧹 오래된 백업 정리 중 (최근 {keep_days}일 유지)...")
    
    if not BACKUP_ROOT.exists():
        return

    # 백업 폴더 목록 가져오기
    backups = sorted(list(BACKUP_ROOT.glob("*")))
    
    # 삭제 기준일 (오늘 - keep_days)
    cutoff_date = datetime.now() - timedelta(days=keep_days)
    
    deleted_cnt = 0
    for backup_path in backups:
        try:
            # 폴더명(YYYY-MM-DD_HHMM)에서 날짜 파싱
            folder_name = backup_path.name
            # 날짜 형식이 아니면 스킵 (안전장치)
            try:
                backup_date = datetime.strptime(folder_name.split("_")[0], "%Y-%m-%d")
            except ValueError:
                continue 

            if backup_date < cutoff_date:
                if backup_path.is_dir():
                    shutil.rmtree(backup_path)
                    print(f"    🗑️ 삭제됨: {folder_name}")
                    deleted_cnt += 1
        except Exception as e:
            print(f"    ⚠️ 정리 중 에러 발생 ({backup_path.name}): {e}")

    if deleted_cnt == 0:
        print("    - 삭제할 오래된 백업이 없습니다.")

def run_backup():
    print(">>> [Pipeline 00] 시스템 전체 백업 (Safety First)")
    
    # 1. 백업 폴더 생성 (이름: 2026-01-10_1320)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    current_backup_dir = BACKUP_ROOT / timestamp
    current_backup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  📦 백업 타겟: {current_backup_dir.relative_to(BASE_DIR)}")

    # 2. 필수 파일 백업 (Master List)
    if MASTER_PATH.exists():
        shutil.copy2(MASTER_PATH, current_backup_dir / "master_ticker_list.csv")
        print("    ✅ Master List 백업 완료")
    else:
        print("    ⚠️ Master List가 없습니다 (백업 건너뜁니다)")

    # 3. 데이터 폴더 백업 (Bronze, Silver, Gold)
    # 데이터가 많으면 시간이 좀 걸리지만, 안전을 위해 필수입니다.
    target_dirs = ["bronze", "silver", "gold"]
    
    for layer in target_dirs:
        src_path = DATA_DIR / layer
        dst_path = current_backup_dir / layer
        
        if src_path.exists():
            print(f"    ⏳ {layer.capitalize()} Layer 복사 중... (데이터 크기에 따라 소요됨)")
            # dirs_exist_ok=True: 덮어쓰기 허용
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            print(f"    ✅ {layer.capitalize()} 백업 완료")
        else:
            print(f"    ℹ️ {layer.capitalize()} 폴더가 비어있어 건너뜁니다.")

    # 4. 오래된 백업 정리 (3일치만 보관)
    clean_old_backups(keep_days=3)
    
    print(f"  ✨ 시스템 백업 완료. 안전하게 작업하세요.")
    print("-" * 40)

if __name__ == "__main__":
    run_backup()