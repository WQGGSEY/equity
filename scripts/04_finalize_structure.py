import os
import shutil
import pandas as pd
from pathlib import Path

# ==========================================
# [설정]
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BRONZE_DIR = DATA_DIR / "bronze"
ARCHIVE_DIR = DATA_DIR / "archive"

# 대상 경로
OLD_FOLDER = BRONZE_DIR / "daily_prices"
NEW_FOLDER = BRONZE_DIR / "daily_prices_combined"
TEMP_REF = DATA_DIR / "temp_reference"

# 마스터 파일
UPDATED_MASTER = BRONZE_DIR / "master_ticker_list_updated.csv"
FINAL_MASTER = BRONZE_DIR / "master_ticker_list.csv" # 최종적으로 덮어쓸 파일

def main():
    print(">>> [Phase 3.5] Finalizing Folder Structure (Deployment)")
    
    # 1. 아카이브 폴더 생성
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. 기존 폴더 백업 (이름 변경하여 보존)
    if OLD_FOLDER.exists():
        backup_name = "daily_prices_backup_kaggle_original"
        backup_path = ARCHIVE_DIR / backup_name
        
        if backup_path.exists():
            print(f"⚠️ 백업 폴더가 이미 존재합니다: {backup_path}")
            # 안전을 위해 덮어쓰지 않고 건너뜀 (필요시 수동 삭제)
        else:
            print(f"📦 기존 폴더 아카이빙: {OLD_FOLDER.name} -> archive/{backup_name}")
            shutil.move(str(OLD_FOLDER), str(backup_path))
    
    # 3. 신규 폴더 승격 (daily_prices_combined -> daily_prices)
    if NEW_FOLDER.exists():
        print(f"🚀 신규 폴더 승격: {NEW_FOLDER.name} -> daily_prices")
        new_main_path = BRONZE_DIR / "daily_prices"
        
        if new_main_path.exists():
            # 혹시라도 이동 중에 에러나서 폴더가 남아있을 경우 대비
            print("❌ 에러: 타겟 폴더(daily_prices)가 비어있지 않습니다. 수동 확인이 필요합니다.")
            return
            
        shutil.move(str(NEW_FOLDER), str(new_main_path))
    else:
        print("❌ 에러: 신규 데이터 폴더(daily_prices_combined)가 없습니다! Phase 3가 제대로 완료되었나요?")
        return

    # 4. 임시 폴더(temp_reference) 정리
    # (이미 통합되었으므로 야후 원본 다운로드 폴더는 삭제해도 안전함)
    if TEMP_REF.exists():
        print(f"🗑️ 임시 참조 데이터 삭제: {TEMP_REF.name}")
        shutil.rmtree(TEMP_REF)

    # 5. 마스터 리스트 경로 보정 및 교체
    print("📝 마스터 리스트 경로 업데이트 중...")
    if UPDATED_MASTER.exists():
        df = pd.read_csv(UPDATED_MASTER)
        
        # 경로 문자열 수정: 'daily_prices_combined' -> 'daily_prices'
        # 파일은 이동했지만, CSV 안에 적힌 텍스트는 옛날 경로일 수 있으므로 수정
        df['file_path'] = df['file_path'].str.replace('daily_prices_combined', 'daily_prices')
        
        # 저장 (기존 master_ticker_list.csv를 덮어씀 -> 이제 이게 정본)
        df.to_csv(FINAL_MASTER, index=False)
        print(f"✅ 마스터 리스트 갱신 완료: {FINAL_MASTER}")
        
        # updated 임시 파일은 이제 필요 없으니 삭제
        os.remove(UPDATED_MASTER)
    else:
        print("⚠️ 경고: 업데이트된 마스터 리스트 파일이 없습니다.")
    
    print("\n" + "="*40)
    print(">>> [Phase 3.5 완료] 폴더 구조가 정리되었습니다.")
    print(f"  📂 메인 데이터: {BRONZE_DIR / 'daily_prices'}")
    print(f"  📜 마스터 파일: {FINAL_MASTER}")
    print("="*40)

if __name__ == "__main__":
    main()