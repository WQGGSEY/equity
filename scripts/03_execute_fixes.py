import os
import json
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ==========================================
# [설정]
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

# 입력 경로
OLD_KAGGLE_DIR = BASE_DIR / "data" / "bronze" / "daily_prices"
NEW_YAHOO_DIR = BASE_DIR / "data" / "temp_reference"
PLAN_PATH = BASE_DIR / "data" / "bronze" / "fix_plan.json"
OLD_MASTER_PATH = BASE_DIR / "data" / "bronze" / "master_ticker_list.csv"

# 출력 경로 (신대륙)
TARGET_DIR = BASE_DIR / "data" / "bronze" / "daily_prices_combined"
NEW_MASTER_PATH = BASE_DIR / "data" / "bronze" / "master_ticker_list_updated.csv"

def main():
    print(">>> [Phase 3] Constructing New Dataset (Copy Mode)")
    
    # 1. 준비
    if not PLAN_PATH.exists():
        raise FileNotFoundError("작전 지도(fix_plan.json)가 없습니다.")
    
    with open(PLAN_PATH, 'r') as f:
        plan = json.load(f)
        
    # 기존 마스터 로드 (메타데이터 참조용)
    if not OLD_MASTER_PATH.exists():
         raise FileNotFoundError("기존 마스터 리스트가 없습니다.")
         
    old_master = pd.read_csv(OLD_MASTER_PATH)
    
    # [FIX] 중복 티커 제거 로직 추가
    # 티커가 중복되면 첫 번째 행만 남기고 제거합니다.
    # (메타데이터 조회용이므로 중복된 것 중 하나만 있어도 무방합니다)
    duplicate_count = old_master.duplicated(subset=['ticker']).sum()
    if duplicate_count > 0:
        print(f"⚠️ 경고: 마스터 리스트에서 {duplicate_count}개의 중복 티커를 발견하여 제거합니다.")
        old_master = old_master.drop_duplicates(subset=['ticker'], keep='first')
    
    # ticker를 인덱스로 만들어 빠른 조회 (이제 에러 안 남)
    master_lookup = old_master.set_index('ticker').to_dict('index')
    
    # 타겟 폴더 생성
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  📂 타겟 폴더: {TARGET_DIR}")
    
    new_master_rows = []
    
    # 2. 실행 루프
    print(">>> 파일 이동 및 통합 시작...")
    
    stats = {"copied_yahoo": 0, "copied_kaggle": 0, "errors": 0}
    
    for item in tqdm(plan, desc="Executing Plan"):
        action = item['action']
        original_ticker = item['ticker']
        
        # 기본 메타데이터 가져오기
        meta = master_lookup.get(original_ticker, {})
        
        # 새 항목을 위한 기본 딕셔너리
        new_entry = {
            "ticker": original_ticker,
            "source": "unknown",
            "is_active": False,
            "original_ticker": original_ticker, 
            "start_date": meta.get('start_date'),
            "end_date": meta.get('end_date'),
            "count": meta.get('count'),
            "file_path": ""
        }

        try:
            # ---------------------------------------------------
            # CASE A: Yahoo 데이터 채택 (MERGE, RENAME)
            # ---------------------------------------------------
            if action in ['MERGE', 'RENAME']:
                target_ticker = item['new_name'] if action == 'RENAME' else original_ticker
                
                # 소스: Yahoo Temp
                src_path = NEW_YAHOO_DIR / f"ticker={target_ticker}" / "price.parquet"
                
                if not src_path.exists():
                    if 'target_path' in item and item['target_path']:
                        src_path = Path(item['target_path'])
                
                if src_path.exists():
                    # 타겟 경로 설정
                    dest_folder = TARGET_DIR / f"ticker={target_ticker}"
                    dest_folder.mkdir(exist_ok=True)
                    dest_path = dest_folder / "price.parquet"
                    
                    # 복사
                    shutil.copy2(src_path, dest_path)
                    
                    # 마스터 정보 업데이트
                    new_entry['ticker'] = target_ticker
                    new_entry['source'] = 'yahoo'
                    new_entry['is_active'] = True
                    new_entry['file_path'] = str(dest_path.relative_to(BASE_DIR))
                    new_entry['last_updated'] = pd.Timestamp.now().strftime('%Y-%m-%d')
                    
                    stats['copied_yahoo'] += 1
                else:
                    # 야후 파일이 없으면 에러 카운트
                    # (하지만 plan 생성 시점과 실행 시점 차이로 없을 수도 있음)
                    print(f"⚠️ [Skip] Yahoo Source Missing: {target_ticker}")
                    stats['errors'] += 1
                    continue

            # ---------------------------------------------------
            # CASE B: Kaggle 데이터 보존 (FORK, MISSING)
            # ---------------------------------------------------
            elif action in ['FORK', 'MISSING']:
                target_ticker = item['new_name'] if action == 'FORK' else original_ticker
                
                # 소스: Old Kaggle
                src_path_str = meta.get('file_path', '')
                if not src_path_str:
                    src_path = OLD_KAGGLE_DIR / f"ticker={original_ticker}" / "price.parquet"
                else:
                    src_path = BASE_DIR / src_path_str
                
                if src_path.exists():
                    dest_folder = TARGET_DIR / f"ticker={target_ticker}"
                    dest_folder.mkdir(exist_ok=True)
                    dest_path = dest_folder / "price.parquet"
                    
                    shutil.copy2(src_path, dest_path)
                    
                    new_entry['ticker'] = target_ticker
                    new_entry['source'] = 'kaggle'
                    new_entry['is_active'] = False
                    new_entry['file_path'] = str(dest_path.relative_to(BASE_DIR))
                    
                    stats['copied_kaggle'] += 1
                else:
                    print(f"⚠️ [Skip] Kaggle Source Missing: {original_ticker}")
                    stats['errors'] += 1
                    continue
            
            # 리스트에 추가
            new_master_rows.append(new_entry)

        except Exception as e:
            print(f"❌ Error processing {original_ticker}: {e}")
            stats['errors'] += 1

    # 3. 새로운 마스터 파일 저장
    new_master_df = pd.DataFrame(new_master_rows)
    # 최종적으로 여기서도 중복 제거 (혹시 모르니)
    new_master_df = new_master_df.drop_duplicates(subset=['ticker'])
    
    new_master_df.to_csv(NEW_MASTER_PATH, index=False)
    
    print("\n" + "="*40)
    print(">>> [Phase 3 완료] 새로운 데이터셋 구축 성공")
    print("="*40)
    print(f"  ✅ Yahoo 데이터 이식: {stats['copied_yahoo']}건")
    print(f"  ✅ Kaggle 데이터 보존: {stats['copied_kaggle']}건")
    print(f"  ❌ 에러/스킵: {stats['errors']}건")
    print(f"  📂 데이터 위치: {TARGET_DIR}")
    print(f"  📝 새 마스터 파일: {NEW_MASTER_PATH}")
    print("\n[안내] 검증 후 기존 폴더(daily_prices)를 삭제하거나 아카이빙하세요.")

if __name__ == "__main__":
    main()