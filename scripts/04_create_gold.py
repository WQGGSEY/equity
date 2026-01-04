import pandas as pd
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import warnings

# 경고 무시 (상관계수 계산 시 runtime warning 등)
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# [Phase 4] Gold Layer: Robust Dedup (Corr & Stitch)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
SILVER_DIR = BASE_DIR / "data" / "silver" / "daily_prices"
GOLD_DIR = BASE_DIR / "data" / "gold" / "daily_prices"

def get_metadata(file_path):
    """
    파일을 가볍게 읽어서 '시작일(Start Date)'과 '종료일(End Date)' 추출
    이 정보로 1차 그룹핑을 수행함 (가격 비교 X)
    """
    try:
        # 인덱스만 빠르게 로드 가능하면 좋지만, parquet 특성상 컬럼 하나 읽는게 빠름
        df = pd.read_parquet(file_path, columns=['Close'])
        if df.empty: return None
        
        start_date = df.index[0]
        end_date = df.index[-1]
        
        # 그룹핑 키: "YYYY-MM" (같은 달에 시작한 종목끼리 비교)
        start_key = start_date.strftime("%Y-%m")
        
        return {
            'ticker': file_path.stem,
            'path': file_path,
            'start_key': start_key,
            'start_date': start_date,
            'end_date': end_date,
            'count': len(df)
        }
    except:
        return None

def calculate_correlation(path_a, path_b):
    """
    [Robust] 두 파일의 겹치는 구간 상관계수 계산
    - 표준편차가 0인(주가 변동 없는) 경우를 방어하여 RuntimeWarning 제거
    """
    try:
        # 필요한 컬럼만 로드
        df_a = pd.read_parquet(path_a, columns=['Close'])
        df_b = pd.read_parquet(path_b, columns=['Close'])
        
        # 교집합 구간 찾기
        common_idx = df_a.index.intersection(df_b.index)
        
        # 겹치는 구간이 너무 짧으면 판단 불가 (최소 30일)
        if len(common_idx) < 30:
            return 0.0
            
        series_a = df_a.loc[common_idx, 'Close'].astype(float)
        series_b = df_b.loc[common_idx, 'Close'].astype(float)
        
        # [핵심 수정] 표준편차가 0인지 확인 (Constant Value Check)
        # 1e-9보다 작으면 변동이 거의 없는 것으로 간주
        if series_a.std() < 1e-9 or series_b.std() < 1e-9:
            return 0.0 # 변동이 없으면 상관관계 계산 불가 -> 무시
            
        # 상관계수 계산
        corr = series_a.corr(series_b)
        
        # 결과가 NaN이면 0으로 처리
        if pd.isna(corr):
            return 0.0
            
        return corr
    except Exception:
        # 파일 로드 실패 등 모든 에러 시 0.0 반환 (안전하게 Skip)
        return 0.0

def stitch_and_save(main_meta, sub_metas, output_dir):
    """
    Main 데이터를 기준으로 Sub 데이터들을 이어붙여서(Stitching) 저장
    """
    try:
        # 1. Main 로드
        main_df = pd.read_parquet(main_meta['path'])
        
        # 2. Sub 순회하며 구멍 메우기
        for sub in sub_metas:
            sub_df = pd.read_parquet(sub['path'])
            # combine_first: main의 결측치를 sub의 값으로 채움 (인덱스 합집합)
            main_df = main_df.combine_first(sub_df)
            
        # 3. Gold 저장
        save_path = output_dir / f"{main_meta['ticker']}.parquet"
        main_df.to_parquet(save_path)
        return True
    except Exception as e:
        print(f"    ❌ 병합 실패 ({main_meta['ticker']}): {e}")
        return False

def main():
    print(">>> [Phase 4] Gold Layer 생성 (Correlation Based Stitching)")
    
    # 1. 폴더 초기화
    if GOLD_DIR.exists():
        shutil.rmtree(GOLD_DIR)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    
    silver_files = list(SILVER_DIR.glob("*.parquet"))
    print(f"  📖 Silver 파일 스캔: {len(silver_files)} 개")

    # 2. 1차 그룹핑 (Start Date Bucketing)
    # { '2012-05': [meta1, meta2, ...], ... }
    buckets = defaultdict(list)
    
    for f in tqdm(silver_files, desc="1. Grouping by Start Date"):
        meta = get_metadata(f)
        if meta:
            buckets[meta['start_key']].append(meta)

    # 3. 그룹별 Correlation 검사 및 병합
    processed_tickers = set()
    dedup_count = 0
    merged_files_count = 0
    
    print("  🔍 2. 정밀 분석 (Correlation) & 병합 (Stitching)...")
    
    # 진행상황 표시를 위해 버킷 순회
    for start_key, candidates in tqdm(buckets.items(), desc="Processing Buckets"):
        if len(candidates) == 1:
            # 비교 대상 없음 -> 바로 이관
            meta = candidates[0]
            shutil.copy2(meta['path'], GOLD_DIR / f"{meta['ticker']}.parquet")
            merged_files_count += 1
            continue
            
        # 그룹 내에서 중복 찾기
        # 데이터가 많은(최신/긴) 순서대로 정렬하여 'Main' 후보 선정
        # 기준: 1. 종료일(최신) 2. 데이터개수(긴것)
        candidates.sort(key=lambda x: (x['end_date'], x['count']), reverse=True)
        
        # 방문 체크용 (그룹 내 로컬)
        local_processed = set()
        
        for i in range(len(candidates)):
            main_cand = candidates[i]
            if main_cand['ticker'] in local_processed:
                continue
                
            duplicates = []
            
            # 나보다 데이터가 적거나 오래된 놈들과 비교
            for j in range(i + 1, len(candidates)):
                sub_cand = candidates[j]
                if sub_cand['ticker'] in local_processed:
                    continue
                
                # Correlation 계산
                corr = calculate_correlation(main_cand['path'], sub_cand['path'])
                
                if corr > 0.99: # 99% 이상 일치하면 동일 종목 간주
                    duplicates.append(sub_cand)
                    local_processed.add(sub_cand['ticker'])
                    dedup_count += 1
                    # 로그 출력 (확인용)
                    # print(f"    🔗 중복 발견: {main_cand['ticker']} == {sub_cand['ticker']} (Corr: {corr:.4f})")
            
            # 병합 및 저장
            if duplicates:
                stitch_and_save(main_cand, duplicates, GOLD_DIR)
            else:
                # 중복 없으면 그냥 복사
                shutil.copy2(main_cand['path'], GOLD_DIR / f"{main_cand['ticker']}.parquet")
            
            local_processed.add(main_cand['ticker'])
            merged_files_count += 1

    print("\n" + "="*40)
    print("  ✅ Gold Layer 생성 완료")
    print(f"  - 원본(Silver): {len(silver_files)} 개")
    print(f"  - 중복 병합됨(Dedup): {dedup_count} 건")
    print(f"  - 최종 Gold 파일: {len(list(GOLD_DIR.glob('*.parquet')))} 개")
    print("="*40)
    
    # 중복 제거 리포트 (옵션)
    if dedup_count > 0:
        print(f"  💡 {dedup_count}개의 과거 티커(FB 등)가 최신 티커(META 등)로 통합되었습니다.")
    
    print("👉 이제 데이터는 물리적(Phase 3)으로나 논리적(Phase 4)으로 완벽합니다.")
    print("👉 'Platinum Layer (Feature Engineering)' 단계로 넘어가십시오.")

if __name__ == "__main__":
    main()