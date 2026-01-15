import pandas as pd
import shutil
import numpy as np
import gc
import sys
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import warnings

# Config 임포트
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import SILVER_DIR, GOLD_DIR, PENNY_STOCK_THRESHOLD, MAX_BUCKET_SIZE

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# [핵심] 허용 가능한 주식 분할/병합 비율 (Whitelist)
# 1.0 (동일), 2(2:1), 0.5(1:2), 3, 0.33, 4, 0.25, 5, 0.2, 10, 0.1, 20, 0.05
VALID_RATIOS = [1.0, 2.0, 0.5, 3.0, 1/3, 4.0, 0.25, 5.0, 0.2, 10.0, 0.1, 20.0, 0.05]
RATIO_TOLERANCE = 0.05  # 오차 허용 범위 (5%)

def get_metadata(file_path):
    try:
        df = pd.read_parquet(file_path, columns=['Close'])
        if df.empty: return None
        return {
            'ticker': file_path.stem,
            'path': file_path,
            'start_key': df.index[0].strftime("%Y-%m"),
            'start_date': df.index[0],
            'end_date': df.index[-1],
            'last_price': float(df['Close'].iloc[-1]),
            'count': len(df)
        }
    except: return None

def is_valid_merge_candidate(meta_a, meta_b, window=120):
    """
    [Strict Logic] 엄격한 병합 후보 검증
    1. 겹치는 구간 확인
    2. 가격 비율(Ratio)이 화이트리스트에 있는지 확인
    3. 보정 후 상관계수 확인
    """
    try:
        # 파일 로드 (Close만)
        df_a = pd.read_parquet(meta_a['path'], columns=['Close'])
        df_b = pd.read_parquet(meta_b['path'], columns=['Close'])
        
        # 1. 교집합 구간 찾기
        common_idx = df_a.index.intersection(df_b.index)
        if len(common_idx) < 30: # 최소 30일 이상 겹쳐야 판단 가능
            return False, 0.0

        # 최근 window만 사용 (너무 먼 과거 데이터 배제)
        if len(common_idx) > window:
            common_idx = common_idx[-window:]

        pa = df_a.loc[common_idx, 'Close']
        pb = df_b.loc[common_idx, 'Close']

        # 2. 가격 비율 검사 (Smart Ratio Check)
        # 평균 가격 비율 계산
        ratio_series = pa / pb
        median_ratio = ratio_series.median()
        
        # 화이트리스트 중 매칭되는 것이 있는지 확인
        is_ratio_valid = False
        target_ratio = 1.0
        
        for valid_r in VALID_RATIOS:
            # 비율이 오차 범위 내에 들어오는지 확인
            if abs(median_ratio - valid_r) / valid_r < RATIO_TOLERANCE:
                is_ratio_valid = True
                target_ratio = valid_r
                break
        
        if not is_ratio_valid:
            # 비율이 이상하면(예: 18.4배) 즉시 탈락
            return False, 0.0

        # 3. 보정 후 상관계수 계산
        # B의 가격을 비율만큼 보정하여 A와 비교
        pb_adjusted = pb * target_ratio
        
        if pa.std() < 1e-6 or pb_adjusted.std() < 1e-6:
            return False, 0.0
            
        corr = pa.corr(pb_adjusted)
        
        # 상관계수 0.99 이상이어야 통과
        return (corr > 0.99), target_ratio

    except Exception:
        return False, 0.0

def stitch_and_save(main_meta, sub_metas, output_dir):
    try:
        main_df = pd.read_parquet(main_meta['path'])
        
        for sub in sub_metas:
            # 검증 로직 재호출하여 정확한 비율 가져오기
            valid, ratio = is_valid_merge_candidate(main_meta, sub)
            if not valid: continue 

            sub_df = pd.read_parquet(sub['path'])
            
            # [Smart Adjust] 검증된 비율로 데이터 보정
            if abs(ratio - 1.0) > 0.01:
                cols = [c for c in ['Open','High','Low','Close','Adj Close'] if c in sub_df.columns]
                sub_df[cols] *= ratio
                if 'Volume' in sub_df.columns:
                    sub_df['Volume'] /= ratio 

            # 병합 (Main이 우선)
            main_df = main_df.combine_first(sub_df)
            
        # 중복 제거 및 정렬
        main_df = main_df[~main_df.index.duplicated(keep='last')]
        main_df.sort_index(inplace=True)

        # 최종 이상치 검사
        cols = [c for c in ['Open','High','Low','Close'] if c in main_df.columns]
        if (main_df[cols] <= 0).any().any(): return False

        # 저장
        save_path = output_dir / f"{main_meta['ticker']}.parquet"
        main_df.to_parquet(save_path)
        return True
    except: return False

def main():
    print(">>> [Script 04] Gold Layer 생성 (Strict Smart Merge)")
    
    if GOLD_DIR.exists(): shutil.rmtree(GOLD_DIR)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    
    silver_files = list(SILVER_DIR.glob("*.parquet"))
    buckets = defaultdict(list)
    for f in tqdm(silver_files, desc="Bucketing"):
        meta = get_metadata(f)
        if meta: buckets[meta['start_key']].append(meta)

    success_cnt = 0
    dedup_cnt = 0
    
    sorted_keys = sorted(buckets.keys())
    pbar = tqdm(sorted_keys)
    
    for key in pbar:
        candidates = buckets[key]
        n_total = len(candidates)
        
        # Smart Cap (이전 로직 유지)
        if n_total > MAX_BUCKET_SIZE:
            candidates.sort(key=lambda x: (x['last_price'] >= PENNY_STOCK_THRESHOLD, x['end_date'], x['count']), reverse=True)
            vips = candidates[:MAX_BUCKET_SIZE]
            others = candidates[MAX_BUCKET_SIZE:]
            
            pbar.set_description(f"Bucket {key} (Cap: {n_total}->{MAX_BUCKET_SIZE})")
            
            for item in others:
                shutil.copy2(item['path'], GOLD_DIR / f"{item['ticker']}.parquet")
                success_cnt += 1
            candidates = vips
        else:
            pbar.set_description(f"Bucket {key} ({n_total})")

        candidates.sort(key=lambda x: (x['end_date'], x['count']), reverse=True)
        processed = set()
        n = len(candidates)
        
        for i in range(n):
            main = candidates[i]
            if main['ticker'] in processed: continue
            
            if main['last_price'] < PENNY_STOCK_THRESHOLD:
                shutil.copy2(main['path'], GOLD_DIR / f"{main['ticker']}.parquet")
                success_cnt += 1
                processed.add(main['ticker'])
                continue

            duplicates = []
            
            # 병합 후보 탐색
            for j in range(i+1, n):
                sub = candidates[j]
                if sub['ticker'] in processed: continue
                
                # [엄격한 검사]
                is_mergeable, _ = is_valid_merge_candidate(main, sub)
                
                if is_mergeable:
                    duplicates.append(sub)
                    processed.add(sub['ticker'])
                    dedup_cnt += 1
            
            if duplicates:
                if stitch_and_save(main, duplicates, GOLD_DIR): success_cnt += 1
            else:
                shutil.copy2(main['path'], GOLD_DIR / f"{main['ticker']}.parquet")
                success_cnt += 1
            processed.add(main['ticker'])
            
        if n > 500: gc.collect()

    print(f"\n  ✅ Gold 생성 완료 (저장: {success_cnt}, 병합: {dedup_cnt})")

if __name__ == "__main__":
    main()