import pandas as pd
import shutil
import numpy as np
import gc
from collections import defaultdict
from tqdm import tqdm
import warnings
from src.config import *

warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_metadata(file_path):
    try:
        # Close만 읽어서 속도 최적화
        df = pd.read_parquet(file_path, columns=['Close'])
        if df.empty: return None
        return {
            'ticker': file_path.stem,
            'path': file_path,
            'start_key': df.index[0].strftime("%Y-%m"), # Bucketing Key
            'start_date': df.index[0],
            'end_date': df.index[-1],
            'last_price': float(df['Close'].iloc[-1]),
            'count': len(df)
        }
    except: return None

def calculate_correlation_optimized(meta_a, meta_b, window=120):
    # [Anomaly 대응] Price Filter: 가격 차이가 50% 이상이면 다른 종목 (O(1) 컷)
    p1, p2 = meta_a['last_price'], meta_b['last_price']
    if p1 == 0 or p2 == 0: return 0.0
    
    # [Anomaly 대응] Penny Stock Skip: 둘 다 동전주면 비교 가치 없음
    if p1 < PENNY_STOCK_THRESHOLD and p2 < PENNY_STOCK_THRESHOLD: return 0.0
    
    if abs(p1 - p2) / max(p1, p2) > 0.5: return 0.0

    # Window Slicing Correlation
    try:
        df_a = pd.read_parquet(meta_a['path'], columns=['Close'])
        df_b = pd.read_parquet(meta_b['path'], columns=['Close'])
        common = df_a.index.intersection(df_b.index)
        
        if len(common) < 30: return 0.0
        if len(common) > window: common = common[-window:]
            
        sa = df_a.loc[common, 'Close'].astype('float32')
        sb = df_b.loc[common, 'Close'].astype('float32')
        
        if sa.std() < 1e-6 or sb.std() < 1e-6: return 0.0
        return sa.corr(sb)
    except: return 0.0

def stitch_and_save(main_meta, sub_metas, output_dir):
    try:
        main_df = pd.read_parquet(main_meta['path'])
        
        for sub in sub_metas:
            sub_df = pd.read_parquet(sub['path'])
            
            # [Anomaly 대응] Ratio Adjusting (데이터 단층 해결)
            common = main_df.index.intersection(sub_df.index)
            if not common.empty:
                pivot = common[-1] # 가장 최신 겹치는 날짜 기준
                p_main = float(main_df.loc[pivot, 'Close'])
                p_sub = float(sub_df.loc[pivot, 'Close'])
                if p_sub != 0:
                    ratio = p_main / p_sub
                    # 1% 이상 차이나면 비율 보정
                    if abs(1.0 - ratio) > 0.01:
                        cols = [c for c in ['Open','High','Low','Close','Adj Close'] if c in sub_df.columns]
                        sub_df[cols] *= ratio
                        if 'Volume' in sub_df.columns: sub_df['Volume'] /= ratio
            
            main_df = main_df.combine_first(sub_df)
            
        main_df = main_df[~main_df.index.duplicated(keep='last')]
        main_df.sort_index(inplace=True)

        # Gatekeeper (음수 및 급등락 방어)
        cols = [c for c in ['Open','High','Low','Close'] if c in main_df.columns]
        if (main_df[cols] < 0).any().any(): return False
        
        pct = main_df['Close'].pct_change().dropna()
        if ((pct > 3.0) | (pct < -0.9)).any(): return False

        save_path = output_dir / f"{main_meta['ticker']}.parquet"
        main_df.to_parquet(save_path)
        return True
    except: return False

def process_gold():
    print(">>> [Phase 4] Gold Processor (Final Logic Sync)")
    
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
        
        # [Anomaly 대응] Smart Safety Cap (무한루프 방지 + 중요 데이터 보존)
        if n_total > MAX_BUCKET_SIZE:
            # 정렬 기준: 1.정상가격(동전주X) 2.최신거래 3.데이터길이
            candidates.sort(key=lambda x: (x['last_price'] >= PENNY_STOCK_THRESHOLD, x['end_date'], x['count']), reverse=True)
            
            vips = candidates[:MAX_BUCKET_SIZE]
            others = candidates[MAX_BUCKET_SIZE:]
            
            pbar.set_description(f"Bucket {key} (Smart Cap: {n_total}->{MAX_BUCKET_SIZE})")
            
            # 탈락한 나머지는 단순 복사 (검사 생략)
            for item in others:
                shutil.copy2(item['path'], GOLD_DIR / f"{item['ticker']}.parquet")
                success_cnt += 1
            candidates = vips
        else:
            pbar.set_description(f"Bucket {key} ({n_total})")

        # Main Stitching Logic
        candidates.sort(key=lambda x: (x['end_date'], x['count']), reverse=True)
        processed = set()
        n = len(candidates)
        
        for i in range(n):
            main = candidates[i]
            if main['ticker'] in processed: continue
            
            # Main이 동전주면 병합 주체 포기
            if main['last_price'] < PENNY_STOCK_THRESHOLD:
                shutil.copy2(main['path'], GOLD_DIR / f"{main['ticker']}.parquet")
                success_cnt += 1
                processed.add(main['ticker'])
                continue

            duplicates = []
            for j in range(i+1, n):
                sub = candidates[j]
                if sub['ticker'] in processed: continue
                # 상관계수 확인
                if calculate_correlation_optimized(main, sub) > 0.99:
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

    print(f"  ✅ Gold 생성 완료 (저장: {success_cnt}, 병합: {dedup_cnt})")

if __name__ == "__main__":
    process_gold()