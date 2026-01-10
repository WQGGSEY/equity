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

def calculate_correlation_optimized(meta_a, meta_b, window=120):
    p1, p2 = meta_a['last_price'], meta_b['last_price']
    if p1 == 0 or p2 == 0: return 0.0
    
    # Penny Stock Skip
    if p1 < PENNY_STOCK_THRESHOLD and p2 < PENNY_STOCK_THRESHOLD: return 0.0
    
    # ❌ [삭제됨] 가격 차이 50% 필터 제거
    # if abs(p1 - p2) / max(p1, p2) > 0.5: return 0.0

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
            # Ratio Adjusting
            common = main_df.index.intersection(sub_df.index)
            if not common.empty:
                pivot = common[-1]
                p_main = float(main_df.loc[pivot, 'Close'])
                p_sub = float(sub_df.loc[pivot, 'Close'])
                if p_sub != 0:
                    ratio = p_main / p_sub
                    if abs(1.0 - ratio) > 0.01:
                        cols = [c for c in ['Open','High','Low','Close','Adj Close'] if c in sub_df.columns]
                        sub_df[cols] *= ratio
                        if 'Volume' in sub_df.columns: sub_df['Volume'] /= ratio
            
            main_df = main_df.combine_first(sub_df)
            
        main_df = main_df[~main_df.index.duplicated(keep='last')]
        main_df.sort_index(inplace=True)

        cols = [c for c in ['Open','High','Low','Close'] if c in main_df.columns]
        if (main_df[cols] < 0).any().any(): return False

        pct = main_df['Close'].pct_change().dropna()
        if ((pct > 3.0) | (pct < -0.9)).any(): return False

        save_path = output_dir / f"{main_meta['ticker']}.parquet"
        main_df.to_parquet(save_path)
        return True
    except: return False

def main():
    print(">>> [Script 04] Gold Layer 생성 (Price Filter Removed)")
    
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
            for j in range(i+1, n):
                sub = candidates[j]
                if sub['ticker'] in processed: continue
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

    print(f"\n  ✅ Gold 생성 완료 (저장: {success_cnt}, 병합: {dedup_cnt})")

if __name__ == "__main__":
    main()