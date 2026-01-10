import pandas as pd
import shutil
import numpy as np
import gc
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import warnings

# 경고 무시
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# ==========================================
# [Phase 4] Gold Layer: Ratio-Adjusted Stitching
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
SILVER_DIR = BASE_DIR / "data" / "silver" / "daily_prices"
GOLD_DIR = BASE_DIR / "data" / "gold" / "daily_prices"

def get_metadata(file_path):
    try:
        df = pd.read_parquet(file_path, columns=['Close'])
        if df.empty: return None
        
        start_date = df.index[0]
        end_date = df.index[-1]
        start_key = start_date.strftime("%Y-%m")
        last_price = float(df['Close'].iloc[-1])
        
        return {
            'ticker': file_path.stem,
            'path': file_path,
            'start_key': start_key,
            'start_date': start_date,
            'end_date': end_date,
            'last_price': last_price,
            'count': len(df)
        }
    except:
        return None

def calculate_correlation_optimized(meta_a, meta_b, window=120):
    """
    상관계수 계산 (가격 필터 + 윈도우 슬라이싱)
    * 주의: 상관계수는 스케일(x10, x0.1)에 영향을 받지 않으므로
      액면분할 전/후 데이터라도 상관계수는 높게 나옵니다.
      따라서 '비율 보정'은 stitch 단계에서 별도로 수행해야 합니다.
    """
    # 1. Price Filter (너무 터무니없는 가격 차이는 필터링하되, 액면분할 고려하여 범위 완화)
    # 액면분할은 보통 1/10, 1/50 등이므로 비율로 체크해야 함.
    # 하지만 여기서는 '상관계수'를 믿고 Price Filter는 최소한의 방어(0원 등)만 수행
    p1, p2 = meta_a['last_price'], meta_b['last_price']
    if p1 == 0 or p2 == 0: return 0.0
    
    # 2. Correlation
    try:
        df_a = pd.read_parquet(meta_a['path'], columns=['Close'])
        df_b = pd.read_parquet(meta_b['path'], columns=['Close'])
        
        common = df_a.index.intersection(df_b.index)
        if len(common) < 30: return 0.0
        
        if len(common) > window:
            common = common[-window:]
            
        sa = df_a.loc[common, 'Close'].astype('float32')
        sb = df_b.loc[common, 'Close'].astype('float32')
        
        if sa.std() < 1e-6 or sb.std() < 1e-6: return 0.0
        
        return sa.corr(sb)
    except:
        return 0.0

def stitch_and_save(main_meta, sub_metas, output_dir):
    """
    [핵심 수정] Ratio-Based Adjusting Stitching
    Main(최신/Yahoo) 데이터를 기준으로, Sub(과거/Kaggle) 데이터의 스케일을 보정하여 병합.
    """
    try:
        # 1. Main 로드 (기준 데이터 - Yahoo/최신)
        main_df = pd.read_parquet(main_meta['path'])
        
        # 2. Sub 순회하며 보정 후 병합
        for sub in sub_metas:
            sub_df = pd.read_parquet(sub['path'])
            
            # --- [Adjusting Logic Start] ---
            # 겹치는 구간 찾기
            common_idx = main_df.index.intersection(sub_df.index)
            
            if not common_idx.empty:
                # 겹치는 구간 중 '가장 최신 날짜'를 기준으로 비율 계산
                # (과거 날짜보다 최신 날짜가 데이터 정합성이 높을 확률이 큼)
                pivot_date = common_idx[-1]
                
                p_main = float(main_df.loc[pivot_date, 'Close'])
                p_sub = float(sub_df.loc[pivot_date, 'Close'])
                
                if p_sub != 0:
                    ratio = p_main / p_sub
                    
                    # 비율이 1.0과 유의미하게 차이나면 (예: 1% 이상) -> 보정 수행
                    # 예: main=12만원, sub=120만원 -> ratio=0.1
                    if abs(1.0 - ratio) > 0.01:
                        # 숫자형 컬럼 전체에 비율 곱하기 (Open, High, Low, Close, Volume 등)
                        # 주의: Volume은 주가가 낮아지면(액면분할) 보통 늘어나므로 반대로 나눠야 할 수도 있으나,
                        # Yahoo의 수정주가(Adj Close) 로직을 따라가기 위해 가격은 곱하고, 볼륨은 나누는게 정석.
                        # 하지만 여기서는 단순화를 위해 가격만 보정하거나, Volume도 같은 비율로 조정(Split의 역)
                        
                        # [Price Correction]
                        price_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close'] if c in sub_df.columns]
                        sub_df[price_cols] = sub_df[price_cols] * ratio
                        
                        # [Volume Correction]
                        # 액면분할(주가 1/10) -> 거래량(10배) 이어야 함.
                        # 주가 ratio가 0.1이면, Volume은 1/0.1 = 10배가 되어야 함.
                        if 'Volume' in sub_df.columns:
                            sub_df['Volume'] = sub_df['Volume'] / ratio
                            
                        # print(f"    🔧 Adjusting {sub['ticker']} by ratio {ratio:.4f} (Pivot: {pivot_date.date()})")
            
            # --- [Adjusting Logic End] ---

            # 3. 병합 (Main 우선, 빈 곳을 보정된 Sub로 채움)
            main_df = main_df.combine_first(sub_df)
            
        # 4. 데이터 정리
        main_df = main_df[~main_df.index.duplicated(keep='last')]
        main_df.sort_index(inplace=True)

        # 5. Gatekeeper (음수 및 급등락 확인)
        cols = [c for c in ['Open','High','Low','Close'] if c in main_df.columns]
        if (main_df[cols] < 0).any().any(): return False

        pct = main_df['Close'].pct_change().dropna()
        # 보정을 했음에도 불구하고 미친 변동성이 있다면 Reject
        if ((pct > 3.0) | (pct < -0.9)).any():
            return False

        # 6. 저장
        save_path = output_dir / f"{main_meta['ticker']}.parquet"
        main_df.to_parquet(save_path)
        return True
    except Exception as e:
        # print(f"Error merging: {e}")
        return False

def main():
    print(">>> [Phase 4] Gold Layer 생성 (Ratio Adjusted)")
    
    if GOLD_DIR.exists(): shutil.rmtree(GOLD_DIR)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    
    silver_files = list(SILVER_DIR.glob("*.parquet"))
    print(f"  📖 Silver 파일 스캔: {len(silver_files)} 개")

    buckets = defaultdict(list)
    for f in tqdm(silver_files, desc="Bucketing"):
        meta = get_metadata(f)
        if meta: buckets[meta['start_key']].append(meta)

    success_count = 0
    dedup_count = 0
    
    print("  🔍 분석 및 병합 (Price Adjusting 적용)...")
    
    sorted_keys = sorted(buckets.keys())
    pbar = tqdm(sorted_keys)
    
    for key in pbar:
        candidates = buckets[key]
        n = len(candidates)
        pbar.set_description(f"Bucket {key} ({n})")
        
        if n == 1:
            meta = candidates[0]
            shutil.copy2(meta['path'], GOLD_DIR / f"{meta['ticker']}.parquet")
            success_count += 1
            continue
            
        candidates.sort(key=lambda x: (x['end_date'], x['count']), reverse=True)
        processed = set()
        
        for i in range(n):
            main = candidates[i]
            if main['ticker'] in processed: continue
            
            duplicates = []
            for j in range(i + 1, n):
                sub = candidates[j]
                if sub['ticker'] in processed: continue
                
                corr = calculate_correlation_optimized(main, sub)
                if corr > 0.99:
                    duplicates.append(sub)
                    processed.add(sub['ticker'])
                    dedup_count += 1
            
            if duplicates:
                saved = stitch_and_save(main, duplicates, GOLD_DIR)
                if saved: success_count += 1
            else:
                shutil.copy2(main['path'], GOLD_DIR / f"{main['ticker']}.parquet")
                success_count += 1
            
            processed.add(main['ticker'])
            
        if n > 1000: gc.collect()

    print("\n" + "="*40)
    print(f"  ✅ Gold Layer 완료")
    print(f"  - 최종 저장: {success_count}")
    print(f"  - 통합 및 보정: {dedup_count}")
    print("="*40)

if __name__ == "__main__":
    main()