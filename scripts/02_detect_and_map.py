import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

# ==========================================
# [설정]
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_PATH = BASE_DIR / "data" / "bronze" / "master_ticker_list.csv"
KAGGLE_DIR = BASE_DIR / "data" / "bronze" / "daily_prices"
REF_DIR = BASE_DIR / "data" / "temp_reference"
OUTPUT_PLAN = BASE_DIR / "data" / "bronze" / "fix_plan.json"

# 병렬 처리 개수 (M2 성능 활용)
N_JOBS = 6

# [강화된 매칭 설정 - User Suggestion: 20 Days]
MIN_OVERLAP_DAYS = 30       
CORRELATION_THRESHOLD = 0.99 

# 지문 대조용 설정
FINGERPRINT_DAYS = 20       # 20일(약 1달) 패턴 비교 -> 오탐 확률 0% 수렴
SEQUENCE_CORR_THRESHOLD = 0.999 # 패턴 일치도 99.9% 요구
PRICE_TOLERANCE = 0.05      # 가격 오차 5% 허용

def load_master():
    if not MASTER_PATH.exists():
        raise FileNotFoundError("Master list not found!")
    df = pd.read_csv(MASTER_PATH)
    if 'start_date' in df.columns:
        df['start_date'] = pd.to_datetime(df['start_date'])
    if 'end_date' in df.columns:
        df['end_date'] = pd.to_datetime(df['end_date'])
    return df

def load_parquet(path):
    try:
        df = pd.read_parquet(path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
        return df
    except:
        return None

def analyze_ticker(row, ref_tickers_set):
    """1차 분석: 이름 충돌 및 지문 추출"""
    ticker = row['ticker']
    kaggle_path = Path(row['file_path'])
    if not kaggle_path.is_absolute():
        kaggle_path = BASE_DIR / kaggle_path
        
    kaggle_df = load_parquet(kaggle_path)
    if kaggle_df is None or len(kaggle_df) < 10:
        return None 

    # CASE 1: 이름 충돌 (Direct Match)
    if ticker in ref_tickers_set:
        ref_path = REF_DIR / f"ticker={ticker}" / "price.parquet"
        ref_df = load_parquet(ref_path)
        
        if ref_df is not None:
            common_idx = kaggle_df.index.intersection(ref_df.index)
            
            if len(common_idx) > MIN_OVERLAP_DAYS:
                k_close = kaggle_df.loc[common_idx, 'Close']
                r_close = ref_df.loc[common_idx, 'Close']
                
                # Zero Variance Check
                if k_close.std() == 0 or r_close.std() == 0:
                    return {"ticker": ticker, "action": "FORK", "reason": "Zero Variance", "new_name": f"{ticker}_legacy"}

                k_ret = k_close.pct_change().dropna()
                r_ret = r_close.pct_change().dropna()
                
                if len(k_ret) < 2 or k_ret.std() == 0 or r_ret.std() == 0:
                     return {"ticker": ticker, "action": "FORK", "reason": "Bad Data for Corr", "new_name": f"{ticker}_legacy"}
                
                corr = k_ret.corr(r_ret)
                
                if pd.isna(corr):
                     return {"ticker": ticker, "action": "FORK", "reason": "NaN Correlation", "new_name": f"{ticker}_legacy"}

                if corr > CORRELATION_THRESHOLD:
                    return {"ticker": ticker, "action": "MERGE", "reason": f"High Correlation ({corr:.4f})", "target_path": str(ref_path)}
                else:
                    return {"ticker": ticker, "action": "FORK", "reason": f"Low Correlation ({corr:.4f})", "new_name": f"{ticker}_legacy"}
            else:
                gap_days = (ref_df.index[0] - kaggle_df.index[-1]).days
                if gap_days > 60:
                    return {"ticker": ticker, "action": "FORK", "reason": f"Gap {gap_days} days", "new_name": f"{ticker}_legacy"}
                else:
                    return {"ticker": ticker, "action": "MERGE", "reason": "Sequential Data", "target_path": str(ref_path)}

    # CASE 2: 지문 추출 (최대 20일)
    # 데이터가 20일보다 적으면 있는 만큼만 사용
    lookback = min(len(kaggle_df), FINGERPRINT_DAYS)
    last_seq = kaggle_df.iloc[-lookback:]
    
    fingerprint = {
        "dates": [d.strftime('%Y-%m-%d') for d in last_seq.index],
        "prices": last_seq['Close'].tolist(),
        "ticker": ticker
    }
    
    return {
        "ticker": ticker,
        "action": "SEARCH_CANDIDATE",
        "fingerprint": fingerprint
    }

def find_alias_batch_v3(candidates, ref_tickers):
    """
    [개선된 2차 수색대]
    - 20일치 가격 시퀀스를 비교하여 Unique Match를 찾음
    """
    print(f"\n>>> 🕵️ 20일치 정밀 패턴 대조 중... (대상: {len(candidates)}개 종목)")
    
    # 날짜별 Lookup Table 구성 (Key: 마지막 날짜)
    lookup_table = {}
    for c in candidates:
        fp = c['fingerprint']
        last_date = fp['dates'][-1]
        
        if last_date not in lookup_table:
            lookup_table[last_date] = []
        lookup_table[last_date].append(c)

    ref_files = list(REF_DIR.glob("ticker=*/price.parquet"))
    
    def scan_ref_file(ref_file):
        matches_found = []
        try:
            # 필요한 컬럼만 로드
            df = pd.read_parquet(ref_file, columns=['Date', 'Close'])
            # 데이터가 너무 적으면 비교 불가 (최소 3일은 있어야 함)
            if df.empty or len(df) < 3: return []
            
            df['date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
            ref_ticker = ref_file.parent.name.replace("ticker=", "")
            
            # 교집합 날짜 확인
            available_dates = set(df['date_str'])
            target_dates = set(lookup_table.keys())
            common_last_dates = available_dates.intersection(target_dates)
            
            for l_date in common_last_dates:
                # l_date 위치 찾기
                curr_rows = df[df['date_str'] == l_date]
                if curr_rows.empty: continue
                curr_idx = curr_rows.index[0] # RangeIndex 가정
                
                loc_idx = df.index.get_loc(curr_idx)
                
                for candidate in lookup_table[l_date]:
                    cand_prices = candidate['fingerprint']['prices']
                    seq_len = len(cand_prices)
                    
                    # Yahoo 데이터 범위 체크
                    if loc_idx < seq_len - 1:
                        continue 
                        
                    # Yahoo 시퀀스 추출
                    ref_seq = df['Close'].iloc[loc_idx - (seq_len - 1) : loc_idx + 1].tolist()
                    
                    # A. 마지막 날 가격 오차 검사
                    p_cand_last = cand_prices[-1]
                    p_ref_last = ref_seq[-1]
                    
                    if abs(p_ref_last - p_cand_last) / p_cand_last > PRICE_TOLERANCE:
                        continue
                        
                    # B. 패턴 매칭 (MSE & Correlation)
                    score = 0
                    if seq_len >= 5: # 5일 이상일 때만 상관계수 신뢰
                        cand_norm = np.array(cand_prices) / cand_prices[0]
                        ref_norm = np.array(ref_seq) / ref_seq[0]
                        
                        mse = np.mean((cand_norm - ref_norm) ** 2)
                        
                        if np.std(cand_norm) > 0 and np.std(ref_norm) > 0:
                            corr = np.corrcoef(cand_norm, ref_norm)[0, 1]
                            if corr < SEQUENCE_CORR_THRESHOLD: 
                                continue
                        
                        score = mse
                    else:
                        # 데이터 짧으면 단순 오차 사용
                        score = abs(p_ref_last - p_cand_last) / p_cand_last

                    matches_found.append({
                        "k_ticker": candidate['ticker'],
                        "y_ticker": ref_ticker,
                        "score": score,
                        "reason": f"Seq Match (len={seq_len}, score={score:.6f})"
                    })
                    
            return matches_found

        except Exception:
            return []

    # 병렬 스캔
    scan_results = Parallel(n_jobs=N_JOBS)(
        delayed(scan_ref_file)(f) for f in tqdm(ref_files, desc="Scanning Reference Universe")
    )
    
    # Best Match Selection (Winner Takes All)
    best_matches = {} 
    
    for batch in scan_results:
        for m in batch:
            k = m['k_ticker']
            if k not in best_matches:
                best_matches[k] = m
            else:
                if m['score'] < best_matches[k]['score']:
                    best_matches[k] = m
                    
    return best_matches

def main():
    print(">>> [Phase 2: V3.1 Final] 20-Day Sequence Fingerprinting")
    
    master = load_master()
    ref_tickers = {p.name.replace("ticker=", "") for p in REF_DIR.glob("ticker=*")}
    
    print(">>> 1차 분석: 이름이 같은 종목 검증 중...")
    results = Parallel(n_jobs=N_JOBS)(
        delayed(analyze_ticker)(row, ref_tickers) 
        for _, row in tqdm(master.iterrows(), total=len(master))
    )
    results = [r for r in results if r is not None]
    
    plan = []
    search_candidates = []
    
    for res in results:
        if res['action'] == 'SEARCH_CANDIDATE':
            search_candidates.append(res)
        else:
            plan.append(res)
            
    # 2차 분석
    if search_candidates:
        best_matches = find_alias_batch_v3(search_candidates, ref_tickers)
        
        matched_set = set()
        for k_ticker, info in best_matches.items():
            plan.append({
                "ticker": k_ticker,
                "action": "RENAME",
                "new_name": info['y_ticker'],
                "reason": info['reason']
            })
            matched_set.add(k_ticker)
            
        for c in search_candidates:
            if c['ticker'] not in matched_set:
                plan.append({
                    "ticker": c['ticker'],
                    "action": "MISSING",
                    "reason": "No sequence match found"
                })

    with open(OUTPUT_PLAN, 'w') as f:
        json.dump(plan, f, indent=4)
        
    df_plan = pd.DataFrame(plan)
    print("\n" + "="*40)
    print(f"  📂 저장 위치: {OUTPUT_PLAN}")
    print("="*40)
    print(df_plan['action'].value_counts())
    
    print("\n[RENAME 제안 예시 - 상위 5개]")
    renames = df_plan[df_plan['action'] == 'RENAME']
    if not renames.empty:
        print(renames.head(5)[['ticker', 'new_name', 'reason']].to_string(index=False))
    else:
        print("  (RENAME 제안 없음)")

if __name__ == "__main__":
    main()