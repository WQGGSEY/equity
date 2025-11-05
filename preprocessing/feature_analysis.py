import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.linalg
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist
import traceback
import random 
from dask.diagnostics import ProgressBar # Dask 진행률 표시줄

# 환경 변수 로드
load_dotenv()
DB_DIR = os.environ.get("DIR_PATH", "./db")

# --- 경로 설정 ---
INDICATORS_DIR = os.path.join(DB_DIR, "indicators_partitioned")
LIST_PATH = os.path.join(DB_DIR, "russell_3000_list.parquet") 
OHLCV_PATH = os.path.join(DB_DIR, "ohlcv.parquet")
HEATMAP_PATH = os.path.join(DB_DIR, "feature_analysis_heatmap.png")
METRICS_PATH = os.path.join(DB_DIR, "feature_analysis_metrics.txt")
# ---

# --- 분석할 17개 지표 목록 ---
FEATURE_COLS = [
    'log_return', 'rsi_20', 'rsi_120', 'mfi_14',
    'b_slope_20', 'b_uncert_20', 'b_slope_240', 'b_uncert_240',
    'bb_width_20', 'bb_width_120',
    'skewness_20', 'skewness_120', 'kurtosis_20', 'kurtosis_120',
    'auto_corr_20', 'auto_corr_120', 'amihud_60'
]

# --- Ticker 샘플링 함수 ---
def get_sampled_tickers(frac: float = 0.10) -> list[str]:
    """Ticker 리스트에서 지정된 비율(frac)만큼을 무작위로 샘플링합니다."""
    print(f"Loading ticker list from: {LIST_PATH}")
    if not os.path.exists(LIST_PATH):
        print(f"[CRITICAL_ERROR] Ticker list file not found at: {LIST_PATH}")
        return []
    try:
        df = pd.read_parquet(LIST_PATH)
        all_tickers = df['ticker'].dropna().unique().tolist()
        if not all_tickers:
            print(f"[CRITICAL_ERROR] Ticker list file is empty.")
            return []
        
        n_sample = int(len(all_tickers) * frac)
        if n_sample < 1: n_sample = 1
            
        sampled_tickers = random.sample(all_tickers, n_sample)
        print(f"Total {len(all_tickers)} tickers. Sampled {len(sampled_tickers)} tickers ({frac*100:.1f}%).")
        return sampled_tickers
        
    except Exception as e:
        print(f"[CRITICAL_ERROR] Failed to read or sample ticker list: {e}")
        print(traceback.format_exc())
        return []

# --- L_uniform (균일성) 계산 함수 ---
def calculate_uniformity(features: np.ndarray, t: float = 2.0) -> float:
    """균일성(Uniformity) L_uniform 손실을 계산합니다."""
    if features.shape[0] < 2:
        print("[WARNING] L_uniform: 2개 미만의 샘플로 계산할 수 없습니다.")
        return np.nan
    try:
        features_normalized = normalize(features, norm='l2', axis=1)
        sq_dists = pdist(features_normalized, metric='sqeuclidean')
        kernel_vals = np.exp(-t * sq_dists)
        mean_potential = kernel_vals.mean()
        l_uniform = np.log(mean_potential)
        return l_uniform
    except Exception as e:
        print(f"[ERROR] L_uniform 계산 중 오류: {e}")
        print(traceback.format_exc())
        return np.nan

# --- L_align (정렬) 계산 함수 ---
def calculate_alignment(features: np.ndarray, labels: np.ndarray, alpha: float = 2.0) -> float:
    """정렬(Alignment) L_align 손실을 계산합니다."""
    if features.shape[0] != labels.shape[0] or features.shape[0] < 2:
        print("[WARNING] L_align: 샘플 또는 레이블이 부족합니다.")
        return np.nan
    try:
        # 피처 이름이 중복될 수 있으므로 임의의 이름 사용
        feature_names = [f"f_{i}" for i in range(features.shape[1])]
        df = pd.DataFrame(features, columns=feature_names)
        df['label'] = labels
        total_dist_sum = 0
        total_pair_count = 0
        
        for label in df['label'].unique():
            current_bin_features = df[df['label'] == label].drop('label', axis=1).values
            n_samples_in_bin = current_bin_features.shape[0]
            if n_samples_in_bin < 2: continue
            
            n_pairs_to_sample = min(2000, n_samples_in_bin * (n_samples_in_bin - 1) // 2)
            if n_pairs_to_sample == 0: continue
            
            idx1 = np.random.randint(0, n_samples_in_bin, n_pairs_to_sample)
            idx2 = np.random.randint(0, n_samples_in_bin, n_pairs_to_sample)
            
            sq_dists = np.sum((current_bin_features[idx1] - current_bin_features[idx2])**2, axis=1)
            
            total_dist_sum += sq_dists.sum()
            total_pair_count += n_pairs_to_sample
            
        if total_pair_count == 0:
            print("[WARNING] L_align: 유효한 positive pair를 찾을 수 없습니다.")
            return np.nan
            
        l_align = total_dist_sum / total_pair_count
        return l_align
    except Exception as e:
        print(f"[ERROR] L_align 계산 중 오류: {e}")
        print(traceback.format_exc())
        return np.nan

# --- 조건수(Condition Number) 계산 함수 ---
def calculate_condition_number(features: np.ndarray, feature_names: list[str]) -> float:
    """피처 상관관계 행렬의 조건수를 계산합니다."""
    try:
        corr_matrix = pd.DataFrame(features, columns=feature_names).corr()
        if corr_matrix.isnull().values.any():
            print("[WARNING] Condition Number: 상관관계 행렬에 NaN이 포함되어 있습니다. (e.g., 상수 피처)")
            corr_matrix = corr_matrix.fillna(0)
            
        cond_num = np.linalg.cond(corr_matrix)
        return cond_num
    except np.linalg.LinAlgError as lae:
        print(f"[ERROR] Condition Number 계산 실패 (특이 행렬 등): {lae}")
        return np.inf
    except Exception as e:
        print(f"[ERROR] Condition Number 계산 중 알 수 없는 오류: {e}")
        return np.nan

# --- 메인 분석 함수 ---
def main_analysis():
    print(f"--- Feature Analysis 시작 ---")
    print(f"입력 데이터 폴더: {INDICATORS_DIR}")
    
    if not os.path.exists(INDICATORS_DIR):
        print(f"[CRITICAL_ERROR] 입력 폴더({INDICATORS_DIR})가 없습니다.")
        print("먼저 preprocessing.py를 실행하세요.")
        return

    # 0. Ticker 샘플링 (10%)
    sampled_tickers = get_sampled_tickers(frac=0.1)
    if not sampled_tickers:
        print("[CRITICAL_ERROR] 샘플링할 Ticker가 없습니다.")
        return

    # 1. Dask로 필터링된 데이터 로드
    print(f"Dask DataFrame 로드 중 (샘플링된 {len(sampled_tickers)}개 Ticker)...")
    try:
        ddf = dd.read_parquet(
            INDICATORS_DIR, 
            engine='pyarrow',
            filters=[('ticker', 'in', sampled_tickers)] 
        )
        available_features = [f for f in FEATURE_COLS if f in ddf.columns]
        
        if 'close' not in ddf.columns:
             ohlcv_df = dd.read_parquet(
                 OHLCV_PATH, 
                 columns=['date', 'ticker', 'close'],
                 filters=[('ticker', 'in', sampled_tickers)]
             )
             ohlcv_df['date'] = dd.to_datetime(ohlcv_df['date'])
             ddf = ddf.merge(ohlcv_df, on=['date', 'ticker'])
        
        if not available_features:
            print("[CRITICAL_ERROR] Dask DataFrame에 분석할 피처가 없습니다.")
            return
        
        print(f"Dask DataFrame 로드 완료 (사용 가능 피처 {len(available_features)}개).")
        
        # --- [핵심 수정 1] ---
        # `groupby.shift`를 사용하기 위해 'ticker'를 인덱스로 설정합니다.
        # 이 작업은 Dask가 데이터를 Ticker별로 재-파티션(셔플)하게 합니다.
        # ProgressBar로 이 셔플 작업의 진행 상황을 봅니다.
        print("Dask DataFrame 인덱스 설정 중 ('ticker')... (셔플 발생, 시간 소요)")
        with ProgressBar():
            ddf = ddf.set_index('ticker')
        print("인덱스 설정 완료.")
        # --- [수정 완료] ---
        
    except Exception as e:
        print(f"[CRITICAL_ERROR] Dask DataFrame 로드 또는 인덱싱 실패: {e}")
        print(traceback.format_exc())
        return

    # 2. L_align을 위한 Target 계산 (인덱스 기반)
    N_DAYS = 20
    EPSILON = 0.05
    print(f"L_align을 위한 Target(미래 {N_DAYS}일 수익률) 계산 중...")
    try:
        # --- [핵심 수정 2] ---
        # 각 파티션(이제 Ticker별로 구성됨)을 'date'로 정렬한 후 shift/pct_change 수행
        def calculate_targets_per_partition(df, n_days, eps):
            # df.index는 'ticker'입니다.
            # 'date' 컬럼으로 파티션 내부를 정렬합니다.
            df = df.sort_values('date') # 1. 파티션(Ticker) 내에서 날짜순 정렬
            
            # 2. 미래 수익률 계산
            future_close = df['close'].shift(-n_days)
            df['future_return'] = (future_close / df['close']) - 1
            
            # 3. Bin 생성
            bins = [-np.inf, -eps, eps, np.inf]
            labels = [-1, 0, 1] 
            df['target_bin'] = pd.cut(
                df['future_return'].replace([np.inf, -np.inf], np.nan),
                bins=bins, 
                labels=labels, 
                include_lowest=True
            )
            return df

        # Dask의 meta를 정의하여 Dask가 출력 타입을 알 수 있게 함
        meta = ddf._meta.copy()
        meta['future_return'] = 'f8'
        # target_bin은 pd.cut의 결과이므로 'category'가 정확합니다.
        meta['target_bin'] = 'category' 
        
        ddf = ddf.map_partitions(
            calculate_targets_per_partition, 
            n_days=N_DAYS, 
            eps=EPSILON, 
            meta=meta
        )
        # --- [수정 완료] ---
        
    except Exception as e:
        print(f"[CRITICAL_ERROR] Target Label 생성 실패: {e}")
        print(traceback.format_exc())
        return

    # 3. Dask -> Pandas 변환 (ProgressBar 적용)
    print("Dask에서 Pandas로 변환 중 (dropna 포함)...")
    try:
        analysis_cols = available_features + ['target_bin']
        # .reset_index(drop=True)로 인덱스를 초기화해야 .compute()가 Pandas DF를 반환
        data_to_compute = ddf[analysis_cols].dropna().reset_index(drop=True) 

        print("계산을 위해 메모리로 로드 중 (Dask Progress)...")
        with ProgressBar():
            sample_df = data_to_compute.compute()
            
        if sample_df.empty:
             print("[CRITICAL_ERROR] Ticker 샘플링 및 dropna() 이후 데이터가 없습니다.")
             return
        
        features_sample = sample_df[available_features].values
        labels_sample = sample_df['target_bin'].values
        print(f"Pandas 변환 완료 (총 {len(sample_df)}개 데이터 포인트)")
        
    except Exception as sample_e:
        print(f"[ERROR] Dask .compute() 또는 dropna() 중 오류 발생: {sample_e}")
        print(traceback.format_exc())
        return

    # 4. 분석 실행 및 결과 저장
    results = []
    
    # --- 균일성 (Uniformity) 평가 ---
    print("\n[분석 1/3] 균일성 (L_uniform) 계산 중...")
    l_uniform = calculate_uniformity(features_sample, t=2.0)
    results.append(f"--- 균일성 (Uniformity) 평가 [L_uniform] ---")
    results.append(f"L_uniform (t=2.0): {l_uniform:<.4e}")
    results.append(f"(낮을수록 균일하게 분포됨)")

    # --- 정렬 (Alignment) 평가 ---
    print("[분석 2/3] 정렬 (L_align) 계산 중...")
    l_align = calculate_alignment(features_sample, labels_sample, alpha=2.0)
    results.append(f"\n--- 정렬 (Alignment) 평가 [L_align] ---")
    results.append(f"L_align (alpha=2.0, N={N_DAYS}, epsilon={EPSILON*100}%): {l_align:<.4e}")
    results.append(f"(낮을수록 '유사한 샘플'이 가깝게 정렬됨)")

    # --- 직교성 (Condition Number) 평가 ---
    print("[분석 3/3] 직교성 (Condition Number) 계산 중...")
    cond_num = calculate_condition_number(features_sample, available_features)
    results.append(f"\n--- 직교성 (Condition Number) 평가 ---")
    matrix_size = len(available_features)
    results.append(f"Condition Number of {matrix_size}x{matrix_size} Correlation Matrix: {cond_num:<.4e}")
    if cond_num > 100:
        results.append(f"[WARNING] Condition Number > 100. 다중공선성(multicollinearity) 위험 존재.")

    # --- 히트맵 생성 ---
    try:
        corr_matrix = pd.DataFrame(features_sample, columns=available_features).corr()
        plt.figure(figsize=(24, 20))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 8})
        plt.title("Feature Correlation Heatmap (Sampled)", fontsize=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(HEATMAP_PATH)
        results.append(f"\n[성공] 상관관계 히트맵 저장: {HEATMAP_PATH}")
    except Exception as e:
        print(f"\n[실패] 히트맵 저장 중 오류 발생: {e}")

    # --- 결과 요약 출력 및 저장 ---
    print("\n" + "="*30)
    print("     Feature Analysis 결과 요약")
    print("="*30)
    final_output = "\n".join(results)
    print(final_output)
    
    try:
        with open(METRICS_PATH, 'w') as f:
            f.write(final_output)
        print(f"\n[성공] 분석 결과 텍스트 파일 저장: {METRICS_PATH}")
    except Exception as e:
        print(f"\n[실패] 분석 결과 저장 중 오류: {e}")


if __name__ == "__main__":
    main_analysis()