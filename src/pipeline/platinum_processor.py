import pandas as pd
import importlib
import sys
import os
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 프로젝트 루트 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import *

def load_feature_class(module_path, class_name):
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        return None

def load_universal_data(ticker, gold_file_path):
    try:
        df = pd.read_parquet(gold_file_path)
        if df.empty: return None
    except Exception:
        return None
    
    # Macro / Fundamental 로드 로직 (필요시 추가)
    return df

def process_single_ticker(file_path, feature_configs):
    """
    [핵심] Feature 생성, 병합, 그리고 **NaN 제거(Cleaning)**를 수행합니다.
    """
    try:
        ticker = file_path.stem
        
        # 1. 데이터 로드
        universal_df = load_universal_data(ticker, file_path)
        if universal_df is None or universal_df.empty: return None
        
        initial_col_count = len(universal_df.columns)

        # 2. Processor 인스턴스화
        processors = []
        for cfg in feature_configs:
            cls = load_feature_class(cfg['module'], cfg['class'])
            if cls:
                processors.append({'name': cfg['class'], 'instance': cls(**cfg['params'])})

        # 3. Feature Generation & Merge
        final_df = universal_df.copy()
        
        for p in processors:
            try:
                res = p['instance'].compute(universal_df)
                if isinstance(res, pd.Series):
                    final_df[res.name] = res
                elif isinstance(res, pd.DataFrame) and not res.empty:
                    # 중복 컬럼 방지하면서 병합
                    new_cols = res.columns.difference(final_df.columns)
                    final_df = pd.concat([final_df, res[new_cols]], axis=1)
            except Exception:
                pass

        # 4. [Filter Logic] Feature 생성 여부 확인
        # FD_ 로 시작하는 컬럼이 없으면 실패로 간주
        has_fd_feature = any(col.startswith('FD_') for col in final_df.columns)
        if len(final_df.columns) == initial_col_count or not has_fd_feature:
            return None

        # 5. [Crucial] Burn-in 구간 제거 (NaN Row Drop)
        # 병합 과정에서 생긴 앞부분의 NaN을 여기서 확실하게 지웁니다.
        before_len = len(final_df)
        final_df.dropna(inplace=True) 
        
        # 만약 dropna 후 남은 데이터가 너무 적으면(예: 20개 미만) 학습 가치가 없으므로 폐기
        # (CEPF는 44개이므로 이 기준을 통과하여 살아남음)
        if final_df.empty or len(final_df) < 20: 
            return None

        # 6. 저장
        save_path = PLATINUM_FEATURES_DIR / f"{ticker}.parquet"
        final_df.to_parquet(save_path)
        
        return ticker

    except Exception as e:
        return None

def process_features():
    print(">>> [Phase 6] Platinum Processor: Parallel Feature Engineering (Strict Mode)")
    
    # 기존 오염된 파일들이 섞이지 않게 폴더를 비우는 것을 권장하지만, 덮어쓰기도 괜찮음
    PLATINUM_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    gold_files = list(GOLD_DIR.glob("*.parquet"))
    num_workers = max(1, os.cpu_count() - 1)
    
    print(f"  ⚡ Running on {num_workers} cores with Strict NaN Dropping...")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_ticker, f, ACTIVE_FEATURES): f for f in gold_files}
        
        for future in tqdm(as_completed(futures), total=len(gold_files), desc="Processing"):
            if future.result():
                success_count += 1
                
    print(f"  ✅ Platinum 생성 완료: {success_count} / {len(gold_files)} 종목")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    process_features()