import pandas as pd
import importlib
import sys
import os
import random
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import *

# ... (load_feature_class, load_universal_data 함수는 그대로 유지) ...
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
    return df

# [NEW] Pre-calibration Logic
def initialize_contrastive_compressor(gold_files):
    """병렬 처리 전, Contrastive Feature의 압축기(Compressor)를 미리 학습시켜 저장함"""
    print(">>> [Pre-Flight] Checking Contrastive Calibration...")
    
    # 1. Find Contrastive Config
    contrastive_cfg = next((f for f in ACTIVE_FEATURES if f['class'] == 'Contrastive_OC_HL'), None)
    if not contrastive_cfg:
        print("   -> No contrastive feature in config. Skipping.")
        return

    # 2. Check if already exists
    compressor_path = MODEL_WEIGHTS_DIR / "ts2vec_compressor.pth"
    if compressor_path.exists():
        print(f"   -> Compressor already exists at {compressor_path}. Using it.")
        return

    # 3. Instantiate Feature Class
    cls = load_feature_class(contrastive_cfg['module'], contrastive_cfg['class'])
    if not cls: return
    feature_instance = cls(**contrastive_cfg['params'])

    # 4. Pick a heavy ticker for calibration
    print("   -> Searching for a valid ticker for calibration...")
    random.shuffle(gold_files) # 셔플해서 랜덤성 부여
    
    for f in gold_files:
        df = load_universal_data(f.stem, f)
        if df is None or len(df) < 1000: continue # 데이터 충분한 것 찾기
        
        print(f"   -> Calibrating on {f.stem} ({len(df)} rows)...")
        success = feature_instance.train_and_save_compressor(df)
        
        if success:
            print("   -> Calibration Success.")
            return
            
    print("⚠️ Warning: Failed to calibrate compressor. Feature generation might fail.")

def process_single_ticker(file_path, feature_configs):
    # ... (기존 로직 그대로) ...
    try:
        ticker = file_path.stem
        universal_df = load_universal_data(ticker, file_path)
        if universal_df is None or universal_df.empty: return None
        
        initial_col_count = len(universal_df.columns)
        processors = []
        for cfg in feature_configs:
            cls = load_feature_class(cfg['module'], cfg['class'])
            if cls:
                processors.append({'name': cfg['class'], 'instance': cls(**cfg['params'])})

        final_df = universal_df.copy()
        
        for p in processors:
            try:
                res = p['instance'].compute(universal_df)
                if isinstance(res, pd.Series):
                    final_df[res.name] = res
                elif isinstance(res, pd.DataFrame) and not res.empty:
                    new_cols = res.columns.difference(final_df.columns)
                    final_df = pd.concat([final_df, res[new_cols]], axis=1)
            except Exception:
                pass

        # [Strict Filter] FD_ 컬럼과 ts2vec_ 컬럼 확인
        # (Contrastive가 켜져 있다면 ts2vec_manifold_ 컬럼이 있어야 함)
        has_fd = any(col.startswith('FD_') for col in final_df.columns)
        # has_ts2vec = any(col.startswith('ts2vec_') for col in final_df.columns) # 필요시 활성화
        
        if len(final_df.columns) == initial_col_count or not has_fd:
            return None

        final_df.dropna(inplace=True) 
        if final_df.empty or len(final_df) < 20: 
            return None

        save_path = PLATINUM_FEATURES_DIR / f"{ticker}.parquet"
        final_df.to_parquet(save_path)
        
        return ticker

    except Exception as e:
        return None

def process_features():
    print(">>> [Phase 6] Platinum Processor: Parallel Feature Engineering")
    PLATINUM_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    gold_files = list(GOLD_DIR.glob("*.parquet"))
    num_workers = max(1, os.cpu_count() - 1)
    
    # [Step 0] Global Calibration (Single Process)
    initialize_contrastive_compressor(gold_files)
    
    # [Step 1] Parallel Processing
    print(f"  ⚡ Running on {num_workers} cores...")
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