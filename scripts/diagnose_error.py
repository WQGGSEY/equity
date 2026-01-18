import sys
import pandas as pd
from pathlib import Path
import importlib

# 프로젝트 루트 경로 설정
FILE_PATH = Path(__file__).resolve()
PROJECT_DIR = FILE_PATH.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import GOLD_DIR, ACTIVE_FEATURES
from src.features.base import GlobalFeature

# 디버깅할 종목 (Gold에 존재하는 파일명으로 변경 가능)
TARGET_TICKER = "AAPL" 

def load_feature_class(module_path, class_name):
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        print(f"    [Error] Failed to load {class_name}: {e}")
        return None

def debug_process():
    print(f"🔍 [DEBUG] Starting Debugging for: {TARGET_TICKER}")
    
    # 1. Gold 파일 로드
    gold_path = GOLD_DIR / f"{TARGET_TICKER}.parquet"
    if not gold_path.exists():
        print(f"❌ Gold file not found: {gold_path}")
        # 아무 파일이나 하나 찾아서 대체
        files = list(GOLD_DIR.glob("*.parquet"))
        if files:
            gold_path = files[0]
            print(f"⚠️ Using alternative file: {gold_path.name}")
        else:
            print("❌ No Gold files found. Aborting.")
            return

    df = pd.read_parquet(gold_path)
    df = df[~df.index.duplicated(keep='last')] # 중복 제거
    
    print("\n" + "="*50)
    print(f"✅ Step 1: Initial Load")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {list(df.columns)}")
    print(f"   - Index Example: {df.index[:3].tolist()} ...")
    print("="*50)

    # 2. Feature Loop 디버깅
    final_df = df.copy()
    final_df['ticker'] = gold_path.stem # Context Injection

    for i, cfg in enumerate(ACTIVE_FEATURES):
        cls_name = cfg['class']
        print(f"\n▶️ [Feature {i+1}] Processing: {cls_name}")
        
        cls = load_feature_class(cfg['module'], cls_name)
        if not cls:
            print("   ❌ Class loading failed.")
            continue
            
        try:
            params = cfg.get('params', {})
            instance = cls(**params)
            
            # [CHECKPOINT 1] Pre-compute state
            print(f"   Before compute shape: {final_df.shape}")
            
            # Compute
            # GlobalFeature 여부 확인
            is_global = False
            try:
                if issubclass(cls, GlobalFeature): is_global = True
            except: pass

            if is_global:
                print("   (Global Feature Mode)")
                res = instance.compute(final_df) # No Copy
            else:
                print("   (Local Feature Mode - Copying DF)")
                res = instance.compute(final_df.copy()) # Copy
            
            # [CHECKPOINT 2] Result Inspection
            if isinstance(res, pd.DataFrame):
                print(f"   👉 Output Shape: {res.shape}")
                print(f"   👉 Output Columns: {list(res.columns)}")
                print(f"   👉 Output Index Match: {res.index.equals(final_df.index)}")
                if not res.index.equals(final_df.index):
                    print(f"      ⚠️ Index Mismatch Detected!")
                    print(f"      Target(Daily): {final_df.index[:3].tolist()}")
                    print(f"      Result: {res.index[:3].tolist()}")
            
            # Merge Logic Debug
            prev_cols = set(final_df.columns)
            
            if isinstance(res, pd.Series):
                final_df[res.name] = res
            elif isinstance(res, pd.DataFrame) and not res.empty:
                new_cols = res.columns.difference(final_df.columns)
                if not new_cols.empty:
                    print(f"   ➕ Merging columns: {list(new_cols)}")
                    # concat 수행
                    final_df = pd.concat([final_df, res[new_cols]], axis=1)
                else:
                    print("   ℹ️ No new columns to merge.")
            
            # [CHECKPOINT 3] Post-merge state
            print(f"   ✅ After Merge Shape: {final_df.shape}")
            print(f"   ✅ Current NaN Count: {final_df.isna().sum().sum()}")
            
        except Exception as e:
            print(f"   ❌ Execution Error: {e}")
            import traceback
            traceback.print_exc()

    # 3. Final Dropna Check
    print("\n" + "="*50)
    print("🧹 Final Cleanup Phase")
    print(f"   Before dropna shape: {final_df.shape}")
    
    # 어디서 NaN이 많은지 확인
    nan_counts = final_df.isna().sum()
    print("   [NaN Distribution]")
    print(nan_counts[nan_counts > 0])
    
    final_df.dropna(inplace=True)
    print(f"   After dropna shape: {final_df.shape}")
    print(f"   Final Columns: {list(final_df.columns)}")
    print("="*50)

if __name__ == "__main__":
    debug_process()