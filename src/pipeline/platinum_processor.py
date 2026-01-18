import pandas as pd
import concurrent.futures
import importlib
import sys
import os
import random
import shutil
import gc
from tqdm import tqdm
from pathlib import Path

# 프로젝트 루트 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import GOLD_DIR, PLATINUM_DIR, DATA_DIR, MODEL_WEIGHTS_DIR, ACTIVE_FEATURES
from src.features.base import GlobalFeature

# 전역 피처 캐시 경로 (임시)
INTERIM_FEATURE_DIR = DATA_DIR / "interim" / "features"

def load_feature_class(module_path, class_name):
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception:
        return None

def load_universal_data(ticker, gold_file_path):
    try:
        df = pd.read_parquet(gold_file_path)
        if df.empty: return None
        df = df[~df.index.duplicated(keep='last')]
    except Exception:
        return None
    return df

def initialize_contrastive_compressor(gold_files):
    """Contrastive Model 초기화"""
    print(">>> [Pre-Flight] Checking Contrastive Calibration...")
    contrastive_cfg = next((f for f in ACTIVE_FEATURES if f['class'] == 'Contrastive_OC_HL'), None)
    if not contrastive_cfg: return

    compressor_path = MODEL_WEIGHTS_DIR / "ts2vec_compressor.pth"
    if compressor_path.exists():
        print(f"   -> Compressor already exists at {compressor_path}. Using it.")
        return

    cls = load_feature_class(contrastive_cfg['module'], contrastive_cfg['class'])
    if not cls: return
    feature_instance = cls(**contrastive_cfg['params'])

    print("   -> Searching for a valid ticker for calibration...")
    sample_files = list(gold_files)
    random.shuffle(sample_files)
    
    for f in sample_files[:20]:
        df = load_universal_data(f.stem, f)
        if df is None or len(df) < 1000: continue
        
        print(f"   -> Calibrating on {f.stem} ({len(df)} rows)...")
        try:
            if feature_instance.train_and_save_compressor(df):
                print("   -> Calibration Success.")
                return
        except: pass
    print("⚠️ Warning: Calibration failed.")

class PlatinumProcessor:
    def __init__(self):
        PLATINUM_DIR.mkdir(parents=True, exist_ok=True)
        (PLATINUM_DIR / "features").mkdir(parents=True, exist_ok=True)
        INTERIM_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_gold_matrix(self):
        print("  📥 Loading Gold Data Matrix for Global Features...")
        files = list(GOLD_DIR.glob("*.parquet"))
        price_dict, vol_dict = {}, {}
        
        def _read(p):
            try:
                df = pd.read_parquet(p, columns=['Close', 'Volume'])
                df = df[~df.index.duplicated(keep='last')]
                return p.stem, df['Close'], df['Volume']
            except: return None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(_read, files), total=len(files), desc="    Reading"))
            
        for res in results:
            if res:
                t, c, v = res
                price_dict[t] = c
                vol_dict[t] = v
        
        return pd.DataFrame(price_dict).sort_index(), pd.DataFrame(vol_dict).sort_index()

    def run_global_features(self):
        """[Phase 1] 전역 피처 계산 및 캐싱"""
        global_tasks = []
        for cfg in ACTIVE_FEATURES:
            cls = load_feature_class(cfg['module'], cfg['class'])
            if cls and issubclass(cls, GlobalFeature):
                global_tasks.append((cfg['class'], cls, cfg.get('params', {})))
        
        if not global_tasks: return

        print(f"\n>>> [Phase 1] Pre-computing {len(global_tasks)} Global Features...")
        prices, volumes = self._load_gold_matrix()
        if prices.empty: return

        for name, cls, params in global_tasks:
            try:
                print(f"  🚀 Global Exec: {name}")
                instance = cls(**params)
                result_matrix = instance.compute_global(prices, volumes)
                
                # 저장 폴더
                save_dir = INTERIM_FEATURE_DIR / name
                if save_dir.exists(): shutil.rmtree(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"    💾 Splitting & Caching to {save_dir}...")
                valid_tickers = result_matrix.columns
                for ticker in tqdm(valid_tickers, desc="    Writing"):
                    s = result_matrix[ticker]
                    s.to_frame(name).to_parquet(save_dir / f"{ticker}.parquet")
                    
            except Exception as e:
                print(f"    🚨 Failed {name}: {e}")
        
        del prices, volumes
        gc.collect()

    def process_single_ticker(self, file_path):
        try:
            ticker = file_path.stem
            universal_df = load_universal_data(ticker, file_path)
            if universal_df is None or universal_df.empty: return None
            
            # [CRITICAL] 원본 보존 Copy
            final_df = universal_df.copy()
            final_df['ticker'] = ticker

            for cfg in ACTIVE_FEATURES:
                cls = load_feature_class(cfg['module'], cfg['class'])
                if not cls: continue
                
                try:
                    instance = cls(**cfg.get('params', {}))
                    
                    # [CRITICAL] 원본 보호를 위해 .copy() 전달
                    if issubclass(cls, GlobalFeature):
                        res = instance.compute(final_df) # 단순 병합 (Safe)
                    else:
                        res = instance.compute(final_df.copy()) # 원본 보호 (Safe)
                    
                    if isinstance(res, pd.Series):
                        final_df[res.name] = res
                    elif isinstance(res, pd.DataFrame) and not res.empty:
                        new_cols = res.columns.difference(final_df.columns)
                        if not new_cols.empty:
                            final_df = pd.concat([final_df, res[new_cols]], axis=1)
                except Exception:
                    pass

            if 'ticker' in final_df.columns:
                final_df.drop(columns=['ticker'], inplace=True)

            # [CHECK] 데이터 생존 확인
            if 'Open' not in final_df.columns: 
                return None
            
            final_df.dropna(inplace=True)
            if final_df.empty or len(final_df) < 20: return None

            output_path = PLATINUM_DIR / "features" / f"{ticker}.parquet"
            final_df.to_parquet(output_path)
            
            del final_df
            gc.collect()
            
            return ticker

        except Exception:
            return None

    def cleanup_interim(self):
        """[Cleanup] 임시 폴더 삭제"""
        if INTERIM_FEATURE_DIR.exists():
            print(f"\n🧹 Cleaning up interim files at {INTERIM_FEATURE_DIR}...")
            try:
                shutil.rmtree(INTERIM_FEATURE_DIR)
                print("   -> Cleanup Complete.")
            except Exception as e:
                print(f"   -> Cleanup Failed: {e}")

    def process_features(self):
        gold_files = list(GOLD_DIR.glob("*.parquet"))
        num_workers = max(1, os.cpu_count() - 1)
        
        # 1. 전역 계산
        self.run_global_features()
        
        # 2. 모델 초기화
        initialize_contrastive_compressor(gold_files)
        
        # 3. 개별 병렬 처리
        print(f"\n[Phase 2] Single Ticker Processing ({num_workers} workers)")
        
        success_count = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.process_single_ticker, f): f for f in gold_files}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(gold_files), desc="Generating Platinum"):
                if future.result():
                    success_count += 1
                    
        print(f"  ✅ Platinum 생성 완료: {success_count} / {len(gold_files)} 종목")
        
        # 4. [NEW] 뒷정리 (interim 삭제)
        self.cleanup_interim()