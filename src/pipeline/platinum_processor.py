import pandas as pd
import importlib
from tqdm import tqdm
from src.config import *

def load_feature_class(module_path, class_name):
    """문자열로 된 클래스 경로를 실제 클래스로 변환"""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def process_features():
    print(">>> [Phase 6] Platinum Processor (Feature Engineering)")
    
    PLATINUM_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature 인스턴스 미리 생성 (준비)
    processors = []
    for cfg in ACTIVE_FEATURES:
        cls = load_feature_class(cfg['module'], cfg['class'])
        processors.append(cls(**cfg['params']))
    
    gold_files = list(GOLD_DIR.glob("*.parquet"))
    
    # 2. 종목별 루프
    for f in tqdm(gold_files, desc="Calculating Features"):
        try:
            df = pd.read_parquet(f)
            if df.empty: continue
            
            # 원본 데이터 복사 (피처만 붙임)
            features_df = df.copy()
            
            # 등록된 모든 피처 계산
            for p in processors:
                res = p.compute(df)
                # 결과가 Series면 이름으로, DataFrame이면 그대로 병합
                if isinstance(res, pd.Series):
                    features_df[res.name] = res
                else:
                    features_df = pd.concat([features_df, res], axis=1)
            
            # 저장
            save_path = PLATINUM_FEATURES_DIR / f.name
            features_df.to_parquet(save_path)
            
        except Exception as e:
            print(f"Error {f.stem}: {e}")

if __name__ == "__main__":
    process_features()