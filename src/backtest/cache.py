import os
import shutil
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"

class CacheManager:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, name):
        return self.cache_dir / name

    def save(self, data, name):
        """
        [Final Fix] Dynamic Type Checking
        이름이 아니라 '실제 값의 크기'를 보고 float16/float32를 결정합니다.
        """
        path = self.get_cache_path(name)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        
        print(f"📦 [Cache] Saving '{name}' (Dynamic Type Check)...")
        
        meta = {
            'tickers': data.get('tickers', []),
            'dates': [d.strftime('%Y-%m-%d') for d in data.get('dates', [])]
        }
        with open(path / 'meta.json', 'w') as f:
            json.dump(meta, f)

        for category in ['prices', 'features']:
            dct = data.get(category, {})
            save_dir = path / category
            save_dir.mkdir(exist_ok=True)
            
            for key, df in dct.items():
                file_path = save_dir / f"{key}.parquet"
                
                # 1. 가격 데이터는 무조건 안전하게 float32 (백테스트 핵심이므로)
                if category == 'prices' or key in ['Open', 'High', 'Low', 'Close']:
                    df.astype('float32').to_parquet(file_path, engine='pyarrow', compression='zstd')
                    continue

                # 2. Features는 값의 범위를 확인하여 동적 결정
                # (1) 절대값의 최댓값 계산 (NaN/Inf 제외)
                # numeric_only=True는 안전장치
                try:
                    # inf가 있으면 max가 inf가 됨 -> float32로 처리됨 (OK)
                    max_val = df.abs().max(numeric_only=True).max()
                except:
                    max_val = float('inf') # 계산 실패시 안전하게 float32로

                # (2) float16 한계(약 65,500) 체크
                # 여유 있게 60,000 넘으면 float32로 전환
                if pd.isna(max_val) or max_val > 60000:
                    # 범위 초과 혹은 inf 포함 시
                    # print(f"   🛡️ Using float32 for '{key}' (Max: {max_val:.1f})")
                    df.astype('float32').to_parquet(file_path, engine='pyarrow', compression='zstd')
                else:
                    # 안전 범위 내라면 압축
                    df.astype('float16').to_parquet(file_path, engine='pyarrow', compression='zstd')

        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024*1024)
        print(f"   -> Save Complete. Total Size: {total_size:.2f} MB")

    def load(self, name, expiration_hours=24):
        path = self.get_cache_path(name)
        if not path.exists(): return None
        
        meta_path = path / 'meta.json'
        if not meta_path.exists(): return None
        
        mtime = datetime.fromtimestamp(os.path.getmtime(meta_path))
        if datetime.now() - mtime > timedelta(hours=expiration_hours):
            print(f"⚠️ [Cache] '{name}' expired. Reloading...")
            return None

        print(f"🚀 [Cache] Loading '{name}'...")
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            dates = [pd.Timestamp(d) for d in meta['dates']]
            tickers = meta['tickers']
            
            data = {'prices': {}, 'features': {}, 'dates': dates, 'tickers': tickers}
            
            for category in ['prices', 'features']:
                target_dir = path / category
                if target_dir.exists():
                    for f in target_dir.glob("*.parquet"):
                        key = f.stem
                        # 로드할 때는 연산 편의를 위해 float32로 통일
                        df = pd.read_parquet(f).astype('float32')
                        data[category][key] = df
            
            return data

        except Exception as e:
            print(f"   -> Cache corrupted: {e}")
            return None