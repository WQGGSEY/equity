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
        # 파일이 아니라 '디렉토리'를 캐시 단위로 씁니다.
        return self.cache_dir / name

    def save(self, data, name):
        """
        데이터를 Parquet 파일들로 쪼개서 저장 (The Crazy Method)
        """
        path = self.get_cache_path(name)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        
        print(f"📦 [Cache] Saving '{name}' (Parquet Sharding Mode)...")
        
        # 1. 메타데이터 저장 (Tickers, Dates)
        meta = {
            'tickers': data.get('tickers', []),
            'dates': [d.strftime('%Y-%m-%d') for d in data.get('dates', [])]
        }
        with open(path / 'meta.json', 'w') as f:
            json.dump(meta, f)

        # 2. DataFrame 저장 (Parquet + Zstd)
        # Prices와 Features를 순회하며 각각 별도 파일로 저장
        for category in ['prices', 'features']:
            dct = data.get(category, {})
            save_dir = path / category
            save_dir.mkdir(exist_ok=True)
            
            for key, df in dct.items():
                # [핵심 1] Feature는 Float16으로 압축 (용량 50% 절감)
                # FD, Return, Correlation 등은 float16으로 충분함
                # 단, Price(가격)와 Amount(거래대금)는 범위가 크므로 float32 유지
                if category == 'features' or key not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Trd_Amt', 'TrdAmount']:
                    df_to_save = df.astype('float16')
                else:
                    df_to_save = df.astype('float32')
                
                # [핵심 2] Parquet + Zstd 압축 (시계열 압축 효율 극대화)
                file_path = save_dir / f"{key}.parquet"
                df_to_save.to_parquet(file_path, engine='pyarrow', compression='zstd')
                
        # 용량 확인
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024*1024)
        print(f"   -> Save Complete. Total Size: {total_size:.2f} MB")

    def load(self, name, expiration_hours=24):
        path = self.get_cache_path(name)
        if not path.exists(): return None
        
        # 시간 체크 (메타파일 기준)
        meta_path = path / 'meta.json'
        if not meta_path.exists(): return None
        
        mtime = datetime.fromtimestamp(os.path.getmtime(meta_path))
        if datetime.now() - mtime > timedelta(hours=expiration_hours):
            print(f"⚠️ [Cache] '{name}' expired. Reloading...")
            return None

        print(f"🚀 [Cache] Loading '{name}' (Parquet Shards)...")
        try:
            # 1. 메타데이터 로드
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            dates = [pd.Timestamp(d) for d in meta['dates']]
            tickers = meta['tickers']
            
            data = {
                'prices': {},
                'features': {},
                'dates': dates,
                'tickers': tickers
            }
            
            # 2. Parquet 로드 (병렬 처리가 가능하지만, 여기선 단순 루프)
            # 필요한 경우 여기서 특정 파일만 읽는 'Lazy Loading' 구현 가능
            for category in ['prices', 'features']:
                target_dir = path / category
                if target_dir.exists():
                    for f in target_dir.glob("*.parquet"):
                        key = f.stem
                        # 읽을 때 다시 float32로 복원 (연산 안정성 위해)
                        df = pd.read_parquet(f).astype('float32')
                        data[category][key] = df
            
            return data

        except Exception as e:
            print(f"   -> Cache corrupted: {e}")
            return None