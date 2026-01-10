import pickle
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

# 프로젝트 루트 경로 추론 (필요 시 config에서 가져와도 됨)
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"

class CacheManager:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, name):
        return self.cache_dir / f"{name}.pkl"

    def save(self, data, name):
        """데이터를 피클 파일로 저장"""
        path = self.get_cache_path(name)
        print(f"📦 [Cache] Saving '{name}' to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print("   -> Save Complete.")

    def load(self, name, expiration_hours=24):
        """유효기간 내의 캐시가 있으면 로드"""
        path = self.get_cache_path(name)
        
        if not path.exists():
            return None
        
        # 파일 수정 시간 확인
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        age = datetime.now() - mtime
        
        if age > timedelta(hours=expiration_hours):
            print(f"⚠️ [Cache] '{name}' expired ({age}). Reloading...")
            return None
        
        print(f"🚀 [Cache] Loading '{name}' (Cached {age.seconds // 3600}h ago)...")
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"   -> Cache corrupted: {e}")
            return None