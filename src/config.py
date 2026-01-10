import os
from pathlib import Path

# ==========================================
# [Config] 시스템 전역 설정
# ==========================================

# 1. 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

BRONZE_DIR = DATA_DIR / "bronze" / "yahoo_price_data"
SILVER_DIR = DATA_DIR / "silver" / "daily_prices"
GOLD_DIR = DATA_DIR / "gold" / "daily_prices"
QUARANTINE_DIR = DATA_DIR / "quarantine"

MASTER_PATH = DATA_DIR / "bronze" / "master_ticker_list.csv"
BACKUP_ROOT = BASE_DIR / "backups"

# 2. 로깅 및 포맷
DATE_FORMAT = "%Y-%m-%d"

# 3. 업데이트 정책
BATCH_SIZE = 30  
USE_THREADS = False 

# 4. Gold Processor 최적화 파라미터 (필수 확인!)
PENNY_STOCK_THRESHOLD = 1.0   # 1달러 미만 동전주는 병합 주체 제외
MAX_BUCKET_SIZE = 300         # 버킷 당 최대 정밀 비교 개수

PLATINUM_DIR = DATA_DIR / "platinum"
PLATINUM_FEATURES_DIR = PLATINUM_DIR / "features"       # Ticker 기준 저장소
PLATINUM_UNIVERSE_DIR = PLATINUM_DIR / "daily_universe" # Date 기준 저장소

# src/config.py 예시

ACTIVE_FEATURES = [
    {
        'class': 'DollarBarStationaryFeature',
        'module': 'src.features.preprocessors',
        'params': {
            'threshold': 50_000,  # 1억원 단위 (종목 유동성에 따라 조절 필요)
            'd': 0.4,                  # 차분 차수
        }
    },
    {
        'class': 'MovingAverage',
        'module': 'src.features.technical',
        'params': {
            'window': 5
        }
    },
    {
        'class': 'FinancialContrastiveDataset',
        'module': 'src.model.dataset',
        'params': {
            # (1) Dataset Arguments (필수)
            'view_1_cols': ["FD_Open", "FD_Close"],
            'view_2_cols': ["FD_High", "FD_Low"],
            'window_size': 64,
            
            # (2) Training Hyperparameters (Dataset은 무시, Script가 사용)
            'batch_size': 128,
            'num_workers': 4,
            'train_split_ratio': 0.5,
            'projection_dim': 128,
            'temperature': 0.1,
            'learning_rate': 1e-3,
            'epochs': 50
        }
    },

]