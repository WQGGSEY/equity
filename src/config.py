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

# Platinum 경로 설정
PLATINUM_DIR = DATA_DIR / "platinum"
PLATINUM_FEATURES_DIR = PLATINUM_DIR / "features"       # Ticker 기준 저장소
PLATINUM_UNIVERSE_DIR = PLATINUM_DIR / "daily_universe" # Date 기준 저장소

MASTER_PATH = DATA_DIR / "bronze" / "master_ticker_list.csv"
BACKUP_ROOT = BASE_DIR / "backups"

# 모델 가중치 저장소 (학습 스크립트가 저장하고, Feature가 참조할 폴더)
MODEL_WEIGHTS_DIR = BASE_DIR / "src" / "models" / "weights"
MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# 2. 로깅 및 포맷
DATE_FORMAT = "%Y-%m-%d"

# 3. 업데이트 정책
BATCH_SIZE = 30  
USE_THREADS = False 

# 4. Gold Processor 최적화 파라미터
PENNY_STOCK_THRESHOLD = 1.0   # 1달러 미만 동전주는 병합 주체 제외
MAX_BUCKET_SIZE = 300         # 버킷 당 최대 정밀 비교 개수


ACTIVE_FEATURES = [
    # 1. 일반 로컬 피처
    {
        'class': 'DollarBarStationaryFeature',
        'module': 'src.features.preprocessors',
        'params': {'threshold': 50_000, 'd': 0.4}
    },
    
    # 2. 글로벌 피처 (Processor가 알아서 Phase 1에서 계산함)
    {
        'class': 'SectorGroup',
        'module': 'src.features.cross_sectional',
        'params': {'window': 252}
    },
    {
        'class': 'LiquidityGroup',
        'module': 'src.features.cross_sectional',
        'params': {'window': 126, 'bins': 10}
    },
    # 3. 모델 피처
    {
        'class': 'Contrastive_OC_HL',
        'module': 'src.features.contrastive', 
        'params': {
           'model_path': str(MODEL_WEIGHTS_DIR / "ts2vec_learnable_tau.pth")
        }
    }
]