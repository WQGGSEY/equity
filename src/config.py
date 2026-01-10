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

# ==========================================
# [Feature Pipeline] 활성화된 Feature 목록
# ==========================================
ACTIVE_FEATURES = [
    {
        'class': 'DollarBarStationaryFeature',
        'module': 'src.features.preprocessors',
        'params': {
            'threshold': 50_000,  # 1억원 단위 (종목 유동성에 따라 조절 필요)
            'd': 0.4,             # 차분 차수
        }
    },
    {
        'class': 'MovingAverage',
        'module': 'src.features.technical',
        'params': {
            'window': 5
        }
    },
    # [NEW] Contrastive Learning 기반 Feature
    # 모델의 구체적 스펙은 저장된 파일(.pth) 헤더에서 자동으로 읽어옴
    {
        'class': 'Contrastive_OC_HL',
        'module': 'src.features.contrastive', 
        'params': {
           'model_path': str(MODEL_WEIGHTS_DIR / "ts2vec_learnable_tau.pth")
        }
    },
]