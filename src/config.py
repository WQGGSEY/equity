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