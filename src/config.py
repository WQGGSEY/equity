import os
from pathlib import Path

# ==========================================
# [Config] 시스템 전역 설정
# ==========================================

# 1. 경로 설정 (프로젝트 루트 기준)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

BRONZE_DIR = DATA_DIR / "bronze" / "yahoo_price_data"
SILVER_DIR = DATA_DIR / "silver" / "daily_prices"
GOLD_DIR = DATA_DIR / "gold" / "daily_prices"

MASTER_PATH = DATA_DIR / "bronze" / "master_ticker_list.csv"
BACKUP_ROOT = BASE_DIR / "backups"

# 2. 로깅 및 포맷
DATE_FORMAT = "%Y-%m-%d"

# 3. 업데이트 정책
# 차단 방지 및 안전 모드
BATCH_SIZE = 20  
USE_THREADS = False 

# 4. Gold Processor 최적화 파라미터
PENNY_STOCK_THRESHOLD = 1.0   # 1달러 미만은 Stitching 주체에서 제외
MAX_BUCKET_SIZE = 300         # 버킷 당 최대 비교 개수 (Smart Cap)