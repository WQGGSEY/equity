import sys
import multiprocessing
from pathlib import Path

# -------------------------------------------------------------------------
# [Setup] Project Root Path
# 스크립트 실행 위치와 상관없이 src 모듈을 찾을 수 있도록 경로를 설정합니다.
# -------------------------------------------------------------------------
FILE_PATH = Path(__file__).resolve()
PROJECT_DIR = FILE_PATH.parent.parent  # equity/
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.pipeline.platinum_processor import process_features

def main():
    """
    [Platinum Layer Initialization]
    Gold Layer 데이터를 로드하여 'Feature Engineering'을 수행합니다.
    - Dollar Bar 변환
    - Fractional Differentiation (정상성 확보)
    - Contrastive Learning Views 생성
    - Universal Data Fusion (Macro, Fundamental 병합)
    """
    print(f"🚀 Initializing Platinum Layer Creation...")
    print(f"📂 Project Root: {PROJECT_DIR}")
    
    # 멀티프로세싱 안전장치 (Windows/macOS 필수)
    multiprocessing.freeze_support()
    
    # Platinum Processor 실행
    process_features()
    
    print("\n✨ Platinum Layer Creation Completed Successfully.")

if __name__ == "__main__":
    main()