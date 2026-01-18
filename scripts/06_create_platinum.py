import sys
import multiprocessing
from pathlib import Path
import time

# 프로젝트 루트 경로 설정
FILE_PATH = Path(__file__).resolve()
PROJECT_DIR = FILE_PATH.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# 이제 PlatinumProcessor 클래스가 존재하므로 정상적으로 import 됩니다.
from src.pipeline.platinum_processor import PlatinumProcessor

def main():
    print("="*60)
    print("🚀 PLATINUM LAYER GENERATION")
    print("="*60)
    
    multiprocessing.freeze_support()
    start_time = time.time()
    
    # Processor 인스턴스 생성 및 실행
    processor = PlatinumProcessor()
    processor.process_features()
    
    end_time = time.time()
    print(f"\n✨ All Done! Total Time: {end_time - start_time:.2f} sec")

if __name__ == "__main__":
    main()