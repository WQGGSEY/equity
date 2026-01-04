import os
from pathlib import Path

# ==========================================
# [Setup] SRC 운영 환경 디렉토리 구축
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

def create_structure():
    print(">>> [System] 운영 환경(src) 디렉토리 구축 시작")
    
    # 생성할 디렉토리 목록
    dirs = [
        SRC_DIR,
        SRC_DIR / "pipeline",
    ]
    
    # 생성할 파일 목록 (__init__.py 등)
    files = [
        SRC_DIR / "__init__.py",
        SRC_DIR / "config.py",
        SRC_DIR / "utils.py",
        SRC_DIR / "main.py",
        SRC_DIR / "pipeline" / "__init__.py",
        SRC_DIR / "pipeline" / "bronze_updater.py",
        SRC_DIR / "pipeline" / "silver_transformer.py",
        SRC_DIR / "pipeline" / "gold_processor.py",
    ]
    
    # 디렉토리 생성
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  📂 Directory: {d.relative_to(BASE_DIR)}")
        
    # 파일 생성 (빈 파일)
    for f in files:
        if not f.exists():
            f.touch()
            print(f"  📄 File: {f.relative_to(BASE_DIR)}")
        else:
            print(f"  ✅ Exists: {f.relative_to(BASE_DIR)}")
            
    print("\n>>> 시스템 뼈대 완성. 이제 각 모듈을 채워넣으면 됩니다.")

if __name__ == "__main__":
    create_structure()