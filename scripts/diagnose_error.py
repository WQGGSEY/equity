import pandas as pd
from pathlib import Path

# 경로 설정 (사용자 환경)
base = Path("/Users/seongje/Desktop/project/domain shift lab/equity/data/bronze/daily_prices")
target_path = base / "ticker=NAN" / "price.parquet"

print(f"📂 파일 확인: {target_path}")

if target_path.exists():
    try:
        df = pd.read_parquet(target_path)
        print("\n--- [데이터 정보] ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\n--- [데이터 내용] ---")
        print(df.head())
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")
else:
    print("❌ 파일이 존재하지 않습니다.")