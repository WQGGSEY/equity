import pandas as pd
from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_PATH = BASE_DIR / "data" / "bronze" / "master_ticker_list.csv"

def main():
    print(">>> [Diagnostic] Master List Duplicate Inspector")
    
    if not MASTER_PATH.exists():
        print("❌ Master List 파일이 없습니다.")
        return

    # CSV 로드
    df = pd.read_csv(MASTER_PATH)
    total_rows = len(df)
    unique_tickers = df['ticker'].nunique()
    
    print(f"  - 전체 행 수: {total_rows}")
    print(f"  - 고유 티커 수: {unique_tickers}")
    print(f"  - 중복된 티커 수: {total_rows - unique_tickers}")
    
    # 중복된 티커 찾기 (모든 중복 항목 표시)
    duplicates = df[df.duplicated(subset=['ticker'], keep=False)].sort_values(by='ticker')
    
    if duplicates.empty:
        print("\n✅ 중복된 티커가 없습니다. (데이터 무결함)")
    else:
        print(f"\n⚠️ 총 {len(duplicates)}개의 중복 행이 발견되었습니다.")
        print("    (아래 리스트를 보고 내용이 완전히 같은지, 아니면 경로가 다른지 확인하세요)\n")
        
        # 보기 좋게 출력
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', 100) # 너무 많으면 잘리니까 적당히
        
        # 상위 20개 그룹만 출력
        print(duplicates.head(50))
        
        # 내용이 완전히 같은지 검사
        is_exact_dup = duplicates.duplicated(keep=False).all()
        
        print("\n" + "-"*60)
        print(">>> [진단 결과 요약]")
        if is_exact_dup:
            print("  🟢 안심하세요: 모든 중복 행의 내용(파일 경로 등)이 완전히 동일합니다.")
            print("     -> 단순 로깅 중복이므로 drop_duplicates()를 써도 데이터 손실이 없습니다.")
        else:
            print("  🔴 위험합니다: 티커는 같지만 내용(start_date, file_path 등)이 다른 행이 있습니다!")
            print("     -> 무작정 지우면 데이터를 잃을 수 있습니다. 상세 확인이 필요합니다.")
            
            # 다른 내용이 있는 놈들만 추출해서 보여줌
            distinct_dups = duplicates[~duplicates.duplicated(keep=False)]
            print("\n  [내용이 충돌하는 중복 행 예시]")
            print(distinct_dups.head(20))

if __name__ == "__main__":
    main()