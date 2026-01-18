import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 설정
file_path = Path(__file__).resolve()
project_dir = file_path.parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# [수정됨] Loader 대신 MarketData 임포트
from src.backtest.loader import MarketData
from src.config import PLATINUM_DIR

def diagnose_crash():
    print("🚑 [DIAGNOSIS] Starting investigation for 2022-03-10 Crash...")

    # 1. MarketData 초기화 및 로드
    print("   Loading Market Data...")
    try:
        # MarketData 인스턴스 생성
        md = MarketData(platinum_dir=PLATINUM_DIR)
        # 데이터 로드 (기본 가격 데이터 로드)
        md.load_all()
    except Exception as e:
        print(f"❌ Failed to load MarketData: {e}")
        return
    
    # 2. 날짜 설정
    date_prev = pd.Timestamp("2022-03-09")
    date_crash = pd.Timestamp("2022-03-10")
    
    # 데이터에 해당 날짜가 있는지 확인 (md.prices['Close'] 사용)
    if 'Close' not in md.prices:
        print("❌ Critical Error: 'Close' price data not found in MarketData.")
        return

    close_prices = md.prices['Close']
    
    if date_crash not in close_prices.index:
        print(f"❌ Error: {date_crash} not found in price data index.")
        print(f"   Available range: {close_prices.index[0]} ~ {close_prices.index[-1]}")
        return

    # 3. 가격 데이터 비교
    price_prev = close_prices.loc[date_prev]
    price_crash = close_prices.loc[date_crash]
    
    # 두 날짜 모두 상장되어 있던 종목만 비교 (NaN 제외)
    common_tickers = price_prev.dropna().index.intersection(price_crash.dropna().index)
    
    print(f"\n📊 Analyzing {len(common_tickers)} tickers active on both days...")
    
    if len(common_tickers) == 0:
        print("⚠️ No common tickers found between the two dates. Something is very wrong.")
        return

    # 4. 수익률 계산
    p_prev = price_prev[common_tickers]
    p_curr = price_crash[common_tickers]
    returns = (p_curr - p_prev) / p_prev
    
    # (A) -30% 이상 폭락한 종목 찾기
    crashers = returns[returns < -0.30].sort_values()
    
    if not crashers.empty:
        print(f"\n📉 [CRASH DETECTED] Top losers on {date_crash.date()}:")
        print(crashers.head(20))
        
        worst = crashers.index[0]
        print(f"\n   -> Worst Ticker: {worst}")
        print(f"      3/09 Price: {p_prev[worst]}")
        print(f"      3/10 Price: {p_curr[worst]}")
    else:
        print("\n✅ No individual stock crashed > 30%.")

    # (B) 데이터 실종 (NaN) 탐지
    # 전날엔 값이 있었는데, 이날 NaN이 된 종목 찾기
    valid_prev_tickers = price_prev.dropna().index
    valid_curr_tickers = price_crash.dropna().index
    missing = valid_prev_tickers.difference(valid_curr_tickers)
    
    if not missing.empty:
        print(f"\n👻 [MISSING DATA] {len(missing)} tickers became NaN on {date_crash.date()}:")
        print(list(missing)[:20]) # 20개만 출력
        
        # 예시 확인
        sample = missing[0]
        print(f"   -> Example '{sample}':")
        # 전후 2일치 데이터 출력
        try:
            window = close_prices.loc[date_prev - pd.Timedelta(days=2) : date_crash + pd.Timedelta(days=2), sample]
            print(window)
        except:
            print("      (Could not fetch window data)")
    else:
        print("\n✅ No missing data found (No tickers disappeared).")

    # (C) 0원 데이터 탐지
    zeros = price_crash[price_crash <= 0].index
    if not zeros.empty:
        print(f"\n0️⃣ [ZERO PRICE] {len(zeros)} tickers have 0.0 price:")
        print(list(zeros))

if __name__ == "__main__":
    diagnose_crash()