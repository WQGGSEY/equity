import pandas as pd
import numpy as np
from .base import Strategy

class GoldenCrossFDStrategy(Strategy):
    """
    [FD Mean Reversion Strategy - Low Turnover Ver.]
    
    1. Entry: Spread가 가장 낮은(Deepest Dip) Top 10 종목 매수
    2. Exit: 랭킹에서 밀려났다고 바로 팔지 않음.
             Spread >= 0 (평균 회귀 완료) 시점에만 매도.
    3. Position Sizing: 최대 10개 슬롯. 빈 자리가 생겨야만 신규 진입.
    """
    def __init__(self, short_window=5, long_window=20, top_n=10, outlier_pct=0.01):
        super().__init__(name="FD Mean Reversion (Wait for Recovery)")
        self.short_window = short_window
        self.long_window = long_window
        self.top_n = top_n
        
        # Pre-computed Matrices
        self.spread = None       
        self.spread_accel = None 

    def initialize(self, market_data):
        self.md = market_data
        
        if 'FD_Close' not in self.md.features:
            raise ValueError("❌ 'FD_Close' feature not found!")
            
        print(f"⚡ [LowTurnover] Pre-computing Spread & Acceleration...")
        
        fd_close = self.md.features['FD_Close']
        ma_s = fd_close.rolling(window=self.short_window).mean()
        ma_l = fd_close.rolling(window=self.long_window).mean()
        
        self.spread = ma_s - ma_l
        self.spread_accel = self.spread.diff()
        
        print("   -> Calculations Complete.")

    def on_bar(self, date, universe_tickers, portfolio):
        orders = []
        current_prices = self.md.prices['Close'].loc[date]
        
        try:
            curr_spread = self.spread.loc[date].reindex(universe_tickers)
            curr_accel = self.spread_accel.loc[date].reindex(universe_tickers)
        except KeyError:
            return []

        # 유효 데이터 필터링
        valid_mask = ~curr_spread.isna() & ~curr_accel.isna()
        curr_spread = curr_spread[valid_mask]
        curr_accel = curr_accel[valid_mask]
        
        if curr_spread.empty: return []

        # -----------------------------
        # 1. 매도 로직 (Exit Logic)
        # -----------------------------
        current_holdings = list(portfolio.holdings.keys())
        kept_holdings = []
        
        for t in current_holdings:
            # 현재 Spread 확인
            s_val = curr_spread.get(t, np.nan)
            
            # (A) 익절 조건: Spread가 0 이상으로 올라오면 (평균 회귀 완료) -> 매도
            if s_val >= 0:
                qty = portfolio.holdings[t]
                orders.append({'ticker': t, 'action': 'SELL', 'quantity': qty})
            
            # (B) 손절/교체 조건: 데이터가 사라졌거나(상폐), 가속도가 꺾였거나 등
            # 여기서는 단순하게 "아직 회복 안 됐으면(-), 그리고 가속도가 살아있으면 들고 간다"
            # 너무 복잡하면 회전율 또 높아지니, 일단은 'Spread < 0 이면 보유'로 단순화
            elif s_val < 0:
                kept_holdings.append(t)
            else:
                # 데이터 없는 경우 등
                orders.append({'ticker': t, 'action': 'SELL', 'quantity': portfolio.holdings[t]})

        # -----------------------------
        # 2. 매수 로직 (Entry Logic)
        # -----------------------------
        # 빈 슬롯 계산 (최대 N개 - 현재 보유 중인 개수)
        slots_available = self.top_n - len(kept_holdings)
        
        if slots_available > 0:
            # 신규 진입 후보군 찾기
            mask_oversold = curr_spread < 0
            mask_turning_up = curr_accel > 0
            
            final_mask = mask_oversold & mask_turning_up
            candidates = curr_spread[final_mask]
            
            if not candidates.empty:
                # 랭킹: Spread가 가장 작은(Deep Dip) 순서
                # 이미 보유한 종목은 제외하고 상위 N개 뽑기
                potential_picks = candidates.nsmallest(self.top_n * 2).index.tolist()
                real_picks = [t for t in potential_picks if t not in kept_holdings][:slots_available]
                
                if real_picks:
                    # 자금 배분: (총 자산) / (목표 슬롯 수) -> 1/N 씩 균등 배분
                    # 주의: 현재 현금 상황에 맞춰야 함
                    equity = portfolio.equity(current_prices)
                    target_amt_per_stock = equity / self.top_n
                    
                    for t in real_picks:
                        if portfolio.cash < target_amt_per_stock * 0.9: break # 현금 부족하면 중단
                        
                        price = current_prices.get(t, 0)
                        if price > 0:
                            qty = int(target_amt_per_stock / price)
                            if qty > 0:
                                orders.append({'ticker': t, 'action': 'BUY', 'quantity': qty})
                            
        return orders