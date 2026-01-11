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
    
class TrendDipStrategy(Strategy):
    """
    [Heuristic V2] Trend-Filtered Deep Dip
    
    Logic:
    1. Market Filter: SPY가 MA200 위에 있을 때만 매매 (Bull Market Only)
    2. Trend Filter: 개별 종목이 MA60 위에 있을 때만 매매 (Uptrend Only)
    3. Entry: FD Spread < 0 (Short-term Dip)
    4. Exit: Spread 회복(0) OR 손절(-5%) OR 타임컷(5일)
    """
    def __init__(self, fd_short=5, fd_long=20, trend_window=60, market_ticker='SPY'):
        super().__init__(name=f"TrendDip_MA{trend_window}_FD{fd_short}_{fd_long}")
        self.fd_short = fd_short
        self.fd_long = fd_long
        self.trend_window = trend_window
        self.market_ticker = market_ticker
        
        # 데이터 컨테이너
        self.spread = None
        self.ma_trend = None
        self.market_ma = None
        self.market_price = None

    def initialize(self, market_data):
        self.md = market_data
        
        # 1. FD Spread 계산 (기존 로직)
        if 'FD_Close' not in self.md.features:
            raise ValueError("FD_Close feature missing!")
            
        fd = self.md.features['FD_Close']
        self.spread = fd.rolling(self.fd_short).mean() - fd.rolling(self.fd_long).mean()
        
        # 2. 개별 종목 추세선 (MA60) 계산
        # Close 가격 기준
        close = self.md.prices['Close']
        self.ma_trend = close.rolling(window=self.trend_window).mean()
        
        # 3. 시장 지수 필터 계산 (SPY)
        if self.market_ticker in close.columns:
            mkt_close = close[self.market_ticker]
            self.market_price = mkt_close
            self.market_ma = mkt_close.rolling(window=200).mean()
        else:
            print(f"⚠️ Warning: {self.market_ticker} not found. Market filter disabled.")
            self.market_ma = None

    def on_bar(self, date, universe_tickers, portfolio):
        orders = []
        current_prices = self.md.prices['Close'].loc[date]
        
        # --- [1] Market Regime Check ---
        # 시장이 MA200 밑이면(하락장), 절대 매수하지 않고 기존 보유분 청산만 고려
        is_bull_market = True
        if self.market_ma is not None:
            mkt_val = self.market_price.loc[date]
            mkt_ma_val = self.market_ma.loc[date]
            if mkt_val < mkt_ma_val:
                is_bull_market = False

        # --- [2] Signal Generation ---
        try:
            curr_spread = self.spread.loc[date].reindex(universe_tickers)
            curr_trend = self.ma_trend.loc[date].reindex(universe_tickers)
        except KeyError:
            return []
            
        # 보유 종목 관리 (Exit Logic)
        for t, qty in list(portfolio.holdings.items()):
            price = current_prices.get(t, np.nan)
            if np.isnan(price) or price <= 0: continue
            
            # 진입가 추적 (포트폴리오 객체에 평균단가가 없으면 대략 계산 필요하나, 
            # 여기선 간소화를 위해 현재가 기반 청산만 구현)
            
            # A. 익절: Spread >= 0 (평균 회귀 완료)
            s_val = curr_spread.get(t, 0)
            if s_val >= 0:
                orders.append({'ticker': t, 'action': 'SELL', 'quantity': qty})
                continue
                
            # B. 손절/시장 악화: 시장이 하락장으로 전환되면 전량 매도 (Safety First)
            if not is_bull_market:
                orders.append({'ticker': t, 'action': 'SELL', 'quantity': qty})
                continue

        # 신규 진입 (Entry Logic)
        # 하락장이면 매수 금지
        if not is_bull_market:
            return orders

        # 매수 후보군 탐색
        candidates = []
        
        for t in universe_tickers:
            if t in portfolio.holdings: continue # 이미 보유중
            
            s_val = curr_spread.get(t, np.nan)
            p_val = current_prices.get(t, np.nan)
            trend_val = curr_trend.get(t, np.nan)
            
            if np.isnan(s_val) or np.isnan(p_val) or np.isnan(trend_val): continue
            
            # [핵심 로직]
            # 1. Dip: Spread < 0
            # 2. Uptrend: 현재가 > 60일 이평선
            if s_val < 0 and p_val > trend_val:
                candidates.append((t, s_val))
        
        # 랭킹: Spread가 가장 낮은(Deepest) 순서로 상위 5개
        candidates.sort(key=lambda x: x[1])
        top_picks = [x[0] for x in candidates[:5]]
        
        # 자금 배분 (1/N)
        if top_picks:
            target_amt = portfolio.cash / len(top_picks)
            for t in top_picks:
                price = current_prices.get(t, 0)
                if price > 0:
                    qty = int(target_amt / price)
                    if qty > 0:
                        orders.append({'ticker': t, 'action': 'BUY', 'quantity': qty})
                        portfolio.cash -= qty * price # 가상 차감 (중복 주문 방지)

        return orders

class FDMomentumTop10Strategy(Strategy):
    """
    [Strategy V4] FD Momentum on Top 10 Giants
    
    Concept:
    - "Don't touch the garbage."
    - 오직 거래대금(Liquidity) 상위 10개 종목(Mega Caps)만 매매 대상.
    - 대형주는 추세가 정직하고, 상폐 위험이 없으며, 슬리피지가 적음.
    
    Logic:
    1. Universe: Daily Top 10 by Trading Amount (Price * Volume)
    2. Entry: FD Spread > 0 (Up-trend) & Accel > 0 (Momentum) & Price > MA60
    3. Exit: 
       - Hold Period 10 days (TSMOM style)
       - OR Safety Stop (Market Crash)
    """
    def __init__(self, fd_short=5, fd_long=20, trend_window=60, hold_period=10):
        super().__init__(name=f"FD_Momentum_Giants_Top10")
        self.fd_short = fd_short
        self.fd_long = fd_long
        self.trend_window = trend_window
        self.hold_period = hold_period
        
        # 데이터 컨테이너
        self.spread = None
        self.spread_diff = None
        self.ma_trend = None
        self.amount_rank = None # 유동성 랭킹
        self.holding_counts = {}

    def initialize(self, market_data):
        self.md = market_data
        print(f"💎 Initializing Strategy: Only Top 10 Giants...")
        
        # 1. FD Features
        if 'FD_Close' not in self.md.features:
            raise ValueError("FD_Close feature missing!")
            
        fd = self.md.features['FD_Close']
        ma_s = fd.rolling(self.fd_short).mean()
        ma_l = fd.rolling(self.fd_long).mean()
        
        self.spread = ma_s - ma_l
        self.spread_diff = self.spread.diff()
        
        # 2. Trend & Liquidity Ranking
        close = self.md.prices['Close']
        volume = self.md.prices['Volume']
        
        # 거래대금(Amount) 계산 및 랭킹 산출
        # (Raw Data에 Amount가 없으면 Close * Volume으로 추정)
        amount = self.md.prices.get('Amount', close * volume)
        
        # 20일 평균 거래대금 기준 (노이즈 제거)
        rolling_amt = amount.rolling(window=20).mean()
        
        # 랭킹: 숫자가 작을수록(1위) 상위
        self.amount_rank = rolling_amt.rank(axis=1, ascending=False)
        
        # 개별 주가 추세 (MA60)
        self.ma_trend = close.rolling(window=self.trend_window).mean()

    def on_bar(self, date, universe_tickers, portfolio):
        orders = []
        current_prices = self.md.prices['Close'].loc[date]
        
        # [Step 1] Top 10 Giants 필터링
        # 엔진에서 3000개를 줬어도, 여기서는 Top 10만 남기고 다 버림
        try:
            daily_ranks = self.amount_rank.loc[date].reindex(universe_tickers)
            # 랭킹 10위 이내인 종목만 선정
            giants_mask = daily_ranks <= 10
            giants_tickers = daily_ranks[giants_mask].index.tolist()
        except KeyError:
            return []
            
        if not giants_tickers:
            return []

        # [Step 2] 데이터 슬라이싱
        curr_spread = self.spread.loc[date].reindex(giants_tickers)
        curr_accel = self.spread_diff.loc[date].reindex(giants_tickers)
        curr_trend = self.ma_trend.loc[date].reindex(giants_tickers)
        
        # -----------------------------
        # 1. 매도 로직 (Exit)
        # -----------------------------
        current_holdings = list(portfolio.holdings.keys())
        for t in current_holdings:
            qty = portfolio.holdings[t]
            self.holding_counts[t] = self.holding_counts.get(t, 0) + 1
            
            # (A) 타임 컷: 10일 보유 후 청산
            if self.holding_counts[t] >= self.hold_period:
                orders.append({'ticker': t, 'action': 'SELL', 'quantity': qty})
                if t in self.holding_counts: del self.holding_counts[t]
                continue
            
            # (B) 랭킹 이탈 시 매도? 
            # -> 굳이 안 함. 한 번 샀으면 Top 10에서 밀려나도 10일은 들고 감 (잦은 매매 방지)

        # -----------------------------
        # 2. 매수 로직 (Entry)
        # -----------------------------
        candidates = []
        
        for t in giants_tickers:
            if t in portfolio.holdings: continue
            
            s_val = curr_spread.get(t, np.nan)
            a_val = curr_accel.get(t, np.nan)
            p_val = current_prices.get(t, np.nan)
            t_val = curr_trend.get(t, np.nan)
            
            if np.isnan(s_val) or np.isnan(p_val): continue
            
            # [조건] 
            # 대형주라도 추세가 꺾이면 사지 않음
            # 1. FD Spread > 0 (상승 모멘텀)
            # 2. Accel > 0 (가속)
            # 3. Price > MA60 (정배열)
            if s_val > 0 and a_val > 0 and p_val > t_val:
                candidates.append((t, s_val))
        
        # 랭킹: 모멘텀이 강한 순서
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_picks = [x[0] for x in candidates] # Top 10 안에서 골랐으므로 개수 제한 굳이 안 해도 됨 (최대 10개)
        
        if top_picks:
            # 현금 배분 (보유 종목 제외하고 남은 슬롯만큼? 아니면 단순 1/N?)
            # 여기서는 공격적으로 가용 현금 전부 투입 (Top Giants니까)
            available_cash = portfolio.cash
            if available_cash > 1000: # 최소 금액
                # 종목당 최대 비중 제한 (예: 자산의 20%)
                # 하지만 간단히: (가용현금 / 종목수)
                target_amt = available_cash / len(top_picks)
                
                for t in top_picks:
                    price = current_prices.get(t, 0)
                    if price > 0:
                        qty = int(target_amt / price)
                        if qty > 0:
                            orders.append({'ticker': t, 'action': 'BUY', 'quantity': qty})

        return orders