import pandas as pd
import numpy as np
from .base import Strategy

class FDRebalanceStrategy(Strategy):
    """
    [FD Based Daily Rebalancing Strategy]
    
    Logic:
    1. Ranking: 매일 'FD_TrdAmount' 기준으로 종목의 순위를 매김.
    2. Selection: 상위 Top N 종목 선정.
    3. Weighting: 선정된 종목에 자산을 1/N 씩 균등 배분 (Equal Weight).
    4. Rebalancing: 매일 리밸런싱 수행 (탈락 종목 매도, 신규/비중 부족 종목 매수).
    
    Default: 'FD_TrdAmount'가 높은(High) 순서대로 추출 (ascending=False)
    """
    def __init__(self, top_n=10, ascending=False, feature_name='FD_TrdAmount'):
        super().__init__(name=f"FD_Rebalance_Top{top_n}")
        self.top_n = top_n
        self.ascending = ascending
        self.feature_name = feature_name
        self.md = None

    def initialize(self, market_data):
        self.md = market_data
        
        # 필수 피처 확인
        if self.feature_name not in self.md.features:
            # 경고 혹은 에러 처리
            available = list(self.md.features.keys())
            raise ValueError(f"❌ Feature '{self.feature_name}' not found in MarketData! Available: {available}")
            
        print(f"⚖️ [FD Rebalance] initialized. Target Feature: {self.feature_name}, Top: {self.top_n}")

    def on_bar(self, date, universe_tickers, portfolio):
        orders = []
        current_prices = self.md.prices['Close'].loc[date]
        
        # 1. Feature 데이터 가져오기
        try:
            # 해당 날짜의 전체 종목 Feature 값
            feature_vals = self.md.features[self.feature_name].loc[date]
        except KeyError:
            return []

        # 2. 유효 종목 필터링 및 랭킹 산출
        # 유니버스에 포함되고, 데이터가 NaN이 아닌 종목만 대상
        valid_candidates = []
        for t in universe_tickers:
            val = feature_vals.get(t, np.nan)
            price = current_prices.get(t, np.nan)
            
            if not np.isnan(val) and not np.isnan(price) and price > 0:
                valid_candidates.append((t, val))
        
        if not valid_candidates:
            return []

        # 정렬 (ascending=False -> 값이 큰 순서가 상위)
        valid_candidates.sort(key=lambda x: x[1], reverse=not self.ascending)
        
        # Top N 선정
        top_picks = [x[0] for x in valid_candidates[:self.top_n]]
        
        # 3. 목표 수량 계산 (Total Equity 기준 1/N)
        # 현재 총 자산 가치 계산 (현금 + 보유 주식 평가액)
        total_equity = portfolio.cash
        for t, qty in portfolio.holdings.items():
            price = current_prices.get(t, 0)
            total_equity += qty * price
            
        target_amt_per_stock = total_equity / len(top_picks) if top_picks else 0
        
        # 4. 주문 생성 (매도 먼저 수행하여 현금 확보)
        
        # (A) 매도 주문: 타겟에 없거나, 비중이 과한 경우
        current_holdings = list(portfolio.holdings.keys())
        for t in current_holdings:
            qty = portfolio.holdings[t]
            price = current_prices.get(t, 0)
            
            if t not in top_picks:
                # 탈락 종목 전량 매도
                orders.append({'ticker': t, 'action': 'SELL', 'quantity': qty})
            else:
                # 보유 중이지만 리밸런싱 필요한 경우 (여기서는 단순화를 위해 줄이는 경우만 매도 처리하고, 늘리는 건 아래 매수 로직에서)
                # 정교한 리밸런싱을 위해 목표 수량과의 차이 계산
                target_qty = int(target_amt_per_stock / price)
                diff = target_qty - qty
                if diff < 0:
                    orders.append({'ticker': t, 'action': 'SELL', 'quantity': abs(diff)})
        
        # (B) 매수 주문: 타겟 종목 신규 진입 또는 비중 확대
        for t in top_picks:
            price = current_prices.get(t, 0)
            target_qty = int(target_amt_per_stock / price)
            
            current_qty = portfolio.holdings.get(t, 0)
            diff = target_qty - current_qty
            
            if diff > 0:
                orders.append({'ticker': t, 'action': 'BUY', 'quantity': diff})
                
        return orders