import pandas as pd
import numpy as np
from .base import Strategy

class FDRebalanceStrategy(Strategy):
    """
    [FD Based Daily Rebalancing Strategy] (Fixed Version)
    """
    def __init__(self, top_n=10, ascending=False, feature_name='FD_TrdAmount'):
        super().__init__(name=f"FD_Rebalance_Top{top_n}")
        self.top_n = top_n
        self.ascending = ascending
        self.feature_name = feature_name
        self.md = None

    def initialize(self, market_data):
        self.md = market_data
        if self.feature_name not in self.md.features:
            available = list(self.md.features.keys())
            raise ValueError(f"❌ Feature '{self.feature_name}' not found in MarketData! Available: {available}")
        print(f"⚖️ [FD Rebalance] initialized. Target Feature: {self.feature_name}, Top: {self.top_n}")

    def on_bar(self, date, universe_tickers, portfolio):
        orders = []
        current_prices = self.md.prices['Close'].loc[date]
        
        # 1. Feature 데이터 가져오기
        try:
            feature_vals = self.md.features[self.feature_name].loc[date]
        except KeyError:
            return []

        # 2. 유효 종목 필터링 및 랭킹 산출
        valid_candidates = []
        for t in universe_tickers:
            val = feature_vals.get(t, np.nan)
            price = current_prices.get(t, np.nan)
            
            # 가격과 피처 값이 모두 유효한 경우만 후보 등록
            if not np.isnan(val) and not np.isnan(price) and price > 0:
                valid_candidates.append((t, val))
        
        if not valid_candidates:
            return []

        # 정렬
        valid_candidates.sort(key=lambda x: x[1], reverse=not self.ascending)
        top_picks = [x[0] for x in valid_candidates[:self.top_n]]
        
        # 3. 목표 수량 계산 (Total Equity 기준 1/N)
        total_equity = portfolio.cash
        for t, qty in portfolio.holdings.items():
            price = current_prices.get(t, np.nan)
            # [수정] 보유 종목의 가격이 NaN이면 0으로 처리하여 전체 자산 가치 오염 방지
            if pd.isna(price) or price <= 0:
                price = 0
            total_equity += qty * price
            
        target_amt_per_stock = total_equity / len(top_picks) if top_picks else 0
        
        # [안전장치] 만약 자산 계산이 잘못되어 NaN이나 음수가 나오면 매매 중단
        if pd.isna(target_amt_per_stock) or target_amt_per_stock <= 0:
            return []
        
        # 4. 주문 생성
        
        # (A) 매도 주문
        current_holdings = list(portfolio.holdings.keys())
        for t in current_holdings:
            qty = portfolio.holdings[t]
            price = current_prices.get(t, np.nan)
            
            # 가격 정보를 알 수 없으면 일단 매도 보류 (또는 시장가 강제 매도 고려 가능)
            if pd.isna(price) or price <= 0:
                continue

            if t not in top_picks:
                orders.append({'ticker': t, 'action': 'SELL', 'quantity': qty})
            else:
                # 리밸런싱 (비중 축소)
                target_qty = int(target_amt_per_stock / price)
                diff = target_qty - qty
                if diff < 0:
                    orders.append({'ticker': t, 'action': 'SELL', 'quantity': abs(diff)})
        
        # (B) 매수 주문
        for t in top_picks:
            price = current_prices.get(t, np.nan)
            
            # [수정] 가격 안전장치
            if pd.isna(price) or price <= 0:
                continue
                
            target_qty = int(target_amt_per_stock / price)
            current_qty = portfolio.holdings.get(t, 0)
            diff = target_qty - current_qty
            
            if diff > 0:
                orders.append({'ticker': t, 'action': 'BUY', 'quantity': diff})
                
        return orders