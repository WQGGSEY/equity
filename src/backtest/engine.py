import pandas as pd
import numpy as np
from tqdm import tqdm

class Portfolio:
    def __init__(self, initial_cash):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = {} # {ticker: quantity}
        self.history = []

    # [FIX] @property 제거 (인자를 받으려면 일반 메서드여야 함)
    def equity(self, current_prices=None):
        """현재 평가금액 계산 (현금 + 보유주식 평가액)"""
        val = self.cash
        if current_prices is not None:
            for t, q in self.holdings.items():
                # current_prices는 Series 또는 dict
                price = current_prices.get(t, np.nan)
                if not np.isnan(price) and price > 0:
                    val += q * price
        return val

class BacktestEngine:
    def __init__(self, market_data, start_date=None, end_date=None):
        self.md = market_data
        self.portfolio = None
        
        # 날짜 필터링
        all_dates = self.md.dates
        if start_date: all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
        if end_date: all_dates = [d for d in all_dates if d <= pd.Timestamp(end_date)]
        self.sim_dates = all_dates
        
        # 유니버스 마스크 (미리 계산)
        self.universe_mask = self._precompute_universe()

    def _precompute_universe(self):
        print("🌌 Pre-computing Dynamic Universe (Top 3000 Liquidity)...")
        # 20일 이동평균 거래대금
        amount = self.md.prices['Amount']
        rolling_amt = amount.rolling(window=20, min_periods=1).mean()
        
        # 랭킹 산출 (내림차순, 1등이 최고 유동성)
        # 3000등 이내이면 True
        rank_matrix = rolling_amt.rank(axis=1, ascending=False)
        mask = (rank_matrix <= 3000)
        return mask

    def run(self, strategy, initial_cash=100_000_000):
        print(f"▶️ Running Strategy: {strategy.name}")
        self.portfolio = Portfolio(initial_cash)
        
        # 전략 초기화 (데이터 접근 권한 부여)
        strategy.initialize(self.md)
        
        for date in tqdm(self.sim_dates, desc="Simulating"):
            # 1. 오늘 유효한 유니버스 티커 식별
            # (loc으로 해당 날짜 행을 가져옴 -> Boolean Series)
            daily_mask = self.universe_mask.loc[date]
            
            # mask가 True인 인덱스(티커)만 추출
            valid_tickers = daily_mask[daily_mask].index.tolist()
            
            # 2. 현재 시장 가격
            current_prices = self.md.prices['Close'].loc[date]
            
            # 3. 전략 실행 (주문 생성)
            orders = strategy.on_bar(date, valid_tickers, self.portfolio)
            
            # 4. 주문 집행
            self._execute_orders(orders, current_prices)
            
            # 5. 로깅
            equity_val = self.portfolio.equity(current_prices)
            self.portfolio.history.append({
                'date': date, 
                'equity': equity_val, 
                'cash': self.portfolio.cash,
                'holdings_count': len(self.portfolio.holdings)
            })
            
        return pd.DataFrame(self.portfolio.history).set_index('date')

    def _execute_orders(self, orders, prices):
        fee_rate = 0.00015 # 0.015%
        
        for order in orders:
            ticker = order['ticker']
            qty = order['quantity']
            action = order['action']
            price = prices.get(ticker, np.nan)
            
            if np.isnan(price) or price <= 0: continue
            
            if action == 'BUY':
                cost = price * qty
                fee = cost * fee_rate
                if self.portfolio.cash >= (cost + fee):
                    self.portfolio.cash -= (cost + fee)
                    self.portfolio.holdings[ticker] = self.portfolio.holdings.get(ticker, 0) + qty
            
            elif action == 'SELL':
                current_qty = self.portfolio.holdings.get(ticker, 0)
                sell_qty = min(current_qty, qty)
                if sell_qty > 0:
                    revenue = price * sell_qty
                    fee = revenue * fee_rate
                    self.portfolio.cash += (revenue - fee)
                    self.portfolio.holdings[ticker] -= sell_qty
                    # 잔고가 0이 되면 딕셔너리에서 제거 (메모리 절약)
                    if self.portfolio.holdings[ticker] == 0:
                        del self.portfolio.holdings[ticker]