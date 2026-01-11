import pandas as pd
import numpy as np
from tqdm import tqdm

class Portfolio:
    def __init__(self, initial_cash):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = {} # {ticker: quantity}
        self.history = []

    def equity(self, current_prices=None):
        val = self.cash
        if current_prices is not None:
            for t, q in self.holdings.items():
                price = current_prices.get(t, np.nan)
                if not np.isnan(price) and price > 0:
                    val += q * price
        return val

class BacktestEngine:
    def __init__(self, market_data, start_date=None, end_date=None):
        self.md = market_data
        self.portfolio = None
        
        all_dates = self.md.dates
        if start_date: all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
        if end_date: all_dates = [d for d in all_dates if d <= pd.Timestamp(end_date)]
        self.sim_dates = all_dates
        self.universe_mask = self._precompute_universe()

    def _precompute_universe(self):
        print("🌌 Pre-computing Dynamic Universe (Top 3000 Liquidity)...")
        amount = self.md.prices['Amount']
        rolling_amt = amount.rolling(window=20, min_periods=1).mean()
        rank_matrix = rolling_amt.rank(axis=1, ascending=False)
        return (rank_matrix <= 3000)

    def run(self, strategy, initial_cash=100_000_000):
        print(f"▶️ Running Strategy: {strategy.name}")
        self.portfolio = Portfolio(initial_cash)
        strategy.initialize(self.md)
        
        for date in tqdm(self.sim_dates, desc="Simulating"):
            daily_mask = self.universe_mask.loc[date]
            valid_tickers = daily_mask[daily_mask].index.tolist()
            current_prices = self.md.prices['Close'].loc[date]
            
            # 1. 전략 실행
            orders = strategy.on_bar(date, valid_tickers, self.portfolio)
            
            # 2. 주문 집행
            daily_turnover = self._execute_orders(orders, current_prices)
            
            # 3. 로깅 (상세 보유 내역 추가)
            equity_val = self.portfolio.equity(current_prices)
            
            # [NEW] 보유 종목 상세 스냅샷 생성
            daily_positions = []
            if equity_val > 0:
                for ticker, qty in self.portfolio.holdings.items():
                    price = current_prices.get(ticker, 0)
                    val = price * qty
                    weight = val / equity_val  # 포트폴리오 내 비중
                    
                    daily_positions.append({
                        'ticker': ticker,
                        'price': price,
                        'qty': qty,
                        'value': int(val),
                        'weight': weight
                    })
            
            self.portfolio.history.append({
                'date': date, 
                'equity': equity_val, 
                'cash': self.portfolio.cash,
                'daily_turnover': daily_turnover,
                'holdings_count': len(self.portfolio.holdings),
                'positions': daily_positions  # [NEW] 상세 내역 저장
            })
            
        return pd.DataFrame(self.portfolio.history).set_index('date')

    def _execute_orders(self, orders, prices):
        fee_rate = 0.00015
        total_traded = 0.0
        
        for order in orders:
            ticker = order['ticker']
            qty = order['quantity']
            action = order['action']
            price = prices.get(ticker, np.nan)
            
            if np.isnan(price) or price <= 0: continue
            
            amt = price * qty
            
            if action == 'BUY':
                cost = amt
                fee = cost * fee_rate
                if self.portfolio.cash >= (cost + fee):
                    self.portfolio.cash -= (cost + fee)
                    self.portfolio.holdings[ticker] = self.portfolio.holdings.get(ticker, 0) + qty
                    total_traded += cost
            
            elif action == 'SELL':
                curr = self.portfolio.holdings.get(ticker, 0)
                sell_qty = min(curr, qty)
                if sell_qty > 0:
                    rev = price * sell_qty
                    fee = rev * fee_rate
                    self.portfolio.cash += (rev - fee)
                    self.portfolio.holdings[ticker] -= sell_qty
                    total_traded += rev
                    
                    if self.portfolio.holdings[ticker] == 0:
                        del self.portfolio.holdings[ticker]
                        
        return total_traded