import pandas as pd
import numpy as np
from tqdm import tqdm

class Portfolio:
    def __init__(self, initial_cash):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = {} 
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
    # [수정 1] __init__에서 fee_rate를 인자로 받도록 변경
    def __init__(self, market_data, start_date=None, end_date=None, fee_rate=0.0, universe_size=3000):
        self.md = market_data
        self.universe_size = universe_size
        
        # [수정 2] 수수료율 저장 (기본값 0.0)
        self.fee_rate = fee_rate 
        
        self.portfolio = None
        
        all_dates = self.md.dates
        if start_date: all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
        if end_date: all_dates = [d for d in all_dates if d <= pd.Timestamp(end_date)]
        self.sim_dates = all_dates
        self.universe_mask = self._precompute_universe()
        
        print("📊 Pre-computing VWAP ((O+H+L+C)/4)...")
        # 데이터가 없는 경우 Close로 대체하거나 NaN 처리
        o = self.md.prices.get('Open', self.md.prices['Close'])
        h = self.md.prices.get('High', self.md.prices['Close'])
        l = self.md.prices.get('Low', self.md.prices['Close'])
        c = self.md.prices['Close']
        self.vwap = (o + h + l + c) / 4.0

    def _precompute_universe(self):
        print(f"🌌 Pre-computing Dynamic Universe (Top {self.universe_size} Liquidity)...")
        # Amount가 없으면 Close * Volume으로 대체
        amount = self.md.prices.get('Amount', self.md.prices['Close'] * self.md.prices['Volume'])
        rolling_amt = amount.rolling(window=20, min_periods=1).mean()
        rank_matrix = rolling_amt.rank(axis=1, ascending=False)
        
        # [수정] 동전주 필터 추가 (1달러 미만 잡주 제외)
        price_filter = (self.md.prices['Close'] > 1.0)
        
        # 랭킹 3000위 이내이면서 & 가격이 1달러 이상인 종목만 True
        return (rank_matrix <= self.universe_size) & price_filter

    def run(self, strategy, initial_cash=100_000_000):
        print(f"▶️ Running Strategy: {strategy.name} (Execution: Next Day VWAP)")
        print(f"   (Settings) Fee Rate: {self.fee_rate * 100:.2f}%") # 확인용 로그 출력

        self.portfolio = Portfolio(initial_cash)
        strategy.initialize(self.md)
        
        last_valid_prices = {} 
        nan_duration = {}
        
        pending_orders = [] 
        
        for date in tqdm(self.sim_dates, desc="Simulating"):
            # 1. 오늘의 데이터
            current_close = self.md.prices['Close'].loc[date]
            current_vwap = self.vwap.loc[date] 
            
            # 메타데이터 업데이트
            for t in list(self.portfolio.holdings.keys()):
                p = current_close.get(t, np.nan)
                if np.isnan(p):
                    nan_duration[t] = nan_duration.get(t, 0) + 1
                else:
                    nan_duration[t] = 0
                    if p > 0: last_valid_prices[t] = p

            # 2. 체결 단계
            daily_turnover = 0.0
            if pending_orders:
                daily_turnover = self._execute_orders(pending_orders, current_vwap)
                pending_orders = [] 

            # 3. 전략 실행 단계
            daily_mask = self.universe_mask.loc[date]
            valid_tickers = daily_mask[daily_mask].index.tolist()
            
            new_orders = strategy.on_bar(date, valid_tickers, self.portfolio)
            pending_orders = new_orders
            
            # 4. 포트폴리오 평가
            equity_val = self.portfolio.cash
            
            daily_positions = []
            
            for ticker, qty in self.portfolio.holdings.items():
                price = current_close.get(ticker, np.nan)
                
                # 상폐/거래정지 처리
                if np.isnan(price):
                    if nan_duration.get(ticker, 0) > 5: price = 0.0 
                    else: price = last_valid_prices.get(ticker, 0.0)
                
                val = price * qty
                if price > 0: equity_val += val
                
                daily_positions.append({
                    'ticker': ticker, 'price': price, 'qty': qty, 'value': val
                })
            
            final_positions_log = []
            if equity_val > 0:
                for pos in daily_positions:
                    pos['weight'] = pos['value'] / equity_val
                    try: pos['value'] = int(pos['value'])
                    except: pos['value'] = 0
                    final_positions_log.append(pos)
            
            self.portfolio.history.append({
                'date': date, 'equity': equity_val, 'cash': self.portfolio.cash,
                'daily_turnover': daily_turnover,
                'holdings_count': len(self.portfolio.holdings),
                'positions': final_positions_log
            })
            
        return pd.DataFrame(self.portfolio.history).set_index('date')

    def _execute_orders(self, orders, prices):
        # [수정 3] 하드코딩 제거 -> self.fee_rate 사용
        fee_rate = self.fee_rate
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
                # 미수 방지
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