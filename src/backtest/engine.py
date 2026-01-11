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
    def __init__(self, market_data, start_date=None, end_date=None):
        self.md = market_data
        self.portfolio = None
        
        all_dates = self.md.dates
        if start_date: all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
        if end_date: all_dates = [d for d in all_dates if d <= pd.Timestamp(end_date)]
        self.sim_dates = all_dates
        self.universe_mask = self._precompute_universe()
        
        # [NEW] VWAP 미리 계산 (O+H+L+C)/4
        print("📊 Pre-computing VWAP ((O+H+L+C)/4)...")
        # 데이터가 없는 경우 Close로 대체하거나 NaN 처리
        o = self.md.prices.get('Open', self.md.prices['Close'])
        h = self.md.prices.get('High', self.md.prices['Close'])
        l = self.md.prices.get('Low', self.md.prices['Close'])
        c = self.md.prices['Close']
        self.vwap = (o + h + l + c) / 4.0

    def _precompute_universe(self):
        print("🌌 Pre-computing Dynamic Universe (Top 3000 Liquidity)...")
        amount = self.md.prices.get('Amount', self.md.prices['Close'] * self.md.prices['Volume'])
        rolling_amt = amount.rolling(window=20, min_periods=1).mean()
        rank_matrix = rolling_amt.rank(axis=1, ascending=False)
        return (rank_matrix <= 3000)

    def run(self, strategy, initial_cash=100_000_000):
        print(f"▶️ Running Strategy: {strategy.name} (Execution: Next Day VWAP)")
        self.portfolio = Portfolio(initial_cash)
        strategy.initialize(self.md)
        
        last_valid_prices = {} 
        nan_duration = {}
        
        # [핵심] 주문 보관함 (오늘 주문 -> 내일 체결)
        pending_orders = [] 
        
        for date in tqdm(self.sim_dates, desc="Simulating"):
            # 1. 오늘의 데이터 (Signal용: Close, Execution용: VWAP)
            current_close = self.md.prices['Close'].loc[date]
            current_vwap = self.vwap.loc[date] # 체결은 이걸로
            
            # 메타데이터 업데이트 (상폐 방지 로직 등)
            for t in list(self.portfolio.holdings.keys()):
                p = current_close.get(t, np.nan)
                if np.isnan(p):
                    nan_duration[t] = nan_duration.get(t, 0) + 1
                else:
                    nan_duration[t] = 0
                    if p > 0: last_valid_prices[t] = p

            # 2. [체결 단계] "어제 접수한 주문"을 "오늘의 VWAP"으로 체결
            # (수수료 0.3% 적용)
            daily_turnover = 0.0
            if pending_orders:
                daily_turnover = self._execute_orders(pending_orders, current_vwap)
                pending_orders = [] # 체결 완료 후 비움

            # 3. [전략 실행 단계] "오늘의 종가(Close)"를 보고 신호 생성
            daily_mask = self.universe_mask.loc[date]
            valid_tickers = daily_mask[daily_mask].index.tolist()
            
            # 전략에게는 'Close' 정보를 줌 (당일 판단)
            new_orders = strategy.on_bar(date, valid_tickers, self.portfolio)
            
            # [핵심] 주문을 바로 체결하지 않고 '내일'로 넘김
            pending_orders = new_orders
            
            # 4. 포트폴리오 평가 (평가는 보수적으로 Close 기준 or VWAP 기준)
            # 보통 자산 평가는 종가(Close)로 하는 것이 원칙
            equity_val = self.portfolio.cash
            
            daily_positions = []
            
            for ticker, qty in self.portfolio.holdings.items():
                price = current_close.get(ticker, np.nan)
                
                # 좀비 기업 처리 (5일 이상 거래 정지 시 0원)
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
        # 수수료 + 슬리피지 포함 0.3% (보수적)
        fee_rate = 0.003
        total_traded = 0.0
        
        for order in orders:
            ticker = order['ticker']
            qty = order['quantity']
            action = order['action']
            
            # 체결 가격은 VWAP
            price = prices.get(ticker, np.nan)
            
            if np.isnan(price) or price <= 0: continue
            
            amt = price * qty
            
            if action == 'BUY':
                cost = amt
                fee = cost * fee_rate
                # 미수 방지: 어제 주문 낼 때 현금 있었어도, 오늘 VWAP이 폭등해서 부족할 수 있음 체크
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