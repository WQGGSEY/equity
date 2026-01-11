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
        """
        현재 포트폴리오의 평가 가치를 계산합니다.
        가격 데이터가 없는 종목은 0원으로 계산될 수 있으니 주의가 필요합니다.
        (BacktestEngine.run 내부에서는 별도의 로직으로 상폐 종목 가치를 보정합니다.)
        """
        val = self.cash
        if current_prices is not None:
            for t, q in self.holdings.items():
                price = current_prices.get(t, np.nan)
                # 가격이 유효한 경우에만 가치 합산
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
        # Amount 데이터가 있으면 사용, 없으면(NaN) 0 처리
        amount = self.md.prices.get('Amount', self.md.prices['Close'] * self.md.prices['Volume'])
        rolling_amt = amount.rolling(window=20, min_periods=1).mean()
        rank_matrix = rolling_amt.rank(axis=1, ascending=False)
        return (rank_matrix <= 3000)

    def run(self, strategy, initial_cash=100_000_000):
        print(f"▶️ Running Strategy: {strategy.name}")
        self.portfolio = Portfolio(initial_cash)
        strategy.initialize(self.md)
        
        # [핵심] 상폐/정지 종목 대비: 각 종목의 '마지막 유효 가격'을 기억하는 메모리
        last_valid_prices = {} 
        
        for date in tqdm(self.sim_dates, desc="Simulating"):
            # 1. 현재 가격 가져오기 (Raw Data, NaN 포함)
            current_prices = self.md.prices['Close'].loc[date]
            
            # [핵심] 유효 가격 업데이트
            # 오늘 가격이 존재하는 종목들은 last_valid_prices를 최신값으로 갱신
            valid_today = current_prices.dropna()
            for t, p in valid_today.items():
                if p > 0:
                    last_valid_prices[t] = p
            
            daily_mask = self.universe_mask.loc[date]
            valid_tickers = daily_mask[daily_mask].index.tolist()
            
            # 2. 전략 실행
            # (Portfolio 클래스에 equity 메서드가 있어야 전략 내부에서 호출 가능함)
            orders = strategy.on_bar(date, valid_tickers, self.portfolio)
            
            # 3. 주문 집행
            daily_turnover = self._execute_orders(orders, current_prices)
            
            # 4. 포트폴리오 평가 및 로깅 (안전 장치 추가)
            equity_val = self.portfolio.cash
            
            daily_positions = []
            
            for ticker, qty in self.portfolio.holdings.items():
                # A. 현재가 우선 조회
                price = current_prices.get(ticker, np.nan)
                
                # B. 현재가가 NaN이면 -> '마지막 유효 가격' 조회 (좀비 종목 평가)
                if np.isnan(price) or price <= 0:
                    price = last_valid_prices.get(ticker, 0.0)
                
                # C. 가치 계산 (NaN 방지)
                val = price * qty
                
                # D. 합산
                if not np.isnan(val):
                    equity_val += val
                else:
                    val = 0 # 끝내 가격을 못 찾은 경우 0 처리
                
                # 상세 내역 임시 저장
                daily_positions.append({
                    'ticker': ticker,
                    'price': price,
                    'qty': qty,
                    'value': val
                })
            
            # 비중(Weight) 계산 및 최종 리스트 생성
            final_positions_log = []
            if equity_val > 0:
                for pos in daily_positions:
                    pos['weight'] = pos['value'] / equity_val
                    # [핵심] 에러 발생 지점 방지: 안전하게 int 변환
                    try:
                        pos['value'] = int(pos['value'])
                    except ValueError:
                        pos['value'] = 0
                    final_positions_log.append(pos)
            
            self.portfolio.history.append({
                'date': date, 
                'equity': equity_val, 
                'cash': self.portfolio.cash,
                'daily_turnover': daily_turnover,
                'holdings_count': len(self.portfolio.holdings),
                'positions': final_positions_log
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
            
            # 가격이 없으면 거래 불가 -> 스킵 (이래서 좀비 종목이 남는 것임)
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