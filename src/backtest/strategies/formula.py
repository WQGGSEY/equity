import pandas as pd
import numpy as np
from .base import Strategy
from src.alpha.parser import AlphaParser

class FormulaStrategy(Strategy):
    """
    [Pure Signal Strategy + Debugger]
    데이터 상태와 시그널 생성 현황을 정밀 진단합니다.
    """
    def __init__(self, expressions, **kwargs):
        kwargs.pop('top_n', None)
        kwargs.pop('threshold', None)
        kwargs.pop('drop_zero', None)
        name = kwargs.pop('name', "FormulaStrategy")
        super().__init__(name=name)
        
        if isinstance(expressions, str):
            self.expressions = [expressions]
        else:
            self.expressions = expressions
            
        self.parser = AlphaParser()
        self.signal_matrix = None

    def initialize(self, market_data):
        self.md = market_data
        print("\n🧪 [Alpha Engine] Initializing & Debugging Data Context...")
        
        # 1. 데이터 컨텍스트 준비
        data_context = {}
        for k, v in market_data.prices.items():
            data_context[k] = v
            data_context[k.lower()] = v
            
        if hasattr(market_data, 'features'):
            for k, v in market_data.features.items():
                data_context[k] = v
                data_context[k.lower()] = v

        # # -------------------------------------------------------------
        # # [🕵️‍♂️ 긴급 점검] 데이터가 진짜 들어있나 확인
        # # -------------------------------------------------------------
        # print(f"   🔍 Checking Data Integrity for {len(market_data.tickers)} tickers, {len(market_data.dates)} days...")
        
        # # (1) FD_Close 확인
        # if 'FD_Close' in data_context:
        #     fd = data_context['FD_Close']
        #     valid_count = fd.notna().sum().sum()
        #     total_cells = fd.shape[0] * fd.shape[1]
        #     fill_rate = (valid_count / total_cells) * 100
        #     print(f"      👉 'FD_Close' Fill Rate: {fill_rate:.2f}% (Valid: {valid_count} / Total: {total_cells})")
        #     if fill_rate < 1.0:
        #         print("      🚨 ERROR: FD_Close 데이터가 거의 비어있습니다! (feature 생성 실패 의심)")
        # else:
        #     print("      🚨 ERROR: 'FD_Close' feature가 로드되지 않았습니다!")

        # # (2) Universe 확인
        # if 'universe' in market_data.prices:
        #     univ = market_data.prices['universe']
        #     univ_ones = (univ == 1.0).sum().sum()
        #     print(f"      👉 Universe (Top 500) Count sum: {univ_ones} (Should be roughly 500 * days)")
        #     if univ_ones == 0:
        #         print("      🚨 ERROR: 유니버스 마스크가 모두 0입니다! (거래대금 계산 실패 의심)")
        # else:
        #     print("      ⚠️ Warning: 'universe' mask not found in prices.")

        # 2. 수식 계산
        final_signal = pd.DataFrame(0.0, index=market_data.dates, columns=market_data.tickers)
        
        for expr in self.expressions:
            print(f"   -> Calculating: {expr}")
            try:
                alpha_val = self.parser.parse(expr, data_context)
                
                # [점검] 수식 결과값 분포 확인
                print(f"      📊 Signal Stats -> Min: {alpha_val.min().min():.4f}, Max: {alpha_val.max().max():.4f}, Mean: {alpha_val.mean().mean():.4f}")
                print(f"      📊 Non-zero Count: {(alpha_val != 0).sum().sum()}")
                
                final_signal = final_signal.add(alpha_val.fillna(0.0), fill_value=0)
            except Exception as e:
                print(f"   🚨 Error in Expression: {expr}")
                raise e
            
        # 3. 유니버스 적용
        if 'universe' in market_data.prices:
            print("   🌌 Applying Universe Mask...")
            final_signal = final_signal * market_data.prices['universe']
            
        final_signal = final_signal.fillna(0.0)
        
        # [최종 점검]
        active_signals = (final_signal != 0).sum().sum()
        print(f"   ✅ Final Active Signals Count: {active_signals}")
        if active_signals == 0:
            print("   ❌ WARNING: 최종 시그널이 하나도 없습니다. 매매가 발생하지 않습니다.")
        
        self.signal_matrix = final_signal

    def on_bar(self, date, valid_tickers, portfolio):
        if date not in self.signal_matrix.index:
            return []
            
        raw_signal = self.signal_matrix.loc[date].reindex(valid_tickers).fillna(0.0)
        active_signal = raw_signal[raw_signal != 0.0]
        
        if active_signal.empty:
            # 시그널 없으면 청산
            orders = []
            for ticker, qty in portfolio.holdings.items():
                if qty != 0:
                    orders.append({'ticker': ticker, 'action': 'SELL', 'quantity': abs(qty)})
            return orders

        total_abs = active_signal.abs().sum()
        if total_abs == 0: return []
        
        target_weights = active_signal / total_abs
        
        orders = []
        current_equity = portfolio.equity()
        
        for ticker in list(portfolio.holdings.keys()):
            if ticker not in target_weights.index:
                qty = portfolio.holdings[ticker]
                if qty > 0:
                    orders.append({'ticker': ticker, 'action': 'SELL', 'quantity': qty})
                elif qty < 0:
                    orders.append({'ticker': ticker, 'action': 'BUY', 'quantity': abs(qty)})

        for ticker, weight in target_weights.items():
            price = self.get_price(date, ticker)
            if price <= 0: continue
            
            target_val = current_equity * weight
            target_qty = int(target_val / price)
            
            current_qty = portfolio.holdings.get(ticker, 0)
            diff = target_qty - current_qty
            
            if diff > 0:
                orders.append({'ticker': ticker, 'action': 'BUY', 'quantity': abs(diff)})
            elif diff < 0:
                orders.append({'ticker': ticker, 'action': 'SELL', 'quantity': abs(diff)})
                
        return orders