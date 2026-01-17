# src/backtest/strategies/formula.py

import pandas as pd
import numpy as np
from .base import Strategy
from ...alpha.parser import AlphaParser
import gc

class FormulaStrategy(Strategy):
    def __init__(self, expressions, top_n=20, **kwargs):
        super().__init__(**kwargs)
        self.expressions = expressions if isinstance(expressions, list) else [expressions]
        self.top_n = top_n
        self.parser = AlphaParser()
        self.signal = None

    def initialize(self, market_data):
        """
        알파 시그널을 미리 계산(Vectorized)하고, 
        **여기서 자동으로 동적 유니버스 필터링을 수행합니다.**
        """
        self.md = market_data
        
        # 1. 데이터 컨텍스트 생성 (Prices + Features)
        # ----------------------------------------------------
        data_context = {}
        
        # (1) 가격 데이터
        for col, df in self.md.prices.items():
            # 대소문자 호환성 (Close -> close)
            data_context[col] = df
            data_context[col.lower()] = df
            
        # (2) 피처 데이터
        for col, df in self.md.features.items():
            data_context[col] = df
            data_context[col.lower()] = df # 대소문자 무시 지원

        # 2. 수식 계산 (Alpha Calculation)
        # ----------------------------------------------------
        print(f"🧪 Calculating {len(self.expressions)} alpha expressions...")
        
        # 최종 시그널 초기화 (모든 값 0.0)
        # shape: (Ticker x Date) or (Date x Ticker) -> ops.py는 (Ticker x Date)를 뱉음
        # 로더가 Transpose를 했으므로, 여기서도 맞춤
        combined_signal = None 

        for i, expr in enumerate(self.expressions):
            try:
                # 파서로 계산
                raw_alpha = self.parser.parse(expr, data_context)
                
                # 차원 확인 및 초기화
                if combined_signal is None:
                    combined_signal = pd.DataFrame(0.0, index=raw_alpha.index, columns=raw_alpha.columns)
                
                # 합산 (정규화 후 합산하는 것이 좋지만, 여기선 단순 합산)
                combined_signal = combined_signal.add(raw_alpha, fill_value=0)
                
                # 메모리 정리
                del raw_alpha
                gc.collect()
                
            except Exception as e:
                print(f"  🚨 Error parsing '{expr}': {e}")
                raise e

        # 3. [핵심] 자동 동적 유니버스 필터링 (Auto-Masking)
        # ----------------------------------------------------
        # 사용자가 YAML에 '* universe'를 안 적어도, 여기서 강제로 적용!
        if 'universe' in self.md.prices:
            print("  🌌 Applying Dynamic Universe Mask (Auto-Filter)...")
            
            # 로더가 만든 universe 마스크 (1.0 or NaN)
            universe_mask = self.md.prices['universe']
            
            # 마스크와 시그널의 모양(Shape)을 강제로 맞춤 (Broadcast Error 방지)
            # reindex로 인덱스/컬럼 순서를 정렬
            aligned_mask = universe_mask.reindex_like(combined_signal)
            
            # 곱하기 연산 (유니버스 밖인 종목은 NaN이 됨)
            self.signal = combined_signal * aligned_mask
            
        else:
            print("  ⚠️ No universe mask found. Using raw signal (Static Universe).")
            self.signal = combined_signal

        # 4. 최종 정리
        # NaN(유니버스 밖 or 데이터 부족)을 -무한대로 보내서 랭킹 꼴찌로 만듦
        # (단, 숏 전략일 경우 처리가 다르지만 기본은 롱 온리 가정)
        self.signal = self.signal.fillna(-np.inf)
        
        print("  ✅ Signal Calculation Complete.")
        del combined_signal, data_context
        gc.collect()

    def on_bar(self, date, valid_tickers, portfolio):
        # ... (기존 매매 로직 유지) ...
        # self.signal에서 해당 날짜(date)의 값을 조회해서 매매
        
        # ops.py 결과는 (Ticker x Date)일 가능성이 높음.
        # 날짜가 컬럼인지 인덱스인지 확인 필요
        try:
            # Case 1: Index가 날짜인 경우
            daily_signal = self.signal.loc[date]
        except:
            # Case 2: Columns가 날짜인 경우 (Transpose된 상태)
            if date in self.signal.columns:
                daily_signal = self.signal[date]
            else:
                return []

        # 상위 N개 선정 (값이 큰 순서)
        # -inf는 자연스럽게 탈락함
        top_picks = daily_signal.nlargest(self.top_n)
        
        # ... (이하 주문 생성 로직) ...
        orders = []
        # (기존 코드의 주문 생성 부분 복사)
        target_weight = 1.0 / self.top_n
        for ticker, score in top_picks.items():
            if score == -np.inf: continue # 유니버스 밖 종목 스킵
            
            # ... 주문 로직 ...
            # (여기서는 생략, 기존 코드 사용)
            price = self.get_price(date, ticker)
            if pd.isna(price) or price <= 0: continue
            
            target_val = portfolio.equity() * target_weight
            current_qty = portfolio.holdings.get(ticker, 0)
            target_qty = int(target_val / price)
            
            diff = target_qty - current_qty
            if diff != 0:
                action = 'BUY' if diff > 0 else 'SELL'
                orders.append({'ticker': ticker, 'action': action, 'quantity': abs(diff)})
                
        return orders