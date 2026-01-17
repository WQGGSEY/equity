# src/alpha/parser.py
import pandas as pd
from src.alpha import ops
import re

class AlphaParser:
    def __init__(self):
        # ops.py에 있는 모든 함수를 가져와서 namespace로 만듦
        self.context = {name: getattr(ops, name) for name in dir(ops) if not name.startswith("_")}
    
    def parse(self, expression: str, data_dict: dict) -> pd.DataFrame:
        """
        expression: "rank(ts_mean(close, 20))" 같은 문자열
        data_dict: {'close': close_df, 'volume': volume_df, ...}
        """
        # 1. 데이터(변수)를 컨텍스트에 추가
        local_ctx = self.context.copy()
        local_ctx.update(data_dict)
        
        try:
            # 2. 문자열 수식 실행 (Vectorized Evaluation)
            # eval("rank(close)", globals=local_ctx)
            result = eval(expression, {"__builtins__": {}}, local_ctx)
            return result
        except Exception as e:
            raise ValueError(f"수식 파싱 오류: '{expression}' -> {e}")
    
    def extract_needed_features(self, expressions: list) -> list:
        """
        수식 리스트를 분석하여 로딩이 필요한 피처 파일명 리스트를 반환합니다.
        예: ["rank(ts2vec_manifold_0)"] -> ["ts2vec_manifold_0"]
        """
        needed = set()
        
        # 변수명 추출 정규식 (문자로 시작, 숫자/언더바 포함 가능)
        token_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')

        for expr in expressions:
            # 1. 수식에서 단어들 추출
            tokens = token_pattern.findall(expr)
            
            for token in tokens:
                # 소문자로 변환하여 비교 (파일명 매칭 유연성)
                token_lower = token.lower()
                
                # 2. 예약어(함수명)나 기본 가격 데이터(OHLCV)는 제외
                if token_lower in self.reserved_keywords:
                    continue
                
                # 3. 숫자로만 된 것은 제외 (이미 정규식에서 걸러지긴 함)
                if token.isdigit():
                    continue

                # 4. 살아남은 것은 외부 피처일 확률이 높음!
                needed.add(token) # 원본 대소문자 유지 (파일명 매칭 위해)

        # 리스트로 변환하여 반환
        return list(needed)