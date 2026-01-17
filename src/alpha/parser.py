import pandas as pd
import re
from src.alpha import ops

class AlphaParser:
    def __init__(self):
        # 1. ops.py 함수들 로드
        self.context = {name: getattr(ops, name) for name in dir(ops) if not name.startswith("_")}
        
        # 2. 예약어 (함수명 등) - 피처 로딩 시 제외용
        self.reserved_keywords = set(self.context.keys())

    def parse(self, expression: str, data_dict: dict) -> pd.DataFrame:
        """
        [Multi-Statement Support]
        세미콜론(;)으로 구분된 문장을 순차적으로 실행합니다.
        마지막 문장의 결과값을 반환합니다.
        
        예: "X = FD_Close * 2; ts_rank(X, 10)"
        """
        local_ctx = self.context.copy()
        local_ctx.update(data_dict)
        
        # 1. 세미콜론으로 문장 분리
        # (문자열 안의 세미콜론은 무시하는 등 복잡한 로직이 필요할 수 있으나, 여기선 단순 split 사용)
        statements = [s.strip() for s in expression.split(';') if s.strip()]
        
        if not statements:
            raise ValueError("빈 수식입니다.")

        try:
            # 2. 마지막 문장을 제외한 앞부분은 '실행(exec)' -> 변수 정의용
            for stmt in statements[:-1]:
                exec(stmt, {"__builtins__": {}}, local_ctx)
                
            # 3. 마지막 문장은 '평가(eval)' -> 결과 반환용
            final_expr = statements[-1]
            return eval(final_expr, {"__builtins__": {}}, local_ctx)
            
        except Exception as e:
            # 디버깅을 위해 어떤 부분에서 에러가 났는지 표시
            raise ValueError(f"수식 실행 오류:\n수식: {expression}\n원인: {e}")

    def extract_needed_features(self, expressions: list) -> list:
        """
        수식에서 필요한 피처(변수)를 추출합니다.
        할당문 왼쪽의 변수(새로 정의된 변수)는 피처로 로딩하면 안 되지만,
        로더가 '없는 파일'은 무시하므로 여기서는 단순하게 추출해도 괜찮습니다.
        """
        needed = set()
        token_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')

        for expr in expressions:
            # 단순화를 위해 전체에서 단어 추출
            tokens = token_pattern.findall(expr)
            for token in tokens:
                token_lower = token.lower()
                if token_lower in self.reserved_keywords:
                    continue
                if token.isdigit():
                    continue
                needed.add(token)
                
        return list(needed)