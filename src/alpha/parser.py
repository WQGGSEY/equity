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
        """
        local_ctx = self.context.copy()
        local_ctx.update(data_dict)
        
        # 입력 방어 로직
        if not isinstance(expression, str):
            raise ValueError(f"수식은 문자열이어야 합니다. 입력값: {expression} ({type(expression)})")

        statements = [s.strip() for s in expression.split(';') if s.strip()]
        
        if not statements:
            raise ValueError("빈 수식입니다.")

        try:
            # 마지막 문장을 제외한 앞부분은 '실행(exec)' -> 변수 정의용
            for stmt in statements[:-1]:
                exec(stmt, {"__builtins__": {}}, local_ctx)
                
            # 마지막 문장은 '평가(eval)' -> 결과 반환용
            final_expr = statements[-1]
            return eval(final_expr, {"__builtins__": {}}, local_ctx)
            
        except Exception as e:
            raise ValueError(f"수식 실행 오류:\n수식: {expression}\n원인: {e}")

    def extract_needed_features(self, expressions: list) -> list:
        """
        수식에서 필요한 피처(변수)를 추출합니다.
        [Smart Filtering]
        1. 예약어(함수명) 제외
        2. 숫자 제외
        3. 수식 내부에서 정의된 변수(x=...) 제외
        """
        needed = set()
        
        # 변수명 추출용 Regex
        token_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')
        # 대입문 감지용 Regex (예: "x =", "ret=")
        assign_pattern = re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=')

        if not expressions:
            return []
            
        for expr in expressions:
            if not isinstance(expr, str): continue
            
            # 세미콜론으로 문장 분리
            statements = [s.strip() for s in expr.split(';') if s.strip()]
            
            # 이 수식 안에서 정의된 로컬 변수들을 추적하는 집합
            local_vars = set()

            for stmt in statements:
                # 1. 이 문장에 등장하는 모든 토큰 추출
                tokens = token_pattern.findall(stmt)
                
                # 2. 이 문장이 변수 선언인지 확인 (LHS 추출)
                match = assign_pattern.match(stmt)
                defined_var = None
                if match:
                    defined_var = match.group(1) # "x = ..." 에서 "x" 추출
                
                # 3. 토큰 필터링
                for token in tokens:
                    # (A) 예약어이거나 숫자인가?
                    if token.lower() in self.reserved_keywords or token.isdigit():
                        continue
                    
                    # (B) 방금 정의된 변수(좌변)인가? -> 피처 아님
                    if token == defined_var:
                        continue
                        
                    # (C) 앞선 문장에서 이미 정의된 로컬 변수인가? -> 피처 아님
                    if token in local_vars:
                        continue
                    
                    # 위 조건을 모두 통과하면 외부 피처로 간주
                    needed.add(token)
                
                # 4. 정의된 변수를 로컬 목록에 추가 (다음 문장부터는 피처로 오인하지 않도록)
                if defined_var:
                    local_vars.add(defined_var)
                
        return sorted(list(needed))