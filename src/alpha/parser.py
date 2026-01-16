# src/alpha/parser.py
import pandas as pd
from src.alpha import ops

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