import pandas as pd
df = pd.read_parquet("data/platinum/features/AAPL.parquet") # 또는 bronze 경로

# 2020년 8월 31일 (4:1 분할일) 가격 확인
print(df.columns)