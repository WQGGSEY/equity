# (예시) train_model.py

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from tqdm import tqdm, trange

load_dotenv()
DB_DIR = os.environ.get("DIR_PATH") 
FEATURE_PATH = f"{DB_DIR}/indicators.parquet"

def load_feature_data(path=FEATURE_PATH):
    print("Loading feature map from Parquet...")
    # 1400만 행 데이터도 몇 초 안에 로드됩니다.
    df = pd.read_parquet(path)
    return df

def log_return_df_to_pivot(indicator_df: pd.DataFrame):
    log_return_df_long = indicator_df[['ticker', 'date', 'log_return']]
    log_return_df_pivot = log_return_df_long.pivot(index='date', columns='ticker', values='log_return')
    return log_return_df_pivot

def create_dynamic_networks(log_return_df_pivot: pd.DataFrame, window_size=60, threshold=0.7):
    np_log_returns = log_return_df_pivot.to_numpy()
    tickers = log_return_df_pivot.columns.tolist()
    ticker_to_idx = {ticker: idx for idx, ticker in enumerate(tickers)}
    idx_to_ticker = {idx: ticker for idx, ticker in enumerate(tickers)}

    valid_dates = log_return_df_pivot.index[window_size - 1:]
    all_edges = []

    for i in trange(len(valid_dates)):
        current_date = valid_dates[i]
        end_idx = i + window_size
        window_data = np_log_returns[i:end_idx]
        corr_matrix = np.corrcoef(window_data, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        corr_matrix[np.abs(corr_matrix) < threshold] = 0
        adj_upper = np.triu(corr_matrix, k=1)
        src_nodes, tgt_nodes = np.where(adj_upper != 0)
        if src_nodes.size > 0:
            # 엣지 가중치(상관관계 값) 추출
            weights = adj_upper[src_nodes, tgt_nodes]
            
            # 이 날짜의 모든 엣지를 DataFrame으로 저장
            date_edges_df = pd.DataFrame({
                'date': current_date,
                'src_idx': src_nodes,
                'tgt_idx': tgt_nodes,
                'weight': weights
            })
            all_edges.append(date_edges_df)
    final_edges_df = pd.concat(all_edges, ignore_index=True)
    final_edges_df['src_ticker'] = final_edges_df['src_idx'].map(idx_to_ticker)
    final_edges_df['tgt_ticker'] = final_edges_df['tgt_idx'].map(idx_to_ticker)
    
    return final_edges_df

if __name__ == "__main__":
    feature_df = load_feature_data()
    log_return_pivot = log_return_df_to_pivot(feature_df)
    dynamic_networks_df = create_dynamic_networks(log_return_pivot, window_size=60, threshold=0.7)
    output_path = f"{DB_DIR}/dynamic_networks.parquet"
    dynamic_networks_df.to_parquet(output_path)
    print(f"Dynamic networks saved to {output_path}")
    





