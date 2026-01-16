import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가 (모듈 import용)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 기존 모듈 Import
from src.config import PLATINUM_FEATURES_DIR
from src.backtest.loader import MarketData
from src.backtest.engine import BacktestEngine
from src.backtest.strategies.formula import FormulaStrategy
from src.alpha import ops # 도움말 표시용

# ----------------------------------------------------------------
# 1. 페이지 설정
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Alpha Studio",
    page_icon="📈",
    layout="wide"
)

st.title("🧪 Alpha Research Studio")
st.markdown("나만의 퀀트 전략을 수식으로 설계하고 즉시 검증하세요.")

# ----------------------------------------------------------------
# 2. 데이터 로드 (캐싱 이용)
# ----------------------------------------------------------------
@st.cache_resource
def load_market_data():
    """데이터는 한 번만 로드하고 메모리에 캐싱"""
    with st.spinner("💾 Loading Market Data (Platinum)..."):
        md = MarketData(PLATINUM_FEATURES_DIR)
        md.load_all()
    return md

try:
    md = load_market_data()
    st.success(f"데이터 로드 완료: {len(md.tickers)} 종목, {len(md.dates)} 거래일", icon="✅")
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

# ----------------------------------------------------------------
# 3. 사이드바 (설정)
# ----------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Backtest Settings")
    
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))
    
    initial_cash = st.number_input("Initial Cash ($)", value=100_000, step=10_000)
    top_n = st.slider("Top N Stocks", min_value=5, max_value=200, value=20)
    
    fee_rate = st.number_input("Fee Rate (%)", value=0.1, step=0.01) / 100
    slippage = st.number_input("Slippage (%)", value=0.1, step=0.01) / 100

    st.markdown("---")
    st.markdown("### 📚 Available Operators")
    
    # ops.py에 있는 함수 목록 보여주기
    op_list = [f for f in dir(ops) if not f.startswith("_")]
    st.code(", ".join(op_list), language="python")

# ----------------------------------------------------------------
# 4. 메인: 수식 입력기
# ----------------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    default_expr = "rank(ts_mean(close, 20) - close) + 0.5 * rank(volume)"
    expression = st.text_area(
        "Alpha Expression (Python Syntax)", 
        value=default_expr,
        height=100,
        help="사용 가능한 변수: close, open, high, low, volume, rsi_14 등"
    )

with col2:
    st.write("") # 여백
    st.write("") 
    run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

# ----------------------------------------------------------------
# 5. 실행 및 결과 표시
# ----------------------------------------------------------------
if run_btn:
    # (1) 엔진 설정
    engine = BacktestEngine(
        md, 
        start_date=start_date, 
        end_date=end_date, 
        fee_rate=fee_rate + slippage # 비용 합산
    )
    
    # (2) 전략 설정
    strategy = FormulaStrategy(
        expressions=[expression], # 리스트로 전달
        top_n=top_n
    )
    
    # (3) 실행
    try:
        with st.spinner("🔄 Simulating Strategy..."):
            result_df = engine.run(strategy, initial_cash=initial_cash)
            
        # (4) 메트릭 계산
        final_equity = result_df['equity'].iloc[-1]
        total_ret = (final_equity / initial_cash - 1) * 100
        cagr = ((final_equity / initial_cash) ** (365 / len(result_df)) - 1) * 100
        mdd = ((result_df['equity'] - result_df['equity'].cummax()) / result_df['equity'].cummax()).min() * 100
        
        # (5) 결과 대시보드
        st.divider()
        st.subheader("📊 Performance Summary")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Return", f"{total_ret:.2f}%", delta_color="normal")
        m2.metric("CAGR", f"{cagr:.2f}%")
        m3.metric("MDD", f"{mdd:.2f}%", delta_color="inverse")
        m4.metric("Final Equity", f"${final_equity:,.0f}")
        
        # (6) 차트 (Plotly)
        tab1, tab2, tab3 = st.tabs(["📈 Equity Curve", "💧 Drawdown", "📝 Trade Log"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result_df.index, y=result_df['equity'], mode='lines', name='Strategy'))
            fig.update_layout(title="Cumulative Wealth", xaxis_title="Date", yaxis_title="Equity ($)")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            dd = (result_df['equity'] - result_df['equity'].cummax()) / result_df['equity'].cummax()
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd.index, y=dd, fill='tozeroy', mode='lines', line=dict(color='red')))
            fig_dd.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown (%)")
            st.plotly_chart(fig_dd, use_container_width=True)
            
        with tab3:
            st.dataframe(result_df.tail(100)) # 최근 100일 로그

    except Exception as e:
        st.error(f"백테스트 중 오류 발생: {e}")
        st.exception(e) # 상세 에러 로그 표시