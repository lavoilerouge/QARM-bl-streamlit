# QUANTITATIVE ASSETS & RISK MANAGEMENT
# Refactored Streamlit App
# Original Author: Valentin Baur
# Refactored for modularity and scalability

import calendar
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

# 1. PAGE CONFIGURATION (Must be first Streamlit command)
st.set_page_config(
    page_title="QARM – Portfolio & Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import local backend modules
# Ensuring these are available globally
try:
    # Try src.optimization first (when running from QARM root via run.py)
    from src.optimization import black_litterman_weights_for_date, sentiment_preprocess, compute_alpha_dynamic
except ImportError:
    try:
        # Try direct import (when running from src directory)
        from optimization import black_litterman_weights_for_date, sentiment_preprocess, compute_alpha_dynamic
    except ImportError as e:
        # Handle case where module not found
        black_litterman_weights_for_date = lambda **kwargs: (None, 0, [])
        alpha = 0.00114  # Fallback constant
        sentiment_preprocess = lambda *args, **kwargs: pd.DataFrame()
        st.warning(
            f"⚠️ Could not import BL optimization functions: {e}. Using default constants.")

# ==========================================
# 2. GLOBAL CONFIGURATION & CONSTANTS
# ==========================================

GLOBAL_BL_CONSTANTS = {
    'ALPHA': 0.00114,
    'LAMBDA_DEFAULT': 3.81,
    'MAX_LONG': 0.25,
    'MAX_SHORT': 0.10,
    'LEVERAGE': 1.2,
}


def init_session_state():
    """Initialize session state variables."""
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "accredited_answer" not in st.session_state:
        st.session_state.accredited_answer = None
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = ['ASTS', 'RKLB', 'MSFT', 'GOOGL', 'TSLA']


def apply_custom_styles():
    """Centralized CSS styling (Dark Luxury Theme)."""
    st.markdown("""
        <style>
        /* --- Fonts & Global Variables --- */
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Lato:wght@300;400;700&display=swap');

        :root {
            --bg-color: #0a0a0a;
            --card-bg: #121212;
            --text-primary: #ffffff; /* Pure White for primary text */
            --text-secondary: #d4d4d4; /* Light Gray for secondary text (high contrast) */
            --accent-gold: #d4af37; /* Brighter Gold */
            --border-color: #333333;
            --font-heading: 'Playfair Display', serif;
            --font-body: 'Lato', sans-serif;
        }

        /* --- Global App Styles --- */
        .stApp {
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: var(--font-body);
        }

        /* Override Streamlit defaults for dark mode consistency */
        h1, h2, h3, h4, h5, h6 {
            font-family: var(--font-heading) !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }

        p, li, div, span {
            font-family: var(--font-body);
            color: var(--text-primary);
        }

        /* --- Centered Layout Container --- */
        .main-container {
            max_width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
        }

        /* --- Navbar Styles --- */
        .topnav {
            background-color: rgba(18, 18, 18, 0.95); /* Slightly more opaque */
            backdrop-filter: blur(10px);
            padding: 15px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: center;
            gap: 60px;
            position: sticky;
            top: 0;
            z-index: 1000;
            margin-bottom: 40px;
        }

        /* Streamlit button overrides for Navbar to look like text links */
        div[data-testid="stHorizontalBlock"] button {
            background: transparent !important;
            border: none !important;
            color: var(--text-secondary) !important;
            font-family: var(--font-body) !important;
            font-size: 15px !important; /* Slightly larger */
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            transition: color 0.3s ease !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        div[data-testid="stHorizontalBlock"] button:hover {
            color: var(--accent-gold) !important;
            text-shadow: 0 0 8px rgba(212, 175, 55, 0.4);
        }

        /* --- Hero Section --- */
        .hero {
            text-align: center;
            padding: 80px 20px;
            background: linear-gradient(180deg, rgba(20,20,20,0.4) 0%, rgba(20,20,20,1) 100%);
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 60px;
        }
        .hero-badge {
            display: inline-block;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--accent-gold);
            margin-bottom: 20px;
            border: 1px solid var(--accent-gold);
            padding: 8px 16px;
            border-radius: 2px;
            background-color: rgba(212, 175, 55, 0.1); /* Subtle gold background */
        }
        .hero-title {
            font-size: 52px !important;
            margin-bottom: 24px;
            line-height: 1.2;
            color: #ffffff !important;
            text-shadow: 0 2px 10px rgba(0,0,0,0.5);
        }
        .hero-subtitle {
            font-size: 20px;
            color: #e0e0e0 !important; /* Much lighter gray */
            max_width: 700px;
            margin: 0 auto;
            line-height: 1.6;
            font-weight: 300;
        }

        /* --- Cards (Info & Model) --- */
        .info-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 30px;
            height: 100%;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border-radius: 4px;
        }
        .info-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.8);
            border-color: var(--accent-gold);
        }
        .info-card h4 {
            margin: 0 0 15px 0;
            font-size: 18px;
            color: var(--accent-gold) !important; /* Gold headers for contrast */
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
            letter-spacing: 0.5px;
        }
        .info-card p {
            font-size: 15px;
            color: #eeeeee; /* Nearly white */
            line-height: 1.6;
        }
        .info-card ul {
            padding-left: 20px;
            margin-top: 15px;
            font-size: 14px;
            color: #cccccc; /* Light gray list items */
        }
        .info-card li {
            margin-bottom: 8px;
        }

        .model-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 40px;
            height: 100%;
            display: flex;
            flex-direction: column;
            gap: 20px;
            position: relative;
            overflow: hidden;
            border-radius: 4px;
        }
        .model-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: var(--accent-gold);
            transform: scaleX(0);
            transition: transform 0.4s ease;
        }
        .model-card:hover::before {
            transform: scaleX(1);
        }

        .model-header {
            display: flex;
            flex-direction: column;
            gap: 15px;
            align-items: center;
            text-align: center;
            margin-bottom: 10px;
        }
        .model-icon {
            font-size: 32px;
            color: var(--accent-gold);
            margin-bottom: 10px;
            text-shadow: 0 0 15px rgba(212, 175, 55, 0.3);
        }
        .model-title {
            font-size: 24px !important;
            letter-spacing: 0.5px;
            color: #ffffff !important;
        }
        .model-subtitle {
            font-size: 15px;
            color: #d0d0d0; /* Lighter gray */
            font-style: italic;
        }
        .model-list {
            font-size: 14px;
            color: #dddddd; /* Lighter gray */
            padding-left: 18px;
            margin: 4px 0 0 0;
        }

        /* --- Pills & Tags --- */
        .pill-tag {
            position: absolute;
            top: 20px;
            right: 20px;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--accent-gold);
            border: 1px solid var(--accent-gold);
            padding: 4px 8px;
            background: rgba(212, 175, 55, 0.05);
        }

        /* --- Headers --- */
        .section-title {
            font-size: 32px !important;
            text-align: center;
            margin-top: 60px;
            margin-bottom: 15px;
            color: var(--accent-gold) !important;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .section-sub {
            font-size: 16px;
            color: #cccccc;
            text-align: center;
            margin-bottom: 50px;
            font-family: var(--font-heading);
            font-style: italic;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }

        /* --- Button Styling (Primary Action) --- */
        .stButton button[kind="primary"] {
            background-color: transparent !important;
            border: 1px solid var(--accent-gold) !important;
            color: var(--accent-gold) !important;
            border-radius: 2px !important;
            padding: 14px 28px !important;
            font-family: var(--font-body) !important;
            font-weight: 600 !important;
            letter-spacing: 2px !important;
            text-transform: uppercase !important;
            transition: all 0.3s ease !important;
            margin-top: 15px !important;
            width: 100% !important;
        }
        .stButton button[kind="primary"]:hover {
            background-color: var(--accent-gold) !important;
            color: #000000 !important;
            box-shadow: 0 0 20px rgba(197, 160, 89, 0.4) !important;
            font-weight: 700 !important;
        }

        /* --- Sidebar Styling --- */
        section[data-testid="stSidebar"] {
            background-color: #0f0f0f;
            border-right: 1px solid var(--border-color);
        }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            font-family: var(--font-heading) !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }
        section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] p {
            color: #cccccc !important;
        }
        /* Sidebar Input Labels */
        .stSlider label, .stSelectbox label, .stNumberInput label, .stDateInput label {
            color: var(--accent-gold) !important;
            font-family: var(--font-heading) !important;
            font-size: 14px !important;
            letter-spacing: 0.5px;
        }

        /* --- Metric Cards --- */
        div[data-testid="metric-container"] {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 20px;
            border-radius: 2px;
            text-align: center;
            box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
        }
        div[data-testid="metric-container"] label {
            font-family: var(--font-body);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 12px;
            color: #888888 !important;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            font-family: var(--font-heading);
            color: var(--accent-gold) !important;
            font-size: 32px;
            text-shadow: 0 0 10px rgba(197, 160, 89, 0.2);
        }

        /* --- Dataframe Styling --- */
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border-color);
        }
        
        /* --- Footer --- */
        .footer {
            font-size: 12px;
            color: #666;
            text-align: center;
            margin-top: 100px;
            padding-top: 30px;
            border-top: 1px solid #222;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
                
        
        /* ------------------------------ */
        /* FIX TOOLTIP TEXT VISIBILITY   */
        /* ------------------------------ */

        /* Tooltip box background (Streamlit help text) */
        div[data-testid="stTooltipContent"] {
            background-color: rgba(30, 30, 30, 0.95) !important; /* dark background */
            color: #ffffff !important; /* force white text */
            border: 1px solid var(--accent-gold) !important; /* optional luxury border */
            font-size: 14px !important;
            padding: 12px !important;
        }

        /* Tooltip arrow fix */
        div[data-testid="stTooltipArrow"] {
            border-bottom-color: rgba(30, 30, 30, 0.95) !important;
        }

        /* Tooltip help icon “?” */
        svg[data-testid="stTooltipIcon"] {
            color: var(--accent-gold) !important; /* gold question mark */
        }
        /* --- FIX: Make dropdown + calendar background dark so white text is visible --- */

        /* Selectbox closed field */
        div[data-baseweb="select"] > div {
            background-color: #121212 !important;
            color: white !important;
        }

        /* Dropdown list */
        ul[role="listbox"] {
            background-color: #121212 !important;
        }

        /* Options inside dropdown */
        li[role="option"] {
            background-color: #121212 !important;
            color: white !important;
        }

        /* On hover */
        li[role="option"]:hover {
            background-color: rgba(212,175,55,0.25) !important;
            color: #d4af37 !important;
        }

        /* Datepicker menu */
        div[data-baseweb="calendar"] {
            background-color: #121212 !important;
        }

        /* Datepicker text */
        div[data-baseweb="calendar"] * {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)


def switch_page(page):
    """Navigation helper."""
    st.session_state.page = page
    st.rerun()


def navbar():
    """Render the top navigation bar."""
    # Container to center the navbar content
    with st.container():
        cols = st.columns([1, 4, 1])  # Centering
        with cols[1]:
            # Using st.columns for layout
            colA, colB, colC = st.columns(3)

            # Using empty containers with buttons to simulate links
            with colA:
                if st.button("Home", key="nav_home", use_container_width=True): switch_page("home")
            with colB:
                if st.button("Optimizer", key="nav_opt", use_container_width=True): switch_page("optimizer")
            with colC:
                if st.button("Black-Litterman", key="nav_bl", use_container_width=True): switch_page("blacklitterman")
    st.markdown("---")  # Subtle separator


# ==========================================
# 3. DATA LOADING & CACHING
# ==========================================

@st.cache_data
def load_bl_data():
    """Load data specifically for the Black-Litterman page."""
    try:
        # Use absolute paths relative to this file's location to work regardless of cwd
        _app_dir = Path(__file__).parent.resolve()
        _project_root = _app_dir.parent
        
        prices_path = _project_root / "data" / "raw" / "prices_cleaned.parquet"
        sentiment_path = _project_root / "data" / "processed" / "sentiment_polarity.parquet"
        
        px_prices = pd.read_parquet(prices_path)
        px_prices = px_prices.sort_index().ffill().bfill()
        rets_all = np.log(px_prices / px_prices.shift(1))

        # Load sentiment_polarity.parquet and transform columns to match optimization code expectations
        sentiment = pd.read_parquet(sentiment_path)
        sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.tz_localize(None)
        
        # Rename msg_count to n_msgs (expected by optimization code)
        sentiment = sentiment.rename(columns={"msg_count": "n_msgs"})
        
        # Compute sentiment_z (z-score of polarity) using rolling statistics
        sentiment = sentiment.sort_values(["ticker", "date"])
        sentiment["polarity_mean"] = sentiment.groupby("ticker")["polarity"].transform(
            lambda x: x.rolling(60, min_periods=20).mean()
        )
        sentiment["polarity_std"] = sentiment.groupby("ticker")["polarity"].transform(
            lambda x: x.rolling(60, min_periods=20).std()
        )
        sentiment["sentiment_z"] = (
            sentiment["polarity"] - sentiment["polarity_mean"]
        ) / sentiment["polarity_std"]

        return px_prices, rets_all.dropna(), sentiment
    except FileNotFoundError as e:
        st.error(f"FATAL ERROR: Missing data files for Black-Litterman in the current directory.")
        st.stop()


@st.cache_data
def load_stock_list():
    """Load stock list for the Optimizer page."""
    from config import STOCK_LIST_CSV
    csv_path = STOCK_LIST_CSV
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df['Display'] = df['Symbol'] + ' - ' + df['Security Name']
        return df
    return None


# ==========================================
# 4. CORE MATH & UTILITY FUNCTIONS
# ==========================================

# --- Portfolio Math ---
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return returns, std


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_std


def portfolio_variance(weights, mean_returns, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix * 252, weights))

def compute_metrics(cum_curve):
    """Compute professional portfolio metrics."""
    rets = cum_curve.pct_change().dropna()

    # Annualized return
    ann_return = (1 + rets.mean())**252 - 1

    # Annualized volatility
    ann_vol = rets.std() * np.sqrt(252)

    # Sharpe
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    # Max Drawdown
    cumulative = (1 + rets).cumprod()
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    max_dd = dd.min()

    # Sortino
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = ann_return / downside if downside > 0 else np.nan

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "Annualized Return (%)": float(ann_return * 100),
        "Annualized Volatility (%)": float(ann_vol * 100),
        "Sharpe Ratio": float(sharpe),
        "Max Drawdown (%)": float(max_dd * 100),
        "Sortino Ratio": float(sortino),
        "Calmar Ratio": float(calmar),
    }


# --- Optimizer Utilities ---
def get_market_caps(tickers, rebalance_date, data_source):
    def get_quarter_for_date(stmt, rebalance_date):
        available_quarters = [pd.to_datetime(q) for q in stmt.columns]
        available_quarters = sorted(available_quarters, reverse=True)
        for q in available_quarters:
            if q <= rebalance_date: return q.strftime('%Y-%m-%d')
        return stmt.columns[0]

    market_caps = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if ticker in data_source.columns and rebalance_date in data_source.index:
                price = data_source.loc[rebalance_date, ticker]
            else:
                price = info.get('regularMarketPrice', None)

            market_cap = info.get('marketCap', None)
            # Attempt to get share count
            try:
                stmt = stock.quarterly_income_stmt
                if not stmt.empty:
                    quarter_date = get_quarter_for_date(stmt, rebalance_date)
                    shares_out = stmt.loc['Basic Average Shares', quarter_date]
                    if not pd.isna(shares_out) and price is not None and not pd.isna(price):
                        market_cap = shares_out * price
            except:
                pass

            if market_cap is None:
                shares_out = info.get('sharesOutstanding', None)
                if shares_out is not None and price is not None and not pd.isna(price):
                    market_cap = shares_out * price

            if market_cap: market_caps[ticker] = market_cap
        except:
            pass
    return market_caps


def optimize_portfolio_at_date(data_window, rf_rate, optimization_method, stocks):
    returns_window = data_window.pct_change().dropna()
    mean_returns_window = returns_window.mean()
    cov_matrix_window = returns_window.cov()

    num_assets = len(stocks)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    if optimization_method == "Max Sharpe Ratio":
        result = minimize(neg_sharpe_ratio, initial_guess, args=(mean_returns_window, cov_matrix_window, rf_rate),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x
    elif optimization_method == "Min Variance":
        result = minimize(portfolio_variance, initial_guess, args=(mean_returns_window, cov_matrix_window),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x
    elif optimization_method == "Value-Weighted":
        rebalance_date = data_window.index[-1]
        market_caps = get_market_caps(stocks, rebalance_date, data_window)
        if market_caps:
            total_market_cap = sum(market_caps.values())
            weights = np.array([market_caps.get(stock, 0) / total_market_cap for stock in stocks])
            return weights / weights.sum()
        else:
            return np.array([1 / num_assets] * num_assets)


def generate_rebalance_dates(start_date, end_date, frequency, trading_index, month=None, day=31):
    if not isinstance(start_date, datetime): start_date = datetime.combine(start_date, datetime.min.time())
    if not isinstance(end_date, datetime): end_date = datetime.combine(end_date, datetime.min.time())

    def last_trading_on_or_before(target) -> Optional[pd.Timestamp]:
        pos = trading_index.searchsorted(pd.Timestamp(target), side='right') - 1
        if pos >= 0: return trading_index[pos]
        return None

    def month_end_day(year, mon):
        return calendar.monthrange(year, mon)[1]

    month_map = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8,
                 "September": 9, "October": 10, "November": 11, "December": 12}
    target_dates = []

    if frequency == "Yearly":
        month_num = month_map.get(month, 12)
        for year in range(start_date.year, end_date.year + 1):
            last_day = month_end_day(year, month_num)
            day_to_use = last_day if day == 31 else min(day, last_day)
            target = datetime(year, month_num, day_to_use)
            if start_date <= target <= end_date: target_dates.append(pd.Timestamp(target))
    elif frequency == "Quarterly":
        quarter_months = [3, 6, 9, 12]
        for year in range(start_date.year, end_date.year + 1):
            for m in quarter_months:
                last_day = month_end_day(year, m)
                target = datetime(year, m, last_day)
                if start_date <= target <= end_date: target_dates.append(pd.Timestamp(target))
    elif frequency == "Monthly":
        cur_year, cur_month = start_date.year, start_date.month
        while (cur_year < end_date.year) or (cur_year == end_date.year and cur_month <= end_date.month):
            last_day = month_end_day(cur_year, cur_month)
            day_to_use = last_day if day == 31 else min(day, last_day)
            target = datetime(cur_year, cur_month, day_to_use)
            if start_date <= target <= end_date: target_dates.append(pd.Timestamp(target))
            if cur_month == 12:
                cur_month = 1; cur_year += 1
            else:
                cur_month += 1

    trading_rebal_dates = []
    seen = set()
    for t in sorted(target_dates):
        trd = last_trading_on_or_before(t)
        if trd is not None and trd not in seen:
            trading_rebal_dates.append(trd)
            seen.add(trd)
    return pd.DatetimeIndex(trading_rebal_dates)


# --- BL Utilities ---

def next_month_window(t, price_index):
    t = pd.to_datetime(t)
    future_dates = price_index[price_index > t]
    if future_dates.empty: return None, None
    start = future_dates.min()

    def get_month_end(start_date):
        month = start_date.month
        year = start_date.year
        month_days = price_index[(price_index.month == month) & (price_index.year == year)]
        if month_days.empty: return None
        return month_days.max()

    end = get_month_end(start)
    return start, end


def get_window_prices(px_prices, tickers, start, end):
    px = px_prices.loc[start:end, tickers]
    px = px.ffill().dropna(how="all", axis=1)
    if px.shape[1] < 2: return pd.DataFrame()
    return px


def extract_sentiment_data(t, sentiment_df, rets_all, train_universe, z_threshold, alpha_val):
    if not train_universe: return pd.DataFrame()

    lookback_days = 180
    rets = rets_all.loc[t - pd.Timedelta(days=lookback_days):t, train_universe].dropna(axis=1, how="any")
    if rets.empty: return pd.DataFrame()
    tickers = rets.columns.tolist()

    sent_agg = sentiment_preprocess(sentiment_df, tickers, t, window_sent=30)
    sent_agg['Q_mens_view'] = sent_agg['z_mean'] * alpha_val
    sent_agg['Polarity'] = sent_agg['z_mean'].apply(
        lambda x: 'Bullish' if x > 0.01 else 'Bearish' if x < -0.01 else 'Neutral')

    display_df = sent_agg.reset_index().rename(
        columns={'ticker': 'Ticker', 'z_mean': 'Z_Score_Mean', 'msgs': 'Messages'})
    display_df = display_df[['Ticker', 'Polarity', 'Z_Score_Mean', 'Messages', 'Q_mens_view']]
    display_df['Relevance'] = np.where(display_df['Z_Score_Mean'].abs() >= z_threshold, 'Active View', 'Weak View')
    display_df = display_df.sort_values(['Relevance', 'Z_Score_Mean'], ascending=[False, False])

    return display_df


def compute_oos_rebalanced_notebook_style(rebal_dates, px_prices, sentiment_df, lam, rets_all, alpha_dyn, z_threshold,
                                          train_universe, max_long, leverage_limit):
    """Wrapper function for OOS simulation with dynamic constraints."""
    price_index = px_prices.index
    last_w = None
    weights_oos = {}

    for t in rebal_dates:
        w_t, K_detected, strong_views = black_litterman_weights_for_date(
            t=t, sentiment_df=sentiment_df, lam=lam, px_prices=px_prices,
            rets_all=rets_all, window_sent=30, lookback_days=180, top_n=50,
            max_long=max_long, max_short=GLOBAL_BL_CONSTANTS['MAX_SHORT'], leverage_limit=leverage_limit,
            alpha_dyn=alpha_dyn, z_threshold=z_threshold, fixed_universe=train_universe,
        )
        if (w_t is None) or (len(w_t) == 0):
            weights_oos[t] = last_w.copy() if last_w is not None else None
        else:
            weights_oos[t] = w_t.copy()
            last_w = w_t.copy()

    returns_oos = []
    for t, w_df in weights_oos.items():
        t = pd.Timestamp(t)
        start, end = next_month_window(t, price_index)
        if start is None or end is None: continue
        if w_df is None:
            dates = price_index[(price_index >= start) & (price_index <= end)]
            returns_oos.append(pd.DataFrame({"date": dates, "ret": 0.0}))
            continue

        tickers = w_df["ticker"].tolist()
        available_cols = [tic for tic in tickers if tic in px_prices.columns]
        if not available_cols: continue

        px_win = get_window_prices(px_prices, available_cols, start, end)
        rets_win = np.log(px_win / px_win.shift(1)).dropna()

        common = [tic for tic in tickers if tic in rets_win.columns]
        if len(common) < 2: continue

        R = rets_win[common].values
        w_df_indexed = w_df.set_index("ticker")
        w_sub = w_df_indexed.loc[common]["w_opt"]
        w_sub_sum = w_sub.sum()
        w_vec = (w_sub / w_sub_sum).values if w_sub_sum != 0 else np.zeros(len(common))

        port_rets = R @ w_vec
        returns_oos.append(pd.DataFrame({"date": rets_win.index, "ret": port_rets}))

    if len(returns_oos) == 0: return None

    nav = pd.concat(returns_oos).groupby("date")["ret"].sum()
    nav = np.exp(nav.cumsum())
    nav /= nav.iloc[0]
    return nav


# ==========================================
# 5. PAGES
# ==========================================

def home_page():
    navbar()

    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # --- HERO ---
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">
                QARM Project v2.0
            </div>
            <div class="hero-title">
                Institutional-Grade<br/>Portfolio Analytics
            </div>
            <div class="hero-subtitle">
                Unifying classical mean-variance optimization with Bayesian Black-Litterman views driven by alternative data.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # --- PROJECT OVERVIEW ---
    st.markdown('<div class="section-title">Platform Capabilities</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="info-card">
            <h4>MARKET DATA</h4>
            <p>Robust ingestion of pre-processed S&P 500 data.</p>
            <ul>
                <li>Adjusted Close Prices</li>
                <li>Logarithmic Returns</li>
                <li>Covariance Matrix Estimation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="info-card">
            <h4>QUANTITATIVE RISK</h4>
            <p>Advanced portfolio theory implementation.</p>
            <ul>
                <li>Markowitz Efficient Frontier</li>
                <li>Ledoit-Wolf Shrinkage</li>
                <li>Constrained Optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div class="info-card">
            <h4>SENTIMENT ALPHA</h4>
            <p>NLP-driven signal generation.</p>
            <ul>
                <li>StockTwits Integration</li>
                <li>Z-Score View Construction</li>
                <li>Bayesian Posterior Updates</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- MODEL SELECTION ---
    st.markdown('<div class="section-title">Select Strategy Module</div>', unsafe_allow_html=True)

    col_markowitz, col_bl = st.columns(2)

    # Markowitz Card
    with col_markowitz:
        st.markdown("""
            <div class="model-card">
                <div class="pill-tag">CLASSIC</div>
                <div class="model-header">
                    <div class="model-icon">❖</div>
                    <div>
                        <div class="model-title">Mean-Variance Optimizer</div>
                        <div class="model-subtitle">Traditional risk-return optimization</div>
                    </div>
                </div>
                <ul class="model-list">
                    <li>Min Variance & Max Sharpe</li>
                    <li>Rolling window backtesting</li>
                    <li>Custom universe selection</li>
                </ul>
            """, unsafe_allow_html=True)

        if st.button("Launch Optimizer", key="btn_opt_home", type="primary"): switch_page("optimizer")
        st.markdown("</div>", unsafe_allow_html=True)

    # Black-Litterman Card
    with col_bl:
        st.markdown("""
            <div class="model-card">
                <div class="pill-tag">ADVANCED</div>
                <div class="model-header">
                    <div class="model-icon">◈</div>
                    <div>
                        <div class="model-title">Black-Litterman Sentiment</div>
                        <div class="model-subtitle">Equilibrium + Alternative Data Views</div>
                    </div>
                </div>
                <ul class="model-list">
                    <li>Sentiment Z-Score Extraction</li>
                    <li>View Confidence Weighting</li>
                    <li>Walk-Forward Analysis</li>
                </ul>
            """, unsafe_allow_html=True)

        if st.button("Launch Black-Litterman", key="btn_bl_home", type="primary"): switch_page("blacklitterman")
        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("""<div class="footer">QARM Project · University of Lausanne · 2025</div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #555; font-size: 12px; margin-bottom: 20px;'>Developed by the QARM Team</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: #555; font-size: 12px; margin-bottom: 20px;'>Valentin Baur, Leonardo Cassano, Maxime Cherix, Maxime David, Damien Grosset-Bourbange</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # Close main container


def blacklitterman_page():
    navbar()

    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Black-Litterman Sentiment Cockpit</div>', unsafe_allow_html=True)

    # --- Authentication Check ---
    if "accredited_answer" not in st.session_state:
        st.session_state.accredited_answer = None

    if st.session_state.accredited_answer is None:
        # Styled Gate
        st.markdown("""
            <style>
            .bl-banner { background: rgba(197, 160, 89, 0.2); padding: 20px; border: 1px solid var(--accent-gold); border-radius: 4px; color: var(--accent-gold); text-align: center; font-weight: 400; margin-bottom: 30px; letter-spacing: 1px; font-family: var(--font-heading); text-transform: uppercase;}
            .stRadio > label { color: var(--text-primary) !important; }
            </style>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="bl-banner">Investor Verification Required</div>', unsafe_allow_html=True)
            st.markdown(
                "<div style='text-align: center; color: #a0a0a0; margin-bottom: 20px;'>This cockpit provides access to an equity strategy that applies the <b>Black-Litterman framework</b> to sentiment-based views derived from Stocktwits data. The resulting portfolios may enter <b>short positions</b> and, in certain cases, employ b>portfolio leverage</b>, which can materially increase both return potential and risk.</div>",
                unsafe_allow_html=True)
            st.markdown(
                """
                <div style='text-align: center; color: #a0a0a0; margin-bottom: 20px;'>
                    Under Swiss regulation, investment approaches that make use of leverage, short selling 
                    or complex allocation models are generally reserved for 
                    <b>professional or institutional clients</b> as defined under the Swiss Federal Act on 
                    Financial Services (<b>FinSA</b>) and the Collective Investment Schemes Act (<b>CISA</b>), 
                    together with the applicable ordinances and guidance issued by <b>FINMA</b> in relation 
                    to complex products and risk management.
                    <br><br>
                    By proceeding, you confirm that you understand these risks and that you qualify as an 
                    <b>accredited or professional investor</b> under the laws and regulations that apply to you.
                </div>
                """,
                unsafe_allow_html=True
            )
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                answer = st.radio("Status Verification:",
                                  ["Yes, I am an accredited investor", "No, I am not an accredited investor"],
                                  index=None)
                if st.button("Enter Environment", type="primary"):
                    st.session_state.accredited_answer = answer
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    if st.session_state.accredited_answer == "No, I am not an accredited investor":
        st.error("Access restricted. Please contact administration or use the standard optimizer.")
        if st.button("Go to Standard Optimizer"): switch_page("optimizer")
        st.stop()

    # --- Main BL Application ---
    with st.spinner("Initializing Market & Sentiment Data..."):
        px_prices, rets_all, sentiment_df = load_bl_data()

    if px_prices is None: return

    # Sidebar
    st.sidebar.header("Simulation Parameters")

    start_date_oos = pd.to_datetime("2016-01-01")
    max_date_price = px_prices.index.max()
    available_dates = px_prices.index[
        (px_prices.index >= start_date_oos) & (px_prices.index <= max_date_price)].normalize().unique()

    if len(available_dates) == 0:
        st.error("No valid OOS simulation period.")
        return

    # Date bounds for calendar
    min_date = available_dates.min().date()
    max_date = available_dates.max().date()
    
    # Default date
    target_default = pd.Timestamp("2018-01-02")
    if target_default in available_dates:
        default_date = target_default.date()
    else:
        default_date = max_date

    date_selected = st.sidebar.date_input(
        "Analysis Date",
        value=default_date,
        min_value=min_date,
        max_value=max_date,
        help="Select the historical date to run the single-period optimization and start the walk-forward backtest."
    )

    st.sidebar.divider()
    st.sidebar.markdown("### Risk Profile")

    # --- ADDED TOOLTIPS HERE ---
    lam_help = (
        "The Market Price of Risk (Lambda). Represents the investor's trade-off between risk and return.\n\n"
        "• Higher (e.g., > 4.0): Conservative. Penalizes volatility heavily.\n"
        "• Lower (e.g., < 2.0): Aggressive. Accepts higher volatility for potential returns.\n"
        f"• Default ({GLOBAL_BL_CONSTANTS['LAMBDA_DEFAULT']}): Implied from S&P 500 historical data (2009-2015)."
    )
    lam = st.sidebar.slider(
        "Risk Aversion (Lambda)",
        0.1, 10.0, GLOBAL_BL_CONSTANTS['LAMBDA_DEFAULT'], 0.1,
        help=lam_help
    )

    st.sidebar.markdown("### Signal Conviction")

    z_help = (
        "The minimum statistical significance (Z-Score) required to form a view from social sentiment.\n\n"
        "• A value of 0.5 means we only act on sentiment signals that are 0.5 standard deviations from the mean.\n"
        "• Higher Threshold: Trades only on extreme, high-conviction signals (less noise, fewer bets).\n"
        "• Lower Threshold: Trades on weaker signals (more bets, higher noise risk)."
    )
    z_threshold = st.sidebar.slider(
        "Min Z-Score (View Threshold)",
        0.0, 1.0, 0.5, 0.1,
        help=z_help
    )

    with st.sidebar.expander("Mandate Constraints"):
        long_help = (
            "The maximum weight allowed for any single asset.\n\n"
            "• Prevents excessive concentration in a few names.\n"
            "• Typical institutional constraints range from 5% to 15% to ensure diversification."
        )
        max_long = st.slider(
            "Max Position Size",
            0.05, 0.50, GLOBAL_BL_CONSTANTS['MAX_LONG'], 0.05,
            help=long_help
        )

        lev_help = (
            "The limit on Gross Exposure (Longs + Absolute Shorts).\n\n"
            "• 1.0 = Fully Invested Long Only (no leverage).\n"
            "• 1.2 = 120% Gross Exposure (e.g., 110% Long, 10% Short).\n"
            "• Allows the strategy to take short positions to fund active long bets (130/30 style)."
        )
        leverage = st.slider(
            "Gross Exposure Limit",
            1.0, 2.0, GLOBAL_BL_CONSTANTS['LEVERAGE'], 0.1,
            help=lev_help
        )

    run_button = st.sidebar.button("Run Optimization", type="primary")

    # Dashboard Body
    date_selected_dt = pd.to_datetime(date_selected)
    train_end = date_selected_dt - pd.Timedelta(days=1)
    train_start = pd.Timestamp("2009-07-10")
    rebal_train = pd.date_range(train_start, train_end, freq="BME")
    Z_THRESHOLD_TRAIN = 0.5
    alpha_dyn = compute_alpha_dynamic(
    sentiment_df,
    rets_all,
    train_end)

    st.markdown(
        f"<h3 style='text-align: center; color: #888; font-weight: 300; margin-bottom: 40px;'>Analysis Date: <span style='color: #fff'>{date_selected_dt.strftime('%B %d, %Y')}</span></h3>",
        unsafe_allow_html=True)

    # Background Universe Calculation
    weights_train, last_w_train = {}, None
    with st.spinner("Scanning Investment Universe..."):
        for t in rebal_train:
            w_t, K_detected, strong_views = black_litterman_weights_for_date(
                t=t, sentiment_df=sentiment_df, lam=lam, px_prices=px_prices,
                rets_all=rets_all, alpha_dyn=alpha_dyn, z_threshold=Z_THRESHOLD_TRAIN, fixed_universe=None,
            )
            if w_t is not None:
                weights_train[t] = w_t.copy()
                last_w_train = w_t.copy()
            else:
                weights_train[t] = last_w_train.copy() if last_w_train is not None else None

    train_universe = set()
    for w in weights_train.values():
        if w is not None: train_universe |= set(w["ticker"].tolist())
    train_universe = sorted(train_universe)

    col1, col2, col3 = st.columns(3)
    col1.metric("Tradeable Universe", f"{len(train_universe)} Assets")
    col2.metric("Training Window", f"{(train_end - train_start).days} Days")
    col3.metric("Alpha Sensitivity", f"{alpha_dyn:.5f}")

    st.markdown("---")

    if run_button:
        with st.spinner("Optimizing Portfolio Structure..."):
            w_df, K, strong_list = black_litterman_weights_for_date(
                t=date_selected_dt,
                sentiment_df=sentiment_df, lam=lam, px_prices=px_prices, rets_all=rets_all,
                alpha_dyn=alpha_dyn, z_threshold=z_threshold, window_sent=30, lookback_days=180,
                top_n=50, max_long=max_long, max_short=GLOBAL_BL_CONSTANTS['MAX_SHORT'], leverage_limit=leverage,
                fixed_universe=train_universe,
            )
            if w_df is None:
                st.error("⚠️ Impossible de générer un portefeuille Black-Litterman avec ces paramètres. "
                "Veuillez ajuster le Z-Score, les messages minimums ou les contraintes.")
                return
            if w_df is not None:
                # 1. Sentiment Views
                st.markdown('<div class="section-sub">Sentiment Signal Analysis</div>', unsafe_allow_html=True)

                sentiment_views_df = extract_sentiment_data(
                    t=date_selected_dt, sentiment_df=sentiment_df, rets_all=rets_all,
                    train_universe=train_universe, z_threshold=z_threshold, alpha_val=alpha_dyn
                )

                if not sentiment_views_df.empty:
                    active_views = sentiment_views_df[sentiment_views_df['Relevance'] == 'Active View']

                    col_view1, col_view2 = st.columns([1, 2])

                    with col_view1:
                        st.markdown(f"""
                            <div class="info-card">
                                <h4>Signal Distribution</h4>
                                <div style="font-size: 36px; color: var(--accent-gold); font-weight: 700; margin-bottom: 10px;">{K}</div>
                                <div style="font-size: 14px; color: #aaa;">Active Views Generated</div>
                                <div style="margin-top: 20px; border-top: 1px solid #333; padding-top: 10px;">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                        <span style="color: #4ade80;">Bullish</span>
                                        <span>{len(sentiment_views_df[sentiment_views_df['Polarity'] == 'Bullish'])}</span>
                                    </div>
                                    <div style="display: flex; justify-content: space-between;">
                                        <span style="color: #f87171;">Bearish</span>
                                        <span>{len(sentiment_views_df[sentiment_views_df['Polarity'] == 'Bearish'])}</span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    with col_view2:
                        st.markdown(f"*Active Views (Z-Score > {z_threshold})*")
                        st.dataframe(
                            active_views.style.format({
                                'Z_Score_Mean': '{:.2f}',
                                'Q_mens_view': '{:.2%}',
                                'Messages': '{:,.0f}'
                            }).background_gradient(subset=['Z_Score_Mean'], cmap='RdYlGn', vmin=-2, vmax=2),
                            use_container_width=True,
                            height=300
                        )
                else:
                    st.info("No active sentiment views.")

                st.markdown("---")

                # 2. Allocation
                st.markdown('<div class="section-sub">Optimal Portfolio Allocation</div>', unsafe_allow_html=True)

                col1, col2 = st.columns([2, 1])
                top = w_df.sort_values("w_opt", ascending=False).head(15)

                with col1:
                    # Use a cleaner, darker theme for Plotly
                    fig_weights = go.Figure(go.Bar(
                        x=top['ticker'],
                        y=top['w_opt'],
                        marker_color='#c5a059'  # Gold
                    ))
                    fig_weights.update_layout(
                        title="Top 15 Active Weights",
                        yaxis_tickformat='.1%',
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e0e0e0')
                    )
                    st.plotly_chart(fig_weights, use_container_width=True)
                    

                with col2:
                    st.dataframe(top.set_index("ticker").style.format("{:.2%}"), use_container_width=True, height=400)

                # 3. Backtest
                st.markdown("---")
                st.markdown('<div class="section-sub">Strategy Backtest (Walk-Forward)</div>', unsafe_allow_html=True)

                future_dates = px_prices.index[px_prices.index > date_selected_dt]

                if not future_dates.empty:
                    rebal_dates = future_dates.to_series().groupby(
                        [future_dates.year, future_dates.month]).last().to_list()
                    with st.spinner("Calculating OOS Equity Curve..."):
                        nav_oos_custom = compute_oos_rebalanced_notebook_style(
                            rebal_dates=rebal_dates, px_prices=px_prices, sentiment_df=sentiment_df,
                            lam=lam, rets_all=rets_all, alpha_dyn=alpha_dyn, z_threshold=z_threshold,
                            train_universe=train_universe, max_long=max_long, leverage_limit=leverage
                        )

                    if nav_oos_custom is not None and len(nav_oos_custom) > 0:
                        sp500_data = yf.download("^GSPC", start=date_selected_dt, end="2020-03-31",
                                                 progress=False)
                        if isinstance(sp500_data.columns,
                                      pd.MultiIndex): sp500_data.columns = sp500_data.columns.get_level_values(0)
                        sp500_rets = np.log(sp500_data["Close"] / sp500_data["Close"].shift(1)).dropna()
                        nav_sp500 = np.exp(sp500_rets.cumsum());
                        nav_sp500 /= nav_sp500.iloc[0]

                        common_dates = nav_oos_custom.index.intersection(nav_sp500.index)

                        fig_perf = go.Figure()
                        fig_perf.add_trace(
                            go.Scatter(x=common_dates, y=nav_oos_custom.loc[common_dates], name="BL Strategy",
                                       line=dict(color='#c5a059', width=2)))
                        fig_perf.add_trace(go.Scatter(x=common_dates, y=nav_sp500.loc[common_dates], name="S&P 500",
                                                      line=dict(color='#666', width=1, dash='dot')))
                        fig_perf.update_layout(
                            title="Cumulative Returns",
                            height=500,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#e0e0e0'),
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=True, gridcolor='#333')
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)

                        # Stats
                        total_return_bl = (nav_oos_custom.iloc[-1] / nav_oos_custom.iloc[0] - 1) * 100
                        total_return_sp500 = (nav_sp500.iloc[-1] / nav_sp500.iloc[0] - 1) * 100

                        col_s1, col_s2, col_s3 = st.columns(3)
                        col_s1.metric("Strategy Return", f"{total_return_bl:.2f}%")
                        col_s2.metric("Benchmark Return", f"{total_return_sp500:.2f}%")
                        col_s3.metric("Active Return", f"{total_return_bl - total_return_sp500:.2f}%")

                    else:
                        st.info("Insufficient OOS data.")
                else:
                    st.warning("End of data reached.")

                st.markdown("### Long / Short Breakdown")
                w_sorted = w_df.copy()
                w_sorted["Direction"] = np.where(w_sorted["w_opt"] >= 0, "LONG", "SHORT")
                w_sorted["AbsWeight"] = w_sorted["w_opt"].abs()
                w_sorted = w_sorted.sort_values("w_opt")

                styled_ls = (
                    w_sorted[["ticker","w_opt","Direction"]]
                    .rename(columns={"ticker": "Ticker", "w_opt": "Weight"})
                    .style.format({"Weight": "{:.2%}"})
                    .applymap(lambda v: "color:#4ade80" if v == "LONG" else "color:#f87171", subset=["Direction"])
                    .applymap(lambda v: "color:#f87171" if v < 0 else "color:#4ade80", subset=["Weight"])
                )
                st.dataframe(styled_ls, use_container_width=True, height=350)
                st.markdown("---")
                st.markdown("### Portfolio Performance Metrics")
                if nav_oos_custom is not None:
                    metrics = compute_metrics(nav_oos_custom)
                    metrics_df = (
                        pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
                        .reset_index()
                        .rename(columns={"index": "Metric"})
                    )
                    styled_metrics= (
                        metrics_df.style
                        .set_properties(subset=["Metric"], **{"color": "white"})
                        .set_properties(subset=["Value"], **{"color": "#c5a059"})
                    )

                    st.table(styled_metrics)
                else:
                    st.info("No performance data available.")

            else:
                st.error("Optimization Failed.")

    st.markdown('</div>', unsafe_allow_html=True)  # Close main container

def optimizer_page():
    navbar()

    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Portfolio Optimizer</div>', unsafe_allow_html=True)

    logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
    if os.path.exists(logo_path):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2: st.image(logo_path, use_container_width=True)

    st.divider()
    st.sidebar.markdown("### Stock Selection")
    st.sidebar.caption("Build your portfolio by adding stocks")
    stock_df = load_stock_list()
    if stock_df is not None:
        st.sidebar.markdown("**📋 Current Portfolio:**")
        if st.session_state.selected_tickers:
            st.sidebar.info(f"**{len(st.session_state.selected_tickers)} stocks** selected")
            for ticker in st.session_state.selected_tickers:
                col1, col2 = st.sidebar.columns([4, 1])
                with col1:
                    stock_info = stock_df[stock_df['Symbol'] == ticker]
                    if not stock_info.empty:
                        st.markdown(f"**{ticker}** · {stock_info.iloc[0]['Security Name'][:22]}...")
                    else:
                        st.markdown(f"**{ticker}**")
                with col2:
                    if st.button("➖", key=f"remove_{ticker}", help="Remove stock"):
                        st.session_state.selected_tickers.remove(ticker)
                        st.rerun()
        else:
            st.sidebar.info("No stocks selected.")
        st.sidebar.divider()
        st.sidebar.markdown("**➕ Add Stocks:**")
        search_term = st.sidebar.text_input("Search stocks",placeholder="Type ticker or company name...",label_visibility="collapsed")
        if search_term:
            filtered = stock_df[
                stock_df['Symbol'].str.contains(search_term.upper(), case=False, na=False) |
                stock_df['Security Name'].str.contains(search_term, case=False, na=False)
            ].head(10)
            if filtered.empty:
                st.sidebar.info("No matching stocks found.")
            else:
                for _, row in filtered.iterrows():
                    col1, col2 = st.sidebar.columns([4, 1])
                    with col1:
                        st.markdown(f"**{row['Symbol']}** · {row['Security Name'][:22]}...")
                    with col2:
                        if row['Symbol'] not in st.session_state.selected_tickers:
                            if st.button("➕", key=f"add_{row['Symbol']}"):
                                st.session_state.selected_tickers.append(row['Symbol'])
                                st.rerun()
                        else:
                            st.write("✓")
        stocks= st.session_state.selected_tickers
    else:
        st.sidebar.warning("Stock list file missing. Using fallback input.")
        ticker_input = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL, NEM, NVDA")
        stocks = [t.strip().upper() for t in ticker_input.split(",")]

    st.sidebar.divider()
    start = st.sidebar.date_input("From", datetime.now() - timedelta(days=365 * 2))
    end = st.sidebar.date_input("To", datetime.now())

    method = st.sidebar.radio("Strategy", ["Max Sharpe Ratio", "Min Variance", "Value-Weighted"])
    rebalance = st.sidebar.selectbox("Rebalance", ["Yearly", "Quarterly", "Monthly"])

    if st.sidebar.button("🚀 Optimize", type="primary"):
        with st.spinner("Fetching Data..."):
            data = yf.download(stocks, start=start, end=end, auto_adjust=False)['Adj Close']

        if data.empty: st.error("No data."); return

        # Run Optimization (Simplified view for brevity, full logic in functions above)
        returns = data.pct_change().dropna()

        # Plot History
        st.subheader("Price History")
        st.line_chart(data)

        # Run Rolling Optimization
        rebal_dates = generate_rebalance_dates(start, end, rebalance, data.index)
        weights_hist = pd.DataFrame(index=data.index, columns=stocks).fillna(0.0)

        current_w = None
        for i, date in enumerate(data.index):
            if pd.Timestamp(date) in rebal_dates or current_w is None:
                if method == "Value-Weighted":
                    current_w = optimize_portfolio_at_date(data.iloc[:i + 1], None, method, stocks)
                else:
                    # Simple MVO window
                    win = data.iloc[max(0, i - 252):i]
                    if len(win) > 50:
                        current_w = optimize_portfolio_at_date(win, 0.02, method, stocks)
                    else:
                        current_w = np.array([1 / len(stocks)] * len(stocks))
            weights_hist.iloc[i] = current_w

        # Perf
        dyn_ret = (returns * weights_hist.shift(1).ffill()).sum(axis=1)
        cum_ret = (1 + dyn_ret).cumprod()

        st.subheader("Performance")
        st.line_chart(cum_ret)
        metrics = compute_metrics(cum_ret)

        st.markdown("### Portfolio Metrics")

        metrics_df = (
            pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
            .reset_index()
            .rename(columns={"index": "Metric"})
        )

        styled_metrics = (
            metrics_df.style
                .set_properties(subset=["Metric"], **{"color": "white"})
                .set_properties(subset=["Value"], **{"color": "#c5a059"})
        )

        st.table(styled_metrics)


        final_w = pd.DataFrame({'Stock': stocks, 'Weight': weights_hist.iloc[-1]}).sort_values('Weight',
                                                                                               ascending=False)
        st.dataframe(final_w.style.format({'Weight': '{:.1%}'}))

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 7. ROUTER
# ==========================================
if __name__ == "__main__":
    init_session_state()
    apply_custom_styles()

    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "optimizer":
        optimizer_page()
    elif st.session_state.page == "blacklitterman":
        blacklitterman_page()