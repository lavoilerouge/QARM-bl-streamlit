"""
QARM Optimization Module
========================
Black-Litterman Model with StockTwits Sentiment Integration

This module contains all optimization logic, including:
- Sentiment preprocessing
- Black-Litterman weight calculation
- Portfolio optimization with CVXPY

Authors: HEC Lausanne QARM Team
Date: November 2025
"""

import duckdb
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import polars as pl
from sklearn.linear_model import LinearRegression
from pandas_datareader import data as pdr
from sklearn.covariance import LedoitWolf
import cvxpy as cp
from numpy.linalg import solve

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================

try:
    from src.config import (
        STOCKTWITS_PARQUET, 
        SENTIMENT_POLARITY_PARQUET, 
        PRICES_CLEANED_PARQUET,
        SP500_DATA_PARQUET, 
        RF_10_PARQUET, 
        DATA_START_DATE, 
        DATA_END_DATE,
        get_path_str,
        ensure_directories 
    )
except ImportError:
    from config import (
        STOCKTWITS_PARQUET, 
        SENTIMENT_POLARITY_PARQUET, 
        PRICES_CLEANED_PARQUET,
        SP500_DATA_PARQUET, 
        RF_10_PARQUET, 
        DATA_START_DATE, 
        DATA_END_DATE,
        get_path_str,
        ensure_directories 
    )

# =============================================================================
# CONSTANTS
# =============================================================================

# Leveraged ETFs to exclude from optimization to prevent skewing results
ETF_LEVERED = {
    "UPRO", "SPXL", "SPXS", "SPXU", "TQQQ", "SQQQ", "UVXY", "VXX",
    "JNUG", "JDST", "LABU", "LABD", "YINN", "YANG", "SRS", "FAZ", "DRV", "FAS",
}

# S&P 500 tickers (2019 version)
SP500_TICKERS_2019 = [
    "A", "AAL", "AAP", "AAPL", "ABBV", "ABC", "ABMD", "ABT", "ACN", "ADBE",
    "ADI", "ADM", "ADP", "ADS", "ADSK", "AEE", "AEP", "AES", "AET", "AFL",
    "AGN", "AIG", "AIV", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK", "ALL",
    "ALLE", "ALXN", "AMAT", "AMD", "AME", "AMG", "AMGN", "AMP", "AMT", "AMZN",
    "AN", "ANSS", "ANTM", "AON", "APA", "APC", "APD", "APH", "APTV", "ARE",
    "ARNC", "ATVI", "AVB", "AVGO", "AVY", "AWK", "AXP", "AZO", "BA", "BAC",
    "BAX", "BBT", "BBY", "BDX", "BEN", "BF-B", "BHGE", "BIIB", "BK", "BKNG",
    "BLK", "BLL", "BMY", "BR", "BRK.A", "BRK.B", "BSX", "BWA", "BXP", "C",
    "CAG", "CAH", "CAKE", "CAT", "CB", "CBOE", "CBS", "CCI", "CCL", "CDNS",
    "CELG", "CERN", "CF", "CFG", "CHD", "CHRW", "CI", "CINF", "CL", "CLX",
    "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COG", "COO",
    "COP", "COST", "COTY", "CPB", "CRM", "CSCO", "CSX", "CTAS", "CTL", "CTSH",
    "CTXS", "CVS", "CVX", "CXO", "D", "DAL", "DD", "DDS", "DE", "DFS", "DG",
    "DGX", "DHI", "DHR", "DIS", "DISCA", "DISCK", "DISH", "DLR", "DLTR", "DOV",
    "DOW", "DRE", "DRI", "DTE", "DUK", "DVA", "DVN", "DWDP", "DXC", "EA",
    "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "EMN", "EMR", "ENPH", "EOG",
    "EQIX", "EQR", "ES", "ESS", "ETFC", "ETN", "ETR", "EVRG", "EW", "EXC",
    "EXPD", "EXPE", "EXR", "F", "FAST", "FB", "FBHS", "FCX", "FDX", "FE",
    "FFIV", "FIS", "FISV", "FITB", "FLIR", "FLR", "FLS", "FLT", "FMC", "FOXA",
    "FOX", "FRC", "FRT", "FTI", "FTNT", "FTV", "GD", "GE", "GGP", "GILD",
    "GIS", "GLW", "GM", "GOOG", "GOOGL", "GPC", "GPN", "GPS", "GRMN", "GS",
    "GT", "GWW", "HAL", "HAS", "HBAN", "HBI", "HCA", "HCP", "HD", "HES",
    "HFC", "HIG", "HII", "HLT", "HOG", "HOLX", "HON", "HP", "HPE", "HPQ",
    "HRB", "HRL", "HSIC", "HST", "HSY", "HUM", "IBM", "ICE", "IDXX", "IFF",
    "ILMN", "INCY", "INFO", "INTC", "INTU", "IP", "IPG", "IPGP", "IQV", "IR",
    "IRM", "ISRG", "IT", "ITW", "IVZ", "JBHT", "JCI", "JEC", "JEF", "JNJ",
    "JNPR", "JPM", "JWN", "K", "KEY", "KHC", "KIM", "KLAC", "KMB", "KMI",
    "KMX", "KO", "KR", "KSS", "KSU", "L", "LB", "LDOS", "LEG", "LEN", "LH",
    "LHX", "LIN", "LKQ", "LLY", "LMT", "LNC", "LNT", "LOW", "LRCX", "LUV",
    "LW", "LYB", "M", "MA", "MAA", "MAC", "MAR", "MAS", "MCD", "MCHP", "MCK",
    "MCO", "MDLZ", "MDT", "MET", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC",
    "MMM", "MNST", "MO", "MOS", "MPC", "MRK", "MRO", "MS", "MSCI", "MSFT",
    "MSI", "MTB", "MTD", "MU", "MXIM", "MYL", "NBL", "NCLH", "NDAQ", "NEE",
    "NEM", "NFLX", "NFX", "NI", "NKE", "NKTR", "NLSN", "NOC", "NOV", "NRG",
    "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NWL", "NWS", "NWSA", "O", "OKE",
    "OMC", "ORCL", "ORLY", "OXY", "PAYX", "PBCT", "PCAR", "PEG", "PEP", "PFE",
    "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PKI", "PLD", "PM", "PNC", "PNR",
    "PNW", "PPG", "PPL", "PRGO", "PRU", "PSA", "PSX", "PVH", "PWR", "PXD",
    "PYPL", "QCOM", "QRVO", "RCL", "RE", "REG", "REGN", "RF", "RHI", "RJF",
    "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTN", "SBAC", "SBUX",
    "SCG", "SCHW", "SEE", "SHW", "SIVB", "SJM", "SLB", "SLG", "SNA", "SNPS",
    "SO", "SPG", "SPGI", "SRCL", "SRE", "STI", "STT", "STX", "STZ", "SWK",
    "SWKS", "SYF", "SYK", "SYMC", "SYY", "T", "TAP", "TDG", "TEL", "TFX",
    "TGT", "TIF", "TJX", "TMO", "TPR", "TRIP", "TROW", "TRV", "TSCO", "TSN",
    "TSS", "TTWO", "TWTR", "TXN", "TXT", "UA", "UAA", "UAL", "UDR", "UHS",
    "ULTA", "UNH", "UNM", "UNP", "UPS", "URI", "USB", "UTX", "V", "VAR",
    "VFC", "VIAB", "VLO", "VMC", "VNO", "VRSK", "VRSN", "VRTX", "VTR", "VZ",
    "WAB", "WAT", "WBA", "WCG", "WDC", "WEC", "WELL", "WFC", "WHR", "WLTW",
    "WM", "WMB", "WMT", "WRK", "WU", "WY", "WYNN", "XEC", "XEL", "XLNX",
    "XOM", "XRAY", "XYL", "YUM", "ZBH", "ZION", "ZTS",
]

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
# =======================================
# MARKET CAP CALCULATION (CONSISTENT WITH VALUE-WEIGHTED)
# =======================================

def get_market_caps_for_date(tickers, rebalance_date, px_prices=None):
    """
    Fetch market capitalizations for given tickers at a specific date.
    Market cap = shares outstanding × price at rebalance_date
    
    This function is consistent with the value-weighted optimization in app.py.
    """
    market_caps = {}
    
    if rebalance_date is not None:
        rebalance_date = pd.to_datetime(rebalance_date)
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get shares outstanding
            shares_out = info.get('sharesOutstanding', None)
            if shares_out is None:
                # Try fast_info as fallback
                shares_out = getattr(stock.fast_info, 'shares_outstanding', None)
            if shares_out is None:
                continue
            
            # Get price at rebalance_date
            price = None
            
            # Try px_prices (price data) first
            if px_prices is not None and rebalance_date is not None and ticker in px_prices.columns:
                available_dates = px_prices.index[px_prices.index <= rebalance_date]
                if len(available_dates) > 0:
                    closest_date = available_dates[-1]
                    price = px_prices.loc[closest_date, ticker]
                    if pd.isna(price):
                        price = None
            
            # Fallback: fetch historical price from Yahoo Finance
            if price is None and rebalance_date is not None:
                start = rebalance_date - pd.Timedelta(days=7)
                end = rebalance_date + pd.Timedelta(days=1)
                hist = stock.history(start=start, end=end)
                if not hist.empty:
                    # Get closest date on or before rebalance_date
                    hist.index = hist.index.tz_localize(None)
                    valid_dates = hist.index[hist.index <= rebalance_date]
                    if len(valid_dates) > 0:
                        price = hist.loc[valid_dates[-1], 'Close']
            
            # Final fallback: current price
            if price is None:
                price = info.get('regularMarketPrice') or info.get('previousClose')
            
            # Compute market cap
            if price is not None and not pd.isna(price):
                market_cap = float(shares_out) * float(price)
                if market_cap > 0:
                    market_caps[ticker] = market_cap
                    
        except Exception:
            continue
    
    return market_caps

def load_stocktwits_info():
    """Load and display StockTwits dataset information."""
    parquet_path = get_path_str(STOCKTWITS_PARQUET)

    period = duckdb.query(f"""
        SELECT 
            MIN(date) as first_tweet,
            MAX(date) as last_tweet
        FROM '{parquet_path}'
    """).df()

    print(f"\nStockTwits Dataset Period:")
    print(f"  First tweet: {period['first_tweet'][0]}")
    print(f"  Last tweet: {period['last_tweet'][0]}")
    duration = (period["last_tweet"][0] - period["first_tweet"][0]).days
    print(f"  Duration: {duration:,} days (~{duration/365:.1f} years)")

    return period


def download_sp500_data():
    """Download S&P 500 data from Yahoo Finance."""
    SP500 = yf.download("^GSPC", start=DATA_START_DATE, end=DATA_END_DATE, progress=False)

    if isinstance(SP500.columns, pd.MultiIndex):
        SP500.columns = SP500.columns.get_level_values(0)

    price_SP500 = SP500["Close"].rename("SP500_Close_Price")
    returns_SP500 = np.log(price_SP500 / price_SP500.shift(1)).rename("SP500_Log_Returns")
    SP500 = pd.concat([price_SP500, returns_SP500], axis=1).dropna()

    ensure_directories()
    SP500.to_parquet(SP500_DATA_PARQUET)
    print(f"✓ S&P 500 data saved to: {SP500_DATA_PARQUET}")

    return SP500


def aggregate_sentiment():
    """Aggregate StockTwits sentiment data daily."""
    parquet_path = get_path_str(STOCKTWITS_PARQUET)

    query = f"""
        WITH base AS (
            SELECT
                REPLACE(ticker, '$', '') AS ticker,
                DATE_TRUNC('day', date) AS date,
                bullish_proba,
                CASE 
                    WHEN bullish_proba >= 0.60 THEN 1
                    WHEN bullish_proba <= 0.40 THEN -1
                    ELSE 0
                END AS sentiment_label
            FROM '{parquet_path}'
            WHERE bullish_proba IS NOT NULL
        )
        SELECT
            ticker,
            date,
            SUM(CASE WHEN sentiment_label = 1 THEN 1 ELSE 0 END) AS n_bullish,
            SUM(CASE WHEN sentiment_label = -1 THEN 1 ELSE 0 END) AS n_bearish,
            SUM(CASE WHEN sentiment_label IN (1,-1) THEN 1 ELSE 0 END) AS n_msgs_valid,
            COUNT(*) AS n_msgs_total
        FROM base
        GROUP BY ticker, date
        ORDER BY ticker, date
    """

    sentiment_df = duckdb.query(query).to_df()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.tz_localize(None)

    # Polarity calculation
    sentiment_df["polarity"] = (
        sentiment_df["n_bullish"] - sentiment_df["n_bearish"]
    ) / sentiment_df["n_msgs_valid"].replace(0, np.nan)
    sentiment_df["n_msgs"] = sentiment_df["n_msgs_valid"]

    # Z-score for polarity
    sentiment_df = sentiment_df.sort_values(["ticker", "date"])
    sentiment_df["polarity_mean"] = sentiment_df.groupby("ticker")["polarity"].transform(
        lambda x: x.rolling(60, min_periods=20).mean()
    )
    sentiment_df["polarity_std"] = sentiment_df.groupby("ticker")["polarity"].transform(
        lambda x: x.rolling(60, min_periods=20).std()
    )
    sentiment_df["sentiment_z"] = (
        sentiment_df["polarity"] - sentiment_df["polarity_mean"]
    ) / sentiment_df["polarity_std"]

    # Save to parquet
    ensure_directories()
    sentiment_pl = pl.from_pandas(sentiment_df)
    (
        sentiment_pl.lazy()
        .with_columns(pl.col("date").cast(pl.Date))
        .sink_parquet(str(SENTIMENT_POLARITY_PARQUET))
    )
    print(f"✓ Sentiment data saved to: {SENTIMENT_POLARITY_PARQUET}")

    return sentiment_df


def download_risk_free_rate():
    """Download 10-year Treasury rate from FRED."""
    RF_10 = pdr.DataReader("DGS10", "fred", start="2008-12-31", end=DATA_END_DATE)

    bdays = pd.date_range(DATA_START_DATE, DATA_END_DATE, freq="B")
    RF_10 = RF_10.reindex(bdays)
    RF_10 = RF_10.interpolate(limit_direction="both") / 100

    daily_rf = (1 + RF_10) ** (1 / 252) - 1
    daily_log_rf = np.log(1 + daily_rf)

    rf_df = pd.DataFrame({
        "RF_10Y": RF_10["DGS10"],
        "Daily_RF_10Y": daily_rf["DGS10"],
        "Daily_Log_RF_10Y": daily_log_rf["DGS10"],
    })

    ensure_directories()
    rf_df.to_parquet(RF_10_PARQUET)
    print(f"✓ Risk-free rate saved to: {RF_10_PARQUET}")

    return rf_df


# =============================================================================
# SENTIMENT PROCESSING FUNCTIONS
# =============================================================================

def dynamic_min_msgs_stocktwits(sentiment_df, t):
    """Calculate dynamic minimum messages threshold."""
    t = pd.to_datetime(t)
    window = sentiment_df[
        (sentiment_df["date"] >= t - pd.Timedelta(days=30)) & (sentiment_df["date"] < t)
    ]
    avg_msgs = window.groupby("ticker", observed=True)["n_msgs"].sum().mean()
    return max(3, int(avg_msgs * 0.25))


def sentiment_preprocess(sentiment_df, tickers, t, window_sent=30):
    """
    Aggregate StockTwits sentiment over a rolling window before t.

    Returns DataFrame indexed by ticker with columns:
      - z_mean: mean of sentiment_z
      - z_last: last value of sentiment_z
      - msgs: total number of messages
    """
    t = pd.to_datetime(t)
    start = t - pd.Timedelta(days=window_sent)

    df = sentiment_df[
        (sentiment_df["ticker"].isin(tickers))
        & (sentiment_df["date"] >= start)
        & (sentiment_df["date"] < t)
    ]

    if df.empty:
        return pd.DataFrame(index=tickers)

    agg = df.groupby("ticker", observed=True).agg(
        z_mean=("sentiment_z", "mean"),
        z_last=("sentiment_z", "last"),
        msgs=("n_msgs", "sum"),
    )

    for tic in tickers:
        if tic not in agg.index:
            agg.loc[tic] = [np.nan, np.nan, 0]

    return agg


def estimate_alpha_from_regression(sentiment_df, returns_clean, min_msgs=10):
    """Estimate alpha (market sensitivity to sentiment) via regression."""
    s = sentiment_df.copy()
    s = s[s["ticker"].isin(returns_clean.columns)]
    s = s[s["n_msgs"] >= min_msgs]

    if s.empty:
        raise ValueError("Not enough data for regression.")

    s_idx = s.set_index(["date", "ticker"])[["sentiment_z"]]
    r = returns_clean.stack().to_frame("ret")
    r.index.set_names(["date", "ticker"], inplace=True)
    r["ret_next"] = r.groupby(level="ticker")["ret"].shift(-1)

    df_reg = s_idx.join(r[["ret_next"]], how="inner").dropna()

    if df_reg.empty:
        raise ValueError("Empty regression dataframe after join.")

    X = df_reg[["sentiment_z"]].values
    y = df_reg["ret_next"].values

    reg = LinearRegression().fit(X, y)
    beta_hat = float(reg.coef_[0])
    sigma_eps2 = float(np.var(y - reg.predict(X), ddof=1))

    return beta_hat, sigma_eps2

# ============================
# GLOBAL ALPHA ESTIMATION
# ============================

def compute_alpha_dynamic(sentiment_df, returns_clean, train_end):
    sentiment_train = sentiment_df[sentiment_df["date"] < train_end]
    returns_train = returns_clean.loc[:train_end]

    alpha_hat, sigma_eps2 = estimate_alpha_from_regression(
        sentiment_train, returns_train)
    
    return alpha_hat

# =============================================================================
# UNIVERSE SELECTION
# =============================================================================

def select_universe(
    sentiment_df,
    px_prices,
    as_of_date,
    top_n=50,
    min_tickers=5,
    min_price=5,
):
    """
    Notebook-style universe selection:
    - NO sentiment filter
    - SP500 ∩ price data
    - Exclude missing prices
    - Select lowest volatility assets
    """

    as_of_date = pd.to_datetime(as_of_date)

    # tickers in SP500 AND in price dataset
    tickers_available = [
        t for t in SP500_TICKERS_2019
        if t in px_prices.columns
    ]

    if len(tickers_available) < min_tickers:
        return None

    # historical price window
    px_hist = px_prices.loc[:as_of_date, tickers_available].ffill()

    # returns & volatility
    rets = px_hist.pct_change()
    vol = rets.std()

    # filter by price (avoid penny stocks)
    last_prices = px_hist.iloc[-1]
    vol = vol[last_prices >= min_price]

    # pick lowest volatility
    vol_sorted = vol.sort_values()
    universe = vol_sorted.index[:top_n].tolist()

    if len(universe) < min_tickers:
        return None

    return universe


# =============================================================================
# BLACK-LITTERMAN VIEWS
# =============================================================================

def build_views_from_sentiment(
    sent_agg,
    tickers,
    alpha_dyn,
    Sigma_mens,
    omega_view=0.05,
    min_msgs=10,
    z_threshold=0.5,
):
    """
    Build Black-Litterman views (P, Q, Omega) from aggregated sentiment.

    Returns:
        P: matrix (K × N)
        Q: vector (K,)
        Omega: matrix (K × K)
        K: number of views
    """
    if sent_agg is None or sent_agg.empty:
        return None, None, None, 0

    N = len(tickers)

    valid = sent_agg[sent_agg["msgs"] >= min_msgs].copy()
    if valid.empty:
        return None, None, None, 0

    strong = valid[valid["z_mean"].abs() >= z_threshold]
    if strong.empty:
        return None, None, None, 0

    K = len(strong)

    P = np.zeros((K, N))
    Q = np.zeros(K)
    Omega = np.zeros((K, K))

    for i, (tic, row) in enumerate(strong.iterrows()):
        if tic in tickers:
            j = tickers.index(tic)
            P[i, j] = 1.0
            Q[i] = alpha_dyn * float(row["z_mean"])
            base_var = omega_view * float(Sigma_mens[j, j])
            confidence = 1 + np.log1p(row["msgs"]) * abs(row["z_mean"])
            confidence = max(confidence, 1.0)
            Omega[i, i] = base_var / confidence

    return P, Q, Omega, K


# =============================================================================
# BLACK-LITTERMAN OPTIMIZATION
# =============================================================================

def black_litterman_weights_for_date(
    t,
    sentiment_df,
    lam,
    px_prices,
    rets_all,
    alpha_dyn,
    window_sent=30,
    lookback_days=180,
    top_n=50,
    max_long=0.25,
    max_short_per_asset=0.10,
    leverage_limit=1.2, # Default value, but should be passed from front-end
    z_threshold=0.5,
    fixed_universe=None,
):
    """
    Calculate Black-Litterman weights at date t using:
      - He-Litterman prior (implicit π)
      - Views constructed from StockTwits (z-scores)

    Fixed: leverage_limit is now a dynamic parameter used in the constraints.
    """
    t = pd.to_datetime(t)

    # Select universe
    if fixed_universe is not None:
        universe = [tic for tic in fixed_universe if tic in px_prices.columns]
    else:
        universe = select_universe(
            sentiment_df=sentiment_df,
            px_prices=px_prices,
            as_of_date=t,
            top_n=top_n,
        )

    if not universe or len(universe) < 5:
        return None, 0, []

    # Exclude leveraged ETFs
    universe = [tic for tic in universe if tic not in ETF_LEVERED]
    if len(universe) < 5:
        return None, 0, []

    # Get price and returns data
    px = px_prices.loc[:t, universe].tail(lookback_days).dropna(axis=1, how="any")
    rets = rets_all.loc[:t, universe].tail(lookback_days).dropna(how="any")

    if rets.shape[0] < 30 or rets.shape[1] < 2:
        return None, 0, []

    tickers = rets.columns.tolist()
    N = len(tickers)

    # Covariance estimation (Ledoit-Wolf)
    lw = LedoitWolf().fit((rets - rets.mean()).values)
    Sigma_daily = lw.covariance_
    Sigma_mens = Sigma_daily * 21
    Sigma_mens += 1e-8 * np.eye(N)

    # Market portfolio weights
    gamma = float(lam)
    vols = np.sqrt(np.diag(Sigma_mens))

    # Try to get market caps at rebalance date
    # On cloud environments, Yahoo Finance may be rate-limited, so we use fallbacks
    try:
        caps_dict = get_market_caps_for_date(tickers, rebalance_date=t, px_prices=px_prices)
        caps = np.array([caps_dict.get(ticker, np.nan) for ticker in tickers])
        valid = ~np.isnan(caps)
        
        if valid.sum() >= 2:
            # Use market-cap weights
            x0 = np.zeros(len(tickers))
            x0[valid] = caps[valid] / caps[valid].sum()
        else:
            # Fallback: inverse volatility weights (risk parity inspired)
            inv_vols = 1.0 / (vols + 1e-8)
            x0 = inv_vols / inv_vols.sum()
    except Exception:
        # Fallback: inverse volatility weights
        inv_vols = 1.0 / (vols + 1e-8)
        x0 = inv_vols / inv_vols.sum()

    # Implied returns (prior)
    pi_raw = gamma * (Sigma_mens @ x0).reshape(-1, 1)
    pi = pi_raw

    # Sentiment preprocessing
    sent_agg = sentiment_preprocess(sentiment_df, tickers, t, window_sent)
    min_msgs_t = dynamic_min_msgs_stocktwits(sentiment_df, t)

    # Build views
    P, Q, Omega, K = build_views_from_sentiment(
        sent_agg,
        tickers,
        alpha_dyn=alpha_dyn,
        Sigma_mens=Sigma_mens,
        min_msgs=min_msgs_t,
        z_threshold=z_threshold,
    )

    if K == 0:
        return None, 0, []

    # Calculate tau (tracking error based)
    TE2 = float(x0 @ Sigma_mens @ x0)
    pi_var = float(pi.T @ pi)
    tau_BL = TE2 / pi_var if pi_var > 1e-9 else 0.05

    # Black-Litterman posterior
    Gamma = tau_BL * Sigma_mens
    Q_vec = Q.reshape(-1, 1)
    M = P @ Gamma @ P.T + Omega
    diff = Q_vec - P @ pi
    mu_BL = pi + Gamma @ P.T @ solve(M + 1e-8 * np.eye(K), diff)
    mu_vec = mu_BL.flatten()

    # Portfolio optimization with CVXPY
    w = cp.Variable(N)
    objective = cp.Maximize(
        mu_vec @ w - 0.5 * gamma * cp.quad_form(w, cp.psd_wrap(Sigma_mens))
    )

    # --- UPDATED CONSTRAINTS ---
    # Use dynamic leverage_limit passed from arguments
    constraints = [
        cp.sum(w) == 1.0,
        cp.norm1(w) <= float(leverage_limit),  # Dynamic gross exposure
        w <= float(max_long),
        w >= -float(max_short_per_asset),
    ]

    prob = cp.Problem(objective, constraints)

    # Solver logic with fallback
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except:
            return None, K, []

    if (w.value is None) or (prob.status not in ["optimal", "optimal_inaccurate"]):
        w_opt = np.ones(N) / N
    else:
        w_opt = np.array(w.value).flatten()

    w_df = pd.DataFrame({"ticker": tickers, "w_opt": w_opt})

    strong_list = sent_agg.index[
        (sent_agg["z_mean"].abs() >= z_threshold) & (sent_agg["msgs"] >= min_msgs_t)
    ].tolist()

    return w_df, K, strong_list


# =============================================================================
# HELPER FUNCTIONS FOR BACKTESTING
# =============================================================================

def next_month_window(t, price_index):
    """
    Return the trading window (start, end) AFTER rebalancing at t.
    """
    t = pd.to_datetime(t)
    future_dates = price_index[price_index > t]

    if future_dates.empty:
        return None, None

    start = future_dates.min()
    month = start.month
    year = start.year
    month_days = price_index[(price_index.month == month) & (price_index.year == year)]

    if month_days.empty:
        return None, None

    end = month_days.max()
    return start, end


def get_window_prices(px_prices, tickers, start, end):
    """Extract prices between start and end for given tickers."""
    px = px_prices.loc[start:end, tickers]
    px = px.ffill().dropna(how="all", axis=1)

    if px.shape[1] < 2:
        return pd.DataFrame()

    return px


def compute_log_returns(px: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from prices."""
    rets = np.log(px / px.shift(1))
    rets = rets.dropna()
    return rets


# =============================================================================
# DATA PREPARATION (run once to prepare all data)
# =============================================================================

def prepare_all_data(force_reload=False):
    """
    Prepare all necessary data files.
    Run this once before using the Streamlit app.
    """
    ensure_directories()

    print("=" * 60)
    print("QARM Data Preparation")
    print("=" * 60)

    # Check if StockTwits data exists
    if not STOCKTWITS_PARQUET.exists():
        print(f"⚠ StockTwits data not found at: {STOCKTWITS_PARQUET}")
        print("  Please place 'stocktwits_optimized.parquet' in data/raw/")
        return False

    # Load StockTwits info
    print("\n1. Loading StockTwits dataset info...")
    load_stocktwits_info()

    # Download S&P 500 data
    if force_reload or not SP500_DATA_PARQUET.exists():
        print("\n2. Downloading S&P 500 data...")
        download_sp500_data()
    else:
        print("\n2. S&P 500 data already exists, skipping...")

    # Aggregate sentiment
    if force_reload or not SENTIMENT_POLARITY_PARQUET.exists():
        print("\n3. Aggregating sentiment data...")
        aggregate_sentiment()
    else:
        print("\n3. Sentiment data already exists, skipping...")

    # Download risk-free rate
    if force_reload or not RF_10_PARQUET.exists():
        print("\n4. Downloading risk-free rate...")
        download_risk_free_rate()
    else:
        print("\n4. Risk-free rate already exists, skipping...")

    print("\n" + "=" * 60)
    print("✓ Data preparation complete!")
    print("=" * 60)

    return True


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    prepare_all_data()