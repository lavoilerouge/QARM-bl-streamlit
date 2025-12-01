# QARM - Quantitative Assets & Risk Management

**Black-Litterman Model with StockTwits Sentiment Integration**

HEC Lausanne - November 2025

---

## 📁 Project Structure

```
QARM/
├── run.py                    # 🚀 Main entry point - RUN THIS
├── config.py                 # ⚙️ Configuration (paths, parameters)
├── requirements.txt          # 📦 Python dependencies
├── README.md                 # 📖 This file
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── app.py               # Streamlit application
│   ├── optimization.py      # Black-Litterman & optimization logic
│   ├── download_all_stocks.py  # NYSE/NASDAQ stock list downloader
│   └── config.py            # Config re-exports
│
├── data/                     # Data directory
│   ├── raw/                  # Raw input data
│   │   ├── stocktwits_optimized.parquet  # StockTwits sentiment data
│   │   └── prices_cleaned.parquet        # Historical stock prices
│   │
│   └── processed/            # Generated data files
│       ├── sentiment_polarity.parquet    # Aggregated daily sentiment polarity
│       ├── SP500_data.parquet            # S&P 500 benchmark data
│       ├── RF_10.parquet                 # Risk-free rate data
│       └── all_nyse_nasdaq_stocks_etfs.csv  # Stock/ETF list
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Data

Place your data files in the `data/raw/` folder:

- **StockTwits data**: `data/raw/stocktwits_optimized.parquet`
- **Prices data**: `data/raw/prices_cleaned.parquet`

### 3. Prepare Data (first time only)

```bash
python run.py --prepare
```

This will:
- ✅ Download NYSE/NASDAQ stock & ETF list (10,000+ securities)
- ✅ Download S&P 500 data from Yahoo Finance
- ✅ Aggregate sentiment from StockTwits
- ✅ Download risk-free rate from FRED

### 4. Run the Application

```bash
python run.py
```

This will start the Streamlit app at `http://localhost:8501`

---

## 📋 Commands

| Command | Description |
|---------|-------------|
| `python run.py` | Run the Streamlit app |
| `python run.py --prepare` | Prepare/regenerate all data files |
| `python run.py --check` | Check configuration and data status |
| `python run.py --deps` | Check if dependencies are installed |
| `python run.py --help` | Show help |

---

## ⚙️ Configuration

All paths are defined in `config.py`. Paths are **automatically relative** to the project directory, so you don't need to change anything when running on a different machine.

### Key Configuration Options

```python
# In config.py

# Data directories (auto-configured)
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model parameters (defaults)
DEFAULT_LOOKBACK_DAYS = 180
DEFAULT_LAMBDA = 2.5  # Risk aversion
DEFAULT_Z_THRESHOLD = 0.5
```

---

## 🎯 Features

### 📘 Portfolio Optimizer
- **Stock Search**: Search 10,000+ NYSE/NASDAQ stocks & ETFs
- **Max Sharpe Ratio** optimization
- **Min Variance** portfolio
- **Equal Weight** allocation
- Real-time data from Yahoo Finance
- Correlation matrix visualization

### 🧠 Black-Litterman Sentiment Model
- **StockTwits sentiment** integration as investor views
- Dynamic universe selection based on sentiment activity
- Out-of-sample backtesting with S&P 500 comparison
- Configurable risk aversion and z-score thresholds

---

## 📊 Data Requirements

### Required Files

1. **StockTwits Data** (`data/raw/stocktwits_optimized.parquet`)
   - Columns: `ticker`, `date`, `bullish_proba`
   - Period: 2009-2020

2. **Price Data** (`data/raw/prices_cleaned.parquet`)
   - Daily adjusted close prices for S&P 500 stocks
   - Same period as StockTwits data

### Auto-Generated Files

These are created when you run `python run.py --prepare`:

- `all_nyse_nasdaq_stocks_etfs.csv` - Stock/ETF search list
- `data/processed/sentiment_polarity.parquet` - Aggregated sentiment polarity
- `data/processed/SP500_data.parquet` - S&P 500 benchmark
- `data/processed/RF_10.parquet` - Risk-free rate

---

## 🔧 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### "Data file not found" errors
```bash
python run.py --check
```
This will show you which files are missing.

### Stock search not working
Run the prepare command to download the stock list:
```bash
python run.py --prepare
```

### Streamlit not starting
Make sure you're in the QARM directory:
```bash
cd /path/to/QARM
python run.py
```

---

## 📝 Authors

HEC Lausanne QARM Team:
Leonardo, Maxime C., Maxime D., Damien, Valentin

---

## 📄 License

Academic use only.
