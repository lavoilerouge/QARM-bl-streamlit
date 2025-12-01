"""
Download ALL NYSE, NASDAQ, and NYSE Arca stocks AND ETFs (tickers and names)
This script uses multiple methods to get comprehensive listings of both stocks and ETFs
"""

import pandas as pd
import requests
from io import StringIO
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import STOCK_LIST_CSV, ensure_directories

# Ensure directories exist
ensure_directories()


def method1_nasdaq_ftp():
    """Download from NASDAQ's official FTP server"""
    print("Method 1: NASDAQ FTP Server...")
    try:
        # NASDAQ stocks
        url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
        nasdaq_data = pd.read_csv(url, sep='|')
        nasdaq_data = nasdaq_data[nasdaq_data['Test Issue'] == 'N']
        
        # Separate stocks and ETFs
        nasdaq_stocks = nasdaq_data[nasdaq_data['ETF'] == 'N'][['Symbol', 'Security Name']].copy()
        nasdaq_stocks['Exchange'] = 'NASDAQ'
        nasdaq_stocks['Type'] = 'Stock'
        
        nasdaq_etfs = nasdaq_data[nasdaq_data['ETF'] == 'Y'][['Symbol', 'Security Name']].copy()
        nasdaq_etfs['Exchange'] = 'NASDAQ'
        nasdaq_etfs['Type'] = 'ETF'
        
        # NYSE, NYSE Arca, and other exchanges
        url2 = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"
        other_data = pd.read_csv(url2, sep='|')
        other_data = other_data[other_data['Test Issue'] == 'N']
        
        # NYSE stocks and ETFs (Exchange code: N)
        nyse_stocks = other_data[(other_data['Exchange'] == 'N') & (other_data['ETF'] == 'N')][['ACT Symbol', 'Security Name']].copy()
        nyse_stocks.columns = ['Symbol', 'Security Name']
        nyse_stocks['Exchange'] = 'NYSE'
        nyse_stocks['Type'] = 'Stock'
        
        nyse_etfs = other_data[(other_data['Exchange'] == 'N') & (other_data['ETF'] == 'Y')][['ACT Symbol', 'Security Name']].copy()
        nyse_etfs.columns = ['Symbol', 'Security Name']
        nyse_etfs['Exchange'] = 'NYSE'
        nyse_etfs['Type'] = 'ETF'
        
        # NYSE Arca stocks and ETFs (Exchange code: P)
        arca_stocks = other_data[(other_data['Exchange'] == 'P') & (other_data['ETF'] == 'N')][['ACT Symbol', 'Security Name']].copy()
        arca_stocks.columns = ['Symbol', 'Security Name']
        arca_stocks['Exchange'] = 'NYSE Arca'
        arca_stocks['Type'] = 'Stock'
        
        arca_etfs = other_data[(other_data['Exchange'] == 'P') & (other_data['ETF'] == 'Y')][['ACT Symbol', 'Security Name']].copy()
        arca_etfs.columns = ['Symbol', 'Security Name']
        arca_etfs['Exchange'] = 'NYSE Arca'
        arca_etfs['Type'] = 'ETF'
        
        # Combine all
        all_securities = pd.concat([
            nasdaq_stocks, nasdaq_etfs, 
            nyse_stocks, nyse_etfs,
            arca_stocks, arca_etfs
        ], ignore_index=True)
        
        print(f"✓ Found {len(all_securities)} securities")
        print(f"  - NASDAQ: {len(nasdaq_stocks)} stocks, {len(nasdaq_etfs)} ETFs")
        print(f"  - NYSE: {len(nyse_stocks)} stocks, {len(nyse_etfs)} ETFs")
        print(f"  - NYSE Arca: {len(arca_stocks)} stocks, {len(arca_etfs)} ETFs")
        print(f"  - Total: {len(nasdaq_stocks) + len(nyse_stocks) + len(arca_stocks)} stocks, {len(nasdaq_etfs) + len(nyse_etfs) + len(arca_etfs)} ETFs")
        
        return all_securities
    except Exception as e:
        print(f"✗ Failed: {e}")
        return None


def method2_wikipedia_indices():
    """Download from Wikipedia major indices"""
    print("\nMethod 2: Wikipedia Indices...")
    try:
        all_stocks = []
        
        # S&P 500
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500 = sp500[['Symbol', 'Security']].copy()
        sp500.columns = ['Symbol', 'Security Name']
        all_stocks.append(sp500)
        
        # NASDAQ-100
        nasdaq100 = pd.read_html('https://en.wikipedia.org/wiki/NASDAQ-100')[4]
        nasdaq100 = nasdaq100[['Ticker', 'Company']].copy()
        nasdaq100.columns = ['Symbol', 'Security Name']
        all_stocks.append(nasdaq100)
        
        # S&P MidCap 400
        try:
            sp400 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')[0]
            sp400 = sp400[['Symbol', 'Security']].copy()
            sp400.columns = ['Symbol', 'Security Name']
            all_stocks.append(sp400)
        except:
            pass
        
        # S&P SmallCap 600
        try:
            sp600 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')[0]
            sp600 = sp600[['Symbol', 'Security']].copy()
            sp600.columns = ['Symbol', 'Security Name']
            all_stocks.append(sp600)
        except:
            pass
        
        stocks = pd.concat(all_stocks, ignore_index=True)
        stocks = stocks.drop_duplicates(subset=['Symbol'])
        print(f"✓ Found {len(stocks)} unique stocks")
        return stocks
    except Exception as e:
        print(f"✗ Failed: {e}")
        return None


def method3_eoddata():
    """Try EODData API"""
    print("\nMethod 3: EODData API...")
    try:
        # This would require an API key, skipping for now
        print("✗ Requires API key (skipped)")
        return None
    except Exception as e:
        print(f"✗ Failed: {e}")
        return None


def save_stocks(stocks_df, filename=None):
    """Save stocks and ETFs to CSV file"""
    if filename is None:
        filename = STOCK_LIST_CSV
    
    if stocks_df is None or len(stocks_df) == 0:
        print("\n❌ No securities to save!")
        return False
    
    # Clean up
    stocks_df = stocks_df.dropna(subset=['Symbol'])
    stocks_df = stocks_df[stocks_df['Symbol'].str.len() > 0]
    stocks_df = stocks_df.sort_values('Symbol').reset_index(drop=True)
    
    # Save
    stocks_df.to_csv(filename, index=False)
    
    print(f"\n{'='*70}")
    print(f"✅ SUCCESS! Saved {len(stocks_df)} securities to: {filename}")
    print(f"{'='*70}")
    
    print("\nSample (first 20 securities):")
    print(stocks_df.head(20).to_string(index=False))
    
    print(f"\nSample (last 10 securities):")
    print(stocks_df.tail(10).to_string(index=False))
    
    # Stats
    if 'Exchange' in stocks_df.columns:
        print("\nBreakdown by Exchange:")
        print(stocks_df['Exchange'].value_counts())
    
    if 'Type' in stocks_df.columns:
        print("\nBreakdown by Type:")
        print(stocks_df['Type'].value_counts())
        
        print("\nDetailed Breakdown:")
        print(stocks_df.groupby(['Exchange', 'Type']).size().to_string())
    
    return True


def main():
    print("="*70)
    print("DOWNLOADING ALL NYSE, NASDAQ & NYSE ARCA STOCKS & ETFs")
    print("="*70)
    
    stocks = None
    
    # Try Method 1 (most comprehensive)
    stocks = method1_nasdaq_ftp()
    
    # If Method 1 fails, try Method 2
    if stocks is None:
        stocks = method2_wikipedia_indices()
    
    # Save results
    if stocks is not None:
        save_stocks(stocks)
    else:
        print("\n❌ All methods failed. Try installing: pip install pandas lxml html5lib")
        print("Or manually download from: https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs")


if __name__ == "__main__":
    main()
