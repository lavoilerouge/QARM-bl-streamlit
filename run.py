#!/usr/bin/env python3
"""
QARM Runner
===========
Single entry point to run the entire QARM project.

Usage:
    python run.py              # Run the Streamlit app
    python run.py --prepare    # Prepare data files (run once)
    python run.py --check      # Check configuration and data files
    python run.py --help       # Show help

Authors: HEC Lausanne QARM Team
Date: November 2025
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'yfinance': 'yfinance',
        'duckdb': 'duckdb',
        'sklearn': 'scikit-learn',
        'cvxpy': 'cvxpy',
        'polars': 'polars',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'pandas_datareader': 'pandas-datareader',
        'scipy': 'scipy',
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print("❌ Missing packages:", ", ".join(missing))
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("✅ All required packages are installed")
    return True


def check_config():
    """Check configuration and data files."""
    from Project.config import print_config, check_data_files, STOCKTWITS_PARQUET
    
    print_config()
    
    files_status = check_data_files()
    all_ready = all(files_status.values())
    
    if not files_status["stocktwits_raw"]:
        print(f"\n⚠️  StockTwits data not found!")
        print(f"   Please place 'stocktwits_optimized.parquet' in:")
        print(f"   {STOCKTWITS_PARQUET.parent}")
    
    return all_ready


def download_stock_list():
    """Download NYSE/NASDAQ stock and ETF list."""
    print("\n" + "=" * 60)
    print("Downloading Stock & ETF List")
    print("=" * 60)
    
    from config import STOCK_LIST_CSV
    csv_path = STOCK_LIST_CSV
    
    # Check if already exists
    if csv_path.exists():
        print(f"✓ Stock list already exists: {csv_path}")
        return True
    
    try:
        from src.download_all_stocks import main as download_main
        download_main()
        return csv_path.exists()
    except Exception as e:
        print(f"⚠️  Could not download stock list: {e}")
        print("   The app will work but without stock search functionality.")
        return False


def prepare_data():
    """Prepare all data files."""
    print("=" * 60)
    print("QARM Data Preparation")
    print("=" * 60)
    
    from config import ensure_directories, check_data_files, STOCKTWITS_PARQUET
    
    # Ensure directories exist
    ensure_directories()
    
    # Download stock list first (doesn't require StockTwits data)
    download_stock_list()
    
    files_status = check_data_files()
    
    if not files_status["stocktwits_raw"]:
        print(f"\n❌ Cannot prepare sentiment data: StockTwits data not found!")
        print(f"   Please place 'stocktwits_optimized.parquet' in:")
        print(f"   {STOCKTWITS_PARQUET.parent}")
        print(f"\n   Then run: python run.py --prepare")
        return False
    
    # Import and run preparation
    from src.optimization import prepare_all_data
    return prepare_all_data()


def run_streamlit():
    """Run the Streamlit application."""
    print("=" * 60)
    print("Starting QARM Streamlit Application")
    print("=" * 60)
    
    app_path = PROJECT_ROOT / "src" / "app.py"
    
    if not app_path.exists():
        print(f"❌ App not found: {app_path}")
        return False
    
    # Run streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    print("=" * 60)
    print("Open your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\n\nStreamlit server stopped.")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="QARM - Quantitative Assets & Risk Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py              Run the Streamlit application
  python run.py --prepare    Prepare data files (run once before using the app)
  python run.py --check      Check configuration and data files status
  
First-time setup:
  1. Place 'stocktwits_optimized.parquet' in data/raw/
  2. Place 'prices_cleaned.parquet' in data/raw/ (if you have it)
  3. Run: python run.py --prepare
     - Downloads NYSE/NASDAQ stock & ETF list
     - Prepares sentiment data
     - Downloads S&P 500 and risk-free rate data
  4. Run: python run.py
        """
    )
    
    parser.add_argument(
        '--prepare', 
        action='store_true',
        help='Prepare data files (download S&P 500, aggregate sentiment, etc.)'
    )
    
    parser.add_argument(
        '--check',
        action='store_true', 
        help='Check configuration and data files status'
    )
    
    parser.add_argument(
        '--deps',
        action='store_true',
        help='Check if required packages are installed'
    )
    
    args = parser.parse_args()
    
    # Check dependencies first
    if args.deps:
        check_dependencies()
        return
    
    # Check configuration
    if args.check:
        check_config()
        return
    
    # Prepare data
    if args.prepare:
        if check_dependencies():
            prepare_data()
        return
    
    # Default: run Streamlit
    if not check_dependencies():
        print("\n⚠️  Please install missing packages first.")
        sys.exit(1)
    
    run_streamlit()


if __name__ == "__main__":
    main()
