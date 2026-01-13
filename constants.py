"""
Constants - Magic numbers and configuration constants for PyFinance Dashboard

This module provides centralized definitions for:
- Portfolio analysis thresholds
- UI configuration values
- API settings
- Cache timeouts
"""

from typing import Dict, List

# =============================================================================
# PORTFOLIO ANALYSIS THRESHOLDS
# =============================================================================

# Concentration risk thresholds (percentage of portfolio)
CONCENTRATION_CRITICAL: float = 25.0  # Severe concentration warning
CONCENTRATION_HIGH: float = 20.0  # High concentration warning
CONCENTRATION_THRESHOLD: float = 15.0  # Standard concentration alert

# Volatility thresholds (annualized)
VOLATILITY_HIGH: float = 30.0  # High volatility warning
VOLATILITY_MODERATE: float = 20.0  # Moderate volatility

# Beta thresholds
BETA_HIGH: float = 1.5  # High market sensitivity
BETA_LOW: float = 0.5  # Low market sensitivity

# Yield thresholds (percentage)
YIELD_LOW: float = 1.0  # Low yield alert
YIELD_HIGH: float = 10.0  # Unusually high yield (may indicate risk)

# Dividend growth thresholds
DIVIDEND_GROWTH_GOOD: float = 5.0  # Good dividend growth
DIVIDEND_GROWTH_EXCELLENT: float = 10.0  # Excellent dividend growth

# Rebalancing thresholds
REBALANCE_DRIFT_MINOR: float = 2.0  # Minor drift, monitor only
REBALANCE_DRIFT_MODERATE: float = 5.0  # Moderate drift, consider action
REBALANCE_DRIFT_MAJOR: float = 10.0  # Major drift, action recommended

# Target allocation tolerance
TARGET_ALLOCATION_TOLERANCE: float = 0.1  # 0.1% tolerance when validating 100%

# =============================================================================
# INSIGHT SEVERITY LEVELS
# =============================================================================

SEVERITY_CRITICAL: int = 1
SEVERITY_HIGH: int = 2
SEVERITY_MEDIUM: int = 3
SEVERITY_LOW: int = 4
SEVERITY_INFO: int = 5

# =============================================================================
# CACHE TIMEOUTS (seconds)
# =============================================================================

CACHE_TTL_PRICES: int = 300  # 5 minutes for price data
CACHE_TTL_DATA: int = 900  # 15 minutes for CSV/transaction data
CACHE_TTL_FX: int = 900  # 15 minutes for FX rates
CACHE_TTL_BENCHMARK: int = 900  # 15 minutes for benchmark data
CACHE_TTL_DIVIDENDS: int = 3600  # 1 hour for dividend calendar data

# =============================================================================
# API CONFIGURATION
# =============================================================================

# Yahoo Finance
YFINANCE_MAX_WORKERS: int = 10  # Max parallel API calls
YFINANCE_TIMEOUT: int = 10  # Timeout in seconds
YFINANCE_MAX_RETRIES: int = 3  # Number of retry attempts
YFINANCE_RETRY_DELAY: float = 1.0  # Delay between retries (seconds)

# Rate limiting
API_RATE_LIMIT_CALLS: int = 100  # Max calls per window
API_RATE_LIMIT_WINDOW: int = 60  # Window size in seconds

# =============================================================================
# UI CONFIGURATION
# =============================================================================

# Table display
TABLE_MAX_ROWS_DEFAULT: int = 50
TABLE_PAGE_SIZE: int = 20

# Chart dimensions
CHART_HEIGHT_DEFAULT: int = 400
CHART_HEIGHT_SMALL: int = 250
CHART_HEIGHT_LARGE: int = 600

# Sparkline dimensions
SPARKLINE_WIDTH: int = 120
SPARKLINE_HEIGHT: int = 36

# Animation durations (milliseconds)
ANIMATION_FAST: int = 150
ANIMATION_NORMAL: int = 300
ANIMATION_SLOW: int = 500

# Auto-refresh interval (seconds)
AUTO_REFRESH_INTERVAL: int = 300  # 5 minutes

# =============================================================================
# INSIGHT CONFIGURATION
# =============================================================================

# Maximum insights to display
MAX_INSIGHTS_DISPLAY: int = 5
MAX_INSIGHTS_GENERATE: int = 20

# Days for recent activity analysis
RECENT_ACTIVITY_DAYS: int = 30
DIVIDEND_LOOKBACK_DAYS: int = 365

# =============================================================================
# GOAL TRACKING
# =============================================================================

GOAL_PROGRESS_EXCELLENT: float = 100.0
GOAL_PROGRESS_GOOD: float = 75.0
GOAL_PROGRESS_MODERATE: float = 50.0
GOAL_PROGRESS_LOW: float = 25.0

# =============================================================================
# BENCHMARKS
# =============================================================================

BENCHMARK_TICKERS: Dict[str, str] = {
    'S&P 500': '^GSPC',
    'FTSE 100': '^FTSE',
    'FTSE All-Share': '^FTAS',
    'NASDAQ': '^IXIC',
    'DAX': '^GDAXI',
}

DEFAULT_BENCHMARK: str = '^GSPC'  # S&P 500

# =============================================================================
# FILE PATTERNS
# =============================================================================

ISA_FILE_PATTERN: str = "freetrade_ISA_*.csv"
SIPP_FILE_PATTERN: str = "freetrade_SIPP_*.csv"
CONFIG_FILE: str = "config.json"
GOALS_FILE: str = "goals.json"
WATCHLIST_FILE: str = "watchlist.json"

# =============================================================================
# SUPPORTED CATEGORIES
# =============================================================================

PORTFOLIO_CATEGORIES: List[str] = ['income', 'growth', 'speculative']

DEFAULT_CATEGORY_TARGETS: Dict[str, float] = {
    'income': 50.0,
    'growth': 35.0,
    'speculative': 15.0,
}

# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

EXPORT_DATE_FORMAT: str = "%Y-%m-%d"
EXPORT_DATETIME_FORMAT: str = "%Y-%m-%d_%H%M%S"
PDF_PAGE_SIZE: str = "A4"

# =============================================================================
# VALIDATION LIMITS
# =============================================================================

MIN_ALLOCATION_PERCENT: float = 0.0
MAX_ALLOCATION_PERCENT: float = 100.0
MIN_SHARES: float = 0.0001  # Minimum fractional shares
MAX_TICKER_LENGTH: int = 20
MAX_CATEGORY_LENGTH: int = 50
