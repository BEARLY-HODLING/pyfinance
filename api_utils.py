"""
API Utilities - Retry logic and rate limiting for external API calls

This module provides:
- Retry decorator with exponential backoff
- Rate limiting utilities
- Safe API wrapper functions
"""

import os
import time
import logging
from typing import TypeVar, Callable, Optional, Any, Dict
from functools import wraps
from datetime import datetime, timedelta
import yfinance as yf

from constants import (
    YFINANCE_MAX_RETRIES,
    YFINANCE_RETRY_DELAY,
    YFINANCE_TIMEOUT,
    API_RATE_LIMIT_CALLS,
    API_RATE_LIMIT_WINDOW,
)

# Set up logging
logger = logging.getLogger(__name__)

T = TypeVar('T')


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, max_calls: int = API_RATE_LIMIT_CALLS, window_seconds: int = API_RATE_LIMIT_WINDOW):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls: list = []

    def can_proceed(self) -> bool:
        """Check if we can make another API call."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)

        # Remove old calls
        self.calls = [t for t in self.calls if t > cutoff]

        return len(self.calls) < self.max_calls

    def record_call(self) -> None:
        """Record an API call."""
        self.calls.append(datetime.now())

    def wait_time(self) -> float:
        """Get time to wait before next call is allowed."""
        if self.can_proceed():
            return 0.0

        oldest_call = min(self.calls)
        wait = (oldest_call + timedelta(seconds=self.window_seconds) - datetime.now()).total_seconds()
        return max(0.0, wait)


# Global rate limiter instance
_rate_limiter = RateLimiter()


def retry_with_backoff(
    max_retries: int = YFINANCE_MAX_RETRIES,
    initial_delay: float = YFINANCE_RETRY_DELAY,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


def safe_execute(
    operation: Callable[[], T],
    context: str,
    default: Optional[T] = None,
    log_error: bool = True
) -> T:
    """Execute operation with automatic error handling.

    Args:
        operation: Function to execute
        context: Human-readable description for logging
        default: Return value on error
        log_error: Whether to log errors

    Returns:
        Operation result or default on error
    """
    try:
        return operation()
    except Exception as e:
        if log_error:
            logger.warning(f"Error in {context}: {type(e).__name__} - {str(e)}")
        return default


@retry_with_backoff(max_retries=YFINANCE_MAX_RETRIES)
def fetch_ticker_info(yahoo_ticker: str) -> Dict[str, Any]:
    """Fetch ticker info from Yahoo Finance with retry logic.

    Args:
        yahoo_ticker: Yahoo Finance ticker symbol

    Returns:
        Ticker info dictionary or empty dict on failure
    """
    # Rate limiting
    if not _rate_limiter.can_proceed():
        wait_time = _rate_limiter.wait_time()
        if wait_time > 0:
            logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
            time.sleep(wait_time)

    _rate_limiter.record_call()

    ticker = yf.Ticker(yahoo_ticker)
    return ticker.info or {}


@retry_with_backoff(max_retries=YFINANCE_MAX_RETRIES)
def fetch_ticker_history(
    yahoo_ticker: str,
    period: str = "1y",
    interval: str = "1d"
) -> Any:
    """Fetch ticker price history from Yahoo Finance with retry logic.

    Args:
        yahoo_ticker: Yahoo Finance ticker symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

    Returns:
        DataFrame with price history
    """
    # Rate limiting
    if not _rate_limiter.can_proceed():
        wait_time = _rate_limiter.wait_time()
        if wait_time > 0:
            logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
            time.sleep(wait_time)

    _rate_limiter.record_call()

    ticker = yf.Ticker(yahoo_ticker)
    return ticker.history(period=period, interval=interval)


@retry_with_backoff(max_retries=YFINANCE_MAX_RETRIES)
def fetch_ticker_dividends(yahoo_ticker: str) -> Any:
    """Fetch dividend history from Yahoo Finance with retry logic.

    Args:
        yahoo_ticker: Yahoo Finance ticker symbol

    Returns:
        Series with dividend history
    """
    # Rate limiting
    if not _rate_limiter.can_proceed():
        wait_time = _rate_limiter.wait_time()
        if wait_time > 0:
            time.sleep(wait_time)

    _rate_limiter.record_call()

    ticker = yf.Ticker(yahoo_ticker)
    return ticker.dividends


def get_api_key(key_name: str, config: Optional[Dict] = None) -> Optional[str]:
    """Get API key from environment variable or config.

    Priority:
    1. Environment variable (most secure)
    2. Config dictionary (less secure)

    Args:
        key_name: Name of the API key (e.g., 'openai_api_key')
        config: Optional config dictionary

    Returns:
        API key string or None if not found
    """
    # Environment variable takes priority (more secure)
    env_key = key_name.upper()
    if env_key in os.environ:
        return os.environ[env_key]

    # Fall back to config (less secure, but user-friendly)
    if config and key_name in config:
        key = config[key_name]
        if key and key.strip():
            return key.strip()

    return None


def mask_api_key(key: str, visible_chars: int = 4) -> str:
    """Mask an API key for display.

    Args:
        key: API key to mask
        visible_chars: Number of characters to show at start and end

    Returns:
        Masked key (e.g., 'sk-ab...xyz')
    """
    if not key:
        return ""

    if len(key) <= visible_chars * 2:
        return "*" * len(key)

    return f"{key[:visible_chars]}...{key[-visible_chars:]}"
