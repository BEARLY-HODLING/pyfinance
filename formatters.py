"""
Formatters - Centralized formatting utilities for PyFinance Dashboard

This module provides:
- Currency formatting (£, $, etc.)
- Percentage formatting with optional sign
- Number formatting with compact notation
- Date/time formatting
"""

from typing import Optional, Union
from datetime import datetime, date
import pandas as pd


class Formatter:
    """Centralized formatting utilities"""

    @staticmethod
    def currency(
        value: Optional[float],
        symbol: str = "£",
        decimals: Optional[int] = None,
        compact: bool = False
    ) -> str:
        """Format value as currency.

        Args:
            value: Numeric value to format
            symbol: Currency symbol (default: £)
            decimals: Fixed decimal places (None = auto: 0 for >=1000, 2 otherwise)
            compact: Use compact notation (K, M, B)

        Returns:
            Formatted currency string or "—" if value is None/NaN

        Examples:
            >>> Formatter.currency(1234.56)
            '£1,235'
            >>> Formatter.currency(1234.56, symbol="$", decimals=2)
            '$1,234.56'
            >>> Formatter.currency(1500000, compact=True)
            '£1.5M'
        """
        if pd.isna(value) or value is None:
            return "—"

        if compact and abs(value) >= 1000:
            for suffix, divisor in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
                if abs(value) >= divisor:
                    return f"{symbol}{value/divisor:.1f}{suffix}"

        if decimals is None:
            decimals = 0 if abs(value) >= 1000 else 2

        return f"{symbol}{value:,.{decimals}f}"

    @staticmethod
    def percent(
        value: Optional[float],
        decimals: int = 1,
        show_sign: bool = True
    ) -> str:
        """Format value as percentage.

        Args:
            value: Numeric value (already in percentage form, e.g., 5.5 for 5.5%)
            decimals: Number of decimal places
            show_sign: Include +/- sign

        Returns:
            Formatted percentage string or "—" if value is None/NaN

        Examples:
            >>> Formatter.percent(5.234)
            '+5.2%'
            >>> Formatter.percent(-3.5, decimals=2)
            '-3.50%'
            >>> Formatter.percent(5.0, show_sign=False)
            '5.0%'
        """
        if pd.isna(value) or value is None:
            return "—"

        if show_sign:
            return f"{value:+.{decimals}f}%"
        return f"{value:.{decimals}f}%"

    @staticmethod
    def number(
        value: Optional[float],
        decimals: int = 0,
        compact: bool = False,
        show_sign: bool = False
    ) -> str:
        """Format number with optional compact notation.

        Args:
            value: Numeric value to format
            decimals: Number of decimal places
            compact: Use compact notation (K, M, B)
            show_sign: Include +/- sign

        Returns:
            Formatted number string or "—" if value is None/NaN

        Examples:
            >>> Formatter.number(1234567)
            '1,234,567'
            >>> Formatter.number(1500000, compact=True)
            '1.5M'
            >>> Formatter.number(42, show_sign=True)
            '+42'
        """
        if pd.isna(value) or value is None:
            return "—"

        if compact:
            for suffix, divisor in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
                if abs(value) >= divisor:
                    formatted = f"{value/divisor:.1f}{suffix}"
                    if show_sign and value > 0:
                        return f"+{formatted}"
                    return formatted

        if show_sign:
            return f"{value:+,.{decimals}f}"
        return f"{value:,.{decimals}f}"

    @staticmethod
    def shares(value: Optional[float], decimals: int = 4) -> str:
        """Format share quantity.

        Args:
            value: Number of shares
            decimals: Number of decimal places (default: 4 for fractional shares)

        Returns:
            Formatted share count or "—" if value is None/NaN
        """
        if pd.isna(value) or value is None:
            return "—"

        # Remove trailing zeros for cleaner display
        if value == int(value):
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}".rstrip('0').rstrip('.')

    @staticmethod
    def date_short(value: Optional[Union[datetime, date, str]]) -> str:
        """Format date in short form (DD Mon YYYY).

        Args:
            value: Date to format

        Returns:
            Formatted date string or "—" if value is None

        Examples:
            >>> Formatter.date_short(datetime(2026, 1, 15))
            '15 Jan 2026'
        """
        if value is None:
            return "—"

        if isinstance(value, str):
            try:
                value = pd.to_datetime(value)
            except Exception:
                return "—"

        if isinstance(value, (datetime, date)):
            return value.strftime("%d %b %Y")

        return "—"

    @staticmethod
    def date_relative(value: Optional[Union[datetime, date]]) -> str:
        """Format date as relative time (e.g., '2 days ago').

        Args:
            value: Date to format

        Returns:
            Relative time string or "—" if value is None
        """
        if value is None:
            return "—"

        if isinstance(value, str):
            try:
                value = pd.to_datetime(value)
            except Exception:
                return "—"

        now = datetime.now()
        if hasattr(value, 'tzinfo') and value.tzinfo is not None:
            value = value.replace(tzinfo=None)

        diff = now - value if isinstance(value, datetime) else now - datetime.combine(value, datetime.min.time())

        days = diff.days
        if days == 0:
            hours = diff.seconds // 3600
            if hours == 0:
                minutes = diff.seconds // 60
                if minutes == 0:
                    return "Just now"
                return f"{minutes} min ago"
            return f"{hours}h ago"
        elif days == 1:
            return "Yesterday"
        elif days < 7:
            return f"{days} days ago"
        elif days < 30:
            weeks = days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        elif days < 365:
            months = days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        else:
            years = days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"

    @staticmethod
    def time_short(value: Optional[datetime]) -> str:
        """Format time in short form (HH:MM).

        Args:
            value: Datetime to format

        Returns:
            Formatted time string or "—" if value is None
        """
        if value is None or not isinstance(value, datetime):
            return "—"
        return value.strftime("%H:%M")

    @staticmethod
    def datetime_full(value: Optional[datetime]) -> str:
        """Format datetime in full form (DD Mon YYYY HH:MM:SS).

        Args:
            value: Datetime to format

        Returns:
            Formatted datetime string or "—" if value is None
        """
        if value is None or not isinstance(value, datetime):
            return "—"
        return value.strftime("%d %b %Y %H:%M:%S")

    @staticmethod
    def ratio(value: Optional[float], decimals: int = 2) -> str:
        """Format ratio (e.g., Sharpe ratio, beta).

        Args:
            value: Ratio value
            decimals: Number of decimal places

        Returns:
            Formatted ratio or "—" if value is None/NaN
        """
        if pd.isna(value) or value is None:
            return "—"
        return f"{value:.{decimals}f}"

    @staticmethod
    def fx_rate(value: Optional[float]) -> str:
        """Format exchange rate with 4 decimal places.

        Args:
            value: FX rate value

        Returns:
            Formatted FX rate or "—" if value is None/NaN
        """
        if pd.isna(value) or value is None:
            return "—"
        return f"{value:.4f}"


# Convenience functions for backward compatibility
def format_currency(value: float, show_decimals: bool = True) -> str:
    """Format value as GBP currency."""
    return Formatter.currency(value, decimals=2 if show_decimals else 0)


def format_percent(value: float, show_sign: bool = True) -> str:
    """Format value as percentage."""
    return Formatter.percent(value, show_sign=show_sign)


def format_number(value: float, decimals: int = 0) -> str:
    """Format number with thousands separator."""
    return Formatter.number(value, decimals=decimals)
