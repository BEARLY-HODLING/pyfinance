"""
Validation - Input validation utilities for PyFinance Dashboard

This module provides:
- Input validation for user inputs
- Data sanitization for CSV imports
- Configuration validation
"""

import re
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

from constants import (
    MIN_ALLOCATION_PERCENT,
    MAX_ALLOCATION_PERCENT,
    MAX_TICKER_LENGTH,
    MAX_CATEGORY_LENGTH,
    PORTFOLIO_CATEGORIES,
    TARGET_ALLOCATION_TOLERANCE,
)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class Validator:
    """Input validation utilities"""

    # Ticker symbol pattern (alphanumeric with optional suffixes like .L)
    TICKER_PATTERN = re.compile(r'^[A-Z0-9]{1,10}(\.[A-Z]{1,4})?$', re.IGNORECASE)

    # API key patterns
    OPENAI_KEY_PATTERN = re.compile(r'^sk-[a-zA-Z0-9]{32,}$')

    @classmethod
    def validate_ticker(cls, ticker: str) -> Tuple[bool, Optional[str]]:
        """Validate a ticker symbol.

        Args:
            ticker: Ticker symbol to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not ticker:
            return False, "Ticker cannot be empty"

        if len(ticker) > MAX_TICKER_LENGTH:
            return False, f"Ticker too long (max {MAX_TICKER_LENGTH} characters)"

        ticker = ticker.strip().upper()

        if not cls.TICKER_PATTERN.match(ticker):
            return False, "Invalid ticker format (use letters/numbers, optional .L suffix)"

        return True, None

    @classmethod
    def validate_allocation(cls, value: float, field_name: str = "Allocation") -> Tuple[bool, Optional[str]]:
        """Validate an allocation percentage.

        Args:
            value: Percentage value
            field_name: Name for error messages

        Returns:
            Tuple of (is_valid, error_message)
        """
        if pd.isna(value):
            return False, f"{field_name} cannot be empty"

        if value < MIN_ALLOCATION_PERCENT:
            return False, f"{field_name} cannot be negative"

        if value > MAX_ALLOCATION_PERCENT:
            return False, f"{field_name} cannot exceed 100%"

        return True, None

    @classmethod
    def validate_category_targets(
        cls,
        targets: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        """Validate category targets sum to 100%.

        Args:
            targets: Dictionary of category -> percentage

        Returns:
            Tuple of (is_valid, error_message)
        """
        total = sum(targets.values())

        if abs(total - 100) > TARGET_ALLOCATION_TOLERANCE:
            return False, f"Category targets must sum to 100% (currently {total:.1f}%)"

        for category, value in targets.items():
            is_valid, error = cls.validate_allocation(value, f"{category} target")
            if not is_valid:
                return False, error

        return True, None

    @classmethod
    def validate_category(cls, category: str) -> Tuple[bool, Optional[str]]:
        """Validate a category name.

        Args:
            category: Category name

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not category:
            return False, "Category cannot be empty"

        if len(category) > MAX_CATEGORY_LENGTH:
            return False, f"Category too long (max {MAX_CATEGORY_LENGTH} characters)"

        category_lower = category.lower()
        if category_lower not in PORTFOLIO_CATEGORIES:
            valid_cats = ', '.join(PORTFOLIO_CATEGORIES)
            return False, f"Invalid category. Must be one of: {valid_cats}"

        return True, None

    @classmethod
    def validate_api_key(
        cls,
        key: str,
        key_type: str = "openai"
    ) -> Tuple[bool, Optional[str]]:
        """Validate an API key format (not validity).

        Args:
            key: API key to validate
            key_type: Type of API key ('openai')

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not key:
            return True, None  # Empty is allowed (optional)

        key = key.strip()

        if key_type == "openai":
            if not cls.OPENAI_KEY_PATTERN.match(key):
                return False, "Invalid OpenAI API key format (should start with 'sk-')"

        return True, None

    @classmethod
    def validate_positive_number(
        cls,
        value: float,
        field_name: str = "Value"
    ) -> Tuple[bool, Optional[str]]:
        """Validate a positive number.

        Args:
            value: Number to validate
            field_name: Name for error messages

        Returns:
            Tuple of (is_valid, error_message)
        """
        if pd.isna(value):
            return False, f"{field_name} cannot be empty"

        if value < 0:
            return False, f"{field_name} must be positive"

        return True, None

    @classmethod
    def validate_proxy_yield(cls, value: float) -> Tuple[bool, Optional[str]]:
        """Validate a proxy yield value.

        Args:
            value: Yield as decimal (e.g., 0.05 for 5%)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if pd.isna(value):
            return False, "Yield cannot be empty"

        if value < 0:
            return False, "Yield cannot be negative"

        if value > 1.0:
            return False, "Yield should be a decimal (e.g., 0.05 for 5%)"

        return True, None


class DataSanitizer:
    """Data sanitization utilities for CSV imports"""

    @classmethod
    def sanitize_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize a DataFrame from CSV import.

        Args:
            df: DataFrame to sanitize

        Returns:
            Sanitized DataFrame
        """
        if df is None or df.empty:
            return df

        # Strip whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]

        return df

    @classmethod
    def sanitize_ticker(cls, ticker: str) -> str:
        """Sanitize a ticker symbol.

        Args:
            ticker: Ticker to sanitize

        Returns:
            Sanitized ticker (uppercase, stripped)
        """
        if not ticker:
            return ""
        return str(ticker).strip().upper()

    @classmethod
    def sanitize_category(cls, category: str) -> str:
        """Sanitize a category name.

        Args:
            category: Category to sanitize

        Returns:
            Sanitized category (lowercase, stripped)
        """
        if not category:
            return "unknown"
        return str(category).strip().lower()

    @classmethod
    def sanitize_numeric(
        cls,
        value: Any,
        default: float = 0.0
    ) -> float:
        """Sanitize a numeric value.

        Args:
            value: Value to sanitize
            default: Default if conversion fails

        Returns:
            Sanitized float value
        """
        if pd.isna(value):
            return default

        try:
            return float(value)
        except (ValueError, TypeError):
            return default


class ConfigValidator:
    """Configuration file validation"""

    REQUIRED_KEYS: List[str] = []  # No strictly required keys
    OPTIONAL_KEYS: List[str] = [
        'yahoo_ticker_map',
        'category_map',
        'proxy_yields',
        'category_targets',
        'target_allocations',
        'theme',
        'openai_api_key',
    ]

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check for required keys
        for key in cls.REQUIRED_KEYS:
            if key not in config:
                errors.append(f"Missing required key: {key}")

        # Validate yahoo_ticker_map
        if 'yahoo_ticker_map' in config:
            ticker_map = config['yahoo_ticker_map']
            if not isinstance(ticker_map, dict):
                errors.append("yahoo_ticker_map must be a dictionary")
            else:
                for freetrade_ticker, yahoo_ticker in ticker_map.items():
                    is_valid, error = Validator.validate_ticker(freetrade_ticker)
                    if not is_valid:
                        errors.append(f"Invalid Freetrade ticker '{freetrade_ticker}': {error}")
                    is_valid, error = Validator.validate_ticker(yahoo_ticker)
                    if not is_valid:
                        errors.append(f"Invalid Yahoo ticker '{yahoo_ticker}': {error}")

        # Validate category_map
        if 'category_map' in config:
            cat_map = config['category_map']
            if not isinstance(cat_map, dict):
                errors.append("category_map must be a dictionary")
            else:
                for ticker, category in cat_map.items():
                    is_valid, error = Validator.validate_category(category)
                    if not is_valid:
                        errors.append(f"Invalid category for '{ticker}': {error}")

        # Validate category_targets
        if 'category_targets' in config:
            targets = config['category_targets']
            if isinstance(targets, dict):
                is_valid, error = Validator.validate_category_targets(targets)
                if not is_valid:
                    errors.append(error)

        # Validate theme
        if 'theme' in config:
            if config['theme'] not in ['dark', 'light']:
                errors.append("theme must be 'dark' or 'light'")

        # Validate API key format (not validity)
        if 'openai_api_key' in config and config['openai_api_key']:
            is_valid, error = Validator.validate_api_key(config['openai_api_key'])
            if not is_valid:
                errors.append(error)

        return len(errors) == 0, errors
