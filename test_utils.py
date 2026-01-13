"""
Unit tests for PyFinance utility modules

Run with: python -m pytest test_utils.py -v
"""

import pytest
import pandas as pd
from datetime import datetime, date

# Import modules to test
from formatters import Formatter, format_currency, format_percent
from validation import Validator, DataSanitizer, ConfigValidator, ValidationError
from constants import (
    CONCENTRATION_THRESHOLD,
    PORTFOLIO_CATEGORIES,
    TARGET_ALLOCATION_TOLERANCE,
)
from theme_manager import ColorPalette, ThemeManager, THEMES


# =============================================================================
# FORMATTER TESTS
# =============================================================================

class TestFormatter:
    """Tests for the Formatter class"""

    def test_currency_basic(self):
        """Test basic currency formatting"""
        assert Formatter.currency(1234.56) == "£1,235"
        assert Formatter.currency(99.99) == "£99.99"
        assert Formatter.currency(0) == "£0.00"

    def test_currency_with_decimals(self):
        """Test currency with explicit decimals"""
        assert Formatter.currency(1234.567, decimals=2) == "£1,234.57"
        assert Formatter.currency(1234.567, decimals=0) == "£1,235"

    def test_currency_with_symbol(self):
        """Test currency with different symbols"""
        assert Formatter.currency(100, symbol="$") == "$100.00"
        assert Formatter.currency(100, symbol="€") == "€100.00"

    def test_currency_compact(self):
        """Test compact currency notation"""
        assert Formatter.currency(1500000, compact=True) == "£1.5M"
        assert Formatter.currency(2500, compact=True) == "£2.5K"
        assert Formatter.currency(1500000000, compact=True) == "£1.5B"

    def test_currency_none_nan(self):
        """Test currency with None/NaN values"""
        assert Formatter.currency(None) == "—"
        assert Formatter.currency(float('nan')) == "—"

    def test_percent_basic(self):
        """Test basic percentage formatting"""
        assert Formatter.percent(5.5) == "+5.5%"
        assert Formatter.percent(-3.2) == "-3.2%"
        assert Formatter.percent(0) == "+0.0%"

    def test_percent_no_sign(self):
        """Test percentage without sign"""
        assert Formatter.percent(5.5, show_sign=False) == "5.5%"
        assert Formatter.percent(-3.2, show_sign=False) == "-3.2%"

    def test_percent_decimals(self):
        """Test percentage with different decimal places"""
        assert Formatter.percent(5.567, decimals=2) == "+5.57%"
        assert Formatter.percent(5.567, decimals=0) == "+6%"

    def test_percent_none_nan(self):
        """Test percentage with None/NaN values"""
        assert Formatter.percent(None) == "—"
        assert Formatter.percent(float('nan')) == "—"

    def test_number_basic(self):
        """Test basic number formatting"""
        assert Formatter.number(1234567) == "1,234,567"
        assert Formatter.number(1234.567, decimals=2) == "1,234.57"

    def test_number_compact(self):
        """Test compact number notation"""
        assert Formatter.number(1500000, compact=True) == "1.5M"
        assert Formatter.number(2500, compact=True) == "2.5K"

    def test_shares_formatting(self):
        """Test share quantity formatting"""
        assert Formatter.shares(100) == "100"
        assert Formatter.shares(10.5) == "10.5"
        assert Formatter.shares(10.12345) == "10.1235"

    def test_date_short(self):
        """Test short date formatting"""
        dt = datetime(2026, 1, 15)
        assert Formatter.date_short(dt) == "15 Jan 2026"

    def test_date_relative(self):
        """Test relative date formatting"""
        now = datetime.now()
        assert Formatter.date_relative(now) == "Just now"

    def test_fx_rate(self):
        """Test FX rate formatting"""
        assert Formatter.fx_rate(1.3456) == "1.3456"
        assert Formatter.fx_rate(1.34567) == "1.3457"


class TestBackwardCompatibility:
    """Test backward compatibility functions"""

    def test_format_currency(self):
        """Test legacy format_currency function"""
        assert format_currency(1234.56) == "£1,234.56"

    def test_format_percent(self):
        """Test legacy format_percent function"""
        assert format_percent(5.5) == "+5.5%"


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidator:
    """Tests for the Validator class"""

    def test_validate_ticker_valid(self):
        """Test valid ticker validation"""
        is_valid, error = Validator.validate_ticker("VUSA")
        assert is_valid
        assert error is None

        is_valid, error = Validator.validate_ticker("VUSA.L")
        assert is_valid

    def test_validate_ticker_invalid(self):
        """Test invalid ticker validation"""
        is_valid, error = Validator.validate_ticker("")
        assert not is_valid
        assert "empty" in error.lower()

        is_valid, error = Validator.validate_ticker("A" * 25)
        assert not is_valid
        assert "long" in error.lower()

    def test_validate_allocation_valid(self):
        """Test valid allocation validation"""
        is_valid, error = Validator.validate_allocation(50.0)
        assert is_valid
        assert error is None

    def test_validate_allocation_invalid(self):
        """Test invalid allocation validation"""
        is_valid, error = Validator.validate_allocation(-5.0)
        assert not is_valid
        assert "negative" in error.lower()

        is_valid, error = Validator.validate_allocation(150.0)
        assert not is_valid
        assert "100" in error

    def test_validate_category_targets(self):
        """Test category targets validation"""
        # Valid targets
        is_valid, error = Validator.validate_category_targets({
            'income': 50.0,
            'growth': 35.0,
            'speculative': 15.0
        })
        assert is_valid

        # Invalid (doesn't sum to 100)
        is_valid, error = Validator.validate_category_targets({
            'income': 50.0,
            'growth': 30.0,
            'speculative': 15.0
        })
        assert not is_valid
        assert "100" in error

    def test_validate_category_valid(self):
        """Test valid category validation"""
        for cat in PORTFOLIO_CATEGORIES:
            is_valid, error = Validator.validate_category(cat)
            assert is_valid

    def test_validate_category_invalid(self):
        """Test invalid category validation"""
        is_valid, error = Validator.validate_category("invalid_cat")
        assert not is_valid

    def test_validate_proxy_yield(self):
        """Test proxy yield validation"""
        is_valid, error = Validator.validate_proxy_yield(0.05)
        assert is_valid

        is_valid, error = Validator.validate_proxy_yield(1.5)
        assert not is_valid
        assert "decimal" in error.lower()


class TestDataSanitizer:
    """Tests for the DataSanitizer class"""

    def test_sanitize_ticker(self):
        """Test ticker sanitization"""
        assert DataSanitizer.sanitize_ticker("  vusa  ") == "VUSA"
        assert DataSanitizer.sanitize_ticker("VUSA.L") == "VUSA.L"
        assert DataSanitizer.sanitize_ticker("") == ""

    def test_sanitize_category(self):
        """Test category sanitization"""
        assert DataSanitizer.sanitize_category("  Income  ") == "income"
        assert DataSanitizer.sanitize_category("GROWTH") == "growth"
        assert DataSanitizer.sanitize_category("") == "unknown"

    def test_sanitize_numeric(self):
        """Test numeric sanitization"""
        assert DataSanitizer.sanitize_numeric(123.45) == 123.45
        assert DataSanitizer.sanitize_numeric("123.45") == 123.45
        assert DataSanitizer.sanitize_numeric("invalid", default=0.0) == 0.0
        assert DataSanitizer.sanitize_numeric(None, default=0.0) == 0.0


class TestConfigValidator:
    """Tests for the ConfigValidator class"""

    def test_validate_config_valid(self):
        """Test valid config validation"""
        config = {
            "yahoo_ticker_map": {"VUSA": "VUSA.L"},
            "category_map": {"VUSA": "growth"},
            "theme": "dark"
        }
        is_valid, errors = ConfigValidator.validate_config(config)
        assert is_valid
        assert len(errors) == 0

    def test_validate_config_invalid_theme(self):
        """Test config with invalid theme"""
        config = {"theme": "invalid"}
        is_valid, errors = ConfigValidator.validate_config(config)
        assert not is_valid
        assert any("theme" in e.lower() for e in errors)


# =============================================================================
# COLOR PALETTE TESTS
# =============================================================================

class TestColorPalette:
    """Tests for the ColorPalette class"""

    def test_get_semantic_color(self):
        """Test getting semantic colors"""
        success = ColorPalette.get('success')
        assert success.startswith('#')
        assert len(success) == 7

    def test_get_with_alpha(self):
        """Test getting color with alpha"""
        color = ColorPalette.get('success', alpha=0.5)
        assert color.startswith('rgba(')
        assert '0.5' in color

    def test_hex_to_rgba(self):
        """Test hex to rgba conversion"""
        rgba = ColorPalette.hex_to_rgba('#22c55e', 0.5)
        assert rgba == 'rgba(34, 197, 94, 0.5)'

    def test_category_scheme(self):
        """Test category color scheme"""
        scheme = ColorPalette.category_scheme('income')
        assert 'bg' in scheme
        assert 'text' in scheme
        assert 'border' in scheme

    def test_error_scheme(self):
        """Test error color scheme"""
        scheme = ColorPalette.error_scheme('error')
        assert 'border' in scheme
        assert 'background' in scheme


# =============================================================================
# THEME MANAGER TESTS
# =============================================================================

class TestThemeManager:
    """Tests for the ThemeManager class"""

    def test_themes_exist(self):
        """Test that both themes are defined"""
        assert 'dark' in THEMES
        assert 'light' in THEMES

    def test_theme_has_required_keys(self):
        """Test that themes have all required keys"""
        required_keys = [
            'background', 'surface', 'text', 'text_secondary',
            'accent', 'success', 'error', 'plotly_template'
        ]
        for theme_name, theme in THEMES.items():
            for key in required_keys:
                assert key in theme, f"Theme '{theme_name}' missing key '{key}'"

    def test_chart_colors_exist(self):
        """Test that chart colors are defined"""
        assert len(ColorPalette.CHART_COLORS) > 0
        for color in ColorPalette.CHART_COLORS:
            assert color.startswith('#')


# =============================================================================
# CONSTANTS TESTS
# =============================================================================

class TestConstants:
    """Tests for constants"""

    def test_threshold_values(self):
        """Test that threshold values are reasonable"""
        assert 0 < CONCENTRATION_THRESHOLD < 100
        assert TARGET_ALLOCATION_TOLERANCE > 0
        assert TARGET_ALLOCATION_TOLERANCE < 1

    def test_categories_defined(self):
        """Test portfolio categories are defined"""
        assert len(PORTFOLIO_CATEGORIES) >= 3
        assert 'income' in PORTFOLIO_CATEGORIES
        assert 'growth' in PORTFOLIO_CATEGORIES


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
