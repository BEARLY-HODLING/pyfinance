"""
Theme Manager - Centralized theme and color management for PyFinance Dashboard

This module provides:
- ColorPalette: Centralized color definitions with semantic naming
- ThemeManager: CSS injection and theme state management
- Type-safe color access with transparency support
"""

from typing import Dict, Optional, Tuple
from functools import lru_cache
import streamlit as st


class ColorPalette:
    """Centralized color definitions with semantic naming"""

    # Base color palette
    BASE: Dict[str, str] = {
        'slate_900': '#0f172a',
        'slate_800': '#1e293b',
        'slate_700': '#334155',
        'slate_600': '#475569',
        'slate_500': '#64748b',
        'slate_400': '#94a3b8',
        'slate_300': '#cbd5e1',
        'slate_200': '#e2e8f0',
        'slate_100': '#f1f5f9',
        'slate_50': '#f8fafc',
        'white': '#ffffff',
        'green': '#22c55e',
        'red': '#ef4444',
        'blue': '#3498db',
        'purple': '#9b59b6',
        'orange': '#f97316',
        'yellow': '#fbbf24',
        'gold': '#d4af37',
        'gold_light': '#fbbf24',
    }

    # Semantic color mappings
    SEMANTIC: Dict[str, str] = {
        'success': '#22c55e',
        'danger': '#ef4444',
        'error': '#ef4444',
        'warning': '#f97316',
        'info': '#3498db',
        'primary': '#3498db',
        'accent': '#d4af37',
        'accent_secondary': '#fbbf24',
    }

    # Category colors for portfolio allocation
    CATEGORIES: Dict[str, str] = {
        'income': '#27ae60',
        'growth': '#3498db',
        'speculative': '#e74c3c',
        'unknown': '#95a5a6',
    }

    # Category color schemes (background, text, border)
    CATEGORY_SCHEMES: Dict[str, Dict[str, str]] = {
        'income': {'bg': '#1e3a5f', 'text': '#60a5fa', 'border': '#3b82f6'},
        'growth': {'bg': '#1a3d2e', 'text': '#4ade80', 'border': '#22c55e'},
        'speculative': {'bg': '#4a3728', 'text': '#fb923c', 'border': '#f97316'},
        'unknown': {'bg': '#374151', 'text': '#9ca3af', 'border': '#6b7280'},
    }

    # Chart color sequences
    CHART_COLORS: list = [
        '#00CC96', '#EF553B', '#636EFA', '#AB63FA',
        '#FFA15A', '#19D3F3', '#FF6692', '#B6E880'
    ]

    @classmethod
    def get(cls, color_key: str, alpha: float = 1.0) -> str:
        """Get color with optional alpha transparency.

        Args:
            color_key: Color name (e.g., 'success', 'danger', 'income')
            alpha: Transparency 0.0-1.0 (1.0 = opaque)

        Returns:
            Color string (hex if alpha=1.0, rgba if alpha<1.0)
        """
        # Try semantic first, then categories, then base
        color = (
            cls.SEMANTIC.get(color_key) or
            cls.CATEGORIES.get(color_key) or
            cls.BASE.get(color_key) or
            '#94a3b8'  # fallback gray
        )

        if alpha < 1.0:
            return cls.hex_to_rgba(color, alpha)
        return color

    @classmethod
    def hex_to_rgba(cls, hex_color: str, alpha: float = 1.0) -> str:
        """Convert hex color to rgba string.

        Args:
            hex_color: Hex color string (e.g., '#22c55e')
            alpha: Transparency 0.0-1.0

        Returns:
            RGBA color string (e.g., 'rgba(34, 197, 94, 0.5)')
        """
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'

    @classmethod
    def hex_to_rgb(cls, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple.

        Args:
            hex_color: Hex color string (e.g., '#22c55e')

        Returns:
            Tuple of (R, G, B) values
        """
        hex_color = hex_color.lstrip('#')
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16)
        )

    @classmethod
    def category_scheme(cls, category: str) -> Dict[str, str]:
        """Get full color scheme for a category badge.

        Args:
            category: Category name (income, growth, speculative)

        Returns:
            Dict with 'bg', 'text', 'border' keys
        """
        cat_key = category.lower() if category else 'unknown'
        return cls.CATEGORY_SCHEMES.get(cat_key, cls.CATEGORY_SCHEMES['unknown'])

    @classmethod
    def error_scheme(cls, error_type: str) -> Dict[str, str]:
        """Get color scheme for error display.

        Args:
            error_type: Type of error ('error', 'warning', 'info')

        Returns:
            Dict with 'border', 'background', 'header_bg', 'text' keys
        """
        base_color = cls.get(error_type)
        return {
            'border': base_color,
            'background': cls.hex_to_rgba(base_color, 0.1),
            'header_bg': cls.hex_to_rgba(base_color, 0.2),
            'text': cls.hex_to_rgba(base_color, 0.9),
        }


# Theme definitions
THEMES: Dict[str, Dict[str, str]] = {
    'dark': {
        'background': '#0f172a',
        'surface': '#1e293b',
        'text': '#f8fafc',
        'text_secondary': '#94a3b8',
        'accent': '#d4af37',
        'accent_secondary': '#fbbf24',
        'success': '#22c55e',
        'warning': '#f97316',
        'error': '#ef4444',
        'border': '#334155',
        'plotly_template': 'plotly_dark',
        'chart_colors': ColorPalette.CHART_COLORS,
    },
    'light': {
        'background': '#f8fafc',
        'surface': '#ffffff',
        'text': '#1e293b',
        'text_secondary': '#64748b',
        'accent': '#d4af37',
        'accent_secondary': '#b8860b',
        'success': '#16a34a',
        'warning': '#ea580c',
        'error': '#dc2626',
        'border': '#e2e8f0',
        'plotly_template': 'plotly_white',
        'chart_colors': ColorPalette.CHART_COLORS,
    }
}


class ThemeManager:
    """Centralized theme and CSS management"""

    _injected: set = set()  # Track what's been injected to avoid duplicates

    @classmethod
    def get_current_theme(cls) -> str:
        """Get the current theme name from session state.

        Returns:
            'dark' or 'light'
        """
        return st.session_state.get('theme', 'dark')

    @classmethod
    def get_theme_colors(cls) -> Dict[str, str]:
        """Get the full color dictionary for the current theme.

        Returns:
            Dict with all theme colors
        """
        return THEMES[cls.get_current_theme()]

    @classmethod
    def get_plotly_template(cls) -> str:
        """Get the Plotly template for the current theme.

        Returns:
            'plotly_dark' or 'plotly_white'
        """
        return THEMES[cls.get_current_theme()]['plotly_template']

    @classmethod
    def get_chart_colors(cls) -> list:
        """Get chart color palette for the current theme.

        Returns:
            List of hex color strings
        """
        return THEMES[cls.get_current_theme()]['chart_colors']

    @classmethod
    def inject_once(cls, css_key: str, css_content: str) -> None:
        """Inject CSS only once per session to avoid duplicates.

        Args:
            css_key: Unique identifier for this CSS block
            css_content: CSS string to inject
        """
        if css_key not in cls._injected:
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            cls._injected.add(css_key)

    @classmethod
    def inject_theme_variables(cls) -> None:
        """Inject CSS variables for the current theme."""
        theme = cls.get_current_theme()
        colors = THEMES[theme]

        css = f"""
        :root {{
            --bg-primary: {colors['background']};
            --bg-surface: {colors['surface']};
            --text-primary: {colors['text']};
            --text-secondary: {colors['text_secondary']};
            --accent: {colors['accent']};
            --accent-secondary: {colors['accent_secondary']};
            --success: {colors['success']};
            --warning: {colors['warning']};
            --error: {colors['error']};
            --border: {colors['border']};
        }}
        """
        cls.inject_once(f'theme_variables_{theme}', css)

    @classmethod
    def reset_injection_state(cls) -> None:
        """Reset the injection tracking (useful for theme changes)."""
        cls._injected.clear()


# Convenience functions for backward compatibility
def get_current_theme() -> str:
    """Get the current theme name."""
    return ThemeManager.get_current_theme()


def get_theme_colors() -> Dict[str, str]:
    """Get the full color dictionary for the current theme."""
    return ThemeManager.get_theme_colors()


def get_plotly_template() -> str:
    """Get the Plotly template for the current theme."""
    return ThemeManager.get_plotly_template()


def get_chart_colors() -> list:
    """Get chart color palette for the current theme."""
    return ThemeManager.get_chart_colors()


def get_category_color(category: str) -> Dict[str, str]:
    """Get color scheme for a category.

    Args:
        category: Category name (income, growth, speculative)

    Returns:
        Dict with 'bg', 'text', 'border' keys
    """
    return ColorPalette.category_scheme(category)
