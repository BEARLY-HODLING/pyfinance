"""
HTML Components - Type-safe HTML component builder for PyFinance Dashboard

This module provides:
- HTMLBuilder: Centralized HTML rendering
- Common UI components (cards, badges, alerts)
- Reduces duplication in st.markdown calls
"""

from typing import Optional, Dict, List, Any
import streamlit as st

from theme_manager import ColorPalette, ThemeManager


class HTMLBuilder:
    """Type-safe HTML component builder"""

    @staticmethod
    def render(html: str) -> None:
        """Render HTML with unsafe_allow_html (single point of control).

        Args:
            html: HTML string to render
        """
        st.markdown(html, unsafe_allow_html=True)

    @staticmethod
    def render_css(css: str, key: Optional[str] = None) -> None:
        """Render CSS styles.

        Args:
            css: CSS string to inject
            key: Optional key for deduplication
        """
        if key:
            ThemeManager.inject_once(key, css)
        else:
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    @classmethod
    def card(
        cls,
        content: str,
        header: Optional[str] = None,
        css_class: str = "",
        style: Optional[Dict[str, str]] = None,
        icon: Optional[str] = None
    ) -> None:
        """Render a card component.

        Args:
            content: Card body HTML
            header: Optional header text
            css_class: CSS class name(s)
            style: Optional inline styles dict
            icon: Optional icon emoji
        """
        style_str = ""
        if style:
            style_str = "; ".join([f"{k}: {v}" for k, v in style.items()])

        header_html = ""
        if header:
            icon_html = f'<span class="card-icon">{icon}</span>' if icon else ''
            header_html = f'<div class="card-header">{icon_html}{header}</div>'

        html = f"""
        <div class="card {css_class}" style="{style_str}">
            {header_html}
            <div class="card-body">{content}</div>
        </div>
        """
        cls.render(html)

    @classmethod
    def metric_card(
        cls,
        title: str,
        value: str,
        icon: str = "",
        delta: Optional[str] = None,
        delta_positive: bool = True,
        accent: str = "blue",
        detail: Optional[str] = None
    ) -> None:
        """Render premium metric card.

        Args:
            title: Metric title
            value: Metric value
            icon: Icon emoji
            delta: Change value
            delta_positive: Whether delta is positive (affects color)
            accent: Accent color name
            detail: Additional detail text
        """
        colors = ThemeManager.get_theme_colors()
        accent_color = ColorPalette.get(accent)

        icon_html = f'<span class="metric-icon">{icon}</span>' if icon else ''

        delta_html = ""
        if delta:
            delta_color = colors['success'] if delta_positive else colors['error']
            delta_html = f'<div class="metric-delta" style="color: {delta_color}">{delta}</div>'

        detail_html = ""
        if detail:
            detail_html = f'<div class="metric-detail">{detail}</div>'

        html = f"""
        <div class="premium-metric-card" style="border-left: 3px solid {accent_color}">
            <div class="metric-header">
                <span class="metric-title">{title}</span>
                {icon_html}
            </div>
            <div class="metric-value">{value}</div>
            {delta_html}
            {detail_html}
        </div>
        """
        cls.render(html)

    @classmethod
    def badge(
        cls,
        text: str,
        variant: str = "default",
        size: str = "normal"
    ) -> str:
        """Generate badge HTML (returns string, doesn't render).

        Args:
            text: Badge text
            variant: Color variant (success, warning, error, info, default)
            size: Size variant (small, normal, large)

        Returns:
            Badge HTML string
        """
        colors = {
            'success': ColorPalette.get('success'),
            'warning': ColorPalette.get('warning'),
            'error': ColorPalette.get('error'),
            'info': ColorPalette.get('info'),
            'default': ColorPalette.get('slate_500'),
        }
        color = colors.get(variant, colors['default'])
        bg_color = ColorPalette.hex_to_rgba(color, 0.2)

        sizes = {
            'small': 'font-size: 0.7rem; padding: 2px 6px;',
            'normal': 'font-size: 0.8rem; padding: 4px 8px;',
            'large': 'font-size: 0.9rem; padding: 6px 12px;',
        }
        size_style = sizes.get(size, sizes['normal'])

        return f"""<span style="
            background: {bg_color};
            color: {color};
            border-radius: 4px;
            {size_style}
            font-weight: 500;
        ">{text}</span>"""

    @classmethod
    def category_badge(cls, category: str) -> str:
        """Generate category badge HTML.

        Args:
            category: Category name (income, growth, speculative)

        Returns:
            Category badge HTML string
        """
        scheme = ColorPalette.category_scheme(category)
        return f"""<span style="
            background: {scheme['bg']};
            color: {scheme['text']};
            border: 1px solid {scheme['border']};
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: capitalize;
        ">{category}</span>"""

    @classmethod
    def alert(
        cls,
        message: str,
        variant: str = "info",
        icon: Optional[str] = None,
        dismissible: bool = False
    ) -> None:
        """Render an alert component.

        Args:
            message: Alert message
            variant: Color variant (success, warning, error, info)
            icon: Optional icon emoji
            dismissible: Whether alert can be dismissed (not implemented)
        """
        scheme = ColorPalette.error_scheme(variant)
        default_icons = {
            'success': '✓',
            'warning': '⚠',
            'error': '✕',
            'info': 'ℹ',
        }
        display_icon = icon or default_icons.get(variant, 'ℹ')

        html = f"""
        <div style="
            background: {scheme['background']};
            border: 1px solid {scheme['border']};
            border-left: 4px solid {scheme['border']};
            border-radius: 4px;
            padding: 12px 16px;
            margin: 8px 0;
            display: flex;
            align-items: center;
            gap: 12px;
        ">
            <span style="font-size: 1.2rem;">{display_icon}</span>
            <span style="color: {scheme['text']}">{message}</span>
        </div>
        """
        cls.render(html)

    @classmethod
    def progress_bar(
        cls,
        value: float,
        max_value: float = 100,
        label: Optional[str] = None,
        show_percentage: bool = True,
        color: Optional[str] = None,
        height: int = 8
    ) -> None:
        """Render a progress bar.

        Args:
            value: Current value
            max_value: Maximum value
            label: Optional label text
            show_percentage: Show percentage text
            color: Bar color (default: accent)
            height: Bar height in pixels
        """
        percentage = min(100, max(0, (value / max_value) * 100)) if max_value > 0 else 0
        bar_color = color or ColorPalette.get('accent')
        colors = ThemeManager.get_theme_colors()

        label_html = f'<div style="margin-bottom: 4px; color: {colors["text_secondary"]}">{label}</div>' if label else ''
        pct_html = f'<span style="margin-left: 8px; color: {colors["text_secondary"]}">{percentage:.0f}%</span>' if show_percentage else ''

        html = f"""
        <div style="margin: 8px 0;">
            {label_html}
            <div style="display: flex; align-items: center;">
                <div style="
                    flex: 1;
                    background: {colors['border']};
                    border-radius: {height // 2}px;
                    height: {height}px;
                    overflow: hidden;
                ">
                    <div style="
                        width: {percentage}%;
                        height: 100%;
                        background: {bar_color};
                        border-radius: {height // 2}px;
                        transition: width 0.3s ease;
                    "></div>
                </div>
                {pct_html}
            </div>
        </div>
        """
        cls.render(html)

    @classmethod
    def stat_row(cls, stats: List[Dict[str, Any]]) -> None:
        """Render a row of statistics.

        Args:
            stats: List of stat dicts with keys: label, value, icon (optional)
        """
        colors = ThemeManager.get_theme_colors()
        stat_html_parts = []

        for stat in stats:
            icon_html = f'<span style="margin-right: 4px">{stat.get("icon", "")}</span>' if stat.get("icon") else ''
            stat_html_parts.append(f"""
                <div style="
                    text-align: center;
                    padding: 8px 16px;
                    background: {colors['surface']};
                    border-radius: 8px;
                    border: 1px solid {colors['border']};
                ">
                    <div style="color: {colors['text_secondary']}; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">
                        {icon_html}{stat['label']}
                    </div>
                    <div style="color: {colors['text']}; font-size: 1.1rem; font-weight: 600; margin-top: 4px;">
                        {stat['value']}
                    </div>
                </div>
            """)

        html = f"""
        <div style="display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0;">
            {''.join(stat_html_parts)}
        </div>
        """
        cls.render(html)

    @classmethod
    def divider(cls, label: Optional[str] = None) -> None:
        """Render a divider line.

        Args:
            label: Optional centered label
        """
        colors = ThemeManager.get_theme_colors()

        if label:
            html = f"""
            <div style="
                display: flex;
                align-items: center;
                margin: 16px 0;
                gap: 12px;
            ">
                <div style="flex: 1; height: 1px; background: {colors['border']}"></div>
                <span style="color: {colors['text_secondary']}; font-size: 0.8rem">{label}</span>
                <div style="flex: 1; height: 1px; background: {colors['border']}"></div>
            </div>
            """
        else:
            html = f'<div style="height: 1px; background: {colors["border"]}; margin: 16px 0;"></div>'

        cls.render(html)

    @classmethod
    def tooltip(cls, text: str, tooltip_text: str) -> str:
        """Generate text with tooltip (returns string).

        Args:
            text: Display text
            tooltip_text: Tooltip content

        Returns:
            HTML string with tooltip
        """
        return f"""<span title="{tooltip_text}" style="
            border-bottom: 1px dotted currentColor;
            cursor: help;
        ">{text}</span>"""


# Convenience functions for common operations
def render_card(content: str, header: Optional[str] = None, **kwargs) -> None:
    """Render a card component."""
    HTMLBuilder.card(content, header, **kwargs)


def render_alert(message: str, variant: str = "info", **kwargs) -> None:
    """Render an alert component."""
    HTMLBuilder.alert(message, variant, **kwargs)


def render_progress(value: float, max_value: float = 100, **kwargs) -> None:
    """Render a progress bar."""
    HTMLBuilder.progress_bar(value, max_value, **kwargs)


def category_badge(category: str) -> str:
    """Get category badge HTML."""
    return HTMLBuilder.category_badge(category)
