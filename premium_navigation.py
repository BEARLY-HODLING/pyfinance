"""
Premium Header & Navigation System for PyFinance Dashboard

Provides premium SaaS-style header, tab navigation, breadcrumbs, and sidebar styling.
"""

import streamlit as st
from datetime import datetime


def render_premium_header(total_value=0, daily_change=0, fx_rate=1.348, theme='dark', colors=None):
    """
    Render a premium SaaS-style header with branding, stats, and theme toggle.

    Args:
        total_value: Total portfolio value in GBP
        daily_change: Daily change percentage
        fx_rate: GBP/USD exchange rate
        theme: Current theme ('dark' or 'light')
        colors: Theme color dictionary
    """
    if colors is None:
        colors = {
            'background': '#0f172a',
            'surface': '#1e293b',
            'text': '#f8fafc',
            'text_secondary': '#94a3b8',
            'accent': '#d4af37',
            'accent_secondary': '#fbbf24',
            'success': '#22c55e',
            'error': '#ef4444',
        }

    header_css = f"""
    <style>
        .premium-header {{
            background: linear-gradient(135deg, {colors['surface']} 0%, {colors['background']} 100%);
            border-bottom: 1px solid {colors['text_secondary']}20;
            padding: 1rem 2rem;
            margin: -1rem -1rem 1.5rem -1rem;
            border-radius: 0 0 16px 16px;
        }}

        .header-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .brand-section {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .logo-area {{
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, {colors['accent']} 0%, {colors['accent_secondary']} 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            font-weight: bold;
            color: {'#0f172a' if theme == 'dark' else '#ffffff'};
            box-shadow: 0 4px 12px {colors['accent']}40;
        }}

        .brand-text {{
            display: flex;
            flex-direction: column;
        }}

        .app-title {{
            font-size: 1.75rem;
            font-weight: 700;
            background: linear-gradient(135deg, {colors['accent']} 0%, {colors['accent_secondary']} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            line-height: 1.2;
        }}

        .tagline {{
            font-size: 0.85rem;
            color: {colors['text_secondary']};
            margin: 0;
            letter-spacing: 0.5px;
        }}

        .stats-bar {{
            display: flex;
            gap: 1.5rem;
            align-items: center;
            flex-wrap: wrap;
        }}

        .stat-item {{
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 0.5rem 1rem;
            background: {colors['background']}80;
            border-radius: 8px;
            border: 1px solid {colors['text_secondary']}20;
            min-width: 100px;
        }}

        .stat-label {{
            font-size: 0.7rem;
            color: {colors['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 2px;
        }}

        .stat-value {{
            font-size: 1.1rem;
            font-weight: 600;
            color: {colors['text']};
        }}

        .stat-value.positive {{
            color: {colors['success']};
        }}

        .stat-value.negative {{
            color: {colors['error']};
        }}

        .header-actions {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .last-updated {{
            font-size: 0.75rem;
            color: {colors['text_secondary']};
            text-align: right;
        }}

        @media (max-width: 768px) {{
            .header-container {{
                flex-direction: column;
                align-items: flex-start;
            }}

            .stats-bar {{
                width: 100%;
                justify-content: space-between;
            }}

            .stat-item {{
                min-width: auto;
                flex: 1;
            }}
        }}
    </style>
    """

    # Format values
    change_class = "positive" if daily_change >= 0 else "negative"
    change_sign = "+" if daily_change >= 0 else ""
    now = datetime.now()

    # Inject CSS first
    st.markdown(header_css, unsafe_allow_html=True)

    # Render header HTML separately
    header_html = f"""
    <div class="premium-header">
        <div class="header-container">
            <div class="brand-section">
                <div class="logo-area">P</div>
                <div class="brand-text">
                    <h1 class="app-title">PyFinance</h1>
                    <p class="tagline">Portfolio Intelligence</p>
                </div>
            </div>
            <div class="stats-bar">
                <div class="stat-item">
                    <span class="stat-label">Total Value</span>
                    <span class="stat-value">£{total_value:,.0f}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Daily Change</span>
                    <span class="stat-value {change_class}">{change_sign}{daily_change:.1f}%</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">GBP/USD</span>
                    <span class="stat-value">{fx_rate:.4f}</span>
                </div>
            </div>
            <div class="header-actions">
                <div class="last-updated">
                    Last updated<br/>
                    <strong>{now.strftime('%d %b %Y')}</strong><br/>
                    {now.strftime('%H:%M:%S')}
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


# Tab configuration
TABS_CONFIG = [
    {'id': 'overview', 'label': 'Overview', 'icon': '🏠'},
    {'id': 'isa', 'label': 'ISA', 'icon': '💰'},
    {'id': 'sipp', 'label': 'SIPP', 'icon': '🏦'},
    {'id': 'transactions', 'label': 'Transactions', 'icon': '📜'},
    {'id': 'dividends', 'label': 'Dividends', 'icon': '📅'},
    {'id': 'rebalancing', 'label': 'Rebalancing', 'icon': '⚖️'},
    {'id': 'watchlist', 'label': 'Watchlist', 'icon': '👁️'},
    {'id': 'assistant', 'label': 'Assistant', 'icon': '🤖'},
    {'id': 'export', 'label': 'Export', 'icon': '📤'},
]


def render_tab_navigation(has_isa=True, has_sipp=True, theme='dark', colors=None):
    """
    Render custom styled tab navigation with icons.
    Returns the selected tab id.

    Args:
        has_isa: Whether ISA data is available
        has_sipp: Whether SIPP data is available
        theme: Current theme ('dark' or 'light')
        colors: Theme color dictionary

    Returns:
        str: The active tab id
    """
    if colors is None:
        colors = {
            'text_secondary': '#94a3b8',
        }

    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 'overview'

    nav_css = f"""
    <style>
        .tab-nav-container {{
            margin-bottom: 1.5rem;
        }}

        .tab-divider {{
            height: 1px;
            background: {colors['text_secondary']}30;
            margin-bottom: 1rem;
        }}
    </style>
    """

    st.markdown(nav_css, unsafe_allow_html=True)

    available_tabs = []
    for tab in TABS_CONFIG:
        if tab['id'] == 'isa' and not has_isa:
            continue
        if tab['id'] == 'sipp' and not has_sipp:
            continue
        available_tabs.append(tab)

    cols = st.columns(len(available_tabs))

    for i, tab in enumerate(available_tabs):
        with cols[i]:
            is_active = st.session_state.active_tab == tab['id']
            btn_label = f"{tab['icon']} {tab['label']}"

            if st.button(
                btn_label,
                key=f"nav_tab_{tab['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.active_tab = tab['id']
                st.rerun()

    st.markdown('<div class="tab-divider"></div>', unsafe_allow_html=True)

    return st.session_state.active_tab


def render_breadcrumb(path_items, theme='dark', colors=None):
    """
    Render a clickable breadcrumb navigation.

    Args:
        path_items: List of tuples (label, tab_id) for breadcrumb path
        theme: Current theme ('dark' or 'light')
        colors: Theme color dictionary
    """
    if colors is None:
        colors = {
            'text': '#f8fafc',
            'text_secondary': '#94a3b8',
            'accent': '#d4af37',
        }

    breadcrumb_css = f"""
    <style>
        .breadcrumb {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0;
            margin-bottom: 1rem;
            font-size: 0.85rem;
        }}

        .breadcrumb-item {{
            color: {colors['text_secondary']};
            transition: color 0.2s ease;
        }}

        .breadcrumb-item.clickable {{
            cursor: pointer;
        }}

        .breadcrumb-item.clickable:hover {{
            color: {colors['accent']};
        }}

        .breadcrumb-item.current {{
            color: {colors['text']};
            font-weight: 500;
        }}

        .breadcrumb-separator {{
            color: {colors['text_secondary']}60;
        }}
    </style>
    """

    breadcrumb_parts = []
    for i, (label, tab_id) in enumerate(path_items):
        is_last = i == len(path_items) - 1
        item_class = "current" if is_last else "clickable"
        breadcrumb_parts.append(f'<span class="breadcrumb-item {item_class}">{label}</span>')
        if not is_last:
            breadcrumb_parts.append('<span class="breadcrumb-separator">&rsaquo;</span>')

    breadcrumb_html = f"""
    {breadcrumb_css}
    <div class="breadcrumb">
        {''.join(breadcrumb_parts)}
    </div>
    """

    st.markdown(breadcrumb_html, unsafe_allow_html=True)


def get_breadcrumb_for_tab(tab_id):
    """
    Get breadcrumb path for a given tab.

    Args:
        tab_id: The tab identifier

    Returns:
        list: List of tuples (label, tab_id) for breadcrumb path
    """
    breadcrumbs = {
        'overview': [('PyFinance', None), ('Overview', 'overview')],
        'isa': [('PyFinance', None), ('ISA', 'isa'), ('Holdings', None)],
        'sipp': [('PyFinance', None), ('SIPP', 'sipp'), ('Holdings', None)],
        'transactions': [('PyFinance', None), ('Transactions', 'transactions'), ('History', None)],
        'dividends': [('PyFinance', None), ('Dividends', 'dividends'), ('Calendar', None)],
        'rebalancing': [('PyFinance', None), ('Rebalancing', 'rebalancing')],
        'watchlist': [('PyFinance', None), ('Watchlist', 'watchlist')],
        'assistant': [('PyFinance', None), ('AI Assistant', 'assistant')],
        'export': [('PyFinance', None), ('Export', 'export')],
    }
    return breadcrumbs.get(tab_id, [('PyFinance', None)])


def render_sidebar_header(theme='dark', colors=None):
    """
    Render a styled sidebar header with logo and collapsible sections.

    Args:
        theme: Current theme ('dark' or 'light')
        colors: Theme color dictionary
    """
    if colors is None:
        colors = {
            'text': '#f8fafc',
            'text_secondary': '#94a3b8',
            'accent': '#d4af37',
            'accent_secondary': '#fbbf24',
        }

    sidebar_header_css = f"""
    <style>
        .sidebar-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 1rem;
            margin: -1rem -1rem 1rem -1rem;
            background: linear-gradient(135deg, {colors['accent']}15 0%, {colors['accent_secondary']}15 100%);
            border-bottom: 1px solid {colors['text_secondary']}20;
            border-radius: 0 0 8px 8px;
        }}

        .sidebar-logo {{
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, {colors['accent']} 0%, {colors['accent_secondary']} 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: bold;
            color: {'#0f172a' if theme == 'dark' else '#ffffff'};
            box-shadow: 0 2px 6px {colors['accent']}30;
        }}

        .sidebar-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: {colors['text']};
            margin: 0;
        }}

        .sidebar-subtitle {{
            font-size: 0.75rem;
            color: {colors['text_secondary']};
            margin: 0;
        }}
    </style>
    """

    sidebar_header_html = f"""
    {sidebar_header_css}
    <div class="sidebar-header">
        <div class="sidebar-logo">S</div>
        <div>
            <p class="sidebar-title">Settings</p>
            <p class="sidebar-subtitle">Configuration & Mappings</p>
        </div>
    </div>
    """

    st.markdown(sidebar_header_html, unsafe_allow_html=True)


def render_section_header(title, icon="", theme='dark', colors=None):
    """
    Render a styled section header within the sidebar.

    Args:
        title: Section title text
        icon: Icon emoji or character
        theme: Current theme ('dark' or 'light')
        colors: Theme color dictionary
    """
    if colors is None:
        colors = {
            'text': '#f8fafc',
            'text_secondary': '#94a3b8',
        }

    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0;
        margin-top: 0.5rem;
        border-bottom: 1px solid {colors['text_secondary']}20;
    ">
        <span style="font-size: 1rem;">{icon}</span>
        <span style="
            font-size: 0.85rem;
            font-weight: 600;
            color: {colors['text']};
            text-transform: uppercase;
            letter-spacing: 0.5px;
        ">{title}</span>
    </div>
    """, unsafe_allow_html=True)
