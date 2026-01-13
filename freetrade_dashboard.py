import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import os
import glob
import io
import base64
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Optional OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Optional Excel export
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# Optional PDF export
try:
    from weasyprint import HTML as WeasyHTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

import calendar
from collections import defaultdict
import uuid
import math
import time
import functools
import traceback
import logging

# Configure logging for error tracking
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Loading states system
from loading_states import (
    inject_loading_css,
    render_skeleton_metric,
    render_skeleton_metrics_row,
    render_skeleton_table,
    render_skeleton_chart,
    render_loading_spinner,
    render_loading_spinner_with_progress,
    show_loading_overlay,
    hide_loading_overlay,
    render_data_loading_status,
    render_skeleton_portfolio_view,
    render_skeleton_overview
)

# Premium metric card components
from premium_metrics import (
    inject_premium_css,
    render_premium_metric,
    render_metric_row,
    render_portfolio_summary_card,
    generate_sparkline_svg
)

# Dividend calendar module
# Export & Reporting System
from export_reports import (
    export_portfolio_csv,
    export_portfolio_excel,
    generate_portfolio_report,
    generate_pdf_report,
    render_export_panel,
    schedule_report_email,
    render_export_sidebar_section,
    OPENPYXL_AVAILABLE,
    WEASYPRINT_AVAILABLE
)

from dividend_calendar import render_dividend_tab

# Smart portfolio insights
from portfolio_insights import (
    generate_portfolio_insights,
    render_insights_panel,
    render_insights_sidebar,
    get_insight_priority_emoji
)

# Premium chart components
from premium_charts import (
    CHART_COLORS,
    get_chart_config,
    get_chart_layout_defaults,
    render_portfolio_value_chart,
    render_allocation_donut,
    render_income_chart,
    render_benchmark_chart,
    render_treemap,
    render_comparison_bar_chart,
    render_drift_gauge
)

# Premium navigation system
from premium_navigation import (
    render_premium_header,
    render_tab_navigation,
    render_breadcrumb,
    get_breadcrumb_for_tab,
    render_sidebar_header,
    render_section_header,
    TABS_CONFIG
)

# Goal tracking system
from goal_tracking import (
    load_goals, save_goals, create_goal, calculate_goal_progress,
    render_goal_card, render_goals_dashboard, render_add_goal_form,
    render_edit_goal_form, render_goal_projections, GOAL_TYPES
)

# Premium animations and micro-interactions
from animations import (
    inject_animation_css,
    render_animated_metric,
    render_success_feedback,
    render_error_feedback,
    render_achievement_badge,
    render_celebration_confetti,
    get_plotly_animation_config,
    apply_chart_animation,
    render_loading_pulse,
    render_shimmer_skeleton
)

# Data refresh system
from refresh_system import (
    init_refresh_state,
    update_refresh_timestamp,
    get_refresh_timestamp,
    get_time_ago_text,
    get_freshness_status,
    get_auto_refresh_countdown,
    clear_cache_and_refresh,
    auto_refresh_handler,
    render_refresh_indicator,
    render_refresh_button,
    render_data_freshness_badge,
    render_refresh_settings,
    render_keyboard_shortcut_handler,
    render_header_with_freshness,
    render_footer_with_freshness,
    track_user_activity
)

# Help and onboarding system
from help_system import (
    render_welcome_modal,
    render_feature_tour,
    render_help_tooltip,
    render_info_popover,
    render_help_sidebar_section,
    render_contextual_help,
    render_keyboard_shortcuts_modal,
    render_inline_help_icon,
    inject_keyboard_shortcut_handlers,
    inject_help_css,
    HELP_CONTENT
)

# Error handling system
from error_handling import (
    ErrorType,
    inject_error_css,
    render_error_message,
    handle_api_error,
    handle_data_error,
    render_empty_state,
    render_partial_data_warning,
    error_boundary,
    render_offline_mode,
    safe_api_call,
    display_api_error,
    display_data_error
)

# Premium data table components
from premium_tables import (
    format_currency,
    format_percent,
    get_category_color,
    render_holdings_table,
    render_dividend_table
)

st.set_page_config(page_title="Freetrade Dashboard", layout="wide", page_icon="📊")

# ============================================================================
# THEME SYSTEM
# ============================================================================

# Theme color definitions
THEMES = {
    'dark': {
        'background': '#0f172a',
        'surface': '#1e293b',
        'text': '#f8fafc',
        'text_secondary': '#94a3b8',
        'accent': '#d4af37',  # gold
        'accent_secondary': '#fbbf24',
        'success': '#22c55e',
        'warning': '#f59e0b',
        'error': '#ef4444',
        'plotly_template': 'plotly_dark',
        'chart_colors': ['#00CC96', '#EF553B', '#636EFA', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
    },
    'light': {
        'background': '#f8fafc',
        'surface': '#ffffff',
        'text': '#1e293b',
        'text_secondary': '#64748b',
        'accent': '#1a365d',  # deep blue
        'accent_secondary': '#2563eb',
        'success': '#16a34a',
        'warning': '#d97706',
        'error': '#dc2626',
        'plotly_template': 'plotly_white',
        'chart_colors': ['#059669', '#dc2626', '#2563eb', '#7c3aed', '#ea580c', '#0891b2', '#db2777', '#65a30d']
    }
}

def get_current_theme() -> str:
    """Get the current theme from session state, defaulting to dark"""
    if 'theme' not in st.session_state:
        # Check config for saved preference
        config_file = 'config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    cfg = json.load(f)
                    st.session_state.theme = cfg.get('theme', 'dark')
            except:
                st.session_state.theme = 'dark'
        else:
            st.session_state.theme = 'dark'
    return st.session_state.theme

def toggle_theme() -> None:
    """Toggle between dark and light themes"""
    current = get_current_theme()
    new_theme = 'light' if current == 'dark' else 'dark'
    st.session_state.theme = new_theme
    # Save preference to config
    config_file = 'config.json'
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                cfg = json.load(f)
            cfg['theme'] = new_theme
            with open(config_file, 'w') as f:
                json.dump(cfg, f, indent=4, sort_keys=True)
        except:
            pass

def get_plotly_template() -> str:
    """Get the Plotly template for the current theme"""
    theme = get_current_theme()
    return THEMES[theme]['plotly_template']

def get_chart_colors() -> list:
    """Get chart color palette for the current theme"""
    theme = get_current_theme()
    return THEMES[theme]['chart_colors']

def get_theme_colors() -> dict:
    """Get the full color dictionary for the current theme"""
    theme = get_current_theme()
    return THEMES[theme]

def load_premium_css():
    """Load the premium CSS theme from external file"""
    css_path = Path(__file__).parent / "assets" / "styles.css"
    css_path.parent.mkdir(parents=True, exist_ok=True)
    if css_path.exists():
        with open(css_path, 'r') as f:
            css_content = f.read()
        st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
        return True
    return False

def inject_theme_css():
    """Inject CSS variables and styles for the current theme"""
    theme = get_current_theme()
    colors = THEMES[theme]

    # Load premium CSS file first (if it exists)
    load_premium_css()

    # Inject premium metric card CSS
    inject_premium_css()

    css = f"""
    <style>
        /* CSS Variables for theme */
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
        }}

        /* Smooth theme transition */
        .stApp, .stApp > div, .main, [data-testid="stAppViewContainer"],
        [data-testid="stHeader"], [data-testid="stSidebar"],
        .stMarkdown, .stDataFrame, .stMetric, .stTabs {{
            transition: background-color 0.3s ease, color 0.3s ease;
        }}

        /* Main app background */
        .stApp {{
            background-color: {colors['background']};
        }}

        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: {colors['surface']};
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] .stMarkdown {{
            color: {colors['text']};
        }}

        /* Header styling */
        [data-testid="stHeader"] {{
            background-color: {colors['background']};
        }}

        /* Text colors */
        .stMarkdown, .stMarkdown p, h1, h2, h3, h4, h5, h6 {{
            color: {colors['text']} !important;
        }}

        /* Caption text */
        .stCaption, small {{
            color: {colors['text_secondary']} !important;
        }}

        /* Metric styling */
        [data-testid="stMetricValue"] {{
            color: {colors['text']} !important;
        }}

        [data-testid="stMetricLabel"] {{
            color: {colors['text_secondary']} !important;
        }}

        /* DataFrames */
        .stDataFrame {{
            background-color: {colors['surface']};
        }}

        .stDataFrame [data-testid="stDataFrameResizable"] {{
            background-color: {colors['surface']};
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {colors['surface']};
            border-radius: 8px;
        }}

        .stTabs [data-baseweb="tab"] {{
            color: {colors['text_secondary']};
        }}

        .stTabs [aria-selected="true"] {{
            color: {colors['accent']} !important;
        }}

        /* Buttons */
        .stButton > button {{
            background-color: {colors['surface']};
            color: {colors['text']};
            border: 1px solid {colors['accent']};
            transition: all 0.2s ease;
        }}

        .stButton > button:hover {{
            background-color: {colors['accent']};
            color: {'#0f172a' if theme == 'dark' else '#ffffff'};
            border-color: {colors['accent']};
        }}

        /* Theme toggle button - special styling */
        .theme-toggle-btn {{
            position: fixed;
            top: 70px;
            right: 20px;
            z-index: 999999;
            background-color: {colors['surface']};
            color: {colors['accent']};
            border: 2px solid {colors['accent']};
            border-radius: 50%;
            width: 50px;
            height: 50px;
            font-size: 24px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }}

        .theme-toggle-btn:hover {{
            background-color: {colors['accent']};
            color: {colors['background']};
            transform: scale(1.1);
        }}

        /* Inputs */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div {{
            background-color: {colors['surface']};
            color: {colors['text']};
            border-color: {colors['text_secondary']};
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {colors['surface']};
            color: {colors['text']};
        }}

        /* Alert boxes */
        .stAlert {{
            background-color: {colors['surface']};
        }}

        /* Dividers */
        hr {{
            border-color: {colors['text_secondary']}40;
        }}

        /* Form styling */
        [data-testid="stForm"] {{
            background-color: {colors['surface']};
            border-radius: 8px;
            padding: 1rem;
        }}

        /* Plotly chart backgrounds - let template handle it */
        .js-plotly-plot .plotly {{
            background-color: transparent !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_theme_toggle():
    """Render the theme toggle button in the top-right corner"""
    theme = get_current_theme()
    icon = "🌙" if theme == 'light' else "☀️"
    tooltip = "Switch to dark mode" if theme == 'light' else "Switch to light mode"

    # Use a container in sidebar for the toggle
    with st.sidebar:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button(icon, key="theme_toggle", help=tooltip, use_container_width=True):
                toggle_theme()
                st.rerun()

# ============================================================================
# CONFIGURATION
# ============================================================================

ISA_CSV_FILES = glob.glob('freetrade_ISA_*.csv')
SIPP_CSV_FILES = glob.glob('freetrade_SIPP_*.csv')
ISA_CSV = max(ISA_CSV_FILES, key=os.path.getmtime) if ISA_CSV_FILES else None
SIPP_CSV = max(SIPP_CSV_FILES, key=os.path.getmtime) if SIPP_CSV_FILES else None
CONFIG_FILENAME = 'config.json'

@st.cache_data(ttl=900)
def load_csv(filepath):
    if not filepath or not os.path.exists(filepath):
        return None
    return pd.read_csv(filepath)

def load_config():
    if os.path.exists(CONFIG_FILENAME):
        with open(CONFIG_FILENAME, 'r') as f:
            config = json.load(f)
            if 'proxy_yields' not in config:
                config['proxy_yields'] = {}
            if 'category_targets' not in config:
                config['category_targets'] = {'income': 50.0, 'growth': 35.0, 'speculative': 15.0}
            if 'target_allocations' not in config:
                config['target_allocations'] = {}
            if 'openai_api_key' not in config:
                config['openai_api_key'] = ''
            return config
    return {
        'yahoo_ticker_map': {},
        'category_map': {},
        'proxy_yields': {},
        'category_targets': {'income': 50.0, 'growth': 35.0, 'speculative': 15.0},
        'target_allocations': {},
        'openai_api_key': ''
    }

def save_config(c):
    with open(CONFIG_FILENAME, 'w') as f:
        json.dump(c, f, indent=4, sort_keys=True)

config = load_config()
yahoo_map = config['yahoo_ticker_map']
category_map = config['category_map']
proxy_yields = config['proxy_yields']
category_targets = config['category_targets']
target_allocations = config['target_allocations']

# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_transactions(df):
    if df is None or df.empty:
        return None, None, 0
    trans = df[df['Type'].isin(['ORDER', 'DIVIDEND', 'INTEREST'])].copy()
    trans['Quantity'] = pd.to_numeric(trans.get('Quantity', 0), errors='coerce').fillna(0)
    trans['Total Amount'] = pd.to_numeric(trans['Total Amount'], errors='coerce')
    trans['Timestamp'] = pd.to_datetime(trans['Timestamp']).dt.tz_localize(None)
    trans = trans.sort_values('Timestamp')
    orders = trans[trans['Type'] == 'ORDER']
    net = orders.groupby('Ticker').agg({'Quantity': 'sum', 'Total Amount': 'sum', 'ISIN': 'first', 'Title': 'first', 'Timestamp': 'min'}).reset_index()
    net = net[net['Quantity'] > 0]
    net.rename(columns={'Total Amount': 'cost_gbp'}, inplace=True)
    div_income = trans[trans['Type'].str.contains('DIVIDEND|INTEREST', na=False)]['Total Amount'].sum()
    return trans, net, div_income

def calculate_dividend_yield(ticker, price, scale, proxy_yields):
    try:
        t = yf.Ticker(ticker)
        divs = t.dividends
        if divs.empty:
            return (proxy_yields[ticker] * 100, 'proxy', 0) if ticker in proxy_yields else (0, 'none', 0)
        divs.index = divs.index.tz_localize(None)
        one_year_ago = pd.Timestamp.now() - timedelta(days=365)
        recent_divs = divs[divs.index > one_year_ago]
        if recent_divs.empty:
            return (proxy_yields[ticker] * 100, 'proxy', 0) if ticker in proxy_yields else (0, 'none', 0)
        if scale > 1:
            recent_divs = recent_divs / 100
        ttm_dividends = recent_divs.sum()
        months_of_data = len(recent_divs)
        calculated_yield = (ttm_dividends / price * 100) if price else 0
        return (calculated_yield, 'calculated_limited' if months_of_data < 6 and ticker in proxy_yields else 'calculated', months_of_data)
    except:
        return (proxy_yields[ticker] * 100, 'proxy', 0) if ticker in proxy_yields else (0, 'error', 0)

@st.cache_data(ttl=900)
def fetch_fx_rate():
    try:
        return yf.Ticker('GBPUSD=X').info.get('regularMarketPrice') or 1.348
    except:
        return 1.348

def fetch_ticker_data(tk, max_retries=2):
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(tk)
            info = t.info
            price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
            if price is None:
                return None
            curr = info.get('currency', '').upper()
            
            # Determine if price is in pence (needs /100 scaling)
            scale = 1
            if '.L' in tk:  # London Stock Exchange
                if curr in ['GBX', 'GBp']:
                    # Explicitly marked as pence
                    scale = 100
                elif curr == 'GBP':
                    # GBP currency - need to check if actually in pence
                    # Most LSE stocks trade in pence, so if price seems reasonable as pence, scale it
                    # Exception: If price is very low (< 1), it's probably already in pounds
                    if price >= 1:
                        scale = 100
            
            hist = t.history(start="2025-10-01")['Close']
            hist.index = hist.index.tz_localize(None)
            if scale > 1 and not hist.empty:
                hist /= 100
            vol = np.std(hist.pct_change()) * np.sqrt(252) * 100 if len(hist) > 1 else None
            dd = (hist.cummax() - hist).max() / hist.cummax().max() * 100 if not hist.empty else None
            beta = info.get('beta') or 1.0
            return {'ticker': tk, 'price': price, 'currency': curr, 'scale': scale, 'vol': vol, 'dd': dd, 'beta': beta, 'hist': hist, 'info': info}
        except:
            if attempt == max_retries - 1:
                return None
    return None

@st.cache_data(ttl=900)
def fetch_all_tickers(tickers):
    results, failed = {}, []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(fetch_ticker_data, tk): tk for tk in tickers}
        for future in as_completed(future_to_ticker):
            tk = future_to_ticker[future]
            result = future.result()
            if result:
                results[tk] = result
            else:
                failed.append(tk)
    return results, failed

# ============================================================================
# MONTHLY INCOME TRACKING
# ============================================================================

def get_monthly_dividends(df):
    """Extract monthly dividend totals from transaction data"""
    if df is None or df.empty:
        return pd.DataFrame(columns=['Month', 'Total Amount'])

    dividends = df[df['Type'].str.contains('DIVIDEND|INTEREST', na=False)].copy()
    if dividends.empty:
        return pd.DataFrame(columns=['Month', 'Total Amount'])

    dividends['Timestamp'] = pd.to_datetime(dividends['Timestamp'])
    dividends['Month'] = dividends['Timestamp'].dt.to_period('M')
    monthly = dividends.groupby('Month')['Total Amount'].sum().reset_index()
    monthly['Month'] = monthly['Month'].astype(str)
    return monthly

def get_combined_monthly_dividends(isa_df, sipp_df):
    """Combine ISA + SIPP dividends for overview chart"""
    isa_divs = get_monthly_dividends(isa_df)
    sipp_divs = get_monthly_dividends(sipp_df)

    if isa_divs.empty and sipp_divs.empty:
        return pd.DataFrame(columns=['Month', 'ISA', 'SIPP', 'Total'])

    # Rename columns before merge
    isa_divs = isa_divs.rename(columns={'Total Amount': 'ISA'})
    sipp_divs = sipp_divs.rename(columns={'Total Amount': 'SIPP'})

    combined = pd.merge(isa_divs, sipp_divs, on='Month', how='outer')
    combined = combined.fillna(0)
    combined['Total'] = combined['ISA'] + combined['SIPP']
    combined = combined.sort_values('Month')
    return combined

def render_monthly_income_chart(monthly_df, title="Monthly Dividend Income", stacked=False, key_suffix="default"):
    """Render a bar chart of monthly dividend income using premium chart component"""
    if monthly_df.empty:
        st.info("ℹ️ No dividend data available yet")
        return

    # Use the premium income chart from premium_charts module
    render_income_chart(
        monthly_df,
        chart_type="bar",
        show_trend=True,
        goal_line=None,
        key_suffix=key_suffix,
        template=get_plotly_template()
    )

def render_income_metrics(monthly_df):
    """Display income summary metrics"""
    if monthly_df.empty:
        return

    col_name = 'Total' if 'Total' in monthly_df.columns else 'Total Amount'

    total_ytd = monthly_df[col_name].sum()
    avg_monthly = monthly_df[col_name].mean()
    best_month = monthly_df.loc[monthly_df[col_name].idxmax()]

    # Growth rate (last 3 months vs previous 3)
    if len(monthly_df) >= 6:
        recent = monthly_df[col_name].tail(3).sum()
        previous = monthly_df[col_name].iloc[-6:-3].sum()
        growth = ((recent - previous) / previous * 100) if previous > 0 else 0
    else:
        growth = 0

    # Projected annual (based on trend)
    if len(monthly_df) > 1:
        x_numeric = np.arange(len(monthly_df))
        z = np.polyfit(x_numeric, monthly_df[col_name], 1)
        projected_next_12 = sum(np.poly1d(z)(np.arange(len(monthly_df), len(monthly_df) + 12)))
    else:
        projected_next_12 = avg_monthly * 12

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Total YTD", f"£{total_ytd:,.0f}")
    c2.metric("📊 Avg Monthly", f"£{avg_monthly:,.0f}")
    c3.metric("📈 Growth", f"{growth:+.1f}%" if growth != 0 else "N/A")
    c4.metric("🎯 Projected Annual", f"£{projected_next_12:,.0f}")

    st.caption(f"📅 Best month: {best_month['Month']} (£{best_month[col_name]:,.0f})")

# ============================================================================
# BENCHMARK COMPARISON
# ============================================================================

BENCHMARKS = {
    "S&P 500": "^GSPC",
    "FTSE 100": "^FTSE",
    "FTSE All-Share": "^FTAS"
}

@st.cache_data(ttl=900)
def fetch_benchmark_data(benchmark_symbol, start_date="2025-10-01"):
    """Fetch benchmark index data"""
    try:
        ticker = yf.Ticker(benchmark_symbol)
        hist = ticker.history(start=start_date)['Close']
        hist.index = hist.index.tz_localize(None)
        return hist
    except:
        return pd.Series()

def normalize_series(series):
    """Normalize a series to start at 100 for fair comparison"""
    if series.empty:
        return series
    return (series / series.iloc[0]) * 100

def render_benchmark_comparison(port_hist, name):
    """Render portfolio vs benchmark comparison chart"""
    if port_hist.empty or 'Total' not in port_hist.columns:
        return

    st.subheader("📊 Benchmark Comparison")

    # Benchmark selector
    benchmark_name = st.selectbox(
        "Compare against:",
        list(BENCHMARKS.keys()),
        key=f"benchmark_{name}"
    )

    benchmark_symbol = BENCHMARKS[benchmark_name]
    benchmark_data = fetch_benchmark_data(benchmark_symbol)

    if benchmark_data.empty:
        st.warning(f"⚠️ Could not fetch {benchmark_name} data")
        return

    # Align dates
    common_dates = port_hist.index.intersection(benchmark_data.index)
    if len(common_dates) < 2:
        st.warning("⚠️ Not enough overlapping data for comparison")
        return

    portfolio_aligned = port_hist.loc[common_dates, 'Total']
    benchmark_aligned = benchmark_data.loc[common_dates]

    # Normalize both to 100
    portfolio_norm = normalize_series(portfolio_aligned)
    benchmark_norm = normalize_series(benchmark_aligned)

    # Calculate alpha (outperformance)
    portfolio_return = (portfolio_aligned.iloc[-1] / portfolio_aligned.iloc[0] - 1) * 100
    benchmark_return = (benchmark_aligned.iloc[-1] / benchmark_aligned.iloc[0] - 1) * 100
    alpha = portfolio_return - benchmark_return

    # Display metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("📈 Portfolio Return", f"{portfolio_return:+.1f}%")
    c2.metric(f"📉 {benchmark_name} Return", f"{benchmark_return:+.1f}%")
    c3.metric("🎯 Alpha", f"{alpha:+.1f}%",
              delta="Beating market" if alpha > 0 else "Underperforming",
              delta_color="normal" if alpha > 0 else "inverse")

    # Use premium benchmark chart
    render_benchmark_chart(
        portfolio_aligned,
        benchmark_aligned,
        benchmark_name,
        key_suffix=f"benchmark_{name}",
        template=get_plotly_template()
    )


# ============================================================================
# PERFORMANCE ATTRIBUTION
# ============================================================================

def calculate_attribution(holdings_df, ticker_data, period='1M'):
    """
    Calculate contribution of each holding to total portfolio return.
    Returns DataFrame with: ticker, weight, return, contribution, attribution_pct
    """
    if holdings_df.empty or not ticker_data:
        return pd.DataFrame()

    period_days = {'1W': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 365}
    days = period_days.get(period, 30)
    cutoff_date = pd.Timestamp.now() - timedelta(days=days)

    results = []
    total_return_contribution = 0

    for _, row in holdings_df.iterrows():
        ticker = row['Ticker']
        weight = row['Weight %'] / 100
        current_value = row['Value £']

        if ticker not in ticker_data or ticker_data[ticker]['hist'].empty:
            continue

        hist = ticker_data[ticker]['hist']
        period_hist = hist[hist.index >= cutoff_date]
        if len(period_hist) < 2:
            continue

        start_price = period_hist.iloc[0]
        end_price = period_hist.iloc[-1]
        holding_return = ((end_price - start_price) / start_price) * 100 if start_price > 0 else 0
        contribution = weight * holding_return
        total_return_contribution += contribution

        results.append({
            'Ticker': ticker,
            'Category': row.get('Category', 'unknown'),
            'Weight %': row['Weight %'],
            'Return %': round(holding_return, 2),
            'Contribution %': round(contribution, 2),
            'Value £': current_value
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    if total_return_contribution != 0:
        df['Attribution %'] = (df['Contribution %'] / abs(total_return_contribution) * 100).round(1)
    else:
        df['Attribution %'] = 0
    df = df.sort_values('Contribution %', ascending=False)
    return df


def calculate_sector_attribution(holdings_df, cat_map, cat_targets, ticker_data, period='1M'):
    """
    Aggregate attribution by category with allocation effect and selection effect.
    Allocation effect = how category weight vs benchmark affected returns
    Selection effect = how stock selection within category affected returns
    """
    if holdings_df.empty or not ticker_data:
        return pd.DataFrame()

    period_days = {'1W': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 365}
    days = period_days.get(period, 30)
    cutoff_date = pd.Timestamp.now() - timedelta(days=days)

    # Calculate portfolio return for each category
    category_data = {}
    total_value = holdings_df['Value £'].sum()

    for category in ['income', 'growth', 'speculative']:
        cat_holdings = holdings_df[holdings_df['Category'] == category]
        if cat_holdings.empty:
            category_data[category] = {
                'actual_weight': 0,
                'target_weight': cat_targets.get(category, 0),
                'return': 0,
                'contribution': 0
            }
            continue

        cat_value = cat_holdings['Value £'].sum()
        actual_weight = (cat_value / total_value * 100) if total_value > 0 else 0
        target_weight = cat_targets.get(category, 0)

        # Calculate category return
        cat_returns = []
        cat_weights = []
        for _, row in cat_holdings.iterrows():
            ticker = row['Ticker']
            if ticker not in ticker_data or ticker_data[ticker]['hist'].empty:
                continue
            hist = ticker_data[ticker]['hist']
            period_hist = hist[hist.index >= cutoff_date]
            if len(period_hist) < 2:
                continue
            ret = ((period_hist.iloc[-1] - period_hist.iloc[0]) / period_hist.iloc[0]) * 100
            cat_returns.append(ret)
            cat_weights.append(row['Value £'])

        if cat_returns and sum(cat_weights) > 0:
            cat_return = np.average(cat_returns, weights=cat_weights)
        else:
            cat_return = 0

        category_data[category] = {
            'actual_weight': actual_weight,
            'target_weight': target_weight,
            'return': cat_return,
            'contribution': (actual_weight / 100) * cat_return
        }

    # Calculate benchmark return (weighted average of category returns at target weights)
    benchmark_return = sum(
        (cat_data['target_weight'] / 100) * cat_data['return']
        for cat_data in category_data.values()
    )

    results = []
    for category, data in category_data.items():
        # Allocation effect: (actual weight - target weight) * (category return - benchmark return)
        allocation_effect = (data['actual_weight'] - data['target_weight']) / 100 * (data['return'] - benchmark_return)

        # Selection effect: actual weight * (category return - benchmark return)
        selection_effect = (data['actual_weight'] / 100) * (data['return'] - benchmark_return)

        results.append({
            'Category': category.title(),
            'Actual Weight %': round(data['actual_weight'], 1),
            'Target Weight %': round(data['target_weight'], 1),
            'Return %': round(data['return'], 2),
            'Contribution %': round(data['contribution'], 2),
            'Allocation Effect %': round(allocation_effect, 2),
            'Selection Effect %': round(selection_effect, 2)
        })

    return pd.DataFrame(results)


def render_attribution_waterfall(attribution_df, period='1M', name=''):
    """Render a waterfall chart showing contribution breakdown - green/red bars."""
    if attribution_df.empty:
        st.info("No attribution data available for this period")
        return

    colors = get_theme_colors()

    # Sort by contribution for better visualization
    df = attribution_df.sort_values('Contribution %', ascending=True).tail(15)  # Top 15

    # Create waterfall-style horizontal bar chart
    fig = go.Figure()

    # Color based on positive/negative
    bar_colors = [colors['success'] if v >= 0 else colors['error'] for v in df['Contribution %']]

    fig.add_trace(go.Bar(
        y=df['Ticker'],
        x=df['Contribution %'],
        orientation='h',
        marker_color=bar_colors,
        text=[f"{v:+.2f}%" for v in df['Contribution %']],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:.2f}%<br>Return: %{customdata[0]:.1f}%<br>Weight: %{customdata[1]:.1f}%<extra></extra>",
        customdata=df[['Return %', 'Weight %']].values
    ))

    # Add total line
    total_return = df['Contribution %'].sum()
    fig.add_vline(x=0, line_dash="dash", line_color=colors['text_secondary'], opacity=0.5)

    fig.update_layout(
        template=get_plotly_template(),
        title=f"Performance Attribution ({period})",
        xaxis_title="Contribution to Return (%)",
        yaxis_title="",
        height=max(400, len(df) * 30),
        showlegend=False,
        margin=dict(l=100, r=50, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True, key=f"waterfall_{name}_{period}")

    # Show total return
    st.metric(f"Total Portfolio Return ({period})", f"{total_return:+.2f}%")


def render_top_contributors(attribution_df, n=5, name=''):
    """Render two columns: top winners and top losers with color-coded bars."""
    if attribution_df.empty:
        return

    colors = get_theme_colors()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top Contributors**")
        winners = attribution_df.nlargest(n, 'Contribution %')

        for _, row in winners.iterrows():
            contrib = row['Contribution %']
            ret = row['Return %']
            weight = row['Weight %']

            # Progress bar style
            st.markdown(f"""
            <div style="margin-bottom: 12px; padding: 10px; background: {colors['surface']}; border-radius: 8px; border-left: 4px solid {colors['success']};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; color: {colors['text']};">{row['Ticker']}</span>
                    <span style="color: {colors['success']}; font-weight: 700;">+{contrib:.2f}%</span>
                </div>
                <div style="font-size: 12px; color: {colors['text_secondary']}; margin-top: 4px;">
                    Return: {ret:+.1f}% | Weight: {weight:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Top Detractors**")
        losers = attribution_df.nsmallest(n, 'Contribution %')

        for _, row in losers.iterrows():
            contrib = row['Contribution %']
            ret = row['Return %']
            weight = row['Weight %']

            st.markdown(f"""
            <div style="margin-bottom: 12px; padding: 10px; background: {colors['surface']}; border-radius: 8px; border-left: 4px solid {colors['error']};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; color: {colors['text']};">{row['Ticker']}</span>
                    <span style="color: {colors['error']}; font-weight: 700;">{contrib:+.2f}%</span>
                </div>
                <div style="font-size: 12px; color: {colors['text_secondary']}; margin-top: 4px;">
                    Return: {ret:+.1f}% | Weight: {weight:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_attribution_heatmap(holdings_df, ticker_data, periods=['1W', '1M', '3M'], name=''):
    """Render a heatmap with tickers as rows and periods as columns."""
    if holdings_df.empty or not ticker_data:
        st.info("No data available for heatmap")
        return

    # Calculate returns for each ticker across periods
    period_days = {'1W': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 365}

    data = []
    for _, row in holdings_df.iterrows():
        ticker = row['Ticker']
        if ticker not in ticker_data or ticker_data[ticker]['hist'].empty:
            continue

        hist = ticker_data[ticker]['hist']
        row_data = {'Ticker': ticker}

        for period in periods:
            days = period_days.get(period, 30)
            cutoff = pd.Timestamp.now() - timedelta(days=days)
            period_hist = hist[hist.index >= cutoff]

            if len(period_hist) >= 2:
                ret = ((period_hist.iloc[-1] - period_hist.iloc[0]) / period_hist.iloc[0]) * 100
            else:
                ret = 0
            row_data[period] = round(ret, 1)

        data.append(row_data)

    if not data:
        return

    heatmap_df = pd.DataFrame(data)
    heatmap_df = heatmap_df.set_index('Ticker')

    # Sort by most recent period
    heatmap_df = heatmap_df.sort_values(periods[-1], ascending=False)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        colorscale=[
            [0, '#ef4444'],      # Red for negative
            [0.5, '#f5f5f5'],    # Neutral
            [1, '#22c55e']       # Green for positive
        ],
        zmid=0,
        text=[[f"{v:+.1f}%" for v in row] for row in heatmap_df.values],
        texttemplate="%{text}",
        textfont={"size": 11},
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:+.1f}%<extra></extra>"
    ))

    fig.update_layout(
        template=get_plotly_template(),
        title="Return Heatmap by Period",
        xaxis_title="Period",
        yaxis_title="",
        height=max(400, len(heatmap_df) * 30),
        yaxis=dict(tickmode='linear'),
        margin=dict(l=100, r=20, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True, key=f"heatmap_{name}")


def render_category_attribution(sector_attr_df, name=''):
    """Render stacked bar chart showing allocation effect vs selection effect per category."""
    if sector_attr_df.empty:
        return

    colors = get_theme_colors()

    fig = go.Figure()

    # Allocation Effect bars
    fig.add_trace(go.Bar(
        name='Allocation Effect',
        x=sector_attr_df['Category'],
        y=sector_attr_df['Allocation Effect %'],
        marker_color=colors['accent'],
        text=[f"{v:+.2f}%" for v in sector_attr_df['Allocation Effect %']],
        textposition='outside'
    ))

    # Selection Effect bars
    fig.add_trace(go.Bar(
        name='Selection Effect',
        x=sector_attr_df['Category'],
        y=sector_attr_df['Selection Effect %'],
        marker_color=colors['accent_secondary'],
        text=[f"{v:+.2f}%" for v in sector_attr_df['Selection Effect %']],
        textposition='outside'
    ))

    fig.update_layout(
        template=get_plotly_template(),
        title="Category Attribution Analysis",
        xaxis_title="Category",
        yaxis_title="Effect (%)",
        barmode='group',
        height=350,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    st.plotly_chart(fig, use_container_width=True, key=f"cat_attr_{name}")

    # Summary table
    st.dataframe(
        sector_attr_df.style.format({
            'Actual Weight %': '{:.1f}%',
            'Target Weight %': '{:.1f}%',
            'Return %': '{:+.2f}%',
            'Contribution %': '{:+.2f}%',
            'Allocation Effect %': '{:+.2f}%',
            'Selection Effect %': '{:+.2f}%'
        }).background_gradient(subset=['Return %', 'Contribution %'], cmap='RdYlGn'),
        use_container_width=True,
        hide_index=True
    )


def render_performance_summary(metrics, attribution_df, benchmark_data=None, benchmark_name="S&P 500", name=''):
    """Render key metrics: total return, alpha, beta, Sharpe with benchmark comparison."""
    if metrics is None:
        return

    colors = get_theme_colors()

    total_return = attribution_df['Contribution %'].sum() if not attribution_df.empty else 0

    # Calculate benchmark return for the period
    benchmark_return = 0
    if benchmark_data is not None and not benchmark_data.empty:
        benchmark_return = ((benchmark_data.iloc[-1] - benchmark_data.iloc[0]) / benchmark_data.iloc[0]) * 100

    alpha = total_return - benchmark_return

    # Create summary card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {colors['surface']} 0%, {colors['background']} 100%);
                border-radius: 16px; padding: 24px; margin-bottom: 20px;
                border: 1px solid {colors['accent']}40;">
        <h3 style="color: {colors['accent']}; margin-bottom: 16px; font-size: 18px;">Performance Summary</h3>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px;">
            <div style="text-align: center;">
                <div style="font-size: 28px; font-weight: 700; color: {'#22c55e' if total_return >= 0 else '#ef4444'};">
                    {total_return:+.2f}%
                </div>
                <div style="font-size: 12px; color: {colors['text_secondary']};">Total Return</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 28px; font-weight: 700; color: {'#22c55e' if alpha >= 0 else '#ef4444'};">
                    {alpha:+.2f}%
                </div>
                <div style="font-size: 12px; color: {colors['text_secondary']};">Alpha vs {benchmark_name}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 28px; font-weight: 700; color: {colors['text']};">
                    {metrics['portfolio_beta']:.2f}
                </div>
                <div style="font-size: 12px; color: {colors['text_secondary']};">Beta</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 28px; font-weight: 700; color: {colors['text']};">
                    {metrics['portfolio_vol']:.1f}%
                </div>
                <div style="font-size: 12px; color: {colors['text_secondary']};">Volatility</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_performance_attribution_section(holdings_df, ticker_data, metrics, cat_targets, name):
    """Render the full Performance Attribution section for a portfolio tab."""
    if holdings_df.empty or metrics is None:
        return

    st.subheader("📊 Performance Attribution")

    # Period selector
    period = st.selectbox(
        "Analysis Period",
        ['1W', '1M', '3M'],
        index=1,
        key=f"attr_period_{name}",
        help="Select time period for attribution analysis"
    )

    # Calculate attribution
    attr_df = calculate_attribution(holdings_df, ticker_data, period)

    if attr_df.empty:
        st.info(f"Not enough data for {period} attribution analysis")
        return

    # Fetch benchmark for comparison
    benchmark_data = fetch_benchmark_data("^GSPC")  # S&P 500
    period_days = {'1W': 7, '1M': 30, '3M': 90}
    days = period_days.get(period, 30)
    cutoff = pd.Timestamp.now() - timedelta(days=days)
    if not benchmark_data.empty:
        benchmark_data = benchmark_data[benchmark_data.index >= cutoff]

    # Performance Summary
    render_performance_summary(metrics, attr_df, benchmark_data, "S&P 500", name)

    # Main attribution views
    tab1, tab2, tab3, tab4 = st.tabs(["Waterfall", "Top Contributors", "Heatmap", "Category Analysis"])

    with tab1:
        render_attribution_waterfall(attr_df, period, name)

    with tab2:
        render_top_contributors(attr_df, n=5, name=name)

    with tab3:
        render_attribution_heatmap(holdings_df, ticker_data, ['1W', '1M', '3M'], name)

    with tab4:
        sector_df = calculate_sector_attribution(holdings_df, category_map, cat_targets, ticker_data, period)
        render_category_attribution(sector_df, name)


# ============================================================================
# TARGET ALLOCATION & REBALANCING
# ============================================================================

def calculate_category_rebalancing(holdings_df, category_targets, total_value):
    """Level 1: Category-level rebalancing analysis"""
    if holdings_df.empty or total_value == 0:
        return pd.DataFrame()

    results = []
    for category, target_pct in category_targets.items():
        current_value = holdings_df[holdings_df['Category'] == category]['Value £'].sum()
        current_pct = (current_value / total_value) * 100 if total_value > 0 else 0
        diff = target_pct - current_pct
        target_value = total_value * (target_pct / 100)
        diff_value = target_value - current_value

        results.append({
            'Category': category.title(),
            'Current £': int(round(current_value)),
            'Target £': int(round(target_value)),
            'Current %': round(current_pct, 1),
            'Target %': target_pct,
            'Drift %': round(diff, 1),
            'Action': "ADD" if diff > 0 else "REDUCE" if diff < 0 else "OK",
            'Amount £': int(round(abs(diff_value)))
        })

    return pd.DataFrame(results)

def calculate_holding_rebalancing(holdings_df, target_allocations, total_value):
    """Level 2: Per-holding rebalancing analysis"""
    if holdings_df.empty or total_value == 0 or not target_allocations:
        return pd.DataFrame()

    results = []
    for ticker, target_pct in target_allocations.items():
        row = holdings_df[holdings_df['Ticker'] == ticker]
        current_value = row['Value £'].sum() if len(row) > 0 else 0
        current_price = row['Price £'].iloc[0] if len(row) > 0 else 0
        current_pct = (current_value / total_value) * 100 if total_value > 0 else 0
        target_value = total_value * (target_pct / 100)
        diff_value = target_value - current_value
        shares_to_trade = int(diff_value / current_price) if current_price > 0 else 0

        results.append({
            'Ticker': ticker,
            'Current %': round(current_pct, 1),
            'Target %': target_pct,
            'Drift %': round(target_pct - current_pct, 1),
            'Action': "BUY" if diff_value > 0 else "SELL" if diff_value < 0 else "HOLD",
            'Amount £': int(round(abs(diff_value))),
            'Shares': abs(shares_to_trade)
        })

    return pd.DataFrame(results)

def render_rebalancing_tab(isa_metrics, sipp_metrics, isa_df, sipp_df):
    """Render the rebalancing tab with dual-level allocation tools"""
    col_r1, col_r2 = st.columns([10, 1])
    with col_r1:
        st.header("⚖️ Target Allocation & Rebalancing")
    with col_r2:
        render_inline_help_icon('rebalancing_help', 'Rebalancing Guide')

    # Show contextual tips for this tab
    render_contextual_help('rebalancing')

    # Combined portfolio value
    total_value = 0
    combined_df = pd.DataFrame()

    if isa_metrics:
        total_value += isa_metrics['total_value']
        combined_df = pd.concat([combined_df, isa_metrics['df']])
    if sipp_metrics:
        total_value += sipp_metrics['total_value']
        combined_df = pd.concat([combined_df, sipp_metrics['df']])

    if combined_df.empty:
        st.warning("⚠️ No portfolio data available")
        return

    # Aggregate by ticker for combined view
    combined_agg = combined_df.groupby(['Ticker', 'Category']).agg({
        'Value £': 'sum',
        'Price £': 'first'
    }).reset_index()

    st.metric("💰 Total Portfolio Value", f"£{total_value:,.0f}")

    # === LEVEL 1: CATEGORY ALLOCATION ===
    st.subheader("📊 Level 1: Category Allocation")

    col1, col2 = st.columns([2, 3])

    with col1:
        st.write("**Set Target Allocation**")

        # Current vs target visualization
        category_results = calculate_category_rebalancing(combined_agg, category_targets, total_value)

        if not category_results.empty:
            # Show current allocation pie
            current_alloc = combined_agg.groupby('Category')['Value £'].sum().reset_index()
            render_allocation_donut(
                current_alloc,
                title="Current Allocation",
                value_col='Value £',
                name_col='Category',
                hole_content={'label': 'Total', 'value': total_value},
                key_suffix="rebalance_cat",
                template=get_plotly_template()
            )

    with col2:
        st.write("**Rebalancing Recommendations**")

        if not category_results.empty:
            # Color-coded table
            def color_action(val):
                if val == "ADD":
                    return 'background-color: rgba(46, 204, 113, 0.3)'
                elif val == "REDUCE":
                    return 'background-color: rgba(231, 76, 60, 0.3)'
                return ''

            styled_df = category_results.style.applymap(color_action, subset=['Action'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            # Bar chart comparison - use premium chart
            render_comparison_bar_chart(
                category_results,
                current_col='Current %',
                target_col='Target %',
                label_col='Category',
                title="Current vs Target Allocation",
                key_suffix="cat_rebalance",
                template=get_plotly_template()
            )

            # Drift gauges for each category
            st.write("**Category Drift Gauges**")
            gauge_cols = st.columns(len(category_results))
            for idx, row in category_results.iterrows():
                with gauge_cols[idx]:
                    render_drift_gauge(
                        row['Drift %'],
                        label=row['Category'],
                        key_suffix=f"drift_{row['Category']}"
                    )

    # === LEVEL 2: HOLDING ALLOCATION ===
    st.subheader("📋 Level 2: Per-Holding Targets")

    if target_allocations:
        holding_results = calculate_holding_rebalancing(combined_agg, target_allocations, total_value)

        if not holding_results.empty:
            # Summary metrics
            total_buy = holding_results[holding_results['Action'] == 'BUY']['Amount £'].sum()
            total_sell = holding_results[holding_results['Action'] == 'SELL']['Amount £'].sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 Total to Buy", f"£{total_buy:,.0f}")
            c2.metric("🔴 Total to Sell", f"£{total_sell:,.0f}")
            c3.metric("📊 Net Rebalance", f"£{abs(total_buy - total_sell):,.0f}")

            # Color-coded holdings table
            def color_holding_action(val):
                if val == "BUY":
                    return 'background-color: rgba(46, 204, 113, 0.3)'
                elif val == "SELL":
                    return 'background-color: rgba(231, 76, 60, 0.3)'
                return ''

            styled_holdings = holding_results.style.applymap(color_holding_action, subset=['Action'])
            st.dataframe(styled_holdings, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ No per-holding targets set. Configure targets in the sidebar.")

    # === TARGET CONFIGURATION ===
    with st.expander("⚙️ Configure Targets"):
        st.write("**Category Targets** (must sum to 100%)")

        col1, col2, col3 = st.columns(3)
        with st.form("category_targets_form"):
            new_income = col1.number_input("Income %", 0.0, 100.0, float(category_targets.get('income', 50)), 1.0)
            new_growth = col2.number_input("Growth %", 0.0, 100.0, float(category_targets.get('growth', 35)), 1.0)
            new_spec = col3.number_input("Speculative %", 0.0, 100.0, float(category_targets.get('speculative', 15)), 1.0)

            total_pct = new_income + new_growth + new_spec
            if abs(total_pct - 100) > 0.1:
                st.error(f"⚠️ Total is {total_pct:.1f}% - must equal 100%")

            if st.form_submit_button("💾 Save Category Targets", use_container_width=True):
                if abs(total_pct - 100) <= 0.1:
                    config['category_targets'] = {
                        'income': new_income,
                        'growth': new_growth,
                        'speculative': new_spec
                    }
                    save_config(config)
                    st.success("✅ Saved!")
                    st.rerun()

        st.divider()
        st.write("**Per-Holding Targets** (optional)")

        # Show all holdings with optional target input
        all_tickers = combined_agg['Ticker'].unique().tolist()

        with st.form("holding_targets_form"):
            new_targets = {}
            cols = st.columns(3)
            for i, ticker in enumerate(all_tickers):
                current_target = target_allocations.get(ticker, 0)
                new_val = cols[i % 3].number_input(
                    ticker,
                    0.0, 100.0,
                    float(current_target),
                    0.5,
                    key=f"target_{ticker}"
                )
                if new_val > 0:
                    new_targets[ticker] = new_val

            if st.form_submit_button("💾 Save Holding Targets", use_container_width=True):
                config['target_allocations'] = new_targets
                save_config(config)
                st.success("✅ Saved!")
                st.rerun()

# ============================================================================
# AI CHAT ASSISTANT
# ============================================================================

def get_portfolio_context(isa_metrics, sipp_metrics, isa_df, sipp_df):
    """Build comprehensive context string for AI assistant"""
    context_parts = []

    # Portfolio summary
    total_value = 0
    total_annual_income = 0
    all_holdings = []

    if isa_metrics:
        total_value += isa_metrics['total_value']
        total_annual_income += isa_metrics['projected_annual']
        for _, row in isa_metrics['df'].iterrows():
            all_holdings.append({
                'Account': 'ISA',
                'Ticker': row['Ticker'],
                'Category': row['Category'],
                'Value': row['Value £'],
                'Weight': row['Weight %'],
                'PL': row['P/L %'],
                'Yield': row['Trail Yield %']
            })

    if sipp_metrics:
        total_value += sipp_metrics['total_value']
        total_annual_income += sipp_metrics['projected_annual']
        for _, row in sipp_metrics['df'].iterrows():
            all_holdings.append({
                'Account': 'SIPP',
                'Ticker': row['Ticker'],
                'Category': row['Category'],
                'Value': row['Value £'],
                'Weight': row['Weight %'],
                'PL': row['P/L %'],
                'Yield': row['Trail Yield %']
            })

    context_parts.append(f"""
PORTFOLIO OVERVIEW:
- Total Portfolio Value: £{total_value:,.0f}
- ISA Value: £{isa_metrics['total_value']:,.0f if isa_metrics else 0}
- SIPP Value: £{sipp_metrics['total_value']:,.0f if sipp_metrics else 0}
- Projected Annual Dividend Income: £{total_annual_income:,.0f}
- Projected Monthly Income: £{total_annual_income/12:,.0f}
- Number of Holdings: {len(all_holdings)}
""")

    # Holdings breakdown
    if all_holdings:
        holdings_df = pd.DataFrame(all_holdings)
        top_holdings = holdings_df.nlargest(10, 'Value')
        context_parts.append("\nTOP 10 HOLDINGS BY VALUE:")
        for _, h in top_holdings.iterrows():
            context_parts.append(f"  - {h['Ticker']} ({h['Account']}): £{h['Value']:,.0f} ({h['Weight']:.1f}%), P/L: {h['PL']:+.1f}%, Yield: {h['Yield']:.1f}%")

        # Category breakdown
        cat_summary = holdings_df.groupby('Category').agg({'Value': 'sum'}).reset_index()
        cat_summary['Pct'] = cat_summary['Value'] / total_value * 100
        context_parts.append("\nCATEGORY ALLOCATION:")
        for _, c in cat_summary.iterrows():
            context_parts.append(f"  - {c['Category'].title()}: £{c['Value']:,.0f} ({c['Pct']:.1f}%)")

        # Best/worst performers
        best = holdings_df.nlargest(3, 'PL')
        worst = holdings_df.nsmallest(3, 'PL')
        context_parts.append("\nBEST PERFORMERS:")
        for _, h in best.iterrows():
            context_parts.append(f"  - {h['Ticker']}: {h['PL']:+.1f}%")
        context_parts.append("\nWORST PERFORMERS:")
        for _, h in worst.iterrows():
            context_parts.append(f"  - {h['Ticker']}: {h['PL']:+.1f}%")

        # Highest yielders
        high_yield = holdings_df.nlargest(5, 'Yield')
        context_parts.append("\nHIGHEST YIELDING HOLDINGS:")
        for _, h in high_yield.iterrows():
            context_parts.append(f"  - {h['Ticker']}: {h['Yield']:.1f}% yield")

    # Recent dividends
    all_divs = pd.DataFrame()
    for df in [isa_df, sipp_df]:
        if df is not None and not df.empty:
            divs = df[df['Type'].str.contains('DIVIDEND|INTEREST', na=False)]
            all_divs = pd.concat([all_divs, divs])

    if not all_divs.empty:
        all_divs['Timestamp'] = pd.to_datetime(all_divs['Timestamp'])
        recent = all_divs.nlargest(10, 'Timestamp')
        context_parts.append("\nRECENT DIVIDEND PAYMENTS:")
        for _, d in recent.iterrows():
            context_parts.append(f"  - {d['Timestamp'].strftime('%Y-%m-%d')}: £{d['Total Amount']:.2f} from {d.get('Ticker', 'Unknown')}")

    return "\n".join(context_parts)

def chat_with_portfolio(user_query, context, api_key):
    """Send query to OpenAI with portfolio context"""
    if not OPENAI_AVAILABLE:
        return "❌ OpenAI package not installed. Run: pip install openai"

    if not api_key:
        return "❌ Please configure your OpenAI API key in the sidebar settings."

    try:
        client = OpenAI(api_key=api_key)

        system_prompt = f"""You are a helpful portfolio assistant for a UK investor. You have access to their current portfolio data.
Be concise but informative. Use £ for currency. When discussing performance, be objective.
If asked about rebalancing or trades, provide specific actionable suggestions based on their holdings.

CURRENT PORTFOLIO DATA:
{context}

Guidelines:
- Be specific with numbers from the data
- For yield calculations, use the provided yield data
- For rebalancing advice, consider their current category allocation
- Be honest about limitations (you only see current snapshot, not full history)
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            max_tokens=800
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"

def render_ai_assistant(isa_metrics, sipp_metrics, isa_df, sipp_df):
    """Render the AI chat assistant interface"""
    col_a1, col_a2 = st.columns([10, 1])
    with col_a1:
        st.header("🤖 AI Portfolio Assistant")
    with col_a2:
        render_inline_help_icon('ai_assistant_help', 'AI Assistant Guide')

    # Show contextual tips for this tab
    render_contextual_help('assistant')

    if not OPENAI_AVAILABLE:
        st.error("❌ OpenAI package not installed. Run: `pip install openai`")
        return

    # API Key configuration
    api_key = config.get('openai_api_key', '') or os.getenv('OPENAI_API_KEY', '')

    if not api_key:
        st.warning("⚠️ No OpenAI API key configured")
        with st.expander("🔑 Configure API Key"):
            new_key = st.text_input("OpenAI API Key", type="password")
            if st.button("💾 Save Key"):
                config['openai_api_key'] = new_key
                save_config(config)
                st.success("✅ Saved!")
                st.rerun()
        return

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Build context
    context = get_portfolio_context(isa_metrics, sipp_metrics, isa_df, sipp_df)

    # Example questions
    st.caption("💡 Try asking:")
    col1, col2 = st.columns(2)
    example_qs = [
        "What's my best performing stock?",
        "How much dividend income do I get monthly?",
        "Am I overweight in any category?",
        "What should I buy next for income?",
        "Compare my ISA vs SIPP",
        "Which holdings have the highest yield?"
    ]

    for i, q in enumerate(example_qs):
        col = col1 if i % 2 == 0 else col2
        if col.button(q, key=f"example_{i}", use_container_width=True):
            st.session_state.pending_query = q

    st.divider()

    # Chat input
    user_query = st.text_input("Ask about your portfolio...", key="chat_input",
                                value=st.session_state.get('pending_query', ''))

    # Clear pending query after it's used
    if 'pending_query' in st.session_state:
        del st.session_state.pending_query

    if user_query:
        with st.spinner("🤔 Thinking..."):
            response = chat_with_portfolio(user_query, context, api_key)

        # Add to history
        st.session_state.chat_history.append({
            'query': user_query,
            'response': response
        })

    # Display chat history (most recent first)
    if st.session_state.chat_history:
        st.subheader("💬 Conversation")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Last 5
            with st.container():
                st.markdown(f"**You:** {chat['query']}")
                st.markdown(f"**Assistant:** {chat['response']}")
                st.divider()

        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


# ============================================================================
# TRANSACTION HISTORY VIEW
# ============================================================================

def get_all_transactions(isa_df, sipp_df):
    """Combine all transactions from ISA and SIPP accounts"""
    all_trans = []

    if isa_df is not None and not isa_df.empty:
        isa_copy = isa_df.copy()
        isa_copy['Account'] = 'ISA'
        all_trans.append(isa_copy)

    if sipp_df is not None and not sipp_df.empty:
        sipp_copy = sipp_df.copy()
        sipp_copy['Account'] = 'SIPP'
        all_trans.append(sipp_copy)

    if not all_trans:
        return pd.DataFrame()

    combined = pd.concat(all_trans, ignore_index=True)
    combined['Timestamp'] = pd.to_datetime(combined['Timestamp'], errors='coerce')
    combined['Total Amount'] = pd.to_numeric(combined['Total Amount'], errors='coerce').fillna(0)
    combined['Quantity'] = pd.to_numeric(combined.get('Quantity', 0), errors='coerce').fillna(0)
    combined['Price per Share in Account Currency'] = pd.to_numeric(
        combined.get('Price per Share in Account Currency', 0), errors='coerce'
    ).fillna(0)
    combined['Title'] = combined['Title'].fillna('Unknown')
    combined['Ticker'] = combined['Ticker'].fillna('')
    combined['Type'] = combined['Type'].fillna('UNKNOWN')
    combined = combined.sort_values('Timestamp', ascending=False)
    return combined

def calculate_transaction_metrics(transactions_df):
    """Calculate key transaction metrics for summary"""
    if transactions_df.empty:
        return {'total_transactions': 0, 'total_invested': 0, 'total_dividends': 0, 'total_interest': 0,
                'avg_transaction_size': 0, 'most_traded_ticker': 'N/A', 'most_traded_count': 0,
                'buy_count': 0, 'sell_count': 0, 'trading_frequency_days': 0,
                'first_transaction': None, 'last_transaction': None}

    orders = transactions_df[transactions_df['Type'] == 'ORDER']
    dividends = transactions_df[transactions_df['Type'] == 'DIVIDEND']
    interest = transactions_df[transactions_df['Type'] == 'INTEREST']
    buy_orders = orders[orders['Buy / Sell'] == 'BUY']
    sell_orders = orders[orders['Buy / Sell'] == 'SELL']
    total_invested = abs(buy_orders['Total Amount'].sum())
    total_dividends = dividends['Total Amount'].sum()
    total_interest = interest['Total Amount'].sum()
    avg_transaction = abs(orders['Total Amount']).mean() if len(orders) > 0 else 0

    if len(orders) > 0:
        ticker_counts = orders[orders['Ticker'].notna() & (orders['Ticker'] != '')]['Ticker'].value_counts()
        most_traded = ticker_counts.index[0] if len(ticker_counts) > 0 else 'N/A'
        most_traded_count = ticker_counts.iloc[0] if len(ticker_counts) > 0 else 0
    else:
        most_traded, most_traded_count = 'N/A', 0

    if len(orders) > 1:
        orders_sorted = orders.sort_values('Timestamp')
        date_range = (orders_sorted['Timestamp'].max() - orders_sorted['Timestamp'].min()).days
        trading_freq = date_range / len(orders) if len(orders) > 0 else 0
    else:
        trading_freq = 0

    return {'total_transactions': len(transactions_df), 'total_invested': total_invested,
            'total_dividends': total_dividends, 'total_interest': total_interest,
            'avg_transaction_size': avg_transaction, 'most_traded_ticker': most_traded,
            'most_traded_count': most_traded_count, 'buy_count': len(buy_orders),
            'sell_count': len(sell_orders), 'trading_frequency_days': trading_freq,
            'first_transaction': transactions_df['Timestamp'].min(),
            'last_transaction': transactions_df['Timestamp'].max()}

def render_transaction_summary(transactions_df):
    """Render transaction summary statistics with charts"""
    if transactions_df.empty:
        st.info("No transactions found")
        return

    metrics = calculate_transaction_metrics(transactions_df)

    st.subheader("Transaction Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{metrics['total_transactions']:,}")
    c2.metric("Total Invested", f"£{metrics['total_invested']:,.0f}")
    c3.metric("Dividends Received", f"£{metrics['total_dividends']:,.0f}")
    c4.metric("Interest Received", f"£{metrics['total_interest']:,.0f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Order Size", f"£{metrics['avg_transaction_size']:,.0f}")
    c6.metric("Most Traded", f"{metrics['most_traded_ticker']} ({metrics['most_traded_count']}x)")
    c7.metric("Buy Orders", f"{metrics['buy_count']:,}")
    c8.metric("Sell Orders", f"{metrics['sell_count']:,}")

    col1, col2 = st.columns(2)
    with col1:
        type_counts = transactions_df['Type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        fig = px.pie(type_counts, values='Count', names='Type',
                     title="Transactions by Type", template=get_plotly_template(), hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True, key="trans_type_pie")

    with col2:
        trans_monthly = transactions_df.copy()
        trans_monthly['Month'] = trans_monthly['Timestamp'].dt.to_period('M').astype(str)
        monthly_counts = trans_monthly.groupby('Month').size().reset_index(name='Count')
        monthly_counts = monthly_counts.sort_values('Month')
        fig = px.bar(monthly_counts, x='Month', y='Count',
                     title="Transactions by Month", template=get_plotly_template(),
                     color_discrete_sequence=['#3498db'])
        fig.update_layout(height=300, xaxis_title="Month", yaxis_title="Transaction Count")
        st.plotly_chart(fig, use_container_width=True, key="trans_monthly_bar")

def render_transaction_filters():
    """Render transaction filter controls and return filter values"""
    with st.expander("Filter Transactions", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            type_filter = st.multiselect("Transaction Type",
                options=['ORDER', 'DIVIDEND', 'INTEREST', 'TOP_UP', 'WITHDRAWAL'],
                default=[], key="trans_type_filter")
        with col2:
            account_filter = st.multiselect("Account", options=['ISA', 'SIPP'],
                default=[], key="trans_account_filter")
        with col3:
            date_range = st.date_input("Date Range", value=[], key="trans_date_range")
        with col4:
            direction_filter = st.multiselect("Buy/Sell", options=['BUY', 'SELL'],
                default=[], key="trans_direction_filter")

        col5, col6 = st.columns([3, 1])
        with col5:
            ticker_filter = st.text_input("Ticker (comma-separated)",
                placeholder="e.g., MSTR, VZLA, JEQP", key="trans_ticker_filter")
        with col6:
            reset_btn = st.button("Reset Filters", key="reset_trans_filters", use_container_width=True)

    return {'type': type_filter if type_filter else None,
            'account': account_filter if account_filter else None,
            'date_range': date_range if len(date_range) == 2 else None,
            'direction': direction_filter if direction_filter else None,
            'ticker': [t.strip().upper() for t in ticker_filter.split(',') if t.strip()] if ticker_filter else None,
            'reset': reset_btn}

def apply_transaction_filters(df, filters):
    """Apply filters to transaction dataframe"""
    if df.empty:
        return df
    filtered = df.copy()
    if filters['type']:
        filtered = filtered[filtered['Type'].isin(filters['type'])]
    if filters['account']:
        filtered = filtered[filtered['Account'].isin(filters['account'])]
    if filters['date_range']:
        start_date, end_date = filters['date_range']
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        filtered = filtered[(filtered['Timestamp'] >= start_dt) & (filtered['Timestamp'] < end_dt)]
    if filters['direction']:
        filtered = filtered[filtered['Buy / Sell'].isin(filters['direction'])]
    if filters['ticker']:
        filtered = filtered[filtered['Ticker'].str.upper().isin(filters['ticker'])]
    return filtered

def render_transaction_search(transactions_df):
    """Render search box for transactions"""
    search_term = st.text_input("Search transactions",
        placeholder="Search by ticker, title, or description...", key="trans_search")
    if search_term and not transactions_df.empty:
        term_lower = search_term.lower()
        mask = (transactions_df['Ticker'].str.lower().str.contains(term_lower, na=False) |
                transactions_df['Title'].str.lower().str.contains(term_lower, na=False) |
                transactions_df['Type'].str.lower().str.contains(term_lower, na=False))
        return transactions_df[mask]
    return transactions_df

def render_transaction_history(transactions_df, page_size=25):
    """Render paginated transaction table"""
    if transactions_df.empty:
        st.info("No transactions match your filters")
        return

    if 'trans_page' not in st.session_state:
        st.session_state.trans_page = 0

    total_records = len(transactions_df)
    total_pages = math.ceil(total_records / page_size)

    if st.session_state.trans_page >= total_pages:
        st.session_state.trans_page = max(0, total_pages - 1)

    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col1:
        if st.button("First", disabled=st.session_state.trans_page == 0, key="trans_first"):
            st.session_state.trans_page = 0
            st.rerun()
    with col2:
        if st.button("Prev", disabled=st.session_state.trans_page == 0, key="trans_prev"):
            st.session_state.trans_page -= 1
            st.rerun()
    with col3:
        st.markdown(f"<div style='text-align: center;'>Page {st.session_state.trans_page + 1} of {total_pages} ({total_records:,} records)</div>", unsafe_allow_html=True)
    with col4:
        if st.button("Next", disabled=st.session_state.trans_page >= total_pages - 1, key="trans_next"):
            st.session_state.trans_page += 1
            st.rerun()
    with col5:
        if st.button("Last", disabled=st.session_state.trans_page >= total_pages - 1, key="trans_last"):
            st.session_state.trans_page = total_pages - 1
            st.rerun()

    start_idx = st.session_state.trans_page * page_size
    end_idx = min(start_idx + page_size, total_records)
    page_data = transactions_df.iloc[start_idx:end_idx]

    display_df = page_data[['Timestamp', 'Type', 'Ticker', 'Title', 'Quantity',
        'Price per Share in Account Currency', 'Total Amount', 'Account', 'Buy / Sell']].copy()
    display_df.columns = ['Date', 'Type', 'Ticker', 'Description', 'Qty', 'Price', 'Total', 'Account', 'Side']
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['Description'] = display_df['Description'].str[:30]
    display_df['Qty'] = display_df['Qty'].apply(lambda x: f"{x:.4f}" if x > 0 else "-")
    display_df['Price'] = display_df['Price'].apply(lambda x: f"£{x:.2f}" if x > 0 else "-")
    display_df['Total'] = display_df['Total'].apply(lambda x: f"£{x:,.2f}")
    display_df['Side'] = display_df['Side'].fillna('-')

    def style_row(row):
        type_val = row['Type']
        if type_val == 'ORDER':
            if row['Side'] == 'BUY':
                return ['background-color: rgba(46, 204, 113, 0.15)'] * len(row)
            elif row['Side'] == 'SELL':
                return ['background-color: rgba(231, 76, 60, 0.15)'] * len(row)
        elif type_val == 'DIVIDEND':
            return ['background-color: rgba(52, 152, 219, 0.15)'] * len(row)
        elif type_val == 'INTEREST':
            return ['background-color: rgba(155, 89, 182, 0.15)'] * len(row)
        elif type_val == 'TOP_UP':
            return ['background-color: rgba(241, 196, 15, 0.15)'] * len(row)
        return [''] * len(row)

    styled_df = display_df.style.apply(style_row, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=500, hide_index=True)
    st.caption("Legend: BUY orders | SELL orders | DIVIDEND | INTEREST | TOP_UP")

def render_transaction_detail_modal(transactions_df):
    """Render a transaction detail view when a ticker is selected"""
    tickers_with_orders = transactions_df[
        (transactions_df['Type'] == 'ORDER') &
        (transactions_df['Ticker'].notna()) &
        (transactions_df['Ticker'] != '')
    ]['Ticker'].unique().tolist()

    if not tickers_with_orders:
        return

    with st.expander("Transaction Details by Ticker"):
        selected_ticker = st.selectbox("Select a ticker to view all transactions",
            options=[''] + sorted(tickers_with_orders), key="trans_detail_ticker")

        if selected_ticker:
            ticker_trans = transactions_df[transactions_df['Ticker'] == selected_ticker].copy()
            if not ticker_trans.empty:
                orders = ticker_trans[ticker_trans['Type'] == 'ORDER']
                dividends = ticker_trans[ticker_trans['Type'] == 'DIVIDEND']

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Orders", len(orders))
                col2.metric("Total Invested", f"£{abs(orders['Total Amount'].sum()):,.0f}")
                col3.metric("Dividends Received", f"£{dividends['Total Amount'].sum():,.0f}")
                col4.metric("Total Shares", f"{orders['Quantity'].sum():,.4f}")

                st.write(f"**All transactions for {selected_ticker}:**")
                detail_df = ticker_trans[['Timestamp', 'Type', 'Quantity', 'Price per Share in Account Currency',
                    'Total Amount', 'Account', 'Buy / Sell']].copy()
                detail_df.columns = ['Date', 'Type', 'Qty', 'Price', 'Total', 'Account', 'Side']
                detail_df['Date'] = detail_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
                detail_df = detail_df.sort_values('Date', ascending=False)
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

                st.write("**Related transactions in same session:**")
                if len(ticker_trans) > 0:
                    sample_date = ticker_trans['Timestamp'].iloc[0].date()
                    same_day = transactions_df[transactions_df['Timestamp'].dt.date == sample_date]
                    other_tickers = same_day[
                        (same_day['Ticker'] != selected_ticker) &
                        (same_day['Ticker'].notna()) &
                        (same_day['Ticker'] != '')
                    ]['Ticker'].unique()
                    if len(other_tickers) > 0:
                        st.write(f"Other tickers traded on {sample_date}: {', '.join(other_tickers[:10])}")
                    else:
                        st.caption("No other tickers traded on the same day")

def render_transactions_tab(isa_df, sipp_df):
    """Render the complete Transactions tab"""
    st.header("Transaction History")
    all_transactions = get_all_transactions(isa_df, sipp_df)

    if all_transactions.empty:
        st.warning("No transaction data available. Ensure CSV files are loaded.")
        return

    render_transaction_summary(all_transactions)
    st.divider()

    filters = render_transaction_filters()
    if filters['reset']:
        st.session_state.trans_page = 0
        st.rerun()

    filtered_transactions = apply_transaction_filters(all_transactions, filters)
    st.subheader("Transaction Ledger")
    filtered_transactions = render_transaction_search(filtered_transactions)
    st.caption(f"Showing {len(filtered_transactions):,} of {len(all_transactions):,} transactions")

    render_transaction_history(filtered_transactions)
    st.divider()
    render_transaction_detail_modal(all_transactions)

# ============================================================================
# WATCHLIST FEATURE
# ============================================================================

# Common tickers for auto-suggest
COMMON_TICKERS = [
    # UK ETFs
    "VUSA.L", "VWRL.L", "VFEM.L", "VUAG.L", "VUKE.L", "ISF.L", "IUKD.L",
    "VHYL.L", "HDLV.L", "IDVY.L", "SUAG.L", "SGLN.L", "PHAU.L",
    # US Stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "MA", "JNJ", "PG", "KO", "PEP", "WMT", "HD", "DIS",
    # UK Stocks
    "SHEL.L", "BP.L", "HSBA.L", "ULVR.L", "RIO.L", "GLEN.L", "LSEG.L",
    # Income ETFs
    "QYLP.L", "JEQP.L", "YMAP.L", "FEPG.L",
    # Mining/Commodities
    "AG", "EXK", "CDE", "VZLA", "THS.L",
    # Other popular
    "SPY", "QQQ", "IWM", "VTI", "VOO", "SCHD", "JEPI", "JEPQ"
]

def load_watchlist():
    """Load watchlist from config.json"""
    if os.path.exists(CONFIG_FILENAME):
        with open(CONFIG_FILENAME, 'r') as f:
            cfg = json.load(f)
            watchlist = cfg.get('watchlist', {})
            if 'items' not in watchlist:
                watchlist = {'items': [], 'alerts': []}
            return watchlist
    return {'items': [], 'alerts': []}

def save_watchlist(watchlist):
    """Save watchlist to config.json, preserving other config data"""
    if os.path.exists(CONFIG_FILENAME):
        with open(CONFIG_FILENAME, 'r') as f:
            cfg = json.load(f)
    else:
        cfg = {}
    cfg['watchlist'] = watchlist
    with open(CONFIG_FILENAME, 'w') as f:
        json.dump(cfg, f, indent=4, sort_keys=True)

def fetch_single_watchlist_ticker(ticker):
    """Fetch data for a single watchlist ticker"""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
        if price is None:
            return None
        prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose') or price
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
        high_52w = info.get('fiftyTwoWeekHigh', 0)
        low_52w = info.get('fiftyTwoWeekLow', 0)
        volume = info.get('regularMarketVolume', 0) or info.get('volume', 0)
        name = info.get('shortName') or info.get('longName') or ticker
        currency = info.get('currency', 'USD').upper()
        try:
            hist_7d = t.history(period='7d')['Close'].tolist()
        except:
            hist_7d = []
        return {
            'ticker': ticker, 'name': name[:30] if len(name) > 30 else name,
            'price': price, 'change_pct': round(change_pct, 2), 'volume': volume,
            '52w_high': high_52w, '52w_low': low_52w, 'currency': currency, 'hist_7d': hist_7d
        }
    except:
        return None

@st.cache_data(ttl=900)
def fetch_watchlist_data(watchlist_items):
    """Fetch current prices for all watchlist items using parallel execution"""
    if not watchlist_items:
        return pd.DataFrame()
    results, failed = [], []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(fetch_single_watchlist_ticker, tk): tk for tk in watchlist_items}
        for future in as_completed(future_to_ticker):
            tk = future_to_ticker[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed.append(tk)
            except:
                failed.append(tk)
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values('ticker')

def create_sparkline(data, width=100, height=30):
    """Create a small sparkline chart for 7-day price trend"""
    if not data or len(data) < 2:
        return None
    fig = go.Figure()
    color = '#00CC96' if data[-1] >= data[0] else '#EF553B'
    fig.add_trace(go.Scatter(y=data, mode='lines', line=dict(color=color, width=2),
        fill='tozeroy', fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)'))
    fig.update_layout(width=width, height=height, margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def render_watchlist_table(watchlist_df, watchlist, portfolio_tickers=None):
    """Render the watchlist table with sparklines and actions"""
    if watchlist_df.empty:
        st.info("Your watchlist is empty. Add some tickers to start tracking!")
        return
    st.subheader("Tracked Securities")
    gainers = len(watchlist_df[watchlist_df['change_pct'] > 0])
    losers = len(watchlist_df[watchlist_df['change_pct'] < 0])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Watching", len(watchlist_df))
    c2.metric("Gainers Today", gainers)
    c3.metric("Losers Today", losers)
    avg_change = watchlist_df['change_pct'].mean()
    c4.metric("Avg Change", f"{avg_change:+.2f}%")
    st.divider()
    items_to_remove = []
    for idx, row in watchlist_df.iterrows():
        col1, col2, col3, col4, col5, col6 = st.columns([2, 1.5, 1, 1.5, 1, 1])
        with col1:
            st.markdown(f"**{row['ticker']}**")
            st.caption(row['name'])
        with col2:
            change_color = "green" if row['change_pct'] >= 0 else "red"
            arrow = "+" if row['change_pct'] >= 0 else ""
            currency_sym = "GBP" if row['currency'] in ['GBP', 'GBX', 'GBp'] else "$"
            display_price = row['price'] / 100 if row['currency'] in ['GBX', 'GBp'] else row['price']
            st.markdown(f"{currency_sym}{display_price:,.2f}")
            st.markdown(f"<span style='color:{change_color}'>{arrow}{row['change_pct']:.2f}%</span>", unsafe_allow_html=True)
        with col3:
            if row['hist_7d'] and len(row['hist_7d']) > 1:
                fig = create_sparkline(row['hist_7d'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"spark_{row['ticker']}")
            else:
                st.caption("No data")
        with col4:
            if row['52w_high'] and row['52w_low']:
                pct_of_high = (row['price'] / row['52w_high'] * 100) if row['52w_high'] else 0
                st.caption(f"52W: {currency_sym}{row['52w_low']:,.0f} - {currency_sym}{row['52w_high']:,.0f}")
                st.progress(min(pct_of_high / 100, 1.0))
            else:
                st.caption("N/A")
        with col5:
            if portfolio_tickers and row['ticker'] in portfolio_tickers:
                st.success("In Portfolio")
            else:
                st.caption("Not owned")
        with col6:
            if st.button("Remove", key=f"remove_{row['ticker']}"):
                items_to_remove.append(row['ticker'])
        st.divider()
    if items_to_remove:
        for ticker in items_to_remove:
            if ticker in watchlist['items']:
                watchlist['items'].remove(ticker)
                watchlist['alerts'] = [a for a in watchlist['alerts'] if a['ticker'] != ticker]
        save_watchlist(watchlist)
        st.success(f"Removed {len(items_to_remove)} item(s) from watchlist")
        st.rerun()

def validate_ticker(ticker):
    """Validate that a ticker exists and can be fetched"""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
        return price is not None
    except:
        return False

def render_add_to_watchlist(watchlist):
    """Render the add ticker form with auto-suggest"""
    st.subheader("Add to Watchlist")
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input("Enter ticker symbol", placeholder="e.g., AAPL, VUSA.L, MSFT",
            help="Enter a Yahoo Finance ticker symbol. UK stocks typically end with .L",
            key="watchlist_ticker_input").upper().strip()
        if ticker_input and len(ticker_input) >= 1:
            suggestions = [t for t in COMMON_TICKERS if t.upper().startswith(ticker_input)][:5]
            if suggestions and ticker_input not in suggestions:
                st.caption(f"Suggestions: {', '.join(suggestions)}")
    with col2:
        add_clicked = st.button("Add", key="add_watchlist_btn", use_container_width=True)
    if add_clicked and ticker_input:
        if ticker_input in watchlist['items']:
            st.warning(f"{ticker_input} is already in your watchlist")
        else:
            with st.spinner(f"Validating {ticker_input}..."):
                if validate_ticker(ticker_input):
                    watchlist['items'].append(ticker_input)
                    save_watchlist(watchlist)
                    st.success(f"Added {ticker_input} to watchlist!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"Could not find ticker: {ticker_input}. Check the symbol and try again.")
    st.caption("Quick add popular tickers:")
    quick_cols = st.columns(6)
    quick_tickers = ["VUSA.L", "AAPL", "MSFT", "NVDA", "SPY", "VWRL.L"]
    for i, qt in enumerate(quick_tickers):
        if qt not in watchlist['items']:
            if quick_cols[i].button(qt, key=f"quick_{qt}", use_container_width=True):
                with st.spinner(f"Adding {qt}..."):
                    if validate_ticker(qt):
                        watchlist['items'].append(qt)
                        save_watchlist(watchlist)
                        st.cache_data.clear()
                        st.rerun()

def render_price_alerts(watchlist):
    """Render price alerts configuration and management"""
    st.subheader("Price Alerts")
    alerts = watchlist.get('alerts', [])
    with st.expander("Set New Alert", expanded=False):
        if not watchlist['items']:
            st.info("Add items to your watchlist first to set alerts")
        else:
            with st.form("new_alert_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    alert_ticker = st.selectbox("Ticker", watchlist['items'], key="alert_ticker")
                with col2:
                    alert_type = st.selectbox("Condition", ["Above", "Below"], key="alert_type")
                with col3:
                    alert_price = st.number_input("Price", min_value=0.0, step=0.01, key="alert_price")
                if st.form_submit_button("Create Alert", use_container_width=True):
                    if alert_ticker and alert_price > 0:
                        new_alert = {'id': str(uuid.uuid4())[:8], 'ticker': alert_ticker,
                            'type': alert_type.lower(), 'price': alert_price,
                            'created': datetime.now().isoformat(), 'triggered': False}
                        if 'alerts' not in watchlist:
                            watchlist['alerts'] = []
                        watchlist['alerts'].append(new_alert)
                        save_watchlist(watchlist)
                        st.success(f"Alert created: {alert_ticker} {alert_type.lower()} {alert_price}")
                        st.rerun()
    if alerts:
        st.write("**Active Alerts**")
        alerts_to_delete = []
        for alert in alerts:
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            with col1:
                condition_icon = "^" if alert['type'] == 'above' else "v"
                st.markdown(f"**{alert['ticker']}** {condition_icon} {alert['price']:.2f}")
            with col2:
                st.caption(f"Type: {alert['type'].title()}")
            with col3:
                if alert.get('triggered'):
                    st.error("TRIGGERED!")
                else:
                    st.success("Active")
            with col4:
                if st.button("X", key=f"del_alert_{alert['id']}", help="Delete alert"):
                    alerts_to_delete.append(alert['id'])
        if alerts_to_delete:
            watchlist['alerts'] = [a for a in watchlist['alerts'] if a['id'] not in alerts_to_delete]
            save_watchlist(watchlist)
            st.rerun()
    else:
        st.caption("No alerts configured. Set alerts to get notified when prices hit your targets.")

def check_price_alerts(watchlist_df, alerts):
    """Check if any price alerts have been triggered"""
    if watchlist_df.empty or not alerts:
        return []
    triggered = []
    for alert in alerts:
        if alert.get('triggered'):
            continue
        ticker = alert['ticker']
        row = watchlist_df[watchlist_df['ticker'] == ticker]
        if row.empty:
            continue
        current_price = row.iloc[0]['price']
        target_price = alert['price']
        if alert['type'] == 'above' and current_price >= target_price:
            triggered.append({'alert': alert, 'current_price': current_price,
                'message': f"{ticker} is above {target_price:.2f} (current: {current_price:.2f})"})
        elif alert['type'] == 'below' and current_price <= target_price:
            triggered.append({'alert': alert, 'current_price': current_price,
                'message': f"{ticker} is below {target_price:.2f} (current: {current_price:.2f})"})
    return triggered

def render_watchlist_tab(portfolio_tickers=None):
    """Render the complete watchlist tab"""
    st.header("Watchlist - Research & Discovery")
    st.caption("Track potential investments and set price alerts")
    watchlist = load_watchlist()
    if watchlist['items']:
        with st.spinner("Fetching watchlist data..."):
            watchlist_df = fetch_watchlist_data(tuple(watchlist['items']))
        triggered_alerts = check_price_alerts(watchlist_df, watchlist.get('alerts', []))
        if triggered_alerts:
            for t in triggered_alerts:
                st.warning(f"ALERT: {t['message']}")
                for alert in watchlist['alerts']:
                    if alert['id'] == t['alert']['id']:
                        alert['triggered'] = True
            save_watchlist(watchlist)
    else:
        watchlist_df = pd.DataFrame()
    tab1, tab2, tab3 = st.tabs(["Watchlist", "Add Ticker", "Alerts"])
    with tab1:
        render_watchlist_table(watchlist_df, watchlist, portfolio_tickers)
        if portfolio_tickers and not watchlist_df.empty:
            st.subheader("Comparison to Portfolio")
            in_portfolio = watchlist_df[watchlist_df['ticker'].isin(portfolio_tickers)]
            not_in_portfolio = watchlist_df[~watchlist_df['ticker'].isin(portfolio_tickers)]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Already Own", len(in_portfolio))
            with col2:
                st.metric("Potential Buys", len(not_in_portfolio))
            if not not_in_portfolio.empty:
                st.caption("**Top performers not in your portfolio:**")
                top_performers = not_in_portfolio.nlargest(3, 'change_pct')
                for _, row in top_performers.iterrows():
                    change_color = "green" if row['change_pct'] >= 0 else "red"
                    st.markdown(f"- **{row['ticker']}** ({row['name']}): <span style='color:{change_color}'>{row['change_pct']:+.2f}%</span>", unsafe_allow_html=True)
    with tab2:
        render_add_to_watchlist(watchlist)
    with tab3:
        render_price_alerts(watchlist)


def calculate_portfolio_metrics(holdings, ticker_data, fx_rate, proxy_yields):
    portfolio = []
    total_value = total_cost = projected_annual = projected_monthly = usd_exposure = 0
    weights, vols, betas, yield_warnings = [], [], [], []
    
    for tk, h in holdings.items():
        if tk not in ticker_data:
            continue
        d = ticker_data[tk]
        price = d['price'] / (fx_rate if d['currency'] == 'USD' else d['scale'])
        value = h['shares'] * price
        pl_pct = (value - h['cost_gbp']) / h['cost_gbp'] * 100 if h['cost_gbp'] else 0
        yield_pct, yield_source, months_data = calculate_dividend_yield(tk, price, d['scale'], proxy_yields)
        
        if yield_source == 'calculated_limited':
            yield_warnings.append(f"{tk} ({months_data}mo)")
        elif yield_source == 'proxy':
            yield_warnings.append(f"{tk} (proxy)")
        
        est_annual = value * (yield_pct / 100)
        projected_annual += est_annual
        projected_monthly += est_annual / 12
        if d['currency'] == 'USD':
            usd_exposure += value
        
        portfolio.append({
            'Ticker': tk, 'Category': h['category'], 'Shares': round(h['shares'], 2), 'Price £': round(price, 2), 
            'Value £': int(round(value)), 'Weight %': 0, 'P/L %': round(pl_pct, 1), 'Trail Yield %': round(yield_pct, 2),
            'Yield Source': yield_source, 'Est Annual £': int(round(est_annual)), 'Vol %': round(d['vol'], 1) if d['vol'] else None,
            'Max DD %': round(d['dd'], 1) if d['dd'] else None, 'Beta': round(d['beta'], 2)
        })
        
        total_value += value
        total_cost += h['cost_gbp']
        if total_value > 0:
            weights.append(value / total_value)
            vols.append(d['vol'] or 0)
            betas.append(d['beta'])
    
    df_port = pd.DataFrame(portfolio).sort_values('Value £', ascending=False)
    df_port['Weight %'] = (df_port['Value £'] / total_value * 100).round(1) if total_value > 0 else 0
    
    return {
        'df': df_port, 'total_value': total_value, 'total_cost': total_cost, 'projected_annual': projected_annual,
        'projected_monthly': projected_monthly, 'usd_exposure': usd_exposure,
        'portfolio_vol': np.average(vols, weights=weights) if weights else 0,
        'portfolio_beta': np.average(betas, weights=weights) if weights else 1.0,
        'yield_warnings': yield_warnings
    }

# ============================================================================
# SIDEBAR - PREMIUM REDESIGN
# ============================================================================

def inject_sidebar_css():
    """Inject premium sidebar CSS styling"""
    theme = get_current_theme()
    colors = THEMES[theme]

    sidebar_css = f"""
    <style>
        /* Sidebar Premium Styling */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {colors['surface']} 0%, {colors['background']} 100%);
        }}

        /* Sidebar Header */
        .sidebar-header {{
            background: linear-gradient(135deg, {colors['accent']} 0%, {colors['accent_secondary']} 100%);
            padding: 1.25rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }}

        .sidebar-header h1 {{
            color: {'#0f172a' if theme == 'dark' else '#ffffff'} !important;
            font-size: 1.5rem;
            margin: 0;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .sidebar-header .portfolio-summary {{
            margin-top: 0.75rem;
            padding-top: 0.75rem;
            border-top: 1px solid rgba(255,255,255,0.2);
        }}

        .sidebar-header .value-display {{
            font-size: 1.75rem;
            font-weight: 800;
            color: {'#0f172a' if theme == 'dark' else '#ffffff'};
            margin: 0.25rem 0;
        }}

        .sidebar-header .refresh-time {{
            font-size: 0.75rem;
            color: {'rgba(15,23,42,0.7)' if theme == 'dark' else 'rgba(255,255,255,0.8)'};
            margin-top: 0.5rem;
        }}

        /* Section Styling */
        .sidebar-section {{
            background: {colors['surface']};
            border-radius: 10px;
            padding: 0.75rem;
            margin-bottom: 0.75rem;
            border: 1px solid {colors['text_secondary']}20;
            transition: all 0.3s ease;
        }}

        .sidebar-section:hover {{
            border-color: {colors['accent']}40;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}

        .sidebar-section-title {{
            font-size: 0.85rem;
            font-weight: 600;
            color: {colors['text']};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        /* Status Indicators */
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 500;
        }}

        .status-badge.success {{
            background: {colors['success']}20;
            color: {colors['success']};
        }}

        .status-badge.warning {{
            background: {colors['warning']}20;
            color: {colors['warning']};
        }}

        .status-badge.error {{
            background: {colors['error']}20;
            color: {colors['error']};
        }}

        /* Ticker Card */
        .ticker-card {{
            background: {colors['background']};
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border-left: 3px solid {colors['warning']};
        }}

        .ticker-card.mapped {{
            border-left-color: {colors['success']};
        }}

        .ticker-card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }}

        .ticker-name {{
            font-weight: 600;
            color: {colors['text']};
            font-size: 0.9rem;
        }}

        .ticker-subtitle {{
            font-size: 0.7rem;
            color: {colors['text_secondary']};
            margin-top: 0.15rem;
        }}

        /* Target Slider Visual */
        .target-slider {{
            position: relative;
            height: 8px;
            background: {colors['background']};
            border-radius: 4px;
            margin: 0.5rem 0;
        }}

        .target-slider-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}

        /* Proxy Yield Indicator */
        .yield-indicator {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem;
            background: {colors['background']};
            border-radius: 6px;
            margin-bottom: 0.4rem;
        }}

        .yield-indicator .ticker {{
            font-weight: 500;
            color: {colors['text']};
        }}

        .yield-indicator .yield-value {{
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }}

        .yield-indicator .yield-badge {{
            background: {colors['accent']}20;
            color: {colors['accent']};
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        /* Sidebar Footer */
        .sidebar-footer {{
            background: {colors['background']};
            border-radius: 10px;
            padding: 0.75rem;
            margin-top: 1rem;
            text-align: center;
        }}

        .sidebar-footer .version {{
            font-size: 0.7rem;
            color: {colors['text_secondary']};
        }}

        .sidebar-footer .links {{
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-top: 0.5rem;
        }}

        .sidebar-footer .links a {{
            color: {colors['accent']};
            font-size: 0.75rem;
            text-decoration: none;
        }}

        .sidebar-footer .links a:hover {{
            text-decoration: underline;
        }}

        /* Streamlit Expander Customization */
        [data-testid="stSidebar"] .streamlit-expanderHeader {{
            background: {colors['surface']};
            border-radius: 8px;
            font-size: 0.85rem;
            padding: 0.5rem 0.75rem;
        }}

        [data-testid="stSidebar"] .streamlit-expanderContent {{
            background: {colors['background']};
            border-radius: 0 0 8px 8px;
            padding: 0.75rem;
        }}

        /* Divider styling */
        .sidebar-divider {{
            height: 1px;
            background: linear-gradient(90deg, transparent, {colors['text_secondary']}30, transparent);
            margin: 1rem 0;
        }}

        /* Animation for collapsibles */
        [data-testid="stSidebar"] details {{
            transition: all 0.3s ease;
        }}

        [data-testid="stSidebar"] details[open] summary {{
            color: {colors['accent']};
        }}

        /* Input field styling in sidebar */
        [data-testid="stSidebar"] .stTextInput > div > div > input {{
            border-radius: 6px;
            font-size: 0.85rem;
        }}

        [data-testid="stSidebar"] .stNumberInput > div > div > input {{
            border-radius: 6px;
            font-size: 0.85rem;
        }}

        /* Button styling in sidebar */
        [data-testid="stSidebar"] .stButton > button {{
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 500;
            padding: 0.5rem 1rem;
        }}
    </style>
    """
    st.markdown(sidebar_css, unsafe_allow_html=True)


def render_sidebar(all_tickers, isa_m=None, sipp_m=None):
    """Render premium sidebar with organized sections"""
    with st.sidebar:
        # Inject sidebar CSS
        inject_sidebar_css()

        theme = get_current_theme()
        colors = THEMES[theme]

        # =====================================================================
        # 1. SIDEBAR HEADER - Portfolio Summary
        # =====================================================================
        total_value = 0
        if isa_m:
            total_value += isa_m.get('total_value', 0)
        if sipp_m:
            total_value += sipp_m.get('total_value', 0)

        # Get refresh timestamp from refresh system
        last_refresh_ts = get_refresh_timestamp('last_refresh')
        last_refresh = last_refresh_ts.strftime('%H:%M') if last_refresh_ts else 'Never'
        refresh_status, refresh_color, refresh_icon = get_freshness_status(last_refresh_ts)

        st.markdown(f"""
        <div class="sidebar-header">
            <h1>PyFinance</h1>
            <div class="portfolio-summary">
                <div style="font-size: 0.75rem; opacity: 0.8;">Total Portfolio Value</div>
                <div class="value-display">{"£{:,.0f}".format(total_value) if total_value > 0 else "Loading..."}</div>
                <div class="refresh-time" style="color: {refresh_color};">
                    {refresh_icon} Last refresh: {last_refresh}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # =====================================================================
        # 2. QUICK ACTIONS
        # =====================================================================
        st.markdown(f'<div class="sidebar-section-title">Quick Actions</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh", key="refresh_btn", use_container_width=True, help="Refresh all data (Press R)"):
                clear_cache_and_refresh()
        with col2:
            # Theme toggle
            theme_icon = "Light" if theme == 'dark' else "Dark"
            if st.button(f"{theme_icon}", key="theme_btn", use_container_width=True, help=f"Switch to {theme_icon.lower()} mode"):
                toggle_theme()
                st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export", key="export_btn", use_container_width=True, help="Export portfolio to CSV"):
                st.session_state.show_export = True
        with col2:
            if st.button("Share", key="share_btn", use_container_width=True, help="Share portfolio (coming soon)"):
                st.toast("Share feature coming soon!", icon="info")

        # Handle export modal
        if st.session_state.get('show_export', False):
            st.info("Export will download your current portfolio data")
            if st.button("Close", key="close_export"):
                st.session_state.show_export = False
                st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # =====================================================================
        # 3. PORTFOLIO FILTERS
        # =====================================================================
        st.markdown(f'<div class="sidebar-section-title">Filters</div>', unsafe_allow_html=True)

        # Account filter
        account_filter = st.selectbox(
            "Account",
            ["All", "ISA", "SIPP"],
            key="account_filter",
            label_visibility="collapsed"
        )
        st.session_state.account_filter_value = account_filter

        # Category filter
        category_filter = st.selectbox(
            "Category",
            ["All", "Income", "Growth", "Speculative"],
            key="category_filter",
            label_visibility="collapsed"
        )
        st.session_state.category_filter_value = category_filter

        # Date range
        date_range = st.selectbox(
            "Period",
            ["All Time", "YTD", "Last 3 Months", "Last Month"],
            key="date_range",
            label_visibility="collapsed"
        )
        st.session_state.date_range_value = date_range

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # =====================================================================
        # 4. TICKER CONFIGURATION (Collapsible)
        # =====================================================================
        unmapped = [ft for ft in all_tickers if ft not in yahoo_map]
        mapped_count = len(all_tickers) - len(unmapped)

        with st.expander(f"Ticker Config ({mapped_count}/{len(all_tickers)} mapped)", expanded=len(unmapped) > 0):
            if unmapped:
                st.markdown(f"""
                <div class="status-badge warning">
                    {len(unmapped)} unmapped ticker{"s" if len(unmapped) > 1 else ""}
                </div>
                """, unsafe_allow_html=True)

                with st.form("ticker_mapping_form"):
                    updates = {}
                    for ft in unmapped:
                        row_data = None
                        for df in [isa_net, sipp_net]:
                            if df is not None and ft in df['Ticker'].values:
                                row_data = df[df['Ticker'] == ft].iloc[0]
                                break

                        # Ticker card with smart suggestions
                        if row_data is not None:
                            title = row_data.get('Title', 'Unknown')[:40]
                            isin = row_data.get('ISIN', '')
                            sugg_yahoo = ft if isin.startswith(('US', 'CA')) else ft + '.L'
                            title_lower = title.lower()
                            sugg_cat = 'income' if any(k in title_lower for k in ['income', 'premium', 'dis', 'dividend']) else \
                                       'growth' if any(k in title_lower for k in ['strategy', 'growth', 'tech']) else \
                                       'speculative' if any(k in title_lower for k in ['silver', 'mining', 'crypto', 'bitcoin']) else 'income'
                        else:
                            title = "Unknown"
                            sugg_yahoo = ft + '.L' if not ft.endswith('.L') else ft
                            sugg_cat = 'income'

                        st.markdown(f"""
                        <div class="ticker-card">
                            <div class="ticker-card-header">
                                <div>
                                    <div class="ticker-name">{ft}</div>
                                    <div class="ticker-subtitle">{title}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        col1, col2 = st.columns(2)
                        y = col1.text_input("Yahoo Symbol", sugg_yahoo, key=f"y_{ft}", label_visibility="collapsed")
                        c = col2.selectbox("Category", ["income", "growth", "speculative"],
                                          index=["income", "growth", "speculative"].index(sugg_cat) if sugg_cat in ["income", "growth", "speculative"] else 0,
                                          key=f"c_{ft}", label_visibility="collapsed")
                        updates[ft] = (y, c)

                    if st.form_submit_button("Save All Mappings", use_container_width=True, type="primary"):
                        for ft, (y, c) in updates.items():
                            if y:
                                yahoo_map[ft] = y
                            category_map[ft] = c
                        config['yahoo_ticker_map'] = yahoo_map
                        config['category_map'] = category_map
                        save_config(config)
                        st.cache_data.clear()
                        st.rerun()
            else:
                st.markdown(f"""
                <div class="status-badge success">
                    All tickers mapped
                </div>
                """, unsafe_allow_html=True)

                # Show mapped tickers summary
                if yahoo_map:
                    st.caption(f"{len(yahoo_map)} tickers configured")

        # =====================================================================
        # 5. PROXY YIELDS (Collapsible)
        # =====================================================================
        proxy_count = len(proxy_yields)
        with st.expander(f"Proxy Yields ({proxy_count} configured)"):
            st.caption("Fallback yields when historical data is limited")

            # Visual display of current proxies
            if proxy_yields:
                for tk, yld in proxy_yields.items():
                    st.markdown(f"""
                    <div class="yield-indicator">
                        <span class="ticker">{tk}</span>
                        <div class="yield-value">
                            <span class="yield-badge">{yld*100:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Add new proxy
            with st.form("add_proxy_form"):
                st.markdown("**Add Proxy Yield**")
                col1, col2 = st.columns([2, 1])
                new_ticker = col1.text_input("Ticker", key="new_proxy_ticker", label_visibility="collapsed", placeholder="Ticker")
                new_yield = col2.number_input("Yield %", 0.0, 30.0, 5.0, 0.5, key="new_proxy_yield", label_visibility="collapsed")
                if st.form_submit_button("Add", use_container_width=True):
                    if new_ticker:
                        proxy_yields[new_ticker] = new_yield / 100
                        config['proxy_yields'] = proxy_yields
                        save_config(config)
                        st.cache_data.clear()
                        st.rerun()

            # Edit existing proxies
            if proxy_yields:
                with st.form("edit_proxies_form"):
                    st.markdown("**Edit Existing**")
                    yield_updates = {}
                    for tk, yld in proxy_yields.items():
                        col1, col2, col3 = st.columns([2, 1.5, 0.5])
                        col1.text(tk)
                        updated = col2.number_input(
                            f"Yield for {tk}",
                            min_value=0.0,
                            max_value=30.0,
                            value=float(yld) * 100,
                            step=0.5,
                            key=f"proxy_{tk}",
                            label_visibility="collapsed"
                        )
                        delete = col3.checkbox("X", key=f"del_{tk}", label_visibility="collapsed")
                        if not delete:
                            yield_updates[tk] = updated / 100

                    if st.form_submit_button("Update Yields", use_container_width=True):
                        config['proxy_yields'] = yield_updates
                        save_config(config)
                        st.cache_data.clear()
                        st.rerun()

        # =====================================================================
        # 6. SETTINGS (Collapsible)
        # =====================================================================
        with st.expander("Settings"):
            st.markdown("**Category Targets**")
            st.caption("Target allocation percentages")

            # Visual sliders for category targets
            cat_colors = {'income': '#22c55e', 'growth': '#3b82f6', 'speculative': '#f59e0b'}
            for cat, target in category_targets.items():
                st.markdown(f"""
                <div style="margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                        <span style="text-transform: capitalize;">{cat}</span>
                        <span style="color: {colors['accent']}; font-weight: 600;">{target:.0f}%</span>
                    </div>
                    <div class="target-slider">
                        <div class="target-slider-fill" style="width: {target}%; background: {cat_colors.get(cat, '#888')};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with st.form("settings_form"):
                col1, col2, col3 = st.columns(3)
                new_income = col1.number_input("Income", 0.0, 100.0, float(category_targets.get('income', 50)), 5.0, key="set_income")
                new_growth = col2.number_input("Growth", 0.0, 100.0, float(category_targets.get('growth', 35)), 5.0, key="set_growth")
                new_spec = col3.number_input("Spec", 0.0, 100.0, float(category_targets.get('speculative', 15)), 5.0, key="set_spec")

                total_pct = new_income + new_growth + new_spec
                if abs(total_pct - 100) > 0.1:
                    st.error(f"Total: {total_pct:.0f}% (must be 100%)")

                if st.form_submit_button("Save Targets", use_container_width=True):
                    if abs(total_pct - 100) <= 0.1:
                        config['category_targets'] = {
                            'income': new_income,
                            'growth': new_growth,
                            'speculative': new_spec
                        }
                        save_config(config)
                        st.success("Saved!")
                        st.rerun()

            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

            # API Key configuration
            st.markdown("**API Configuration**")
            api_key = config.get('openai_api_key', '') or os.getenv('OPENAI_API_KEY', '')

            if api_key:
                st.markdown(f"""
                <div class="status-badge success">
                    OpenAI API configured
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-badge warning">
                    No API key set
                </div>
                """, unsafe_allow_html=True)

            with st.form("api_key_form"):
                new_key = st.text_input("OpenAI API Key", type="password", value="", key="api_key_input",
                                       placeholder="sk-..." if not api_key else "Key configured")
                if st.form_submit_button("Save API Key", use_container_width=True):
                    if new_key:
                        config['openai_api_key'] = new_key
                        save_config(config)
                        st.success("API key saved!")
                        st.rerun()

            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

            # Enhanced data refresh settings
            render_refresh_settings(THEMES[get_current_theme()])

        # =====================================================================
        # =====================================================================
        # HELP SECTION
        # =====================================================================
        render_help_sidebar_section()

        # 7. FOOTER
        # =====================================================================
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sidebar-footer">
            <div class="version">PyFinance Dashboard v2.0</div>
            <div style="font-size: 0.65rem; color: {colors['text_secondary']}; margin-top: 0.3rem;">
                Data: Yahoo Finance
            </div>
            <div class="links">
                <span style="font-size: 0.7rem; color: {colors['text_secondary']};">
                    Made for Freetrade users
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)



# ============================================================================
# PORTFOLIO VIEW
# ============================================================================

def render_portfolio_view(name, csv_file, net, div_income, ticker_data, fx_rate):
    if net is None or net.empty:
        render_empty_state(message=f"No {name} data found", icon="📂", hint=f"Upload your Freetrade export file: freetrade_{name}_*.csv")
        return None
    
    # Debug: Show which tickers we're processing
    st.caption(f"📊 Processing {len(net)} positions from {name}")
    
    holdings = {yahoo_map[row['Ticker']]: {'shares': row['Quantity'], 'cost_gbp': abs(row['cost_gbp']), 
                'category': category_map.get(row['Ticker'], 'unknown')} 
                for _, row in net.iterrows() if row['Ticker'] in yahoo_map}
    
    if not holdings:
        render_error_message(error_type=ErrorType.WARNING, message=f"No mapped tickers found for {name}", suggestions=["Map your Freetrade tickers to Yahoo Finance symbols in the sidebar", "LSE stocks need a .L suffix (e.g., VUSA.L)", "US stocks do not need a suffix"])
        return None
    
    valid = {tk: h for tk, h in holdings.items() if tk in ticker_data}
    if not valid:
        render_error_message(error_type=ErrorType.ERROR, message=f"Could not fetch price data for {name} holdings", suggestions=["Check your internet connection", "Verify ticker mappings are correct", "Yahoo Finance may be temporarily unavailable"], show_retry=True, retry_key=f"retry_{name}_prices")
        return None
    
    m = calculate_portfolio_metrics(valid, ticker_data, fx_rate, proxy_yields)
    
    if m['yield_warnings']:
        st.info(f"ℹ️ Limited data: {', '.join(m['yield_warnings'])}")
    
    # Calculate P/L percentage
    pl_pct = (m['total_value'] - m['total_cost']) / m['total_cost'] * 100 if m['total_cost'] else 0
    pl_amount = m['total_value'] - m['total_cost']

    # Render premium metric cards
    render_metric_row([
        {
            'title': 'Portfolio Value',
            'value': f"£{m['total_value']:,.0f}",
            'delta': f"{pl_pct:+.1f}%",
            'icon': '💰',
            'accent': 'green',
            'detail': f"Cost basis: £{m['total_cost']:,.0f}"
        },
        {
            'title': 'Profit / Loss',
            'value': f"£{pl_amount:,.0f}",
            'delta': f"{pl_pct:+.1f}%",
            'icon': '📈',
            'accent': 'blue' if pl_pct >= 0 else 'purple',
            'detail': 'Unrealized gains/losses'
        },
        {
            'title': 'Monthly Income',
            'value': f"£{int(m['projected_monthly']):,.0f}",
            'icon': '💵',
            'accent': 'gold',
            'detail': 'Projected dividend income'
        },
        {
            'title': 'Annual Income',
            'value': f"£{int(m['projected_annual']):,.0f}",
            'icon': '🎯',
            'accent': 'green',
            'detail': f"Yield: {(m['projected_annual']/m['total_value']*100):.2f}%" if m['total_value'] else ''
        },
        {
            'title': 'Realised Income',
            'value': f"£{div_income:,.0f}",
            'icon': '💸',
            'accent': 'purple',
            'detail': 'Total dividends received'
        }
    ])
    
    col_h1, col_h2 = st.columns([10, 1])
    with col_h1:
        st.subheader("📋 Holdings")
    with col_h2:
        render_inline_help_icon('holdings_help', 'Holdings Table')
    search = st.text_input("🔍 Search", "", key=f"search_{name}")
    
    df_display = m['df'].copy()
    if search:
        df_display = df_display[df_display['Ticker'].str.contains(search, case=False, na=False) | 
                                df_display['Category'].str.contains(search, case=False, na=False)]
    
    col1, col2 = st.columns(2)
    with col1:
        top = df_display.nlargest(1, 'P/L %')
        if not top.empty:
            st.success(f"🚀 Top: {top.iloc[0]['Ticker']} ({top.iloc[0]['P/L %']:+.1f}%)")
    with col2:
        btm = df_display.nsmallest(1, 'P/L %')
        if not btm.empty:
            st.error(f"📉 Bottom: {btm.iloc[0]['Ticker']} ({btm.iloc[0]['P/L %']:+.1f}%)")
    
    # Render premium holdings table with sparklines
    st.caption("🤖=Auto | ⚠️=Limited | 📌=Proxy")
    render_holdings_table(df_display, ticker_data=ticker_data, show_sparklines=True)
    
    st.subheader("🥧 Allocation")
    col1, col2 = st.columns(2)
    with col1:
        render_allocation_donut(
            m['df'],
            title="By Holding",
            value_col='Value £',
            name_col='Ticker',
            hole_content={'label': 'Total', 'value': m['total_value']},
            key_suffix=f"holding_{name}",
            template=get_plotly_template()
        )
    with col2:
        alloc = m['df'].groupby('Category')['Value £'].sum().reset_index()
        render_allocation_donut(
            alloc,
            title="By Category",
            value_col='Value £',
            name_col='Category',
            hole_content={'label': 'Total', 'value': m['total_value']},
            key_suffix=f"category_{name}",
            template=get_plotly_template()
        )

    # Treemap visualization
    with st.expander("🗺️ Allocation Treemap", expanded=False):
        render_treemap(
            m['df'],
            value_col='Value £',
            category_col='Category',
            label_col='Ticker',
            key_suffix=f"treemap_{name}",
            template=get_plotly_template()
        )

    # Historical portfolio value chart
    st.subheader("📈 Portfolio Value Over Time")
    if csv_file:
        # Reload transaction data for this portfolio
        portfolio_df = load_csv(csv_file)
        trans_hist, _, _ = process_transactions(portfolio_df)
        
        if trans_hist is not None and not trans_hist.empty:
            # Calculate historical portfolio
            start_date = pd.Timestamp("2025-10-01")
            all_dates = pd.date_range(start=start_date, end=pd.Timestamp.now().normalize(), freq='D')
            
            shares_hist = {tk: pd.Series(0.0, index=all_dates) for tk in valid}
            cash_additions = pd.Series(0.0, index=all_dates)
            
            for _, row in trans_hist.iterrows():
                ts = row['Timestamp']
                if ts < start_date:
                    continue
                ts_date = ts.normalize()
                ft = row.get('Ticker')
                row_type = str(row['Type'])
                
                if row_type == 'ORDER' and ft in yahoo_map:
                    tk = yahoo_map[ft]
                    if tk in valid:
                        qty_change = row['Quantity']
                        shares_hist[tk].loc[ts_date:] += qty_change
                elif 'DIVIDEND' in row_type or 'INTEREST' in row_type:
                    cash_additions.loc[ts_date:] += row['Total Amount']
            
            port_hist = pd.DataFrame(index=all_dates)
            for tk in valid:
                if tk in ticker_data:
                    hist = ticker_data[tk]['hist']
                    price_series = hist.reindex(all_dates).ffill()
                    port_hist[tk] = shares_hist[tk] * price_series
            
            port_hist['Cash/Income'] = cash_additions.cumsum()
            port_hist['Total'] = port_hist.sum(axis=1, skipna=True)
            port_hist = port_hist[port_hist['Total'] > 0]
            
            if not port_hist.empty:
                returns = port_hist['Total'].pct_change().dropna()
                
                col1, col2, col3 = st.columns(3)
                if len(returns) > 0 and returns.std() > 0:
                    sharpe = np.sqrt(252) * (returns.mean() - 0.04/252) / returns.std()
                    downside = returns[returns < 0]
                    sortino = np.sqrt(252) * (returns.mean() - 0.04/252) / downside.std() if len(downside) > 0 and downside.std() > 0 else 0
                    max_dd = port_hist['Total'].expanding().max().sub(port_hist['Total']).div(port_hist['Total'].expanding().max()).max() * 100
                    
                    col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    col2.metric("Sortino Ratio", f"{sortino:.2f}")
                    col3.metric("Max Drawdown", f"{max_dd:.1f}%")
                
                # Use premium portfolio value chart
                render_portfolio_value_chart(
                    port_hist,
                    show_annotations=True,
                    key_suffix=f"hist_{name}",
                    template=get_plotly_template()
                )

                # Benchmark comparison
                render_benchmark_comparison(port_hist, name)
                
                # Performance Attribution
                render_performance_attribution_section(m['df'], ticker_data, m, category_targets, name)
            else:
                st.info("ℹ️ No historical data available yet")
        else:
            st.info("ℹ️ No transaction history available")

    st.subheader("⚠️ Risk Metrics & Insights")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Vol", f"{m['portfolio_vol']:.1f}%")
    c2.metric("Avg Beta", f"{m['portfolio_beta']:.2f}")
    c3.metric("USD Exposure", f"{(m['usd_exposure']/m['total_value']*100):.1f}%")
    c4.metric("Yield on Value", f"{(m['projected_annual']/m['total_value']*100):.2f}%")
    
    alerts = []
    if m['projected_monthly'] > 400:
        alerts.append("⚠️ Very high monthly income - sustainability check")
    if m['portfolio_vol'] > 30:
        alerts.append("⚠️ High volatility - consider defensive holdings")
    if m['portfolio_beta'] > 1.2:
        alerts.append("⚠️ Aggressive portfolio - more volatile than market")
    if m['usd_exposure'] / m['total_value'] > 0.3:
        alerts.append("⚠️ High USD exposure - currency risk")
    for _, r in m['df'].iterrows():
        if r['Weight %'] > 15:
            alerts.append(f"⚠️ Consider trimming {r['Ticker']} ({r['Weight %']:.1f}%)")
    if not alerts:
        alerts.append("✅ Portfolio balanced")
    
    st.write("**📌 Insights:**")
    for a in alerts:
        st.write(f"- {a}")

    # Monthly Income Chart for this account
    st.subheader(f"📅 {name} Monthly Dividend Income")
    if csv_file:
        portfolio_df_income = load_csv(csv_file)
        monthly_income = get_monthly_dividends(portfolio_df_income)
        if not monthly_income.empty:
            render_income_metrics(monthly_income)
            render_monthly_income_chart(monthly_income, title=f"{name} Monthly Income", key_suffix=f"income_{name}")
        else:
            st.info("ℹ️ No dividend data available yet")

    st.caption(f"📄 {os.path.basename(csv_file)}")
    return m

# ============================================================================
# MAIN
# ============================================================================

def main():
    global isa_net, sipp_net

    # Initialize refresh state tracking
    init_refresh_state()
    track_user_activity()

    # Inject theme CSS, loading CSS, animations, error CSS, help CSS, and render toggle
    inject_theme_css()
    inject_loading_css()
    inject_animation_css()
    inject_error_css()
    inject_help_css()
    inject_keyboard_shortcut_handlers()
    render_theme_toggle()

    # First-time user welcome and feature tour
    render_welcome_modal()
    render_feature_tour()
    render_keyboard_shortcuts_modal()

    # Inject keyboard shortcut handler (R to refresh)
    render_keyboard_shortcut_handler()

    # Get theme colors for refresh UI
    theme_colors = get_theme_colors()

    has_isa = ISA_CSV is not None
    has_sipp = SIPP_CSV is not None

    if not has_isa and not has_sipp:
        render_empty_state(
            message="No portfolio data found",
            icon="📊",
            hint="Export your transaction history from Freetrade and place the CSV files in this folder"
        )
        render_error_message(
            error_type=ErrorType.INFO,
            message="How to get started",
            suggestions=[
                "Open the Freetrade app on your phone",
                "Go to Activity > Export > Download CSV",
                "Place the file in this folder as: freetrade_ISA_*.csv or freetrade_SIPP_*.csv",
                "Refresh this page"
            ],
            show_report=False
        )
        st.stop()

    isa_df = load_csv(ISA_CSV) if has_isa else None
    sipp_df = load_csv(SIPP_CSV) if has_sipp else None

    _, isa_net, isa_div = process_transactions(isa_df)
    _, sipp_net, sipp_div = process_transactions(sipp_df)

    all_tickers = set()
    if isa_net is not None:
        all_tickers.update(isa_net['Ticker'].unique())
    if sipp_net is not None:
        all_tickers.update(sipp_net['Ticker'].unique())

    # Sidebar rendered after metrics calculation (see below)

    all_yahoo = set()
    if isa_net is not None:
        all_yahoo.update([yahoo_map[ft] for ft in isa_net['Ticker'] if ft in yahoo_map])
    if sipp_net is not None:
        all_yahoo.update([yahoo_map[ft] for ft in sipp_net['Ticker'] if ft in yahoo_map])

    if not all_yahoo:
        render_error_message(
            error_type=ErrorType.WARNING,
            message="No tickers mapped yet",
            suggestions=[
                "Open the sidebar to map your Freetrade tickers to Yahoo Finance symbols",
                "LSE stocks need a '.L' suffix (e.g., VUSA.L)",
                "US stocks don't need a suffix"
            ],
            show_report=False
        )
        st.stop()

    # Create placeholder for loading states
    loading_placeholder = st.empty()

    # Show skeleton loading state while fetching data
    with loading_placeholder.container():
        render_data_loading_status(f"Fetching market data for {len(all_yahoo)} tickers...")
        render_skeleton_overview()

    # Fetch data (cached) and track timestamps
    fx = fetch_fx_rate()
    update_refresh_timestamp('fx_rate')
    
    tickers, failed = fetch_all_tickers(list(all_yahoo))
    update_refresh_timestamp('prices')

    # Clear loading state
    loading_placeholder.empty()

    # Show partial data warning if some tickers failed
    if failed:
        render_partial_data_warning(failed, context="ticker prices")
    
    # Pre-calculate metrics for all tabs
    isa_m = None
    sipp_m = None

    if isa_net is not None and not isa_net.empty:
        isa_holdings = {yahoo_map[r['Ticker']]: {'shares': r['Quantity'], 'cost_gbp': abs(r['cost_gbp']),
                        'category': category_map.get(r['Ticker'], 'unknown')}
                        for _, r in isa_net.iterrows() if r['Ticker'] in yahoo_map}
        isa_valid = {tk: h for tk, h in isa_holdings.items() if tk in tickers}
        isa_m = calculate_portfolio_metrics(isa_valid, tickers, fx, proxy_yields) if isa_valid else None

    if sipp_net is not None and not sipp_net.empty:
        sipp_holdings = {yahoo_map[r['Ticker']]: {'shares': r['Quantity'], 'cost_gbp': abs(r['cost_gbp']),
                         'category': category_map.get(r['Ticker'], 'unknown')}
                         for _, r in sipp_net.iterrows() if r['Ticker'] in yahoo_map}
        sipp_valid = {tk: h for tk, h in sipp_holdings.items() if tk in tickers}
        sipp_m = calculate_portfolio_metrics(sipp_valid, tickers, fx, proxy_yields) if sipp_valid else None

    # Render the premium sidebar with portfolio metrics
    render_sidebar(list(all_tickers), isa_m, sipp_m)

    # Calculate total portfolio value for premium header
    total_val_header = (isa_m['total_value'] if isa_m else 0) + (sipp_m['total_value'] if sipp_m else 0)
    total_cost_header = (isa_m['total_cost'] if isa_m else 0) + (sipp_m['total_cost'] if sipp_m else 0)
    daily_change_header = ((total_val_header - total_cost_header) / total_cost_header * 100) if total_cost_header else 0

    # Render the premium header with branding, stats bar, and timestamp
    theme = get_current_theme()
    colors = THEMES[theme]
    render_premium_header(
        total_value=total_val_header,
        daily_change=daily_change_header,
        fx_rate=fx,
        theme=theme,
        colors=colors
    )

    # Render custom styled tab navigation with icons
    active_tab = render_tab_navigation(has_isa=has_isa, has_sipp=has_sipp, theme=theme, colors=colors)

    # Render breadcrumb showing current location
    render_breadcrumb(get_breadcrumb_for_tab(active_tab), theme=theme, colors=colors)

    # TABS
    if has_isa and has_sipp:
        t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs(["🏠 Overview", "💰 ISA", "🏦 SIPP", "📜 Transactions", "📅 Dividends", "⚖️ Rebalancing", "👁️ Watchlist", "🤖 Assistant", "📤 Export"])

        with t1:
            st.header("🏠 Combined Portfolio")

            total_val = (isa_m['total_value'] if isa_m else 0) + (sipp_m['total_value'] if sipp_m else 0)
            total_cost = (isa_m['total_cost'] if isa_m else 0) + (sipp_m['total_cost'] if sipp_m else 0)
            total_annual = (isa_m['projected_annual'] if isa_m else 0) + (sipp_m['projected_annual'] if sipp_m else 0)
            total_realized = (isa_div if isa_div else 0) + (sipp_div if sipp_div else 0)
            pl_pct = ((total_val - total_cost) / total_cost * 100) if total_cost else 0

            # Render premium portfolio summary card
            render_portfolio_summary_card(
                total_value=total_val,
                pl_pct=pl_pct,
                monthly_income=int(total_annual / 12),
                annual_income=int(total_annual),
                realized_income=total_realized
            )

            # ISA vs SIPP breakdown with premium cards
            render_metric_row([
                {
                    'title': 'ISA Portfolio',
                    'value': f"£{isa_m['total_value']:,.0f}" if isa_m else "£0",
                    'delta': f"{((isa_m['total_value']-isa_m['total_cost'])/isa_m['total_cost']*100):+.1f}%" if isa_m and isa_m['total_cost'] else None,
                    'icon': '💰',
                    'accent': 'green',
                    'detail': f"Annual income: £{int(isa_m['projected_annual']):,}" if isa_m else None
                },
                {
                    'title': 'SIPP Portfolio',
                    'value': f"£{sipp_m['total_value']:,.0f}" if sipp_m else "£0",
                    'delta': f"{((sipp_m['total_value']-sipp_m['total_cost'])/sipp_m['total_cost']*100):+.1f}%" if sipp_m and sipp_m['total_cost'] else None,
                    'icon': '🏦',
                    'accent': 'blue',
                    'detail': f"Annual income: £{int(sipp_m['projected_annual']):,}" if sipp_m else None
                }
            ])


            # Smart Portfolio Insights Panel
            st.subheader("🧠 Smart Portfolio Insights")

            # Combine holdings for insights
            insights_holdings = pd.DataFrame()
            if isa_m and 'df' in isa_m:
                insights_holdings = pd.concat([insights_holdings, isa_m['df']])
            if sipp_m and 'df' in sipp_m:
                insights_holdings = pd.concat([insights_holdings, sipp_m['df']])

            if not insights_holdings.empty:
                # Aggregate combined metrics
                insights_metrics = {
                    'total_value': total_val,
                    'total_cost': total_cost,
                    'projected_annual': total_annual,
                    'usd_exposure': (isa_m.get('usd_exposure', 0) if isa_m else 0) + (sipp_m.get('usd_exposure', 0) if sipp_m else 0),
                    'portfolio_vol': ((isa_m.get('portfolio_vol', 0) if isa_m else 0) + (sipp_m.get('portfolio_vol', 0) if sipp_m else 0)) / 2,
                    'portfolio_beta': ((isa_m.get('portfolio_beta', 1) if isa_m else 1) + (sipp_m.get('portfolio_beta', 1) if sipp_m else 1)) / 2
                }

                # Combine transactions for dividend insights
                insights_transactions = pd.DataFrame()
                if isa_df is not None and not isa_df.empty:
                    insights_transactions = pd.concat([insights_transactions, isa_df])
                if sipp_df is not None and not sipp_df.empty:
                    insights_transactions = pd.concat([insights_transactions, sipp_df])

                # Generate insights
                portfolio_insights = generate_portfolio_insights(
                    holdings_df=insights_holdings,
                    metrics=insights_metrics,
                    transactions_df=insights_transactions,
                    ticker_data=tickers,
                    category_targets=category_targets
                )

                # Get theme for styling
                current_theme = get_current_theme()
                theme_colors = THEMES[current_theme]

                # Render the insights panel
                render_insights_panel(portfolio_insights, max_display=5, theme=current_theme, colors=theme_colors)
            else:
                st.info("No holdings data available to generate insights")

            # Combined Monthly Income Chart
            st.subheader("📅 Monthly Dividend Income")
            combined_income = get_combined_monthly_dividends(isa_df, sipp_df)
            if not combined_income.empty:
                render_income_metrics(combined_income)
                render_monthly_income_chart(combined_income, title="Combined Monthly Income (ISA + SIPP)", stacked=True, key_suffix="combined_overview")
            else:
                st.info("ℹ️ No dividend data available yet")

            # Goals section
            st.divider()

            # Combine holdings DataFrames for goal tracking
            combined_holdings_df = pd.DataFrame()
            if isa_m and 'df' in isa_m:
                combined_holdings_df = pd.concat([combined_holdings_df, isa_m['df']])
            if sipp_m and 'df' in sipp_m:
                combined_holdings_df = pd.concat([combined_holdings_df, sipp_m['df']])

            # Combine transaction data
            combined_trans_df = pd.DataFrame()
            if isa_df is not None and not isa_df.empty:
                combined_trans_df = pd.concat([combined_trans_df, isa_df])
            if sipp_df is not None and not sipp_df.empty:
                combined_trans_df = pd.concat([combined_trans_df, sipp_df])

            # Build combined metrics
            combined_metrics = {
                'total_value': total_val,
                'total_cost': total_cost,
                'projected_annual': total_annual,
                'projected_monthly': total_annual / 12
            }

            # Load and render goals
            goals = load_goals()
            render_goals_dashboard(
                goals=goals,
                metrics=combined_metrics,
                holdings_df=combined_holdings_df,
                monthly_income_df=combined_income,
                transaction_df=combined_trans_df,
                plotly_template=get_plotly_template()
            )

        with t2:
            st.header("💰 ISA Portfolio")
            # Force fresh load of ISA data
            _, isa_net_tab, isa_div_tab = process_transactions(isa_df)
            render_portfolio_view("ISA", ISA_CSV, isa_net_tab, isa_div_tab, tickers, fx)
        
        with t3:
            st.header("🏦 SIPP Portfolio")
            # Force fresh load of SIPP data
            _, sipp_net_tab, sipp_div_tab = process_transactions(sipp_df)
            render_portfolio_view("SIPP", SIPP_CSV, sipp_net_tab, sipp_div_tab, tickers, fx)

        with t4:
            render_transactions_tab(isa_df, sipp_df)

        with t5:
            render_dividend_tab(isa_df, sipp_df, isa_m, sipp_m, get_plotly_template(), get_chart_colors())

        with t6:
            render_rebalancing_tab(isa_m, sipp_m, isa_df, sipp_df)

        with t7:
            # Get portfolio tickers for watchlist comparison
            portfolio_tickers = set()
            if isa_m and 'df' in isa_m:
                portfolio_tickers.update(isa_m['df']['Ticker'].tolist())
            if sipp_m and 'df' in sipp_m:
                portfolio_tickers.update(sipp_m['df']['Ticker'].tolist())
            render_watchlist_tab(portfolio_tickers)

        with t8:
            render_ai_assistant(isa_m, sipp_m, isa_df, sipp_df)

        with t9:
            render_export_panel(
                isa_metrics=isa_m,
                sipp_metrics=sipp_m,
                isa_df=isa_df,
                sipp_df=sipp_df,
                get_monthly_dividends_func=get_monthly_dividends,
                get_combined_monthly_dividends_func=get_combined_monthly_dividends
            )
            # Add scheduled reports section
            schedule_report_email()

    elif has_isa:
        st.header("💰 ISA Portfolio")
        render_portfolio_view("ISA", ISA_CSV, isa_net, isa_div, tickers, fx)
    
    else:
        st.header("🏦 SIPP Portfolio")
        render_portfolio_view("SIPP", SIPP_CSV, sipp_net, sipp_div, tickers, fx)
    
    # Render enhanced footer with freshness info
    render_footer_with_freshness(fx, theme_colors)
    
    # Check for auto-refresh
    auto_refresh_handler()

if __name__ == "__main__":
    main()