"""
Data Refresh System for Freetrade Dashboard
============================================

Provides comprehensive data freshness tracking, auto-refresh capabilities,
and user-friendly refresh controls for the portfolio dashboard.

Features:
1. render_refresh_indicator() - Prominent data freshness display
2. render_refresh_button() - Premium styled refresh button with spinner
3. auto_refresh_handler() - Configurable auto-refresh at intervals
4. clear_cache_and_refresh() - Force fresh API calls with progress
5. render_data_freshness_badge() - Compact inline freshness badge
6. render_refresh_settings() - Sidebar settings panel
7. render_keyboard_shortcut_handler() - R key to refresh
8. render_header_with_freshness() - Header with freshness badge
9. render_footer_with_freshness() - Footer with freshness info
"""

import streamlit as st
from datetime import datetime
import time

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_refresh_state():
    """Initialize session state for refresh tracking"""
    if 'refresh_timestamps' not in st.session_state:
        st.session_state.refresh_timestamps = {
            'prices': None,
            'holdings': None,
            'fx_rate': None,
            'benchmarks': None,
            'csv_data': None,
            'last_refresh': None
        }
    if 'auto_refresh_enabled' not in st.session_state:
        st.session_state.auto_refresh_enabled = True
    if 'auto_refresh_interval' not in st.session_state:
        st.session_state.auto_refresh_interval = 15  # minutes
    if 'is_refreshing' not in st.session_state:
        st.session_state.is_refreshing = False
    if 'last_user_activity' not in st.session_state:
        st.session_state.last_user_activity = datetime.now()


def update_refresh_timestamp(data_type: str):
    """Update the refresh timestamp for a specific data type"""
    init_refresh_state()
    st.session_state.refresh_timestamps[data_type] = datetime.now()
    st.session_state.refresh_timestamps['last_refresh'] = datetime.now()


def get_refresh_timestamp(data_type: str = 'last_refresh') -> datetime:
    """Get the last refresh timestamp for a data type"""
    init_refresh_state()
    return st.session_state.refresh_timestamps.get(data_type)


def track_user_activity():
    """Track user activity to pause auto-refresh during interaction"""
    init_refresh_state()
    st.session_state.last_user_activity = datetime.now()


# ============================================================================
# TIME FORMATTING UTILITIES
# ============================================================================

def get_time_ago_text(timestamp: datetime) -> str:
    """Convert timestamp to human-readable 'X minutes ago' text"""
    if timestamp is None:
        return "Never"

    delta = datetime.now() - timestamp
    seconds = delta.total_seconds()

    if seconds < 60:
        return "Just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"


def get_freshness_status(timestamp: datetime) -> tuple:
    """
    Determine freshness status based on timestamp.
    Returns (status, color, icon) tuple.

    Status levels:
    - fresh (< 5 min): Green - data is current
    - stale (5-15 min): Yellow - data may be outdated
    - very_stale (> 15 min): Red - data needs refresh
    - unknown: Gray - no timestamp available
    """
    if timestamp is None:
        return ('unknown', '#6b7280', '?')

    delta = datetime.now() - timestamp
    minutes = delta.total_seconds() / 60

    if minutes < 5:
        return ('fresh', '#22c55e', '🟢')  # Green - fresh
    elif minutes < 15:
        return ('stale', '#f59e0b', '🟡')  # Yellow - stale
    else:
        return ('very_stale', '#ef4444', '🔴')  # Red - very stale


def get_auto_refresh_countdown() -> str:
    """Get countdown text until next auto-refresh"""
    init_refresh_state()

    if not st.session_state.auto_refresh_enabled:
        return "Auto-refresh disabled"

    last_refresh = get_refresh_timestamp('last_refresh')
    if last_refresh is None:
        return "Waiting for initial load"

    interval_minutes = st.session_state.auto_refresh_interval
    elapsed = (datetime.now() - last_refresh).total_seconds() / 60
    remaining = max(0, interval_minutes - elapsed)

    if remaining < 1:
        return "Refreshing soon..."
    elif remaining < 2:
        return "~1 minute until refresh"
    else:
        return f"~{int(remaining)} minutes until refresh"


# ============================================================================
# CORE REFRESH FUNCTIONALITY
# ============================================================================

def clear_cache_and_refresh():
    """
    Clear all cached data and trigger a fresh data fetch.
    Shows progress indicators during refresh.
    """
    init_refresh_state()
    st.session_state.is_refreshing = True

    # Create progress containers
    progress_container = st.empty()
    status_container = st.empty()

    try:
        with progress_container:
            progress_bar = st.progress(0, text="Starting refresh...")

        # Step 1: Clear cache
        with status_container:
            st.info("Clearing cached data...")
        progress_bar.progress(10, text="Clearing cache...")
        st.cache_data.clear()
        time.sleep(0.2)

        # Step 2: Update timestamps
        progress_bar.progress(20, text="Resetting timestamps...")
        for key in st.session_state.refresh_timestamps:
            st.session_state.refresh_timestamps[key] = None

        # Step 3: Signal completion
        progress_bar.progress(100, text="Refresh complete!")
        with status_container:
            st.success("Cache cleared! Reloading data...")

        time.sleep(0.5)

    finally:
        st.session_state.is_refreshing = False
        progress_container.empty()
        status_container.empty()

    # Trigger rerun to reload data
    st.rerun()


def auto_refresh_handler():
    """
    Handle automatic data refresh based on configured interval.
    Pauses auto-refresh during active user interaction.
    """
    init_refresh_state()

    if not st.session_state.auto_refresh_enabled:
        return

    last_refresh = get_refresh_timestamp('last_refresh')
    if last_refresh is None:
        return

    interval_minutes = st.session_state.auto_refresh_interval
    time_since_refresh = (datetime.now() - last_refresh).total_seconds() / 60

    # Check if user has been active recently (within 2 minutes)
    last_activity = st.session_state.last_user_activity
    user_active = (datetime.now() - last_activity).total_seconds() < 120

    # Auto-refresh if interval exceeded and user not actively interacting
    if time_since_refresh >= interval_minutes and not user_active:
        clear_cache_and_refresh()


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_refresh_indicator(theme_colors: dict):
    """
    Render a prominent data freshness indicator.
    Shows last refresh time with color-coded status.

    Args:
        theme_colors: Dictionary with theme colors (surface, text, text_secondary)
    """
    init_refresh_state()
    last_refresh = get_refresh_timestamp('last_refresh')
    status, color, icon = get_freshness_status(last_refresh)
    time_ago = get_time_ago_text(last_refresh)

    # Determine status text
    if status == 'fresh':
        status_text = "Data is current"
    elif status == 'stale':
        status_text = "Data may be outdated"
    elif status == 'very_stale':
        status_text = "Data needs refresh"
    else:
        status_text = "No data loaded"

    indicator_html = f"""
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: linear-gradient(135deg, {theme_colors['surface']}ee, {theme_colors['surface']}cc);
        border: 1px solid {color}40;
        border-radius: 24px;
        font-size: 14px;
        color: {theme_colors['text']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    ">
        <span style="font-size: 16px;">{icon}</span>
        <span style="font-weight: 500;">Data as of: {time_ago}</span>
        <span style="color: {theme_colors['text_secondary']}; font-size: 12px;">({status_text})</span>
    </div>
    """
    st.markdown(indicator_html, unsafe_allow_html=True)


def render_refresh_button(theme_colors: dict):
    """
    Render a premium-styled refresh button with spinner animation.

    Args:
        theme_colors: Dictionary with theme colors (accent, accent_secondary, background)
    """
    init_refresh_state()

    # Button styling CSS
    button_css = f"""
    <style>
    .refresh-btn-container {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .refresh-btn {{
        background: linear-gradient(135deg, {theme_colors['accent']}, {theme_colors['accent_secondary']});
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        color: {theme_colors['background']};
        font-weight: 600;
        font-size: 14px;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px {theme_colors['accent']}40;
    }}
    .refresh-btn:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px {theme_colors['accent']}60;
    }}
    .refresh-btn:disabled {{
        opacity: 0.6;
        cursor: not-allowed;
        transform: none;
    }}
    @keyframes spin {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    .spinning {{
        animation: spin 1s linear infinite;
    }}
    </style>
    """
    st.markdown(button_css, unsafe_allow_html=True)

    is_refreshing = st.session_state.get('is_refreshing', False)

    button_label = "Refreshing..." if is_refreshing else "🔄 Refresh Data"
    button_disabled = is_refreshing

    if st.button(
        button_label,
        key="main_refresh_btn",
        disabled=button_disabled,
        help="Press R to refresh (keyboard shortcut)"
    ):
        clear_cache_and_refresh()


def render_data_freshness_badge(data_type: str = 'last_refresh', compact: bool = True) -> str:
    """
    Render a compact freshness badge.
    Can be used inline in headers/footers.

    Args:
        data_type: Which timestamp to check (default: 'last_refresh')
        compact: Whether to use compact styling (default: True)

    Returns:
        HTML string for the badge
    """
    init_refresh_state()
    timestamp = get_refresh_timestamp(data_type)
    status, color, icon = get_freshness_status(timestamp)
    time_ago = get_time_ago_text(timestamp)

    if compact:
        badge_html = f"""
        <span style="
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 10px;
            background: {color}20;
            border: 1px solid {color}40;
            border-radius: 12px;
            font-size: 12px;
            color: {color};
            font-weight: 500;
        ">
            {icon} {time_ago}
        </span>
        """
    else:
        badge_html = f"""
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            background: {color}15;
            border: 1px solid {color}30;
            border-radius: 16px;
            font-size: 13px;
            color: {color};
        ">
            <span>{icon}</span>
            <span>{time_ago}</span>
        </div>
        """

    return badge_html


def render_refresh_settings(theme_colors: dict):
    """
    Render refresh settings panel for sidebar.
    Includes auto-refresh toggle, interval slider, and manual refresh button.

    Args:
        theme_colors: Dictionary with theme colors
    """
    init_refresh_state()

    st.divider()
    st.subheader("🔄 Data Refresh")

    # Show current status
    last_refresh = get_refresh_timestamp('last_refresh')
    status, color, icon = get_freshness_status(last_refresh)
    time_ago = get_time_ago_text(last_refresh)

    st.markdown(f"""
    <div style="
        padding: 12px;
        background: {color}15;
        border-left: 3px solid {color};
        border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
    ">
        <div style="font-size: 14px; color: {theme_colors['text']};">
            {icon} Last updated: <strong>{time_ago}</strong>
        </div>
        <div style="font-size: 12px; color: {theme_colors['text_secondary']}; margin-top: 4px;">
            {get_auto_refresh_countdown()}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Auto-refresh toggle
    auto_refresh = st.toggle(
        "Auto-refresh",
        value=st.session_state.auto_refresh_enabled,
        help="Automatically refresh data at the specified interval"
    )
    if auto_refresh != st.session_state.auto_refresh_enabled:
        st.session_state.auto_refresh_enabled = auto_refresh
        st.session_state.last_user_activity = datetime.now()

    # Interval slider (only show if auto-refresh enabled)
    if st.session_state.auto_refresh_enabled:
        interval = st.slider(
            "Refresh interval (minutes)",
            min_value=5,
            max_value=60,
            value=st.session_state.auto_refresh_interval,
            step=5,
            help="How often to automatically refresh data"
        )
        if interval != st.session_state.auto_refresh_interval:
            st.session_state.auto_refresh_interval = interval
            st.session_state.last_user_activity = datetime.now()

    # Manual refresh button
    if st.button("🔄 Refresh Now", key="sidebar_refresh_btn", use_container_width=True):
        st.session_state.last_user_activity = datetime.now()
        clear_cache_and_refresh()

    # Show detailed timestamps in expander
    with st.expander("📊 Data timestamps"):
        timestamps = st.session_state.refresh_timestamps
        for key, ts in timestamps.items():
            if key != 'last_refresh':
                ts_text = ts.strftime('%H:%M:%S') if ts else 'Not loaded'
                st.caption(f"**{key.replace('_', ' ').title()}:** {ts_text}")

    st.caption("Press **R** to refresh (keyboard shortcut)")


def render_keyboard_shortcut_handler():
    """
    Inject JavaScript for keyboard shortcut handling.
    R key triggers refresh.
    """
    shortcut_js = """
    <script>
    // Keyboard shortcut handler for refresh
    document.addEventListener('keydown', function(e) {
        // Only trigger on 'R' key, not in input fields
        if (e.key === 'r' || e.key === 'R') {
            const activeElement = document.activeElement;
            const isInputField = activeElement.tagName === 'INPUT' ||
                                activeElement.tagName === 'TEXTAREA' ||
                                activeElement.isContentEditable;

            if (!isInputField) {
                // Find and click the refresh button
                const refreshBtns = document.querySelectorAll('button');
                for (let btn of refreshBtns) {
                    if (btn.innerText.includes('Refresh')) {
                        btn.click();
                        break;
                    }
                }
            }
        }
    });
    </script>
    """
    st.markdown(shortcut_js, unsafe_allow_html=True)


def render_header_with_freshness(theme_colors: dict):
    """
    Render the dashboard header with integrated freshness indicator.

    Args:
        theme_colors: Dictionary with theme colors
    """
    freshness_badge = render_data_freshness_badge('last_refresh', compact=True)

    header_html = f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        margin-bottom: 16px;
        border-bottom: 1px solid {theme_colors['text_secondary']}20;
    ">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 28px;">📊</span>
            <span style="font-size: 24px; font-weight: 700; color: {theme_colors['text']};">
                Freetrade Portfolio
            </span>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            {freshness_badge}
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_footer_with_freshness(fx_rate: float, theme_colors: dict):
    """
    Render footer with freshness info and FX rate.

    Args:
        fx_rate: Current GBP/USD exchange rate
        theme_colors: Dictionary with theme colors
    """
    last_refresh = get_refresh_timestamp('last_refresh')
    refresh_time = last_refresh.strftime('%Y-%m-%d %H:%M:%S') if last_refresh else 'Never'
    status, color, icon = get_freshness_status(last_refresh)

    footer_html = f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 0;
        margin-top: 24px;
        border-top: 1px solid {theme_colors['text_secondary']}20;
        font-size: 12px;
        color: {theme_colors['text_secondary']};
    ">
        <div style="display: flex; align-items: center; gap: 16px;">
            <span>{icon} Last refresh: {refresh_time}</span>
            <span>💱 £1 = ${fx_rate:.4f}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="
                padding: 4px 8px;
                background: {color}20;
                border-radius: 8px;
                color: {color};
                font-weight: 500;
            ">
                {get_auto_refresh_countdown()}
            </span>
        </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
