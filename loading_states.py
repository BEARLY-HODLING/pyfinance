"""
Loading States & Skeleton Components for Freetrade Dashboard

Professional loading states system with:
- Skeleton placeholders for metrics, tables, and charts
- Loading spinners with customizable text
- Full-page loading overlays
- Progress indicators

Usage:
    from loading_states import (
        inject_loading_css,
        render_skeleton_metric,
        render_skeleton_metrics_row,
        render_skeleton_table,
        render_skeleton_chart,
        render_loading_spinner,
        show_loading_overlay,
        render_skeleton_portfolio_view,
        render_skeleton_overview
    )

    # At app start
    inject_loading_css()

    # While loading data
    render_skeleton_metrics_row(5)
"""

import streamlit as st

# ============================================================================
# CSS DEFINITIONS
# ============================================================================

LOADING_CSS = """
<style>
/* Skeleton base styles */
@keyframes skeleton-pulse {
    0% { opacity: 0.6; }
    50% { opacity: 0.3; }
    100% { opacity: 0.6; }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

@keyframes spinner-rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.skeleton-base {
    background: linear-gradient(90deg, #2d2d2d 25%, #3d3d3d 50%, #2d2d2d 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
    border-radius: 4px;
}

.skeleton-pulse {
    background-color: #2d2d2d;
    animation: skeleton-pulse 1.5s ease-in-out infinite;
    border-radius: 4px;
}

/* Skeleton metric card */
.skeleton-metric {
    padding: 1rem;
    border-radius: 8px;
    background: #1e1e1e;
    border: 1px solid #333;
}

.skeleton-metric-label {
    width: 60%;
    height: 14px;
    margin-bottom: 8px;
}

.skeleton-metric-value {
    width: 80%;
    height: 28px;
    margin-bottom: 4px;
}

.skeleton-metric-delta {
    width: 40%;
    height: 12px;
}

/* Skeleton table */
.skeleton-table {
    width: 100%;
    border-collapse: collapse;
    background: #1e1e1e;
    border-radius: 8px;
    overflow: hidden;
}

.skeleton-table th,
.skeleton-table td {
    padding: 12px 8px;
    border-bottom: 1px solid #333;
}

.skeleton-table-header {
    height: 16px;
    width: 70%;
}

.skeleton-table-cell {
    height: 14px;
}

/* Skeleton chart */
.skeleton-chart {
    width: 100%;
    border-radius: 8px;
    background: #1e1e1e;
    border: 1px solid #333;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
}

.skeleton-chart::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.05) 50%, transparent 100%);
    animation: shimmer 2s ease-in-out infinite;
    background-size: 200% 100%;
}

.skeleton-chart-icon {
    font-size: 48px;
    color: #444;
    z-index: 1;
}

/* Loading spinner */
.loading-spinner-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem;
}

.loading-spinner {
    width: 48px;
    height: 48px;
    border: 4px solid #333;
    border-top: 4px solid #00CC96;
    border-radius: 50%;
    animation: spinner-rotate 1s linear infinite;
}

.loading-spinner-text {
    margin-top: 1rem;
    color: #888;
    font-size: 14px;
}

/* Full page overlay */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(14, 17, 23, 0.92);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 9999;
}

.loading-overlay-spinner {
    width: 64px;
    height: 64px;
    border: 5px solid #333;
    border-top: 5px solid #00CC96;
    border-radius: 50%;
    animation: spinner-rotate 1s linear infinite;
}

.loading-overlay-text {
    margin-top: 1.5rem;
    color: #ccc;
    font-size: 16px;
    font-weight: 500;
}

.loading-overlay-subtext {
    margin-top: 0.5rem;
    color: #666;
    font-size: 12px;
}

/* Progress indicator */
.progress-container {
    width: 200px;
    height: 4px;
    background: #333;
    border-radius: 2px;
    margin-top: 1rem;
    overflow: hidden;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #00CC96, #00b386);
    border-radius: 2px;
    animation: shimmer 2s ease-in-out infinite;
    background-size: 200% 100%;
}

/* Data loading status */
.data-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: #1e293b;
    border-radius: 6px;
    margin-bottom: 1rem;
}

.data-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #00CC96;
    animation: skeleton-pulse 1s ease-in-out infinite;
}

.data-status-text {
    color: #94a3b8;
    font-size: 13px;
}
</style>
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def inject_loading_css():
    """Inject the CSS for loading states (call once at app start)"""
    st.markdown(LOADING_CSS, unsafe_allow_html=True)


def render_skeleton_metric():
    """Render a skeleton placeholder for a metric card"""
    return st.markdown("""
        <div class="skeleton-metric">
            <div class="skeleton-base skeleton-metric-label"></div>
            <div class="skeleton-base skeleton-metric-value"></div>
            <div class="skeleton-base skeleton-metric-delta"></div>
        </div>
    """, unsafe_allow_html=True)


def render_skeleton_metrics_row(count=5):
    """Render a row of skeleton metric cards"""
    cols = st.columns(count)
    for col in cols:
        with col:
            render_skeleton_metric()


def render_skeleton_table(rows=5, cols=8):
    """Render a skeleton placeholder for a data table"""
    header_html = "".join([
        f'<th><div class="skeleton-base skeleton-table-header" style="width: {60 + (i * 5) % 30}%"></div></th>'
        for i in range(cols)
    ])

    rows_html = ""
    for r in range(rows):
        cells = "".join([
            f'<td><div class="skeleton-base skeleton-table-cell" style="width: {40 + ((r + c) * 7) % 40}%"></div></td>'
            for c in range(cols)
        ])
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(f"""
        <table class="skeleton-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    """, unsafe_allow_html=True)


def render_skeleton_chart(height=400):
    """Render a skeleton placeholder for a chart"""
    st.markdown(f"""
        <div class="skeleton-chart" style="height: {height}px;">
            <div class="skeleton-chart-icon">📊</div>
        </div>
    """, unsafe_allow_html=True)


def render_loading_spinner(text="Loading..."):
    """Render a centered loading spinner with customizable text"""
    st.markdown(f"""
        <div class="loading-spinner-container">
            <div class="loading-spinner"></div>
            <div class="loading-spinner-text">{text}</div>
        </div>
    """, unsafe_allow_html=True)


def render_loading_spinner_with_progress(text="Loading...", subtext=""):
    """Render a loading spinner with an indeterminate progress bar"""
    st.markdown(f"""
        <div class="loading-spinner-container">
            <div class="loading-spinner"></div>
            <div class="loading-spinner-text">{text}</div>
            {f'<div style="color: #666; font-size: 12px; margin-top: 4px;">{subtext}</div>' if subtext else ''}
            <div class="progress-container">
                <div class="progress-bar" style="width: 100%"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def show_loading_overlay(text="Fetching live data...", subtext="This may take a few seconds"):
    """Show a full-page loading overlay (use sparingly)"""
    st.markdown(f"""
        <div class="loading-overlay" id="loading-overlay">
            <div class="loading-overlay-spinner"></div>
            <div class="loading-overlay-text">{text}</div>
            <div class="loading-overlay-subtext">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)


def hide_loading_overlay():
    """Hide the loading overlay via JavaScript"""
    st.markdown("""
        <script>
            var overlay = document.getElementById('loading-overlay');
            if (overlay) overlay.style.display = 'none';
        </script>
    """, unsafe_allow_html=True)


def render_data_loading_status(text="Fetching market data..."):
    """Render a subtle inline loading status indicator"""
    st.markdown(f"""
        <div class="data-status">
            <div class="data-status-dot"></div>
            <div class="data-status-text">{text}</div>
        </div>
    """, unsafe_allow_html=True)


def render_skeleton_portfolio_view():
    """Render a complete skeleton for the portfolio view"""
    # Metrics row
    render_skeleton_metrics_row(5)

    st.markdown("<br>", unsafe_allow_html=True)

    # Holdings section header
    st.markdown("### 📋 Holdings")
    render_skeleton_table(rows=6, cols=10)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts section
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### By Holding")
        render_skeleton_chart(height=300)
    with col2:
        st.markdown("#### By Category")
        render_skeleton_chart(height=300)


def render_skeleton_overview():
    """Render skeleton for the overview tab"""
    render_skeleton_metrics_row(4)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        render_skeleton_metric()
    with col2:
        render_skeleton_metric()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📅 Monthly Dividend Income")
    render_skeleton_chart(height=350)
