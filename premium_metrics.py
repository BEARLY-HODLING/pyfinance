"""
Premium Metric Card Components for Freetrade Dashboard

Provides glassmorphism-styled metric cards with Bloomberg Terminal aesthetic,
including sparklines, color-coded deltas, and responsive layouts.
"""

import streamlit as st


def inject_premium_css():
    """Inject glassmorphism CSS styles for premium metric cards"""
    st.markdown("""
    <style>
    /* Premium Glassmorphism Metric Cards */
    .premium-metric-card {
        background: linear-gradient(135deg, rgba(30, 35, 45, 0.95) 0%, rgba(20, 25, 35, 0.98) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 8px 0;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .premium-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    }

    .premium-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.12);
    }

    .metric-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }

    .metric-title {
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: rgba(255, 255, 255, 0.5);
        margin: 0;
    }

    .metric-icon {
        font-size: 20px;
        opacity: 0.7;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff;
        font-family: 'SF Mono', 'Fira Code', 'Monaco', monospace;
        letter-spacing: -0.5px;
        line-height: 1.1;
        margin: 0 0 8px 0;
    }

    .metric-value-large {
        font-size: 42px;
    }

    .metric-delta {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 14px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        margin-top: 4px;
    }

    .metric-delta.positive {
        color: #00E676;
        background: rgba(0, 230, 118, 0.12);
    }

    .metric-delta.negative {
        color: #FF5252;
        background: rgba(255, 82, 82, 0.12);
    }

    .metric-delta.neutral {
        color: rgba(255, 255, 255, 0.6);
        background: rgba(255, 255, 255, 0.08);
    }

    .metric-sparkline {
        margin-top: 16px;
        height: 40px;
        width: 100%;
    }

    .metric-sparkline svg {
        width: 100%;
        height: 100%;
    }

    /* Premium Summary Card */
    .premium-summary-card {
        background: linear-gradient(145deg, rgba(30, 40, 55, 0.98) 0%, rgba(15, 20, 30, 0.99) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 28px 32px;
        margin: 16px 0;
        backdrop-filter: blur(24px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.5);
        position: relative;
    }

    .premium-summary-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 24px;
        right: 24px;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 230, 118, 0.3), transparent);
    }

    .summary-hero {
        text-align: center;
        margin-bottom: 24px;
        padding-bottom: 24px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .summary-hero-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: rgba(255, 255, 255, 0.4);
        margin-bottom: 8px;
    }

    .summary-hero-value {
        font-size: 56px;
        font-weight: 800;
        color: #ffffff;
        font-family: 'SF Mono', 'Fira Code', 'Monaco', monospace;
        letter-spacing: -2px;
        text-shadow: 0 0 60px rgba(0, 230, 118, 0.3);
    }

    .summary-hero-delta {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 18px;
        font-weight: 600;
        margin-top: 12px;
    }

    .summary-hero-delta.positive {
        color: #00E676;
    }

    .summary-hero-delta.negative {
        color: #FF5252;
    }

    .summary-metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
    }

    .summary-metric-item {
        text-align: center;
        padding: 16px 12px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        transition: all 0.2s ease;
    }

    .summary-metric-item:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(255, 255, 255, 0.08);
    }

    .summary-metric-label {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: rgba(255, 255, 255, 0.4);
        margin-bottom: 8px;
    }

    .summary-metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        font-family: 'SF Mono', 'Fira Code', 'Monaco', monospace;
    }

    .summary-metric-sub {
        font-size: 11px;
        color: rgba(255, 255, 255, 0.35);
        margin-top: 4px;
    }

    /* Metric Row Container */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 16px 0;
    }

    /* Progress Bar */
    .metric-progress-container {
        margin-top: 12px;
        height: 6px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 3px;
        overflow: hidden;
    }

    .metric-progress-bar {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s ease;
    }

    .metric-progress-bar.income {
        background: linear-gradient(90deg, #00E676, #00C853);
    }

    .metric-progress-bar.growth {
        background: linear-gradient(90deg, #2196F3, #1976D2);
    }

    .metric-progress-bar.speculative {
        background: linear-gradient(90deg, #FF9800, #F57C00);
    }

    /* Accent colors for different metric types */
    .premium-metric-card.accent-green {
        border-left: 3px solid #00E676;
    }

    .premium-metric-card.accent-blue {
        border-left: 3px solid #2196F3;
    }

    .premium-metric-card.accent-gold {
        border-left: 3px solid #FFD700;
    }

    .premium-metric-card.accent-purple {
        border-left: 3px solid #9C27B0;
    }

    /* Hover detail tooltip */
    .metric-detail {
        font-size: 11px;
        color: rgba(255, 255, 255, 0.35);
        margin-top: 8px;
        opacity: 0;
        transform: translateY(4px);
        transition: all 0.2s ease;
    }

    .premium-metric-card:hover .metric-detail {
        opacity: 1;
        transform: translateY(0);
    }

    /* Light theme overrides for premium cards */
    .light-theme .premium-metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.98) 100%);
        border: 1px solid rgba(0, 0, 0, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.9);
    }

    .light-theme .premium-metric-card::before {
        background: linear-gradient(90deg, transparent, rgba(0, 0, 0, 0.06), transparent);
    }

    .light-theme .metric-title {
        color: rgba(0, 0, 0, 0.5);
    }

    .light-theme .metric-value {
        color: #1e293b;
    }

    .light-theme .premium-summary-card {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 250, 252, 0.99) 100%);
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.12);
    }

    .light-theme .premium-summary-card::before {
        background: linear-gradient(90deg, transparent, rgba(0, 100, 50, 0.2), transparent);
    }

    .light-theme .summary-hero-label,
    .light-theme .summary-metric-label {
        color: rgba(0, 0, 0, 0.5);
    }

    .light-theme .summary-hero-value,
    .light-theme .summary-metric-value {
        color: #1e293b;
        text-shadow: none;
    }

    .light-theme .summary-metric-sub,
    .light-theme .metric-detail {
        color: rgba(0, 0, 0, 0.4);
    }

    .light-theme .summary-metric-item {
        background: rgba(0, 0, 0, 0.02);
        border: 1px solid rgba(0, 0, 0, 0.04);
    }

    .light-theme .summary-hero {
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    }

    /* Mobile Responsive */
    @media (max-width: 768px) {
        .metric-value {
            font-size: 24px;
        }
        .summary-hero-value {
            font-size: 36px;
        }
        .summary-metrics-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        .premium-metric-card {
            padding: 16px 18px;
        }
        .premium-summary-card {
            padding: 20px 24px;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def generate_sparkline_svg(data, width=120, height=36, positive=True):
    """
    Generate an inline SVG sparkline from data points.

    Args:
        data: List of numeric values
        width: SVG width
        height: SVG height
        positive: If True, use green color; if False, use red

    Returns:
        SVG string for embedding in HTML
    """
    if not data or len(data) < 2:
        return ""

    # Normalize data to fit in the SVG viewbox
    data = list(data)
    min_val = min(data)
    max_val = max(data)
    val_range = max_val - min_val if max_val != min_val else 1

    # Create path points
    points = []
    for i, val in enumerate(data):
        x = (i / (len(data) - 1)) * width
        y = height - ((val - min_val) / val_range) * (height - 4) - 2
        points.append(f"{x:.1f},{y:.1f}")

    path_d = "M " + " L ".join(points)
    fill_d = f"M 0,{height} L " + " L ".join(points) + f" L {width},{height} Z"

    stroke_color = "#00E676" if positive else "#FF5252"

    return f'''
    <svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">
        <defs>
            <linearGradient id="sparkGrad_{id(data)}" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:{stroke_color};stop-opacity:0.3" />
                <stop offset="100%" style="stop-color:{stroke_color};stop-opacity:0" />
            </linearGradient>
        </defs>
        <path d="{fill_d}" fill="url(#sparkGrad_{id(data)})" />
        <path d="{path_d}" fill="none" stroke="{stroke_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
    '''


def get_theme_class():
    """Get the appropriate theme class based on current theme"""
    try:
        # Check Streamlit session state directly
        if 'theme' in st.session_state:
            return "light-theme" if st.session_state.theme == 'light' else ""
        return ""
    except Exception:
        # Default to dark theme
        return ""


def render_premium_metric(title, value, delta=None, delta_color="normal", icon=None, trend_data=None, accent=None, detail=None):
    """
    Render a premium metric card using native Streamlit components for reliability.

    Args:
        title: Metric label
        value: Main display value (pre-formatted string)
        delta: Change value (e.g., "+5.2%" or "-100")
        delta_color: "normal" (green=positive), "inverse" (red=positive), or "neutral"
        icon: Emoji or icon string
        trend_data: List of recent values for sparkline (not used with native)
        accent: Color accent (not used with native)
        detail: Additional detail shown below
    """
    # Build title with icon
    display_title = f"{icon} {title}" if icon else title

    # Use native st.metric for reliability
    st.metric(
        label=display_title.upper(),
        value=value,
        delta=delta,
        delta_color="normal" if delta_color != "inverse" else "inverse",
        help=detail
    )

    # Show detail as caption if provided
    if detail:
        st.caption(detail)


def render_metric_row(metrics):
    """
    Render multiple metrics in a responsive row.

    Args:
        metrics: List of dicts with keys: title, value, delta, delta_color, icon, trend, accent, detail

    Example:
        render_metric_row([
            {'title': 'Value', 'value': '£10,000', 'delta': '+5.2%', 'icon': '💰', 'accent': 'green'},
            {'title': 'P/L', 'value': '+£500', 'delta': '+5.2%', 'icon': '📈', 'accent': 'blue'},
        ])
    """
    if not metrics:
        return

    cols = st.columns(len(metrics))
    for i, metric in enumerate(metrics):
        with cols[i]:
            render_premium_metric(
                title=metric.get('title', ''),
                value=metric.get('value', ''),
                delta=metric.get('delta'),
                delta_color=metric.get('delta_color', 'normal'),
                icon=metric.get('icon'),
                trend_data=metric.get('trend'),
                accent=metric.get('accent'),
                detail=metric.get('detail')
            )


def render_portfolio_summary_card(total_value, pl_pct, monthly_income, annual_income, realized_income, prev_total=None):
    """
    Render a premium portfolio summary card with all key metrics.

    Args:
        total_value: Total portfolio value
        pl_pct: Profit/loss percentage
        monthly_income: Projected monthly dividend income
        annual_income: Projected annual dividend income
        realized_income: Total realized dividend income to date
        prev_total: Previous period total for comparison (optional)
    """
    theme_class = get_theme_class()

    # Determine delta display
    pl_class = "positive" if pl_pct >= 0 else "negative"
    pl_arrow = "^" if pl_pct >= 0 else "v"

    # Calculate period change if previous value available
    period_change_html = ""
    if prev_total and prev_total > 0:
        period_pct = ((total_value - prev_total) / prev_total) * 100
        period_class = "positive" if period_pct >= 0 else "negative"
        period_arrow = "^" if period_pct >= 0 else "v"
        period_change_html = f'<span style="color: rgba(128,128,128,0.6); margin-left: 12px;">vs last month: </span><span class="{period_class}">{period_arrow} {period_pct:+.1f}%</span>'

    # Calculate yield on portfolio
    portfolio_yield = (annual_income / total_value * 100) if total_value > 0 else 0

    # Calculate P/L amount
    pl_amount = int(total_value * pl_pct / 100)
    pl_color = '#00E676' if pl_pct >= 0 else '#FF5252'

    # Use Streamlit native components for reliability
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Monthly Income", f"£{monthly_income:,}", help="Projected monthly dividend income")
    with col2:
        st.metric("Annual Income", f"£{annual_income:,}", delta=f"{portfolio_yield:.1f}% yield")
    with col3:
        st.metric("Realized", f"£{realized_income:,}", help="Total dividends received")
    with col4:
        delta_str = f"£{pl_amount:+,}"
        st.metric("P/L Amount", delta_str, delta=f"{pl_pct:+.1f}%")
