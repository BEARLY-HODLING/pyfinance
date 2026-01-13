"""
Premium Data Table Components for Freetrade Dashboard

Provides professional trading platform-style tables with:
- Premium CSS styling with gradients, shadows, transitions
- Color-coded P/L (green positive, red negative)
- Category badges with custom colors
- Mini sparkline charts for 7D trend
- Hover row highlights
- Sticky headers
- Custom scrollbars
"""

import streamlit as st
import pandas as pd
from premium_metrics import generate_sparkline_svg


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_currency(value: float) -> str:
    """Format value as GBP currency with proper commas"""
    if pd.isna(value) or value is None:
        return "—"
    if abs(value) >= 1000:
        return f"£{value:,.0f}"
    return f"£{value:.2f}"


def format_percent(value: float, show_sign: bool = True) -> str:
    """Format value as percentage with optional sign"""
    if pd.isna(value) or value is None:
        return "—"
    if show_sign:
        return f"{value:+.1f}%"
    return f"{value:.1f}%"


def get_category_color(category: str) -> dict:
    """Returns color scheme for category badge

    Args:
        category: Category name (income, growth, speculative, or unknown)

    Returns:
        Dictionary with 'bg', 'text', and 'border' color values
    """
    colors = {
        'income': {'bg': '#1e3a5f', 'text': '#60a5fa', 'border': '#3b82f6'},
        'growth': {'bg': '#1a3d2e', 'text': '#4ade80', 'border': '#22c55e'},
        'speculative': {'bg': '#4a3728', 'text': '#fb923c', 'border': '#f97316'},
        'unknown': {'bg': '#374151', 'text': '#9ca3af', 'border': '#6b7280'}
    }
    return colors.get(category.lower() if category else 'unknown', colors['unknown'])


# ============================================================================
# CSS STYLES
# ============================================================================

def get_premium_table_css() -> str:
    """Return comprehensive CSS for premium data tables"""
    return """
    <style>
    /* Premium Table Container */
    .premium-table-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 1px rgba(212, 175, 55, 0.3);
        overflow: hidden;
        margin: 1rem 0;
    }

    .premium-table-wrapper {
        max-height: 500px;
        overflow-y: auto;
        overflow-x: auto;
    }

    /* Custom scrollbar */
    .premium-table-wrapper::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    .premium-table-wrapper::-webkit-scrollbar-track {
        background: #1e293b;
        border-radius: 4px;
    }

    .premium-table-wrapper::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #d4af37, #b8972e);
        border-radius: 4px;
    }

    .premium-table-wrapper::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #e5c158, #d4af37);
    }

    /* Premium Table */
    .premium-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sticky Header */
    .premium-table thead {
        position: sticky;
        top: 0;
        z-index: 10;
    }

    .premium-table thead th {
        background: linear-gradient(180deg, #334155 0%, #1e293b 100%);
        color: #d4af37;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 16px 12px;
        text-align: left;
        border-bottom: 2px solid #d4af37;
        white-space: nowrap;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .premium-table thead th:hover {
        background: linear-gradient(180deg, #475569 0%, #334155 100%);
        color: #fbbf24;
    }

    .premium-table thead th .sort-icon {
        margin-left: 4px;
        opacity: 0.5;
    }

    /* Table Body */
    .premium-table tbody tr {
        background: rgba(30, 41, 59, 0.5);
        transition: all 0.2s ease;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
    }

    .premium-table tbody tr:nth-child(even) {
        background: rgba(15, 23, 42, 0.5);
    }

    .premium-table tbody tr:hover {
        background: rgba(212, 175, 55, 0.1);
        transform: scale(1.005);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .premium-table tbody td {
        padding: 14px 12px;
        color: #f8fafc;
        font-size: 0.9rem;
        vertical-align: middle;
    }

    /* Ticker Column */
    .ticker-cell {
        font-weight: 700;
        font-size: 0.95rem;
        color: #f8fafc;
    }

    /* Category Badge */
    .category-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        border: 1px solid;
    }

    /* P/L Colors */
    .pl-positive {
        color: #22c55e;
        font-weight: 600;
    }

    .pl-negative {
        color: #ef4444;
        font-weight: 600;
    }

    .pl-neutral {
        color: #94a3b8;
    }

    /* Value Cells */
    .value-cell {
        font-weight: 500;
        font-family: 'SF Mono', 'Monaco', monospace;
    }

    /* Sparkline Container */
    .sparkline-cell {
        padding: 8px 12px;
    }

    .sparkline-cell svg {
        vertical-align: middle;
    }

    /* Yield Badge */
    .yield-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        background: rgba(34, 197, 94, 0.15);
        border-radius: 8px;
        font-size: 0.85rem;
        color: #4ade80;
    }

    .yield-badge.proxy {
        background: rgba(251, 191, 36, 0.15);
        color: #fbbf24;
    }

    .yield-badge.limited {
        background: rgba(249, 115, 22, 0.15);
        color: #fb923c;
    }

    /* Expandable Row Details */
    .row-details {
        background: rgba(15, 23, 42, 0.8);
        padding: 16px 24px;
        border-left: 3px solid #d4af37;
        margin: 0 12px 12px 12px;
        border-radius: 0 8px 8px 0;
    }

    .row-details-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 16px;
    }

    .detail-item {
        text-align: center;
    }

    .detail-label {
        font-size: 0.7rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .detail-value {
        font-size: 1rem;
        color: #f8fafc;
        font-weight: 600;
    }

    /* Dividend Table Styles */
    .dividend-table-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        overflow: hidden;
        margin: 1rem 0;
    }

    .dividend-table {
        width: 100%;
        border-collapse: collapse;
    }

    .dividend-table thead th {
        background: linear-gradient(180deg, #334155 0%, #1e293b 100%);
        color: #22c55e;
        padding: 14px 16px;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        text-align: left;
        border-bottom: 2px solid #22c55e;
    }

    .dividend-table tbody tr {
        transition: background 0.2s ease;
    }

    .dividend-table tbody tr:hover {
        background: rgba(34, 197, 94, 0.1);
    }

    .dividend-table tbody td {
        padding: 12px 16px;
        color: #f8fafc;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
    }

    .date-cell {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    .amount-cell {
        font-weight: 600;
        color: #22c55e;
        font-family: 'SF Mono', monospace;
    }

    .running-total-cell {
        color: #d4af37;
        font-weight: 500;
    }

    .ticker-badge {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid #6366f1;
        border-radius: 6px;
        color: #a5b4fc;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """


# ============================================================================
# TABLE COMPONENTS
# ============================================================================

def render_holdings_table(
    df: pd.DataFrame,
    ticker_data: dict = None,
    show_sparklines: bool = True
) -> None:
    """Render a premium HTML holdings table with sparklines and category badges.

    Args:
        df: DataFrame with holdings data (requires columns: Ticker, Category,
            Shares, Price £, Value £, Weight %, P/L %, Trail Yield %,
            Yield Source, Est Annual £)
        ticker_data: Dictionary of ticker data with 'hist' key for sparklines
        show_sparklines: Whether to show 7D trend sparklines
    """
    if df is None or df.empty:
        st.info("No holdings data available")
        return

    # Inject CSS
    st.markdown(get_premium_table_css(), unsafe_allow_html=True)

    # Build HTML table
    html_parts = ['<div class="premium-table-container"><div class="premium-table-wrapper">']
    html_parts.append('<table class="premium-table">')

    # Header
    html_parts.append('<thead><tr>')
    columns = ['Ticker', 'Category', 'Shares', 'Price', 'Value', 'Weight', 'P/L', 'Yield', 'Est Annual']
    if show_sparklines:
        columns.append('7D Trend')
    for col in columns:
        html_parts.append(f'<th>{col} <span class="sort-icon">↕</span></th>')
    html_parts.append('</tr></thead>')

    # Body
    html_parts.append('<tbody>')
    for _, row in df.iterrows():
        ticker = row.get('Ticker', 'N/A')
        category = row.get('Category', 'unknown')
        shares = row.get('Shares', 0)
        price = row.get('Price £', 0)
        value = row.get('Value £', 0)
        weight = row.get('Weight %', 0)
        pl = row.get('P/L %', 0)
        trail_yield = row.get('Trail Yield %', 0)
        yield_source = row.get('Yield Source', 'none')
        est_annual = row.get('Est Annual £', 0)

        # Category badge colors
        cat_colors = get_category_color(category)

        # P/L class
        pl_class = 'pl-positive' if pl > 0 else 'pl-negative' if pl < 0 else 'pl-neutral'

        # Yield badge class
        yield_class = 'proxy' if yield_source == 'proxy' else 'limited' if yield_source == 'calculated_limited' else ''
        yield_icon = '📌' if yield_source == 'proxy' else '⚠️' if yield_source == 'calculated_limited' else '🤖'

        html_parts.append('<tr>')
        html_parts.append(f'<td class="ticker-cell">{ticker}</td>')
        html_parts.append(f'<td><span class="category-badge" style="background:{cat_colors["bg"]};color:{cat_colors["text"]};border-color:{cat_colors["border"]}">{category}</span></td>')
        html_parts.append(f'<td class="value-cell">{shares:.2f}</td>')
        html_parts.append(f'<td class="value-cell">{format_currency(price)}</td>')
        html_parts.append(f'<td class="value-cell">{format_currency(value)}</td>')
        html_parts.append(f'<td>{weight:.1f}%</td>')
        html_parts.append(f'<td class="{pl_class}">{format_percent(pl)}</td>')
        html_parts.append(f'<td><span class="yield-badge {yield_class}">{yield_icon} {trail_yield:.1f}%</span></td>')
        html_parts.append(f'<td class="value-cell">{format_currency(est_annual)}</td>')

        # Sparkline
        if show_sparklines and ticker_data and ticker in ticker_data:
            hist = ticker_data[ticker].get('hist')
            if hist is not None and len(hist) >= 7:
                last_7 = hist.tail(7).tolist()
                sparkline = generate_sparkline_svg(last_7, width=60, height=20)
                html_parts.append(f'<td class="sparkline-cell">{sparkline}</td>')
            else:
                html_parts.append('<td class="sparkline-cell">—</td>')
        elif show_sparklines:
            html_parts.append('<td class="sparkline-cell">—</td>')

        html_parts.append('</tr>')

    html_parts.append('</tbody></table></div></div>')

    st.markdown(''.join(html_parts), unsafe_allow_html=True)


def render_dividend_table(transactions_df: pd.DataFrame, limit: int = 15) -> None:
    """Render a premium table showing recent dividend payments.

    Args:
        transactions_df: DataFrame with transaction data (requires columns:
            Type, Timestamp, Ticker, Total Amount)
        limit: Maximum number of recent dividends to show
    """
    if transactions_df is None or transactions_df.empty:
        st.info("No dividend data available")
        return

    # Filter for dividends/interest
    divs = transactions_df[transactions_df['Type'].str.contains('DIVIDEND|INTEREST', na=False)].copy()
    if divs.empty:
        st.info("No dividend payments recorded yet")
        return

    # Sort by date descending and limit
    divs['Timestamp'] = pd.to_datetime(divs['Timestamp'])
    divs = divs.sort_values('Timestamp', ascending=False).head(limit)

    # Calculate running total (from oldest to newest, then reverse back)
    divs_reversed = divs.iloc[::-1].copy()
    divs_reversed['Running Total'] = divs_reversed['Total Amount'].cumsum()
    divs = divs_reversed.iloc[::-1]

    # Inject CSS
    st.markdown(get_premium_table_css(), unsafe_allow_html=True)

    # Build HTML
    html_parts = ['<div class="dividend-table-container">']
    html_parts.append('<table class="dividend-table">')
    html_parts.append('<thead><tr>')
    html_parts.append('<th>Date</th><th>Ticker</th><th>Type</th><th>Amount</th><th>Running Total</th>')
    html_parts.append('</tr></thead>')

    html_parts.append('<tbody>')
    for _, row in divs.iterrows():
        date_str = row['Timestamp'].strftime('%d %b %Y')
        ticker = row.get('Ticker', 'N/A')
        pay_type = 'Dividend' if 'DIVIDEND' in str(row['Type']) else 'Interest'
        amount = row['Total Amount']
        running = row['Running Total']

        html_parts.append('<tr>')
        html_parts.append(f'<td class="date-cell">{date_str}</td>')
        html_parts.append(f'<td><span class="ticker-badge">{ticker}</span></td>')
        html_parts.append(f'<td>{pay_type}</td>')
        html_parts.append(f'<td class="amount-cell">{format_currency(amount)}</td>')
        html_parts.append(f'<td class="running-total-cell">{format_currency(running)}</td>')
        html_parts.append('</tr>')

    html_parts.append('</tbody></table></div>')

    st.markdown(''.join(html_parts), unsafe_allow_html=True)
