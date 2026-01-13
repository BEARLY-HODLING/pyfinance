"""
Smart Portfolio Insights System
Generates intelligent, actionable insights for portfolio management
"""

import streamlit as st
import pandas as pd
from datetime import timedelta


def get_insight_priority_emoji(priority):
    """Get emoji indicator for insight priority level"""
    priority_map = {
        1: "🔴",  # Critical - immediate attention
        2: "🟠",  # High - review soon
        3: "🟡",  # Medium - be aware
        4: "🔵",  # Low - informational
        5: "⚪",  # Minimal - nice to know
    }
    return priority_map.get(priority, "⚪")


def generate_portfolio_insights(holdings_df, metrics, transactions_df, ticker_data=None, category_targets=None, benchmark_return=None):
    """
    Generate smart, actionable portfolio insights like a financial advisor.

    Returns a list of insight objects, each with:
    - type: 'alert' | 'opportunity' | 'achievement' | 'info'
    - icon: emoji
    - title: short headline
    - description: detailed explanation
    - action: optional suggested action
    - priority: 1-5 (1 = most urgent)
    """
    insights = []

    if holdings_df is None or holdings_df.empty:
        return insights

    total_value = metrics.get('total_value', 0)
    if total_value == 0:
        return insights

    # =========================================================================
    # CONCENTRATION RISK (any holding > 15%)
    # =========================================================================
    concentrated = holdings_df[holdings_df['Weight %'] > 15]
    for _, row in concentrated.iterrows():
        weight = row['Weight %']
        severity = 1 if weight > 25 else 2 if weight > 20 else 3
        insights.append({
            'type': 'alert',
            'icon': '⚠️',
            'title': f"Concentration Risk: {row['Ticker']}",
            'description': f"{row['Ticker']} represents {weight:.1f}% of your portfolio. Single-stock concentration above 15% increases unsystematic risk that could be diversified away.",
            'action': f"Consider trimming by £{int((weight - 10) / 100 * total_value):,} to reduce to 10%",
            'priority': severity
        })

    # =========================================================================
    # HIGH VOLATILITY HOLDINGS (vol > 30%)
    # =========================================================================
    if 'Vol %' in holdings_df.columns:
        high_vol = holdings_df[(holdings_df['Vol %'].notna()) & (holdings_df['Vol %'] > 30)]
        for _, row in high_vol.iterrows():
            vol = row['Vol %']
            weight = row['Weight %']
            insights.append({
                'type': 'alert',
                'icon': '📊',
                'title': f"High Volatility: {row['Ticker']}",
                'description': f"{row['Ticker']} has {vol:.1f}% annualized volatility - significantly above market average (~15-20%). This position ({weight:.1f}% of portfolio) may experience large swings.",
                'action': "Review position sizing or add defensive holdings to balance" if weight > 5 else None,
                'priority': 2 if weight > 10 else 3
            })

    # =========================================================================
    # UNDERPERFORMING VS BENCHMARK
    # =========================================================================
    if benchmark_return is not None:
        portfolio_return = (total_value - metrics.get('total_cost', total_value)) / metrics.get('total_cost', total_value) * 100
        alpha = portfolio_return - benchmark_return
        if alpha < -5:
            insights.append({
                'type': 'alert',
                'icon': '📉',
                'title': f"Underperforming Benchmark by {abs(alpha):.1f}%",
                'description': f"Your portfolio return ({portfolio_return:+.1f}%) is trailing the market benchmark ({benchmark_return:+.1f}%). This underperformance may warrant a strategy review.",
                'action': "Review worst performers and consider rebalancing to index-like weights",
                'priority': 2
            })
        elif alpha > 5:
            insights.append({
                'type': 'achievement',
                'icon': '🏆',
                'title': f"Outperforming Benchmark by {alpha:.1f}%",
                'description': f"Congratulations! Your portfolio ({portfolio_return:+.1f}%) is beating the market ({benchmark_return:+.1f}%). Your stock selection is adding value.",
                'action': None,
                'priority': 4
            })

    # =========================================================================
    # CATEGORY IMBALANCE VS TARGETS
    # =========================================================================
    if category_targets:
        for category, target in category_targets.items():
            cat_holdings = holdings_df[holdings_df['Category'] == category]
            current_pct = cat_holdings['Value £'].sum() / total_value * 100 if total_value > 0 else 0
            drift = current_pct - target

            if abs(drift) > 10:
                is_over = drift > 0
                insights.append({
                    'type': 'alert' if is_over else 'opportunity',
                    'icon': '⚖️',
                    'title': f"{category.title()}: {'Overweight' if is_over else 'Underweight'} by {abs(drift):.1f}%",
                    'description': f"Your {category} allocation is {current_pct:.1f}% vs target {target:.1f}%. {'Excess concentration may increase sector-specific risk.' if is_over else 'You may be missing exposure to this strategy.'}",
                    'action': f"{'Reduce' if is_over else 'Increase'} {category} holdings by ~£{int(abs(drift) / 100 * total_value):,}",
                    'priority': 2
                })
            elif abs(drift) > 5:
                insights.append({
                    'type': 'info',
                    'icon': '📋',
                    'title': f"{category.title()}: Slight {'Overweight' if drift > 0 else 'Underweight'}",
                    'description': f"{category.title()} at {current_pct:.1f}% (target: {target:.1f}%). Minor drift within acceptable range.",
                    'action': None,
                    'priority': 4
                })

    # =========================================================================
    # HIGH USD EXPOSURE
    # =========================================================================
    usd_exposure = metrics.get('usd_exposure', 0)
    usd_pct = (usd_exposure / total_value * 100) if total_value > 0 else 0
    if usd_pct > 30:
        insights.append({
            'type': 'alert' if usd_pct > 50 else 'info',
            'icon': '💵',
            'title': f"High USD Exposure: {usd_pct:.1f}%",
            'description': f"£{usd_exposure:,.0f} of your portfolio is in USD-denominated assets. GBP/USD fluctuations directly impact your returns.",
            'action': "Consider GBP-hedged alternatives or adding UK-listed holdings" if usd_pct > 50 else None,
            'priority': 2 if usd_pct > 50 else 3
        })

    # =========================================================================
    # YIELD ANOMALIES (unusually high/low)
    # =========================================================================
    if 'Trail Yield %' in holdings_df.columns:
        # Unusually high yield (might indicate distress or unsustainability)
        high_yielders = holdings_df[(holdings_df['Trail Yield %'] > 10) & (holdings_df['Weight %'] > 2)]
        for _, row in high_yielders.iterrows():
            insights.append({
                'type': 'alert',
                'icon': '⚡',
                'title': f"Unusually High Yield: {row['Ticker']} ({row['Trail Yield %']:.1f}%)",
                'description': f"Yields above 10% may indicate market skepticism about dividend sustainability, potential capital erosion, or elevated risk.",
                'action': "Research recent news and earnings - very high yields often precede dividend cuts",
                'priority': 2
            })

        # Low/no yield in income holdings
        income_holdings = holdings_df[holdings_df['Category'] == 'income']
        low_income = income_holdings[(income_holdings['Trail Yield %'] < 2) & (income_holdings['Weight %'] > 2)]
        for _, row in low_income.iterrows():
            insights.append({
                'type': 'info',
                'icon': '💭',
                'title': f"Low Yield for Income Holding: {row['Ticker']}",
                'description': f"{row['Ticker']} is categorized as 'income' but only yields {row['Trail Yield %']:.1f}%. Consider if this still fits your income strategy.",
                'action': "Review if holding should be recategorized to 'growth'",
                'priority': 4
            })

    # =========================================================================
    # HOLDINGS APPROACHING 52-WEEK HIGH/LOW
    # =========================================================================
    if ticker_data:
        for _, row in holdings_df.iterrows():
            ticker = row['Ticker']
            if ticker in ticker_data:
                hist = ticker_data[ticker].get('hist')

                if hist is not None and not hist.empty:
                    current_price = hist.iloc[-1] if len(hist) > 0 else None
                    week_52_high = hist.max()
                    week_52_low = hist.min()

                    if current_price and week_52_high and week_52_low:
                        # Near 52-week high (within 5%)
                        if current_price >= week_52_high * 0.95 and row['Weight %'] > 2:
                            pct_from_high = (week_52_high - current_price) / week_52_high * 100
                            insights.append({
                                'type': 'achievement',
                                'icon': '🎯',
                                'title': f"Near 52-Week High: {ticker}",
                                'description': f"{ticker} is trading within {pct_from_high:.1f}% of its 52-week high. Strong momentum, but consider taking partial profits.",
                                'action': "Consider trailing stop-loss or trimming if significantly overweight",
                                'priority': 3
                            })

                        # Near 52-week low (within 10%)
                        elif current_price <= week_52_low * 1.10 and row['Weight %'] > 1:
                            pct_from_low = (current_price - week_52_low) / week_52_low * 100
                            pl = row['P/L %']
                            insights.append({
                                'type': 'opportunity' if pl < -10 else 'info',
                                'icon': '📍',
                                'title': f"Near 52-Week Low: {ticker}",
                                'description': f"{ticker} is trading near 52-week lows ({pct_from_low:.1f}% above). {'Could be an averaging-down opportunity if thesis intact.' if pl < -10 else 'Monitor for potential support.'}",
                                'action': "Review investment thesis - is this temporary weakness or fundamental deterioration?",
                                'priority': 3
                            })

    # =========================================================================
    # BIG WINNERS (based on P/L)
    # =========================================================================
    big_winners = holdings_df[(holdings_df['P/L %'] > 50) & (holdings_df['Weight %'] > 3)]
    for _, row in big_winners.iterrows():
        insights.append({
            'type': 'achievement',
            'icon': '🚀',
            'title': f"Big Winner: {row['Ticker']} +{row['P/L %']:.0f}%",
            'description': f"Excellent performance! {row['Ticker']} has gained {row['P/L %']:.1f}% and now represents {row['Weight %']:.1f}% of your portfolio.",
            'action': "Consider taking some profits to lock in gains and rebalance",
            'priority': 3
        })

    # =========================================================================
    # RECENT DIVIDEND ACTIVITY (from transactions)
    # =========================================================================
    if transactions_df is not None and not transactions_df.empty:
        transactions_copy = transactions_df.copy()
        transactions_copy['Timestamp'] = pd.to_datetime(transactions_copy['Timestamp'])
        # Remove timezone info for comparison
        if transactions_copy['Timestamp'].dt.tz is not None:
            transactions_copy['Timestamp'] = transactions_copy['Timestamp'].dt.tz_localize(None)

        # Recent dividends (last 30 days)
        thirty_days_ago = pd.Timestamp.now() - timedelta(days=30)
        recent_divs = transactions_copy[
            (transactions_copy['Type'].str.contains('DIVIDEND|INTEREST', na=False)) &
            (transactions_copy['Timestamp'] > thirty_days_ago)
        ]

        if not recent_divs.empty:
            total_recent = recent_divs['Total Amount'].sum()
            count = len(recent_divs)
            insights.append({
                'type': 'achievement',
                'icon': '💰',
                'title': f"Recent Income: £{total_recent:.0f} ({count} payments)",
                'description': f"You've received £{total_recent:.2f} in dividends/interest over the past 30 days across {count} payment(s).",
                'action': None,
                'priority': 4
            })

    # =========================================================================
    # PORTFOLIO HEALTH INDICATORS
    # =========================================================================
    portfolio_vol = metrics.get('portfolio_vol', 0)
    if portfolio_vol > 25:
        insights.append({
            'type': 'alert',
            'icon': '🌊',
            'title': f"High Portfolio Volatility: {portfolio_vol:.1f}%",
            'description': "Your portfolio volatility exceeds typical market levels. Consider adding defensive positions like bonds or low-beta stocks.",
            'action': "Review position sizing in highest-volatility holdings",
            'priority': 2
        })
    elif portfolio_vol < 10:
        insights.append({
            'type': 'info',
            'icon': '🛡️',
            'title': f"Low Portfolio Volatility: {portfolio_vol:.1f}%",
            'description': "Your portfolio is quite defensive with low volatility. Good for capital preservation but may underperform in bull markets.",
            'action': None,
            'priority': 5
        })

    # High beta warning
    portfolio_beta = metrics.get('portfolio_beta', 1.0)
    if portfolio_beta > 1.3:
        insights.append({
            'type': 'alert',
            'icon': '📈',
            'title': f"Aggressive Portfolio Beta: {portfolio_beta:.2f}",
            'description': f"Your portfolio moves {(portfolio_beta - 1) * 100:.0f}% more than the market. This amplifies both gains and losses.",
            'action': "Consider adding lower-beta holdings to reduce market sensitivity",
            'priority': 3
        })

    # =========================================================================
    # DIVERSIFICATION CHECK
    # =========================================================================
    num_holdings = len(holdings_df)
    if num_holdings < 5:
        insights.append({
            'type': 'alert',
            'icon': '🎲',
            'title': f"Limited Diversification: Only {num_holdings} Holdings",
            'description': "With fewer than 5 positions, your portfolio lacks diversification benefits. A single stock's poor performance could significantly impact your wealth.",
            'action': "Consider adding 5-10 more positions across different sectors",
            'priority': 2
        })
    elif num_holdings > 30:
        insights.append({
            'type': 'info',
            'icon': '📚',
            'title': f"Many Holdings: {num_holdings} Positions",
            'description': "A large number of holdings can be difficult to monitor and may result in closet indexing. Consider consolidating into your highest-conviction ideas.",
            'action': "Review smallest positions for potential consolidation",
            'priority': 4
        })

    # =========================================================================
    # OVERALL YIELD HEALTH
    # =========================================================================
    yield_on_value = (metrics.get('projected_annual', 0) / total_value * 100) if total_value > 0 else 0
    if yield_on_value > 6:
        insights.append({
            'type': 'achievement',
            'icon': '🌟',
            'title': f"Strong Income Generation: {yield_on_value:.1f}% Yield",
            'description': f"Your portfolio yields £{metrics.get('projected_annual', 0):,.0f} annually ({yield_on_value:.1f}% on current value). Well above typical market yield of 2-3%.",
            'action': None,
            'priority': 4
        })

    # Sort by priority
    insights.sort(key=lambda x: x['priority'])

    return insights


def render_insights_panel(insights, max_display=5, theme='dark', colors=None):
    """
    Render a premium card-based insights panel with smooth animations.
    Color-coded by type with expandable details.
    """
    if not insights:
        st.info("🎉 No insights at this time - your portfolio looks well-balanced!")
        return

    # Default colors if not provided
    if colors is None:
        colors = {
            'text': '#f8fafc' if theme == 'dark' else '#1e293b',
            'text_secondary': '#94a3b8' if theme == 'dark' else '#64748b',
            'accent': '#d4af37' if theme == 'dark' else '#1a365d'
        }

    insight_css = f"""
    <style>
        .insight-container {{
            margin-bottom: 1rem;
        }}

        .insight-card {{
            background: {'rgba(30, 41, 59, 0.8)' if theme == 'dark' else 'rgba(255, 255, 255, 0.9)'};
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            border-left: 4px solid;
            box-shadow: 0 4px 15px rgba(0, 0, 0, {'0.2' if theme == 'dark' else '0.08'});
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }}

        .insight-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, {'0.3' if theme == 'dark' else '0.12'});
        }}

        .insight-card.alert {{
            border-left-color: #ef4444;
            background: linear-gradient(135deg, {'rgba(239, 68, 68, 0.1)' if theme == 'dark' else 'rgba(239, 68, 68, 0.05)'}, {'rgba(30, 41, 59, 0.8)' if theme == 'dark' else 'rgba(255, 255, 255, 0.9)'});
        }}

        .insight-card.opportunity {{
            border-left-color: #3b82f6;
            background: linear-gradient(135deg, {'rgba(59, 130, 246, 0.1)' if theme == 'dark' else 'rgba(59, 130, 246, 0.05)'}, {'rgba(30, 41, 59, 0.8)' if theme == 'dark' else 'rgba(255, 255, 255, 0.9)'});
        }}

        .insight-card.achievement {{
            border-left-color: #22c55e;
            background: linear-gradient(135deg, {'rgba(34, 197, 94, 0.1)' if theme == 'dark' else 'rgba(34, 197, 94, 0.05)'}, {'rgba(30, 41, 59, 0.8)' if theme == 'dark' else 'rgba(255, 255, 255, 0.9)'});
        }}

        .insight-card.info {{
            border-left-color: #6b7280;
            background: linear-gradient(135deg, {'rgba(107, 114, 128, 0.1)' if theme == 'dark' else 'rgba(107, 114, 128, 0.05)'}, {'rgba(30, 41, 59, 0.8)' if theme == 'dark' else 'rgba(255, 255, 255, 0.9)'});
        }}

        .insight-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }}

        .insight-icon {{
            font-size: 1.5rem;
            line-height: 1;
        }}

        .insight-title {{
            font-weight: 600;
            font-size: 1rem;
            color: {colors['text']};
            flex: 1;
        }}

        .insight-priority {{
            font-size: 0.75rem;
            opacity: 0.7;
        }}

        .insight-description {{
            font-size: 0.875rem;
            color: {colors['text_secondary']};
            line-height: 1.5;
            margin-bottom: 0.5rem;
        }}

        .insight-action {{
            font-size: 0.8rem;
            color: {colors['accent']};
            font-weight: 500;
            padding: 0.5rem 0.75rem;
            background: {'rgba(212, 175, 55, 0.1)' if theme == 'dark' else 'rgba(26, 54, 93, 0.1)'};
            border-radius: 6px;
            margin-top: 0.5rem;
            display: inline-block;
        }}

        .insight-action::before {{
            content: ">>> ";
        }}

        .insights-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
        }}

        .insights-title {{
            font-size: 1.25rem;
            font-weight: 600;
            color: {colors['text']};
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .insights-count {{
            font-size: 0.75rem;
            background: {colors['accent']};
            color: {'#0f172a' if theme == 'dark' else '#ffffff'};
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-weight: 600;
        }}

        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateX(-10px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}

        .insight-card {{
            animation: slideIn 0.3s ease-out forwards;
        }}

        .insight-card:nth-child(1) {{ animation-delay: 0ms; }}
        .insight-card:nth-child(2) {{ animation-delay: 50ms; }}
        .insight-card:nth-child(3) {{ animation-delay: 100ms; }}
        .insight-card:nth-child(4) {{ animation-delay: 150ms; }}
        .insight-card:nth-child(5) {{ animation-delay: 200ms; }}
    </style>
    """
    st.markdown(insight_css, unsafe_allow_html=True)

    # Header
    st.markdown(f"""
        <div class="insights-header">
            <div class="insights-title">
                Smart Insights
                <span class="insights-count">{len(insights)}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Show top insights
    displayed = insights[:max_display]
    hidden = insights[max_display:]

    # Initialize dismissed insights in session state
    if 'dismissed_insights' not in st.session_state:
        st.session_state.dismissed_insights = set()

    for idx, insight in enumerate(displayed):
        insight_key = f"{insight['title']}_{insight['type']}"

        if insight_key in st.session_state.dismissed_insights:
            continue

        priority_emoji = get_insight_priority_emoji(insight['priority'])
        action_html = f'<div class="insight-action">{insight["action"]}</div>' if insight.get('action') else ''

        st.markdown(f"""
            <div class="insight-container">
                <div class="insight-card {insight['type']}">
                    <div class="insight-header">
                        <span class="insight-icon">{insight['icon']}</span>
                        <span class="insight-title">{insight['title']}</span>
                        <span class="insight-priority">{priority_emoji}</span>
                    </div>
                    <div class="insight-description">{insight['description']}</div>
                    {action_html}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Dismiss button (using columns for alignment)
        col1, col2 = st.columns([8, 1])
        with col2:
            if st.button("x", key=f"dismiss_{idx}_{insight_key[:20]}", help="Dismiss this insight"):
                st.session_state.dismissed_insights.add(insight_key)
                st.rerun()

    # Show more button
    if hidden:
        with st.expander(f"Show {len(hidden)} more insights"):
            for idx, insight in enumerate(hidden):
                priority_emoji = get_insight_priority_emoji(insight['priority'])
                action_html = f'<div class="insight-action">{insight["action"]}</div>' if insight.get('action') else ''

                st.markdown(f"""
                    <div class="insight-container">
                        <div class="insight-card {insight['type']}">
                            <div class="insight-header">
                                <span class="insight-icon">{insight['icon']}</span>
                                <span class="insight-title">{insight['title']}</span>
                                <span class="insight-priority">{priority_emoji}</span>
                            </div>
                            <div class="insight-description">{insight['description']}</div>
                            {action_html}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # Clear dismissed button
    if st.session_state.dismissed_insights:
        if st.button("Reset dismissed insights", key="reset_dismissed"):
            st.session_state.dismissed_insights = set()
            st.rerun()


def render_insights_sidebar(insights, theme='dark', colors=None):
    """
    Render a compact sidebar version of insights.
    Shows icons and titles only, with click to expand.
    """
    if not insights:
        return

    # Default colors if not provided
    if colors is None:
        colors = {
            'text': '#f8fafc' if theme == 'dark' else '#1e293b',
        }

    sidebar_css = f"""
    <style>
        .sidebar-insight {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
            margin-bottom: 0.25rem;
            border-radius: 6px;
            background: {'rgba(30, 41, 59, 0.5)' if theme == 'dark' else 'rgba(0, 0, 0, 0.03)'};
            cursor: pointer;
            transition: background 0.2s ease;
        }}

        .sidebar-insight:hover {{
            background: {'rgba(30, 41, 59, 0.8)' if theme == 'dark' else 'rgba(0, 0, 0, 0.06)'};
        }}

        .sidebar-insight.alert {{
            border-left: 3px solid #ef4444;
        }}

        .sidebar-insight.opportunity {{
            border-left: 3px solid #3b82f6;
        }}

        .sidebar-insight.achievement {{
            border-left: 3px solid #22c55e;
        }}

        .sidebar-insight.info {{
            border-left: 3px solid #6b7280;
        }}

        .sidebar-insight-icon {{
            font-size: 1rem;
        }}

        .sidebar-insight-title {{
            font-size: 0.75rem;
            color: {colors['text']};
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
    </style>
    """
    st.markdown(sidebar_css, unsafe_allow_html=True)

    st.markdown("**Quick Insights**")

    # Show top 5 in sidebar
    for insight in insights[:5]:
        st.markdown(f"""
            <div class="sidebar-insight {insight['type']}">
                <span class="sidebar-insight-icon">{insight['icon']}</span>
                <span class="sidebar-insight-title">{insight['title']}</span>
            </div>
        """, unsafe_allow_html=True)

    if len(insights) > 5:
        st.caption(f"+{len(insights) - 5} more in Overview tab")
