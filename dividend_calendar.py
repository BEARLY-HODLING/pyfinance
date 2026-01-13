"""
Dividend Calendar & Tracking Module for Freetrade Dashboard

Provides comprehensive dividend tracking with:
- Calendar view (month/year)
- Timeline view with predictions
- Statistics and analytics
- Per-ticker breakdown
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import calendar
from datetime import datetime, timedelta
from collections import defaultdict


def get_dividend_history(transactions_df):
    """
    Extract all dividend and interest payments from transaction data.

    Returns DataFrame with columns: date, ticker, amount, type, title
    """
    if transactions_df is None or transactions_df.empty:
        return pd.DataFrame(columns=['date', 'ticker', 'amount', 'type', 'title'])

    # Filter for dividend and interest transactions
    dividends = transactions_df[transactions_df['Type'].str.contains('DIVIDEND|INTEREST', na=False)].copy()

    if dividends.empty:
        return pd.DataFrame(columns=['date', 'ticker', 'amount', 'type', 'title'])

    # Process and standardize
    dividends['Timestamp'] = pd.to_datetime(dividends['Timestamp'])
    dividends['date'] = dividends['Timestamp'].dt.date

    # Determine type
    dividends['type'] = dividends['Type'].apply(
        lambda x: 'interest' if 'INTEREST' in str(x).upper() else 'dividend'
    )

    result = pd.DataFrame({
        'date': dividends['date'],
        'ticker': dividends['Ticker'].fillna('INTEREST'),
        'amount': dividends['Total Amount'].abs(),
        'type': dividends['type'],
        'title': dividends.get('Title', '').fillna('')
    })

    return result.sort_values('date', ascending=False).reset_index(drop=True)


def predict_next_dividends(dividend_history, holdings_df):
    """
    Analyze payment patterns and predict next expected payment dates.

    Returns DataFrame with columns: ticker, expected_date, estimated_amount, confidence, frequency
    """
    if dividend_history.empty:
        return pd.DataFrame(columns=['ticker', 'expected_date', 'estimated_amount', 'confidence', 'frequency'])

    predictions = []
    today = pd.Timestamp.now().date()

    # Group by ticker and analyze patterns
    for ticker in dividend_history['ticker'].unique():
        if ticker == 'INTEREST':
            continue  # Skip interest predictions

        ticker_divs = dividend_history[dividend_history['ticker'] == ticker].copy()
        ticker_divs['date'] = pd.to_datetime(ticker_divs['date'])
        ticker_divs = ticker_divs.sort_values('date')

        if len(ticker_divs) < 2:
            # Single payment - assume quarterly
            last_date = ticker_divs['date'].iloc[-1]
            last_amount = ticker_divs['amount'].iloc[-1]
            next_date = last_date + timedelta(days=90)

            if next_date.date() > today:
                predictions.append({
                    'ticker': ticker,
                    'expected_date': next_date.date(),
                    'estimated_amount': last_amount,
                    'confidence': 'low',
                    'frequency': 'unknown'
                })
            continue

        # Calculate gaps between payments
        gaps = ticker_divs['date'].diff().dropna().dt.days.tolist()
        avg_gap = np.mean(gaps)
        std_gap = np.std(gaps) if len(gaps) > 1 else 30

        # Determine frequency
        if avg_gap <= 35:
            frequency = 'monthly'
            expected_gap = 30
        elif avg_gap <= 100:
            frequency = 'quarterly'
            expected_gap = 91
        elif avg_gap <= 200:
            frequency = 'semi-annual'
            expected_gap = 182
        else:
            frequency = 'annual'
            expected_gap = 365

        # Calculate confidence based on regularity
        regularity = 1 - min(std_gap / max(avg_gap, 1), 1)
        if regularity > 0.8:
            confidence = 'high'
        elif regularity > 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Estimate next payment date
        last_date = ticker_divs['date'].iloc[-1]

        # Average recent amounts
        recent_amounts = ticker_divs['amount'].tail(4).mean()

        # Generate predictions for next 3 periods
        for i in range(3):
            pred_date = last_date + timedelta(days=expected_gap * (i + 1))
            if pred_date.date() > today:
                predictions.append({
                    'ticker': ticker,
                    'expected_date': pred_date.date(),
                    'estimated_amount': recent_amounts,
                    'confidence': confidence if i == 0 else 'low',
                    'frequency': frequency
                })

    if not predictions:
        return pd.DataFrame(columns=['ticker', 'expected_date', 'estimated_amount', 'confidence', 'frequency'])

    result = pd.DataFrame(predictions)
    return result.sort_values('expected_date').reset_index(drop=True)


def get_ticker_color_map(tickers, chart_colors):
    """Generate consistent colors for tickers"""
    color_map = {}
    for i, ticker in enumerate(sorted(tickers)):
        color_map[ticker] = chart_colors[i % len(chart_colors)]
    return color_map


def render_dividend_calendar(history_df, predictions_df, plotly_template, chart_colors, view='month'):
    """
    Render a calendar grid view of dividends.

    Args:
        history_df: Historical dividend data
        predictions_df: Predicted future dividends
        plotly_template: Plotly template string
        chart_colors: List of chart colors
        view: 'month' or 'year'
    """
    st.subheader("📅 Dividend Calendar")

    # View toggle
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        view_mode = st.radio("View", ['Month', 'Year'], horizontal=True, key="cal_view")

    with col2:
        today = pd.Timestamp.now()
        if view_mode == 'Month':
            selected_month = st.selectbox(
                "Month",
                options=range(1, 13),
                index=today.month - 1,
                format_func=lambda x: calendar.month_name[x],
                key="cal_month"
            )
            selected_year = st.selectbox("Year", options=[today.year - 1, today.year, today.year + 1], index=1, key="cal_year")
        else:
            selected_year = st.selectbox("Year", options=[today.year - 1, today.year, today.year + 1], index=1, key="cal_year_only")
            selected_month = None

    # Prepare data
    all_tickers = set()
    if not history_df.empty:
        all_tickers.update(history_df['ticker'].unique())
    if not predictions_df.empty:
        all_tickers.update(predictions_df['ticker'].unique())

    color_map = get_ticker_color_map(all_tickers, chart_colors)

    if view_mode == 'Month':
        _render_month_calendar(history_df, predictions_df, selected_year, selected_month, color_map, plotly_template)
    else:
        _render_year_calendar(history_df, predictions_df, selected_year, color_map, plotly_template)


def _render_month_calendar(history_df, predictions_df, year, month, color_map, plotly_template):
    """Render a single month calendar view"""

    # Get calendar data
    cal = calendar.Calendar(firstweekday=0)  # Monday start
    month_days = cal.monthdayscalendar(year, month)

    # Prepare dividend data for this month
    div_by_day = defaultdict(list)
    pred_by_day = defaultdict(list)

    # Historical dividends
    if not history_df.empty:
        for _, row in history_df.iterrows():
            d = pd.to_datetime(row['date'])
            if d.year == year and d.month == month:
                div_by_day[d.day].append({
                    'ticker': row['ticker'],
                    'amount': row['amount'],
                    'type': row['type']
                })

    # Predicted dividends
    if not predictions_df.empty:
        for _, row in predictions_df.iterrows():
            d = pd.to_datetime(row['expected_date'])
            if d.year == year and d.month == month:
                pred_by_day[d.day].append({
                    'ticker': row['ticker'],
                    'amount': row['estimated_amount'],
                    'confidence': row['confidence']
                })

    # Build heatmap data
    week_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Create figure
    fig = go.Figure()

    # Add cells for each day
    today = pd.Timestamp.now()
    annotations = []

    for week_idx, week in enumerate(month_days):
        for day_idx, day in enumerate(week):
            if day == 0:
                continue

            x = day_idx
            y = len(month_days) - week_idx - 1

            # Get dividends for this day
            divs = div_by_day.get(day, [])
            preds = pred_by_day.get(day, [])
            total_amount = sum(d['amount'] for d in divs) + sum(p['amount'] for p in preds)

            # Determine cell color
            if divs:
                cell_color = 'rgba(46, 204, 113, 0.3)'  # Green for received
            elif preds:
                cell_color = 'rgba(155, 89, 182, 0.2)'  # Purple for predicted
            else:
                cell_color = 'rgba(128, 128, 128, 0.05)'  # Gray for empty

            # Highlight today
            is_today = (year == today.year and month == today.month and day == today.day)
            if is_today:
                cell_color = 'rgba(241, 196, 15, 0.3)'  # Yellow for today

            # Add cell
            fig.add_shape(
                type="rect",
                x0=x - 0.45, x1=x + 0.45,
                y0=y - 0.45, y1=y + 0.45,
                fillcolor=cell_color,
                line=dict(color='rgba(128,128,128,0.3)', width=1),
            )

            # Day number annotation
            annotations.append(dict(
                x=x - 0.35, y=y + 0.35,
                text=str(day),
                showarrow=False,
                font=dict(size=10, color='gray'),
                xanchor='left', yanchor='top'
            ))

            # Amount annotation
            if total_amount > 0:
                annotations.append(dict(
                    x=x, y=y - 0.1,
                    text=f"£{total_amount:.0f}",
                    showarrow=False,
                    font=dict(size=11, color='white' if divs else '#9b59b6', weight='bold'),
                    xanchor='center', yanchor='middle'
                ))

            # Ticker dots
            all_items = [(d['ticker'], True) for d in divs] + [(p['ticker'], False) for p in preds]
            for i, (ticker, is_received) in enumerate(all_items[:3]):  # Max 3 dots
                color = color_map.get(ticker, '#888888')
                if is_received:
                    fig.add_trace(go.Scatter(
                        x=[x - 0.3 + i * 0.2],
                        y=[y + 0.15],
                        mode='markers',
                        marker=dict(size=8, color=color),
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=f"{ticker}: £{divs[i]['amount']:.2f}" if i < len(divs) else f"{ticker}: ~£{preds[i-len(divs)]['amount']:.2f} (predicted)"
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=[x - 0.3 + i * 0.2],
                        y=[y + 0.15],
                        mode='markers',
                        marker=dict(size=8, color=color, line=dict(width=2, color=color), symbol='circle-open'),
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=f"{ticker}: ~£{preds[i-len(divs)]['amount']:.2f} (predicted)" if i >= len(divs) else ""
                    ))

    fig.update_layout(
        template=plotly_template,
        title=f"{calendar.month_name[month]} {year}",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(7)),
            ticktext=week_labels,
            showgrid=False,
            zeroline=False,
            range=[-0.5, 6.5]
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-0.5, len(month_days) - 0.5]
        ),
        annotations=annotations,
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True, key="month_cal")

    # Legend
    st.caption("● Received | ○ Predicted | 🟡 Today")


def _render_year_calendar(history_df, predictions_df, year, color_map, plotly_template):
    """Render a year heatmap view"""

    # Aggregate by month
    monthly_data = defaultdict(lambda: {'received': 0, 'predicted': 0, 'tickers': set()})

    if not history_df.empty:
        for _, row in history_df.iterrows():
            d = pd.to_datetime(row['date'])
            if d.year == year:
                monthly_data[d.month]['received'] += row['amount']
                monthly_data[d.month]['tickers'].add(row['ticker'])

    if not predictions_df.empty:
        for _, row in predictions_df.iterrows():
            d = pd.to_datetime(row['expected_date'])
            if d.year == year:
                monthly_data[d.month]['predicted'] += row['estimated_amount']
                monthly_data[d.month]['tickers'].add(row['ticker'])

    months = list(range(1, 13))
    received = [monthly_data[m]['received'] for m in months]
    predicted = [monthly_data[m]['predicted'] for m in months]
    month_names = [calendar.month_abbr[m] for m in months]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=month_names,
        y=received,
        name='Received',
        marker_color='#2ecc71'
    ))

    fig.add_trace(go.Bar(
        x=month_names,
        y=predicted,
        name='Predicted',
        marker_color='rgba(155, 89, 182, 0.5)',
        marker_line=dict(width=2, color='#9b59b6')
    ))

    fig.update_layout(
        template=plotly_template,
        title=f"{year} Dividend Income by Month",
        barmode='stack',
        xaxis_title="Month",
        yaxis_title="Amount £",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, use_container_width=True, key="year_cal")


def render_dividend_timeline(history_df, predictions_df, plotly_template, chart_colors):
    """
    Render a horizontal timeline view showing past 6 months + next 3 months predicted.
    """
    st.subheader("📊 Dividend Timeline")

    if history_df.empty and predictions_df.empty:
        st.info("No dividend data available")
        return

    today = pd.Timestamp.now()
    start_date = today - timedelta(days=180)
    end_date = today + timedelta(days=90)

    # Prepare data
    timeline_data = []

    if not history_df.empty:
        for _, row in history_df.iterrows():
            d = pd.to_datetime(row['date'])
            if start_date <= d <= today:
                timeline_data.append({
                    'date': d,
                    'ticker': row['ticker'],
                    'amount': row['amount'],
                    'is_predicted': False
                })

    if not predictions_df.empty:
        for _, row in predictions_df.iterrows():
            d = pd.to_datetime(row['expected_date'])
            if today < d <= end_date:
                timeline_data.append({
                    'date': d,
                    'ticker': row['ticker'],
                    'amount': row['estimated_amount'],
                    'is_predicted': True
                })

    if not timeline_data:
        st.info("No dividend data in the selected timeframe")
        return

    df = pd.DataFrame(timeline_data)
    df['month'] = df['date'].dt.to_period('M').astype(str)

    # Get unique tickers for coloring
    all_tickers = df['ticker'].unique()
    color_map = get_ticker_color_map(all_tickers, chart_colors)

    # Aggregate by month and ticker
    monthly = df.groupby(['month', 'ticker', 'is_predicted'])['amount'].sum().reset_index()

    fig = go.Figure()

    # Add bars for each ticker
    for ticker in sorted(all_tickers):
        ticker_data = monthly[monthly['ticker'] == ticker]

        # Received
        received = ticker_data[~ticker_data['is_predicted']]
        if not received.empty:
            fig.add_trace(go.Bar(
                x=received['month'],
                y=received['amount'],
                name=ticker,
                marker_color=color_map[ticker],
                legendgroup=ticker
            ))

        # Predicted
        predicted = ticker_data[ticker_data['is_predicted']]
        if not predicted.empty:
            fig.add_trace(go.Bar(
                x=predicted['month'],
                y=predicted['amount'],
                name=f"{ticker} (pred)",
                marker_color=color_map[ticker],
                marker_line=dict(width=2, color='white'),
                opacity=0.5,
                legendgroup=ticker,
                showlegend=False
            ))

    # Add running total line
    monthly_totals = df.groupby('month')['amount'].sum().cumsum()
    fig.add_trace(go.Scatter(
        x=monthly_totals.index,
        y=monthly_totals.values,
        mode='lines+markers',
        name='Cumulative',
        line=dict(color='#f39c12', width=3, dash='dot'),
        yaxis='y2'
    ))

    # Add vertical line for today
    today_month = today.to_period('M').strftime('%Y-%m')
    fig.add_vline(x=today_month, line_dash="dash", line_color="red", annotation_text="Today")

    fig.update_layout(
        template=plotly_template,
        barmode='stack',
        xaxis_title="Month",
        yaxis_title="Monthly Income £",
        yaxis2=dict(
            title="Cumulative £",
            overlaying='y',
            side='right'
        ),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, use_container_width=True, key="div_timeline")


def render_dividend_stats(history_df, predictions_df, plotly_template):
    """Display comprehensive dividend statistics"""
    st.subheader("📈 Dividend Statistics")

    if history_df.empty:
        st.info("No dividend history available")
        return

    today = pd.Timestamp.now()
    current_year = today.year

    # Filter for current year
    history_df = history_df.copy()
    history_df['date'] = pd.to_datetime(history_df['date'])
    ytd_data = history_df[history_df['date'].dt.year == current_year]

    # Calculate metrics
    total_ytd = ytd_data['amount'].sum() if not ytd_data.empty else 0

    # Monthly average
    if not ytd_data.empty:
        ytd_data = ytd_data.copy()
        ytd_data['month'] = ytd_data['date'].dt.month
        monthly_totals = ytd_data.groupby('month')['amount'].sum()
        avg_monthly = monthly_totals.mean()
        best_month_idx = monthly_totals.idxmax()
        best_month_val = monthly_totals.max()
        best_month_name = calendar.month_name[best_month_idx]
    else:
        avg_monthly = 0
        best_month_name = "N/A"
        best_month_val = 0

    # Growth rate (compare last 3 months to previous 3 months)
    if len(history_df) >= 6:
        history_df_copy = history_df.copy()
        history_df_copy['month'] = history_df_copy['date'].dt.to_period('M')
        monthly = history_df_copy.groupby('month')['amount'].sum().sort_index()
        if len(monthly) >= 6:
            recent = monthly.tail(3).sum()
            previous = monthly.iloc[-6:-3].sum()
            growth_rate = ((recent - previous) / previous * 100) if previous > 0 else 0
        else:
            growth_rate = 0
    else:
        growth_rate = 0

    # Next expected total (sum of next 3 months predictions)
    if not predictions_df.empty:
        predictions_df = predictions_df.copy()
        predictions_df['expected_date'] = pd.to_datetime(predictions_df['expected_date'])
        next_3_months = predictions_df[
            (predictions_df['expected_date'] >= today) &
            (predictions_df['expected_date'] <= today + timedelta(days=90))
        ]
        next_expected = next_3_months['estimated_amount'].sum()
    else:
        next_expected = 0

    # Display metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Total YTD", f"£{total_ytd:,.0f}")
    c2.metric("📊 Avg Monthly", f"£{avg_monthly:,.0f}")
    c3.metric("🏆 Best Month", f"£{best_month_val:,.0f}", best_month_name)
    c4.metric("📈 Growth Rate", f"{growth_rate:+.1f}%")
    c5.metric("🔮 Next 3 Mo", f"£{next_expected:,.0f}")

    # Dividend type breakdown
    if not history_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            type_breakdown = history_df.groupby('type')['amount'].sum().reset_index()
            fig = px.pie(
                type_breakdown,
                values='amount',
                names='type',
                title="Income by Type",
                template=plotly_template,
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True, key="div_type_pie")

        with col2:
            # Year over year comparison if data exists
            history_df_yoy = history_df.copy()
            history_df_yoy['year'] = history_df_yoy['date'].dt.year
            yearly = history_df_yoy.groupby('year')['amount'].sum().reset_index()
            if len(yearly) > 1:
                fig = px.bar(
                    yearly,
                    x='year',
                    y='amount',
                    title="Year over Year",
                    template=plotly_template,
                    color_discrete_sequence=['#2ecc71']
                )
                fig.update_layout(xaxis_title="Year", yaxis_title="Total £")
                st.plotly_chart(fig, use_container_width=True, key="div_yoy")
            else:
                st.info("More data needed for YoY comparison")


def render_dividend_by_ticker(history_df, holdings_df, plotly_template):
    """Render breakdown of dividends by ticker with yield calculation"""
    st.subheader("📋 Dividends by Ticker")

    if history_df.empty:
        st.info("No dividend history available")
        return

    # Filter out interest
    div_only = history_df[history_df['type'] == 'dividend']

    if div_only.empty:
        st.info("No dividend payments recorded")
        return

    # Aggregate by ticker
    by_ticker = div_only.groupby('ticker').agg({
        'amount': ['sum', 'count', 'mean']
    }).reset_index()
    by_ticker.columns = ['Ticker', 'Total', 'Payments', 'Avg Payment']
    by_ticker = by_ticker.sort_values('Total', ascending=False)

    # Add yield calculation if holdings data available
    if holdings_df is not None and not holdings_df.empty and 'Value £' in holdings_df.columns:
        yields = []
        for ticker in by_ticker['Ticker']:
            holding = holdings_df[holdings_df['Ticker'] == ticker]
            if not holding.empty:
                value = holding['Value £'].iloc[0]
                annual_div = by_ticker[by_ticker['Ticker'] == ticker]['Total'].iloc[0]
                yield_pct = (annual_div / value * 100) if value > 0 else 0
                yields.append(round(yield_pct, 2))
            else:
                yields.append(0)
        by_ticker['Yield %'] = yields

    # Bar chart
    fig = px.bar(
        by_ticker,
        x='Ticker',
        y='Total',
        title="Total Dividends Received by Ticker",
        template=plotly_template,
        color='Total',
        color_continuous_scale='Greens'
    )
    fig.update_layout(height=350, xaxis_title="Ticker", yaxis_title="Total £")
    st.plotly_chart(fig, use_container_width=True, key="div_by_ticker_bar")

    # Table
    format_dict = {
        'Total': '£{:,.2f}',
        'Avg Payment': '£{:,.2f}',
    }
    if 'Yield %' in by_ticker.columns:
        format_dict['Yield %'] = '{:.2f}%'

    st.dataframe(
        by_ticker.style.format(format_dict),
        use_container_width=True,
        hide_index=True
    )


def render_upcoming_payments(predictions_df):
    """Display upcoming predicted dividend payments"""
    st.subheader("🔮 Upcoming Payments")

    if predictions_df.empty:
        st.info("No predictions available - need more dividend history")
        return

    # Sort by date and show next 10
    predictions_df = predictions_df.copy()
    predictions_df['expected_date'] = pd.to_datetime(predictions_df['expected_date'])
    upcoming = predictions_df.sort_values('expected_date').head(10)

    if upcoming.empty:
        st.info("No upcoming payments predicted")
        return

    # Summary metrics
    total_upcoming = upcoming['estimated_amount'].sum()
    unique_tickers = upcoming['ticker'].nunique()

    c1, c2, c3 = st.columns(3)
    c1.metric("📅 Next 10 Payments", f"£{total_upcoming:,.0f}")
    c2.metric("🏷️ Unique Tickers", unique_tickers)
    c3.metric("📆 First Expected", upcoming['expected_date'].iloc[0].strftime('%Y-%m-%d'))

    # Color by confidence
    def confidence_color(val):
        if val == 'high':
            return 'background-color: rgba(46, 204, 113, 0.3)'
        elif val == 'medium':
            return 'background-color: rgba(241, 196, 15, 0.3)'
        return 'background-color: rgba(231, 76, 60, 0.2)'

    display_df = upcoming[['ticker', 'expected_date', 'estimated_amount', 'confidence', 'frequency']].copy()
    display_df.columns = ['Ticker', 'Expected Date', 'Est. Amount', 'Confidence', 'Frequency']
    display_df['Expected Date'] = display_df['Expected Date'].dt.strftime('%Y-%m-%d')

    # Use map instead of deprecated applymap (pandas 2.1+)
    st.dataframe(
        display_df.style
            .map(confidence_color, subset=['Confidence'])
            .format({'Est. Amount': '£{:,.2f}'}),
        use_container_width=True,
        hide_index=True
    )


def render_dividend_tab(isa_df, sipp_df, isa_m, sipp_m, plotly_template, chart_colors):
    """Main function to render the complete Dividend Calendar tab"""
    st.header("📅 Dividend Calendar & Tracking")

    # Combine transaction data
    combined_trans = pd.DataFrame()
    if isa_df is not None and not isa_df.empty:
        combined_trans = pd.concat([combined_trans, isa_df])
    if sipp_df is not None and not sipp_df.empty:
        combined_trans = pd.concat([combined_trans, sipp_df])

    if combined_trans.empty:
        st.warning("No transaction data available")
        return

    # Get dividend history
    history_df = get_dividend_history(combined_trans)

    # Get combined holdings for predictions
    combined_holdings = pd.DataFrame()
    if isa_m is not None and 'df' in isa_m:
        combined_holdings = pd.concat([combined_holdings, isa_m['df']])
    if sipp_m is not None and 'df' in sipp_m:
        combined_holdings = pd.concat([combined_holdings, sipp_m['df']])

    # Get predictions
    predictions_df = predict_next_dividends(history_df, combined_holdings)

    # View toggle
    view_type = st.radio(
        "View Mode",
        ['Calendar', 'Timeline', 'Statistics'],
        horizontal=True,
        key="div_view_mode"
    )

    st.divider()

    if view_type == 'Calendar':
        render_dividend_calendar(history_df, predictions_df, plotly_template, chart_colors)
        st.divider()
        render_upcoming_payments(predictions_df)

    elif view_type == 'Timeline':
        render_dividend_timeline(history_df, predictions_df, plotly_template, chart_colors)
        st.divider()
        render_dividend_by_ticker(history_df, combined_holdings, plotly_template)

    else:  # Statistics
        render_dividend_stats(history_df, predictions_df, plotly_template)
        st.divider()
        render_dividend_by_ticker(history_df, combined_holdings, plotly_template)
        st.divider()
        render_upcoming_payments(predictions_df)
