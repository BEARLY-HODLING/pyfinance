"""
Premium Chart Components for Freetrade Dashboard

This module provides enhanced, visually consistent chart components using Plotly.
All charts share a common color palette and configuration for a cohesive look.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Import from centralized color management
from theme_manager import ColorPalette

# ============================================================================
# CHART COLOR PALETTE
# ============================================================================

# Build CHART_COLORS using centralized ColorPalette
CHART_COLORS = {
    # Primary colors (from ColorPalette.SEMANTIC)
    'primary': ColorPalette.get('primary'),
    'secondary': ColorPalette.BASE.get('purple', '#9b59b6'),
    'success': ColorPalette.get('success'),
    'danger': ColorPalette.get('danger'),
    'warning': ColorPalette.get('warning'),
    'info': ColorPalette.get('info'),

    # Category colors (from ColorPalette.CATEGORIES)
    'income': ColorPalette.CATEGORIES['income'],
    'growth': ColorPalette.CATEGORIES['growth'],
    'speculative': ColorPalette.CATEGORIES['speculative'],
    'unknown': ColorPalette.CATEGORIES['unknown'],

    # Chart-specific
    'portfolio_line': ColorPalette.CHART_COLORS[0],  # '#00CC96'
    'benchmark_line': ColorPalette.CHART_COLORS[1],  # '#EF553B'
    'trend_line': ColorPalette.get('warning'),
    'fill_positive': ColorPalette.hex_to_rgba(ColorPalette.get('success'), 0.2),
    'fill_negative': ColorPalette.hex_to_rgba(ColorPalette.get('danger'), 0.2),
    'grid': 'rgba(255, 255, 255, 0.1)',

    # Gradient fills
    'gradient_start': 'rgba(0, 204, 150, 0.4)',
    'gradient_end': 'rgba(0, 204, 150, 0.0)',
}

# ============================================================================
# CHART CONFIGURATION
# ============================================================================

def get_chart_config():
    """Returns consistent Plotly config for all charts"""
    return {
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d'],
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'portfolio_chart',
            'height': 600,
            'width': 1200,
            'scale': 2
        }
    }

def get_chart_layout_defaults(template='plotly_dark'):
    """Returns consistent layout defaults for all charts"""
    return {
        'template': template,
        'font': {'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'},
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50},
        'hovermode': 'x unified',
        'hoverlabel': {
            'bgcolor': 'rgba(30, 30, 30, 0.95)',
            'bordercolor': CHART_COLORS['primary'],
            'font': {'size': 12}
        },
        'xaxis': {
            'gridcolor': CHART_COLORS['grid'],
            'showgrid': True,
            'zeroline': False,
        },
        'yaxis': {
            'gridcolor': CHART_COLORS['grid'],
            'showgrid': True,
            'zeroline': False,
        }
    }

# ============================================================================
# PREMIUM CHART COMPONENTS
# ============================================================================

def render_portfolio_value_chart(history_df, show_annotations=True, key_suffix="", template='plotly_dark'):
    """
    Premium area chart with gradient fill for portfolio value over time.

    Args:
        history_df: DataFrame with DateTimeIndex and 'Total' column
        show_annotations: Whether to show milestone annotations (ATH, max drawdown)
        key_suffix: Unique key suffix for Streamlit
        template: Plotly template to use
    """
    if history_df.empty or 'Total' not in history_df.columns:
        st.info("No portfolio history data available")
        return

    fig = go.Figure()

    # Main area chart with gradient fill and smooth line
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df['Total'],
        mode='lines',
        name='Portfolio Value',
        line=dict(
            width=3,
            color=CHART_COLORS['portfolio_line'],
            shape='spline',
            smoothing=1.3
        ),
        fill='tozeroy',
        fillcolor=CHART_COLORS['gradient_start'],
        hovertemplate='<b>%{x|%d %b %Y}</b><br>Value: £%{y:,.0f}<extra></extra>'
    ))

    # Add milestone annotations
    if show_annotations and len(history_df) > 5:
        values = history_df['Total']

        # Find all-time high
        ath_idx = values.idxmax()
        ath_val = values.max()

        # Find significant drop (if any)
        rolling_max = values.expanding().max()
        drawdown = (rolling_max - values) / rolling_max * 100
        if drawdown.max() > 5:  # Only show if >5% drawdown
            max_dd_idx = drawdown.idxmax()
            max_dd_val = values[max_dd_idx]

            fig.add_annotation(
                x=max_dd_idx,
                y=max_dd_val,
                text=f"Max DD<br>-{drawdown.max():.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor=CHART_COLORS['danger'],
                arrowwidth=2,
                ax=40,
                ay=40,
                font=dict(color=CHART_COLORS['danger'], size=10),
                bgcolor='rgba(231, 76, 60, 0.2)',
                bordercolor=CHART_COLORS['danger'],
                borderwidth=1,
                borderpad=4
            )

        # Add ATH annotation
        fig.add_annotation(
            x=ath_idx,
            y=ath_val,
            text=f"ATH<br>£{ath_val:,.0f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=CHART_COLORS['success'],
            arrowwidth=2,
            ax=-40,
            ay=-40,
            font=dict(color=CHART_COLORS['success'], size=10),
            bgcolor='rgba(46, 204, 113, 0.2)',
            bordercolor=CHART_COLORS['success'],
            borderwidth=1,
            borderpad=4
        )

    # Layout with range selector
    layout = get_chart_layout_defaults(template)
    layout.update({
        'title': {
            'text': 'Portfolio Value Over Time',
            'font': {'size': 18},
            'x': 0.5,
            'xanchor': 'center'
        },
        'xaxis': {
            **layout['xaxis'],
            'title': '',
            'rangeselector': {
                'buttons': [
                    dict(count=7, label='1W', step='day', stepmode='backward'),
                    dict(count=1, label='1M', step='month', stepmode='backward'),
                    dict(count=3, label='3M', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(label='ALL', step='all')
                ],
                'bgcolor': 'rgba(50, 50, 50, 0.8)',
                'activecolor': CHART_COLORS['primary'],
                'bordercolor': CHART_COLORS['grid'],
                'font': {'size': 11}
            },
            'rangeslider': {'visible': False},
            'type': 'date',
            'showspikes': True,
            'spikecolor': CHART_COLORS['primary'],
            'spikethickness': 1,
            'spikedash': 'dot',
            'spikemode': 'across'
        },
        'yaxis': {
            **layout['yaxis'],
            'title': 'Value (£)',
            'tickformat': '£,.0f',
            'showspikes': True,
            'spikecolor': CHART_COLORS['primary'],
            'spikethickness': 1,
            'spikedash': 'dot'
        },
        'height': 450,
        'showlegend': False
    })

    fig.update_layout(**layout)

    st.plotly_chart(fig, use_container_width=True, config=get_chart_config(), key=f"portfolio_value_{key_suffix}")


def render_allocation_donut(df, title, value_col='Value £', name_col='Ticker', hole_content=None, key_suffix="", template='plotly_dark'):
    """
    Premium donut chart for allocation visualization.

    Args:
        df: DataFrame with values and names
        title: Chart title
        value_col: Column name for values
        name_col: Column name for labels
        hole_content: Dict with 'value' and 'label' for center content
        key_suffix: Unique key suffix for Streamlit
        template: Plotly template to use
    """
    if df.empty:
        st.info("No allocation data available")
        return

    # Color mapping based on category if available
    colors = []
    if 'Category' in df.columns:
        for cat in df['Category']:
            colors.append(CHART_COLORS.get(cat.lower() if isinstance(cat, str) else 'unknown', CHART_COLORS['info']))
    else:
        # Use a pleasant color palette
        color_palette = [
            CHART_COLORS['primary'], CHART_COLORS['success'], CHART_COLORS['secondary'],
            CHART_COLORS['warning'], CHART_COLORS['info'], CHART_COLORS['danger'],
            '#16a085', '#8e44ad', '#d35400', '#2c3e50'
        ]
        colors = [color_palette[i % len(color_palette)] for i in range(len(df))]

    fig = go.Figure(data=[go.Pie(
        labels=df[name_col],
        values=df[value_col],
        hole=0.55,
        marker=dict(
            colors=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=2)
        ),
        textposition='inside',
        textinfo='percent',
        textfont=dict(size=11),
        hovertemplate='<b>%{label}</b><br>Value: £%{value:,.0f}<br>Share: %{percent}<extra></extra>',
        pull=[0.02] * len(df),  # Slight pull-out effect
        rotation=90
    )])

    # Add center annotation
    if hole_content:
        fig.add_annotation(
            text=f"<b>{hole_content.get('label', 'Total')}</b><br><span style='font-size:20px'>£{hole_content.get('value', 0):,.0f}</span>",
            x=0.5, y=0.5,
            font=dict(size=14),
            showarrow=False
        )

    layout = get_chart_layout_defaults(template)
    layout.update({
        'title': {
            'text': title,
            'font': {'size': 16},
            'x': 0.5,
            'xanchor': 'center'
        },
        'legend': {
            'orientation': 'v',
            'yanchor': 'middle',
            'y': 0.5,
            'xanchor': 'left',
            'x': 1.02,
            'font': {'size': 11},
            'bgcolor': 'rgba(0,0,0,0)',
            'itemsizing': 'constant'
        },
        'height': 350,
        'margin': {'l': 20, 'r': 120, 't': 50, 'b': 20}
    })

    fig.update_layout(**layout)

    st.plotly_chart(fig, use_container_width=True, config=get_chart_config(), key=f"donut_{key_suffix}")


def render_income_chart(monthly_df, chart_type="bar", show_trend=True, goal_line=None, key_suffix="", template='plotly_dark'):
    """
    Premium income chart with trend line and goal tracking.

    Args:
        monthly_df: DataFrame with 'Month' and income columns (ISA, SIPP, Total, or Total Amount)
        chart_type: "bar" or "line"
        show_trend: Whether to show trend line
        goal_line: Optional monthly goal amount
        key_suffix: Unique key suffix for Streamlit
        template: Plotly template to use
    """
    if monthly_df.empty:
        st.info("No income data available yet")
        return

    fig = go.Figure()

    # Determine if stacked (ISA + SIPP) or single
    is_stacked = 'ISA' in monthly_df.columns and 'SIPP' in monthly_df.columns

    if is_stacked:
        if chart_type == "bar":
            fig.add_trace(go.Bar(
                x=monthly_df['Month'],
                y=monthly_df['ISA'],
                name='ISA',
                marker_color=CHART_COLORS['primary'],
                hovertemplate='<b>ISA</b><br>%{x}<br>£%{y:,.0f}<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                x=monthly_df['Month'],
                y=monthly_df['SIPP'],
                name='SIPP',
                marker_color=CHART_COLORS['secondary'],
                hovertemplate='<b>SIPP</b><br>%{x}<br>£%{y:,.0f}<extra></extra>'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=monthly_df['Month'],
                y=monthly_df['ISA'],
                name='ISA',
                mode='lines+markers',
                line=dict(width=2, color=CHART_COLORS['primary']),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)'
            ))
            fig.add_trace(go.Scatter(
                x=monthly_df['Month'],
                y=monthly_df['SIPP'],
                name='SIPP',
                mode='lines+markers',
                line=dict(width=2, color=CHART_COLORS['secondary']),
                marker=dict(size=6)
            ))

        value_col = 'Total'
    else:
        value_col = 'Total Amount' if 'Total Amount' in monthly_df.columns else 'Total'

        if chart_type == "bar":
            # Find best month for annotation
            best_idx = monthly_df[value_col].idxmax()
            colors = [CHART_COLORS['success'] if i == best_idx else CHART_COLORS['primary']
                     for i in range(len(monthly_df))]

            fig.add_trace(go.Bar(
                x=monthly_df['Month'],
                y=monthly_df[value_col],
                name='Income',
                marker_color=colors,
                marker_line=dict(color='rgba(0,0,0,0.3)', width=1),
                hovertemplate='<b>%{x}</b><br>Income: £%{y:,.0f}<extra></extra>'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=monthly_df['Month'],
                y=monthly_df[value_col],
                name='Income',
                mode='lines+markers',
                line=dict(width=3, color=CHART_COLORS['success'], shape='spline'),
                marker=dict(size=8, color=CHART_COLORS['success']),
                fill='tozeroy',
                fillcolor='rgba(46, 204, 113, 0.2)'
            ))

    # Add trend line
    if show_trend and len(monthly_df) > 1:
        x_numeric = np.arange(len(monthly_df))
        values = monthly_df[value_col] if not is_stacked else monthly_df['Total']
        z = np.polyfit(x_numeric, values, 1)
        trend = np.poly1d(z)(x_numeric)

        trend_direction = "Up" if z[0] > 0 else "Down"

        fig.add_trace(go.Scatter(
            x=monthly_df['Month'],
            y=trend,
            name=f'Trend ({trend_direction})',
            mode='lines',
            line=dict(dash='dash', color=CHART_COLORS['trend_line'], width=2),
            hoverinfo='skip'
        ))

    # Add goal line
    if goal_line:
        fig.add_hline(
            y=goal_line,
            line_dash="dot",
            line_color=CHART_COLORS['info'],
            line_width=2,
            annotation_text=f"Goal: £{goal_line:,.0f}",
            annotation_position="top right",
            annotation_font=dict(color=CHART_COLORS['info'])
        )

    # Add YTD running total as secondary axis
    if len(monthly_df) > 1:
        ytd_values = monthly_df[value_col].cumsum() if not is_stacked else monthly_df['Total'].cumsum()
        fig.add_trace(go.Scatter(
            x=monthly_df['Month'],
            y=ytd_values,
            name='YTD Total',
            mode='lines+markers',
            line=dict(width=2, color=CHART_COLORS['warning'], dash='dot'),
            marker=dict(size=5),
            yaxis='y2',
            hovertemplate='<b>YTD Total</b><br>%{x}<br>£%{y:,.0f}<extra></extra>'
        ))

    # Best month annotation
    if not is_stacked and len(monthly_df) > 0:
        best_month = monthly_df.loc[monthly_df[value_col].idxmax()]
        fig.add_annotation(
            x=best_month['Month'],
            y=best_month[value_col],
            text=f"Best<br>£{best_month[value_col]:,.0f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=CHART_COLORS['success'],
            arrowwidth=1.5,
            ax=0,
            ay=-40,
            font=dict(color=CHART_COLORS['success'], size=10),
            bgcolor='rgba(46, 204, 113, 0.2)',
            borderpad=3
        )

    layout = get_chart_layout_defaults(template)
    layout.update({
        'title': {
            'text': 'Monthly Dividend Income',
            'font': {'size': 16},
            'x': 0.5
        },
        'barmode': 'stack' if is_stacked and chart_type == "bar" else 'group',
        'xaxis': {
            **layout['xaxis'],
            'title': '',
            'tickangle': -45 if len(monthly_df) > 6 else 0
        },
        'yaxis': {
            **layout['yaxis'],
            'title': 'Monthly Income (£)',
            'tickformat': '£,.0f'
        },
        'yaxis2': {
            'title': 'YTD Total (£)',
            'overlaying': 'y',
            'side': 'right',
            'tickformat': '£,.0f',
            'showgrid': False,
            'zeroline': False
        },
        'height': 400,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'center',
            'x': 0.5,
            'bgcolor': 'rgba(0,0,0,0)'
        }
    })

    fig.update_layout(**layout)

    st.plotly_chart(fig, use_container_width=True, config=get_chart_config(), key=f"income_premium_{key_suffix}")


def render_benchmark_chart(portfolio_series, benchmark_series, benchmark_name, key_suffix="", template='plotly_dark'):
    """
    Dual line chart comparing portfolio vs benchmark, normalized to 100.

    Args:
        portfolio_series: Series with portfolio values (DateTimeIndex)
        benchmark_series: Series with benchmark values (DateTimeIndex)
        benchmark_name: Name of the benchmark for display
        key_suffix: Unique key suffix for Streamlit
        template: Plotly template to use
    """
    if portfolio_series.empty or benchmark_series.empty:
        st.info("Insufficient data for benchmark comparison")
        return

    # Normalize both to 100
    port_norm = (portfolio_series / portfolio_series.iloc[0]) * 100
    bench_norm = (benchmark_series / benchmark_series.iloc[0]) * 100

    fig = go.Figure()

    # Portfolio line
    fig.add_trace(go.Scatter(
        x=port_norm.index,
        y=port_norm.values,
        mode='lines',
        name='Your Portfolio',
        line=dict(width=3, color=CHART_COLORS['portfolio_line'], shape='spline'),
        hovertemplate='<b>Portfolio</b><br>%{x|%d %b %Y}<br>Value: %{y:.1f}<extra></extra>'
    ))

    # Benchmark line
    fig.add_trace(go.Scatter(
        x=bench_norm.index,
        y=bench_norm.values,
        mode='lines',
        name=benchmark_name,
        line=dict(width=2, color=CHART_COLORS['benchmark_line'], dash='dash'),
        hovertemplate=f'<b>{benchmark_name}</b><br>%{{x|%d %b %Y}}<br>Value: %{{y:.1f}}<extra></extra>'
    ))

    # Fill between lines to show outperformance/underperformance
    for i in range(len(port_norm) - 1):
        x_range = [port_norm.index[i], port_norm.index[i+1]]
        port_vals = [port_norm.iloc[i], port_norm.iloc[i+1]]
        bench_vals = [bench_norm.iloc[i], bench_norm.iloc[i+1]]

        # Determine if outperforming
        outperforming = (port_vals[0] + port_vals[1]) / 2 > (bench_vals[0] + bench_vals[1]) / 2
        fill_color = CHART_COLORS['fill_positive'] if outperforming else CHART_COLORS['fill_negative']

        fig.add_trace(go.Scatter(
            x=x_range + x_range[::-1],
            y=port_vals + bench_vals[::-1],
            fill='toself',
            fillcolor=fill_color,
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Calculate final alpha
    portfolio_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1) * 100
    benchmark_return = (benchmark_series.iloc[-1] / benchmark_series.iloc[0] - 1) * 100
    alpha = portfolio_return - benchmark_return

    # Add alpha annotation
    alpha_color = CHART_COLORS['success'] if alpha > 0 else CHART_COLORS['danger']
    fig.add_annotation(
        x=port_norm.index[-1],
        y=port_norm.iloc[-1],
        text=f"Alpha: {alpha:+.1f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor=alpha_color,
        ax=50,
        ay=0,
        font=dict(color=alpha_color, size=12),
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor=alpha_color,
        borderwidth=1,
        borderpad=6
    )

    # Add baseline
    fig.add_hline(y=100, line_dash="dot", line_color=CHART_COLORS['grid'], line_width=1)

    layout = get_chart_layout_defaults(template)
    layout.update({
        'title': {
            'text': f'Portfolio vs {benchmark_name} (Normalized)',
            'font': {'size': 16},
            'x': 0.5
        },
        'xaxis': {
            **layout['xaxis'],
            'title': ''
        },
        'yaxis': {
            **layout['yaxis'],
            'title': 'Value (Indexed to 100)',
            'tickformat': '.0f'
        },
        'height': 400,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'center',
            'x': 0.5,
            'bgcolor': 'rgba(0,0,0,0)'
        }
    })

    fig.update_layout(**layout)

    st.plotly_chart(fig, use_container_width=True, config=get_chart_config(), key=f"benchmark_premium_{key_suffix}")


def render_treemap(df, value_col, category_col, label_col, key_suffix="", template='plotly_dark'):
    """
    Premium treemap for hierarchical allocation visualization.

    Args:
        df: DataFrame with values, categories, and labels
        value_col: Column name for values
        category_col: Column name for categories
        label_col: Column name for labels
        key_suffix: Unique key suffix for Streamlit
        template: Plotly template to use
    """
    if df.empty:
        st.info("No data available for treemap")
        return

    # Prepare data for treemap
    df_treemap = df.copy()
    df_treemap['Category_Display'] = df_treemap[category_col].str.title()

    # Create color mapping
    color_map = {
        'Income': CHART_COLORS['income'],
        'Growth': CHART_COLORS['growth'],
        'Speculative': CHART_COLORS['speculative'],
        'Unknown': CHART_COLORS['unknown']
    }

    fig = px.treemap(
        df_treemap,
        path=['Category_Display', label_col],
        values=value_col,
        color='Category_Display',
        color_discrete_map=color_map,
        hover_data={value_col: ':,.0f'},
        custom_data=[value_col]
    )

    fig.update_traces(
        textinfo='label+percent root',
        textfont=dict(size=12),
        hovertemplate='<b>%{label}</b><br>Value: £%{customdata[0]:,.0f}<br>Share: %{percentRoot:.1%}<extra></extra>',
        marker=dict(
            cornerradius=5,
            line=dict(width=2, color='rgba(0,0,0,0.3)')
        )
    )

    layout = get_chart_layout_defaults(template)
    layout.update({
        'title': {
            'text': 'Portfolio Allocation Treemap',
            'font': {'size': 16},
            'x': 0.5
        },
        'height': 450,
        'margin': {'l': 10, 'r': 10, 't': 50, 'b': 10}
    })

    fig.update_layout(**layout)

    st.plotly_chart(fig, use_container_width=True, config=get_chart_config(), key=f"treemap_{key_suffix}")


def render_comparison_bar_chart(df, current_col, target_col, label_col, title="Current vs Target", key_suffix="", template='plotly_dark'):
    """
    Grouped bar chart for comparing current vs target allocations.

    Args:
        df: DataFrame with current and target values
        current_col: Column name for current values
        target_col: Column name for target values
        label_col: Column name for labels
        title: Chart title
        key_suffix: Unique key suffix for Streamlit
        template: Plotly template to use
    """
    if df.empty:
        return

    fig = go.Figure()

    # Current bars
    fig.add_trace(go.Bar(
        x=df[label_col],
        y=df[current_col],
        name='Current',
        marker_color=CHART_COLORS['primary'],
        marker_line=dict(color='rgba(0,0,0,0.3)', width=1),
        text=df[current_col].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Current: %{y:.1f}%<extra></extra>'
    ))

    # Target bars
    fig.add_trace(go.Bar(
        x=df[label_col],
        y=df[target_col],
        name='Target',
        marker_color=CHART_COLORS['success'],
        marker_line=dict(color='rgba(0,0,0,0.3)', width=1),
        text=df[target_col].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Target: %{y:.1f}%<extra></extra>'
    ))

    layout = get_chart_layout_defaults(template)
    layout.update({
        'title': {
            'text': title,
            'font': {'size': 16},
            'x': 0.5
        },
        'barmode': 'group',
        'xaxis': {
            **layout['xaxis'],
            'title': ''
        },
        'yaxis': {
            **layout['yaxis'],
            'title': 'Allocation %',
            'tickformat': '.0f',
            'range': [0, max(df[current_col].max(), df[target_col].max()) * 1.2]
        },
        'height': 300,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'center',
            'x': 0.5
        }
    })

    fig.update_layout(**layout)

    st.plotly_chart(fig, use_container_width=True, config=get_chart_config(), key=f"comparison_{key_suffix}")


def render_drift_gauge(drift_pct, label, key_suffix=""):
    """
    Gauge chart showing allocation drift from target.

    Args:
        drift_pct: Drift percentage (-100 to 100)
        label: Category or ticker label
        key_suffix: Unique key suffix for Streamlit
    """
    # Determine color based on drift
    if abs(drift_pct) < 2:
        color = CHART_COLORS['success']
        status = "On Target"
    elif abs(drift_pct) < 5:
        color = CHART_COLORS['warning']
        status = "Minor Drift"
    else:
        color = CHART_COLORS['danger']
        status = "Significant Drift"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=drift_pct,
        title={'text': f"{label}<br><span style='font-size:12px;color:{color}'>{status}</span>"},
        delta={'reference': 0, 'increasing': {'color': CHART_COLORS['success']}, 'decreasing': {'color': CHART_COLORS['danger']}},
        gauge={
            'axis': {'range': [-15, 15], 'tickwidth': 1, 'tickcolor': CHART_COLORS['grid']},
            'bar': {'color': color},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 2,
            'bordercolor': CHART_COLORS['grid'],
            'steps': [
                {'range': [-15, -5], 'color': 'rgba(231, 76, 60, 0.3)'},
                {'range': [-5, -2], 'color': 'rgba(243, 156, 18, 0.3)'},
                {'range': [-2, 2], 'color': 'rgba(46, 204, 113, 0.3)'},
                {'range': [2, 5], 'color': 'rgba(243, 156, 18, 0.3)'},
                {'range': [5, 15], 'color': 'rgba(231, 76, 60, 0.3)'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 4},
                'thickness': 0.75,
                'value': drift_pct
            }
        },
        number={'suffix': '%', 'font': {'size': 24}}
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin={'l': 30, 'r': 30, 't': 50, 'b': 10}
    )

    st.plotly_chart(fig, use_container_width=True, config=get_chart_config(), key=f"gauge_{key_suffix}")
