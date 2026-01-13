# PyFinance - Development Documentation

## Project Overview

An agency-quality Streamlit-based portfolio dashboard for tracking Freetrade ISA and SIPP investments. Features premium UX, professional visualizations, AI-powered insights, and comprehensive analytics.

**Codebase Size:** ~14,000 lines across 14 premium modules

## Tech Stack

- **Frontend**: Streamlit, Plotly, Custom CSS
- **Data**: pandas, numpy, yfinance
- **AI**: OpenAI API (optional), rule-based insights
- **Config**: JSON-based configuration
- **Themes**: Dark/Light mode with CSS variables

## Architecture

### Core Files

| File                     | Purpose                                     | Lines  |
| ------------------------ | ------------------------------------------- | ------ |
| `freetrade_dashboard.py` | Main dashboard application                  | ~3,600 |
| `config.json`            | Ticker mappings, categories, targets, theme | -      |

### Premium Modules

| Module                  | Purpose        | Description                            |
| ----------------------- | -------------- | -------------------------------------- |
| `premium_navigation.py` | UI Navigation  | Premium header, tab nav, breadcrumbs   |
| `premium_metrics.py`    | Metric Cards   | Portfolio summary, metric displays     |
| `premium_charts.py`     | Visualizations | Enhanced Plotly charts with themes     |
| `premium_tables.py`     | Data Tables    | Sortable, filterable holdings tables   |
| `portfolio_insights.py` | AI Insights    | Smart recommendations, health scoring  |
| `dividend_calendar.py`  | Dividends      | Calendar/timeline/statistics views     |
| `goal_tracking.py`      | Goals          | Financial goal tracking dashboard      |
| `animations.py`         | Polish         | CSS animations, micro-interactions     |
| `loading_states.py`     | UX             | Skeleton loaders, loading states       |
| `refresh_system.py`     | Data           | Auto-refresh, freshness indicators     |
| `help_system.py`        | Onboarding     | Tooltips, guided tours, help           |
| `error_handling.py`     | Reliability    | Error boundaries, graceful degradation |
| `export_reports.py`     | Export         | PDF, CSV, Excel reports                |

## Development Patterns

### Key Patterns

- **Native Streamlit**: Use `st.metric` over custom HTML for reliability
- **CSS Injection**: Split CSS and HTML into separate `st.markdown` calls
- **Timezone Handling**: Use `dt.tz_localize(None)` before datetime comparisons
- **Theme Colors**: Access via `get_theme_colors()` function

### Module Integration

All premium modules are imported at the top of `freetrade_dashboard.py`:

```python
from premium_navigation import render_premium_header, render_tab_navigation
from premium_metrics import render_premium_metric, render_portfolio_summary_card
from portfolio_insights import generate_portfolio_insights, render_insights_panel
# ... etc
```

### Theme System

```python
THEMES = {
    'dark': {
        'background': '#0f172a',
        'surface': '#1e293b',
        'text': '#f8fafc',
        'accent': '#d4af37',  # Gold
        # ...
    },
    'light': {
        'background': '#f8fafc',
        'surface': '#ffffff',
        'text': '#1e293b',
        # ...
    }
}
```

## Common Tasks

### Add New Holding

1. Export latest CSV from Freetrade app
2. Place in project folder
3. Refresh dashboard - unmapped tickers appear in sidebar
4. Map Yahoo ticker and assign category

### Set Financial Goals

1. Go to Overview tab
2. Scroll to Goals section
3. Click "Add Goal"
4. Set type, target, deadline

### Switch Theme

- Click sun/moon icon in top-right corner
- Preference saved to config.json

## Troubleshooting

| Issue                | Solution                                        |
| -------------------- | ----------------------------------------------- |
| No data showing      | Check CSV files exist with correct naming       |
| Ticker not found     | Map Freetrade ticker to Yahoo ticker in sidebar |
| Wrong prices         | LSE stocks may need .L suffix (e.g., VUSA.L)    |
| Yield showing 0      | Add proxy yield in sidebar settings             |
| Raw HTML showing     | Restart dashboard - CSS injection issue         |
| Timezone errors      | Update pandas/numpy versions                    |
| Charts not rendering | Check Plotly installation                       |

## Adding New Features

1. Create new module in project root
2. Import functions in `freetrade_dashboard.py`
3. Integrate into appropriate tab
4. Use theme colors via `get_theme_colors()`
