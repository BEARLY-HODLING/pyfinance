# PyFinance - Premium Portfolio Dashboard

A professional-grade Streamlit dashboard for tracking UK investment portfolios with Freetrade ISA and SIPP accounts. Features premium UX, real-time market data, AI-powered insights, and comprehensive analytics.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

### Portfolio Management

- **Multi-Account Support**: Track ISA and SIPP accounts simultaneously
- **Real-Time Prices**: Live market data via Yahoo Finance API
- **Holdings Analysis**: P/L, yield, volatility, beta metrics
- **Allocation Tracking**: By holding and category (income/growth/speculative)

### Smart Analytics

- **AI-Powered Insights**: Automated portfolio recommendations
- **Risk Metrics**: Sharpe ratio, Sortino ratio, max drawdown
- **Benchmark Comparison**: Compare against S&P 500, FTSE 100, FTSE All-Share
- **Rebalancing Tools**: Target allocation with buy/sell recommendations

### Premium UI/UX

- **Dark/Light Themes**: Toggle with persistent preferences
- **Responsive Design**: Works on desktop and tablet
- **Animated Interactions**: Smooth transitions and hover effects
- **Loading States**: Skeleton loaders during data fetch

### Additional Features

- **Dividend Calendar**: Calendar, timeline, and statistics views
- **Transaction History**: Searchable and filterable
- **Goal Tracking**: Set and monitor financial targets
- **Export Reports**: PDF, CSV, and Excel formats
- **AI Assistant**: Natural language portfolio queries (requires OpenAI API)

## Screenshots

_Dashboard screenshots would go here_

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/pyfinance.git
   cd pyfinance
   ```

2. **Create virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up configuration**

   ```bash
   cp config.example.json config.json
   # Edit config.json with your ticker mappings
   ```

5. **Add your data**
   - Export transactions from Freetrade app
   - Save as `freetrade_ISA_*.csv` and/or `freetrade_SIPP_*.csv`
   - Or use sample files to test: rename `sample_ISA.csv` to `freetrade_ISA_sample.csv`

6. **Run the dashboard**

   ```bash
   streamlit run freetrade_dashboard.py
   ```

7. **Open browser**
   Navigate to http://localhost:8501

## Project Structure

```
pyfinance/
├── freetrade_dashboard.py    # Main application (~3,600 lines)
├── config.example.json       # Configuration template
├── requirements.txt          # Python dependencies
├── sample_ISA.csv           # Sample ISA transaction data
├── sample_SIPP.csv          # Sample SIPP transaction data
│
├── # Premium Modules
├── premium_navigation.py     # Header, tabs, breadcrumbs
├── premium_metrics.py        # Metric cards, portfolio summary
├── premium_charts.py         # Enhanced Plotly visualizations
├── premium_tables.py         # Data tables with styling
│
├── # Feature Modules
├── portfolio_insights.py     # AI-powered recommendations
├── dividend_calendar.py      # Calendar/timeline views
├── goal_tracking.py          # Financial goal tracking
├── export_reports.py         # PDF/CSV/Excel export
│
├── # System Modules
├── animations.py             # CSS animations
├── loading_states.py         # Skeleton loaders
├── refresh_system.py         # Auto-refresh system
├── help_system.py            # Tooltips, onboarding
├── error_handling.py         # Error boundaries
│
└── assets/
    └── css/                  # CSS stylesheets
```

## Configuration

### config.json

```json
{
  "yahoo_ticker_map": {
    "VUSA": "VUSA.L",
    "AAPL": "AAPL"
  },
  "category_map": {
    "VUSA": "growth",
    "AAPL": "growth"
  },
  "proxy_yields": {
    "FEPG.L": 0.23
  },
  "category_targets": {
    "income": 50.0,
    "growth": 35.0,
    "speculative": 15.0
  },
  "theme": "dark",
  "openai_api_key": ""
}
```

### Configuration Options

| Key                  | Description                                     | Example            |
| -------------------- | ----------------------------------------------- | ------------------ |
| `yahoo_ticker_map`   | Map Freetrade tickers to Yahoo Finance symbols  | `"VUSA": "VUSA.L"` |
| `category_map`       | Assign holdings to categories                   | `"VUSA": "growth"` |
| `proxy_yields`       | Override yield for accumulating ETFs            | `"FEPG.L": 0.23`   |
| `category_targets`   | Target allocation percentages (must sum to 100) | `"income": 50.0`   |
| `target_allocations` | Per-holding target percentages                  | `"VUSA": 15.0`     |
| `theme`              | Default theme (`dark` or `light`)               | `"dark"`           |
| `openai_api_key`     | OpenAI API key for AI assistant                 | `"sk-..."`         |

### Categories

- **income**: Dividend-focused holdings (ETFs like FEPG, JEQP, QYLP)
- **growth**: Capital appreciation focused (VUSA, VWRL, VUAG)
- **speculative**: Higher risk/reward positions

## Data Format

### Freetrade CSV Export

The dashboard expects CSV exports from the Freetrade app with these columns:

| Column           | Description                  |
| ---------------- | ---------------------------- |
| Type             | ORDER, DIVIDEND, or INTEREST |
| Timestamp        | ISO 8601 format              |
| Account Currency | GBP                          |
| Total Amount     | Transaction value            |
| Buy / Sell       | Buy or Sell                  |
| Ticker           | Freetrade ticker symbol      |
| Name             | Security name                |
| Order Status     | EXECUTED                     |
| No. of Shares    | Share quantity               |
| FX Rate          | Exchange rate used           |
| Base Currency    | Currency of the security     |
| Price per Share  | Execution price              |
| Total Shares     | Cumulative position          |

## Tabs Overview

| Tab              | Description                                        |
| ---------------- | -------------------------------------------------- |
| **Overview**     | Combined portfolio, insights, income charts, goals |
| **ISA**          | ISA account holdings, charts, benchmarks           |
| **SIPP**         | SIPP account holdings, charts, benchmarks          |
| **Transactions** | Full transaction history with filters              |
| **Dividends**    | Calendar, timeline, and dividend statistics        |
| **Rebalancing**  | Target allocation and rebalancing suggestions      |
| **Watchlist**    | Track stocks you're interested in                  |
| **Assistant**    | AI-powered portfolio Q&A                           |
| **Export**       | Generate PDF, CSV, Excel reports                   |

## Smart Portfolio Insights

The dashboard automatically analyzes your portfolio and provides insights:

- **Concentration Risk**: Alerts when holdings exceed 15% of portfolio
- **Sector Balance**: Identifies over/under exposure
- **Yield Opportunities**: Suggests income alternatives
- **Volatility Alerts**: Flags high-volatility positions
- **Rebalancing Needs**: Shows drift from targets
- **Dividend Tracking**: Recent payments and projections

## Environment Variables

For enhanced security, set these environment variables instead of using config.json:

```bash
export OPENAI_API_KEY="sk-your-api-key"
```

## Dependencies

Core dependencies (see `requirements.txt`):

```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.30
plotly>=5.17.0
```

Optional dependencies:

```
openai>=1.0.0        # AI Assistant
openpyxl>=3.1.0      # Excel export
reportlab>=4.0.0     # PDF export
weasyprint>=60.0     # Advanced PDF styling
```

## Development

### Running in Development

```bash
streamlit run freetrade_dashboard.py --server.runOnSave true
```

### Code Structure

The codebase follows these patterns:

- **Native Streamlit**: Uses `st.metric` for reliable rendering
- **CSS Injection**: Separate CSS and HTML markdown calls
- **Theme System**: Centralized color management via `THEMES` dict
- **Error Handling**: Graceful degradation with user-friendly messages

### Adding New Features

1. Create a new module in project root
2. Import functions in `freetrade_dashboard.py`
3. Integrate into appropriate tab
4. Use `get_theme_colors()` for styling

## Troubleshooting

| Issue            | Solution                                         |
| ---------------- | ------------------------------------------------ |
| No data showing  | Ensure CSV files match `freetrade_*.csv` pattern |
| Ticker not found | Add mapping in sidebar or config.json            |
| Wrong prices     | LSE stocks need `.L` suffix (e.g., `VUSA.L`)     |
| Yield showing 0  | Add proxy yield for accumulating ETFs            |
| Charts blank     | Check Plotly installation: `pip install plotly`  |
| API rate limits  | Wait a few minutes, yfinance has rate limits     |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) - The fastest way to build data apps
- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance market data
- [Plotly](https://plotly.com/) - Interactive graphing library
- [Freetrade](https://freetrade.io/) - Commission-free investing platform

## Disclaimer

This software is for informational purposes only. It is not financial advice. Always do your own research and consult a qualified financial advisor before making investment decisions. The authors are not responsible for any financial losses incurred from using this software.

---

**Made with Python and Streamlit**
