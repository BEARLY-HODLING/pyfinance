"""
Help & Onboarding System for Freetrade Portfolio Dashboard

This module provides:
- Welcome modal for first-time users
- Feature tour with step-by-step walkthrough
- Tooltips and popovers for contextual help
- FAQ and keyboard shortcuts
- Sidebar help section

Usage:
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
        HELP_CONTENT
    )
"""

import streamlit as st
import json
import os


# ============================================================================
# HELP CONTENT DICTIONARY
# ============================================================================

HELP_CONTENT = {
    'overview_intro': """
**Welcome to your Portfolio Dashboard!**

This is your command center for tracking your Freetrade investments. Here you can:
- See your combined portfolio value across all accounts
- Track your dividend income month by month
- Compare your performance against market benchmarks
- Get AI-powered insights about your portfolio
""",
    'holdings_help': """
**Understanding Your Holdings Table**

Each row shows one of your investments with:
- **Value**: Current market value in GBP
- **Weight %**: How much of your portfolio this holding represents
- **P/L %**: Your profit or loss since purchase
- **Trail Yield %**: Annual dividend yield (based on trailing 12 months)
- **Vol %**: Volatility - how much the price moves
- **Beta**: Market sensitivity (>1 = more volatile than market)

**Yield Icons:**
- Robot icon: Auto-calculated from dividend history
- Warning icon: Limited data (less than 6 months)
- Pin icon: Using your manually set proxy yield
""",
    'rebalancing_help': """
**Rebalancing Your Portfolio**

**Level 1 - Category Allocation:**
Set target percentages for Income, Growth, and Speculative categories. The dashboard shows how far you've drifted from targets.

**Level 2 - Per-Holding Targets:**
Set specific percentage targets for individual holdings. Get precise buy/sell recommendations with share counts.

**Tips:**
- Category targets must sum to 100%
- Green = need to buy more, Red = consider selling
- Rebalance when drift exceeds 5%
""",
    'ai_assistant_help': """
**AI Portfolio Assistant**

Ask natural language questions about your portfolio:
- "What's my best performing stock?"
- "How much dividend income will I get this year?"
- "Am I overweight in any category?"
- "What should I buy next for income?"

**Requirements:**
- OpenAI API key (set in sidebar or as environment variable)
- Uses GPT-4o-mini for cost-effective responses

**Privacy:** Your portfolio data is sent to OpenAI for analysis. No data is stored permanently.
""",
    'dividend_tracking_help': """
**Dividend Tracking**

The monthly income chart shows your dividend and interest payments over time.

**Metrics:**
- **Total YTD**: All dividend income this year
- **Avg Monthly**: Your average monthly income
- **Growth**: Comparing recent 3 months vs previous 3 months
- **Projected Annual**: Based on current trend

**Tips:**
- Export dividends from Freetrade regularly for accurate tracking
- Proxy yields help estimate future income for newer holdings
""",
    'faq_items': [
        {
            'q': 'How do I add a new holding?',
            'a': 'Export your latest transactions from Freetrade and place the CSV file in the PyFinance folder. Unmapped tickers will appear in the sidebar for configuration.'
        },
        {
            'q': 'Why is my yield showing 0%?',
            'a': 'The holding may be new or not paying dividends. You can set a "proxy yield" in the sidebar to estimate future income.'
        },
        {
            'q': 'What does the .L suffix mean?',
            'a': 'The .L suffix indicates London Stock Exchange listings. UK stocks need this suffix for Yahoo Finance to find the correct prices.'
        },
        {
            'q': 'How often does data refresh?',
            'a': 'Prices are cached for 15 minutes. Click the browser refresh button or press R to fetch fresh data.'
        },
        {
            'q': 'Can I track multiple accounts?',
            'a': 'Yes! Place both freetrade_ISA_*.csv and freetrade_SIPP_*.csv files in the folder. The Overview tab shows combined totals.'
        },
        {
            'q': 'What is Beta?',
            'a': 'Beta measures how much a stock moves relative to the market. Beta > 1 means more volatile than market, < 1 means less volatile.'
        },
        {
            'q': 'How is dividend yield calculated?',
            'a': 'Trailing 12-month dividends divided by current price. For LSE stocks, we automatically convert pence to pounds.'
        }
    ],
    'keyboard_shortcuts': [
        {'key': 'R', 'action': 'Refresh data'},
        {'key': '1', 'action': 'Go to Overview tab'},
        {'key': '2', 'action': 'Go to ISA tab'},
        {'key': '3', 'action': 'Go to SIPP tab'},
        {'key': '4', 'action': 'Go to Rebalancing tab'},
        {'key': '5', 'action': 'Go to Assistant tab'},
        {'key': '?', 'action': 'Show keyboard shortcuts'},
        {'key': 'T', 'action': 'Toggle dark/light theme'}
    ],
    'tour_steps': [
        {
            'title': 'Portfolio Overview',
            'content': 'See your total portfolio value, profit/loss, and projected income at a glance.',
            'tab': 'Overview'
        },
        {
            'title': 'Holdings Table',
            'content': 'View all your investments with real-time prices, yields, and performance metrics.',
            'tab': 'ISA/SIPP'
        },
        {
            'title': 'Allocation Charts',
            'content': 'Visualize how your money is spread across holdings and categories.',
            'tab': 'ISA/SIPP'
        },
        {
            'title': 'Dividend Tracking',
            'content': 'Track your monthly income and see growth trends over time.',
            'tab': 'Overview'
        },
        {
            'title': 'Benchmark Comparison',
            'content': 'Compare your portfolio performance against S&P 500, FTSE 100, or FTSE All-Share.',
            'tab': 'ISA/SIPP'
        },
        {
            'title': 'Rebalancing Tools',
            'content': 'Set target allocations and get specific buy/sell recommendations.',
            'tab': 'Rebalancing'
        },
        {
            'title': 'AI Assistant',
            'content': 'Ask questions about your portfolio in plain English.',
            'tab': 'Assistant'
        }
    ],
    'contextual_tips': {
        'overview': [
            'The Overview tab shows your combined ISA and SIPP performance.',
            'Monthly income trends help predict future dividend payments.',
            'Check if your portfolio is beating market benchmarks.'
        ],
        'isa': [
            'ISA contributions are tax-free up to your annual allowance.',
            'Use the search box to quickly find specific holdings.',
            'Sort by any column by clicking the column header.'
        ],
        'sipp': [
            'SIPP funds are locked until retirement age.',
            'Consider tax implications when planning withdrawals.',
            'Higher growth assets may be better suited for long-term SIPP.'
        ],
        'rebalancing': [
            'Category targets help maintain your risk profile.',
            'Rebalance when any category drifts more than 5% from target.',
            'Consider tax implications before selling in taxable accounts.'
        ],
        'assistant': [
            'Try asking about your highest yielding stocks.',
            'Ask for rebalancing suggestions based on your goals.',
            'The AI can compare performance across your accounts.'
        ]
    }
}


# ============================================================================
# WELCOME MODAL
# ============================================================================

def render_welcome_modal():
    """Render first-time user welcome modal with feature overview and quick start steps.

    Shows a welcome screen for new users with:
    - Brief feature overview
    - Quick start steps
    - "Don't show again" checkbox
    - Stores preference in session_state and config.json
    """
    # Check if user has dismissed the welcome
    if st.session_state.get('hide_welcome', False):
        return

    # Check config for saved preference
    config_file = 'config.json'
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                cfg = json.load(f)
                if cfg.get('hide_welcome', False):
                    st.session_state.hide_welcome = True
                    return
        except:
            pass

    # Show welcome modal
    with st.container():
        st.markdown("""
        <style>
        .welcome-modal {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 2px solid #d4af37;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .welcome-title {
            font-size: 2rem;
            color: #d4af37;
            margin-bottom: 1rem;
        }
        .quick-start-step {
            background: rgba(212, 175, 55, 0.1);
            border-left: 3px solid #d4af37;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
        }
        </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("## Welcome to Your Portfolio Dashboard!")
        with col2:
            if st.button("X Close", key="close_welcome"):
                st.session_state.hide_welcome = True
                st.rerun()

        st.markdown("""
        Track your Freetrade ISA and SIPP investments with real-time prices,
        dividend tracking, and intelligent portfolio analytics.
        """)

        st.markdown("### Features")

        feat_cols = st.columns(3)
        with feat_cols[0]:
            st.markdown("""
            **Portfolio Tracking**
            - Real-time prices
            - Multi-account support
            - P/L calculations
            """)
        with feat_cols[1]:
            st.markdown("""
            **Income Analysis**
            - Dividend tracking
            - Monthly income charts
            - Yield calculations
            """)
        with feat_cols[2]:
            st.markdown("""
            **Smart Tools**
            - Rebalancing advisor
            - Benchmark comparison
            - AI assistant
            """)

        st.markdown("### Quick Start")

        st.markdown("""
        <div class="quick-start-step">
        <strong>Step 1:</strong> Export your transactions from Freetrade app
        </div>
        <div class="quick-start-step">
        <strong>Step 2:</strong> Place CSV files in the PyFinance folder (freetrade_ISA_*.csv, freetrade_SIPP_*.csv)
        </div>
        <div class="quick-start-step">
        <strong>Step 3:</strong> Map any unmapped tickers in the sidebar
        </div>
        <div class="quick-start-step">
        <strong>Step 4:</strong> Explore your portfolio across the tabs!
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            if st.button("Take a Tour", key="start_tour", use_container_width=True):
                st.session_state.show_tour = True
                st.session_state.tour_step = 0
                st.session_state.hide_welcome = True
                st.rerun()
        with col2:
            if st.button("Show Keyboard Shortcuts", key="show_shortcuts_welcome", use_container_width=True):
                st.session_state.show_shortcuts = True
        with col3:
            dont_show = st.checkbox("Don't show again", key="dont_show_welcome")
            if dont_show:
                st.session_state.hide_welcome = True
                # Save to config
                try:
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            cfg = json.load(f)
                    else:
                        cfg = {}
                    cfg['hide_welcome'] = True
                    with open(config_file, 'w') as f:
                        json.dump(cfg, f, indent=4, sort_keys=True)
                except:
                    pass

        st.divider()


# ============================================================================
# FEATURE TOUR
# ============================================================================

def render_feature_tour():
    """Render guided tour of main features with step-by-step walkthrough.

    Shows a step-by-step tour with:
    - Progress dots
    - Next/Back/Skip buttons
    - Highlights current feature area
    """
    if not st.session_state.get('show_tour', False):
        return

    tour_steps = HELP_CONTENT['tour_steps']
    current_step = st.session_state.get('tour_step', 0)
    total_steps = len(tour_steps)

    if current_step >= total_steps:
        st.session_state.show_tour = False
        return

    step = tour_steps[current_step]

    # Tour overlay
    st.markdown(f"""
    <style>
    .tour-overlay {{
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.98) 0%, rgba(15, 23, 42, 0.98) 100%);
        border: 2px solid #d4af37;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        position: relative;
    }}
    .tour-progress {{
        display: flex;
        justify-content: center;
        gap: 8px;
        margin-top: 1rem;
    }}
    .tour-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #64748b;
    }}
    .tour-dot.active {{
        background: #d4af37;
    }}
    .tour-step-indicator {{
        color: #d4af37;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }}
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown(f"<div class='tour-step-indicator'>Step {current_step + 1} of {total_steps}</div>",
                    unsafe_allow_html=True)

        st.markdown(f"### {step['title']}")
        st.markdown(step['content'])
        st.caption(f"Find this in: **{step['tab']}** tab")

        # Progress dots
        dots_html = '<div class="tour-progress">'
        for i in range(total_steps):
            active = 'active' if i == current_step else ''
            dots_html += f'<div class="tour-dot {active}"></div>'
        dots_html += '</div>'
        st.markdown(dots_html, unsafe_allow_html=True)

        # Navigation buttons
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            if current_step > 0:
                if st.button("Back", key="tour_back", use_container_width=True):
                    st.session_state.tour_step = current_step - 1
                    st.rerun()

        with col2:
            if st.button("Skip Tour", key="tour_skip", use_container_width=True):
                st.session_state.show_tour = False
                st.rerun()

        with col3:
            pass  # Spacer

        with col4:
            if current_step < total_steps - 1:
                if st.button("Next", key="tour_next", type="primary", use_container_width=True):
                    st.session_state.tour_step = current_step + 1
                    st.rerun()
            else:
                if st.button("Finish", key="tour_finish", type="primary", use_container_width=True):
                    st.session_state.show_tour = False
                    st.rerun()

        st.divider()


# ============================================================================
# TOOLTIP COMPONENT
# ============================================================================

def render_help_tooltip(text, tooltip_text, position='top'):
    """Render a reusable tooltip component that shows on hover.

    Args:
        text: The text/element to attach the tooltip to
        tooltip_text: The tooltip content to show on hover
        position: Tooltip position ('top', 'bottom', 'left', 'right')
    """
    tooltip_id = f"tooltip_{hash(text + tooltip_text) % 10000}"

    position_styles = {
        'top': 'bottom: 125%; left: 50%; transform: translateX(-50%);',
        'bottom': 'top: 125%; left: 50%; transform: translateX(-50%);',
        'left': 'right: 125%; top: 50%; transform: translateY(-50%);',
        'right': 'left: 125%; top: 50%; transform: translateY(-50%);'
    }

    st.markdown(f"""
    <style>
    .tooltip-container-{tooltip_id} {{
        position: relative;
        display: inline-block;
        cursor: help;
    }}
    .tooltip-container-{tooltip_id} .tooltip-text {{
        visibility: hidden;
        background-color: #1e293b;
        color: #f8fafc;
        text-align: left;
        border-radius: 8px;
        padding: 10px 14px;
        position: absolute;
        z-index: 1000;
        {position_styles.get(position, position_styles['top'])}
        width: 250px;
        font-size: 0.85rem;
        line-height: 1.4;
        border: 1px solid #d4af37;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        opacity: 0;
        transition: opacity 0.2s, visibility 0.2s;
    }}
    .tooltip-container-{tooltip_id}:hover .tooltip-text {{
        visibility: visible;
        opacity: 1;
    }}
    </style>
    <span class="tooltip-container-{tooltip_id}">
        {text}
        <span class="tooltip-text">{tooltip_text}</span>
    </span>
    """, unsafe_allow_html=True)


# ============================================================================
# INFO POPOVER
# ============================================================================

def render_info_popover(title, content):
    """Render a click-to-show detailed info popover with question mark icon trigger.

    Args:
        title: The popover title
        content: The detailed content to show (supports markdown)
    """
    with st.popover("(?)"):
        st.markdown(f"**{title}**")
        st.markdown(content)


# ============================================================================
# SIDEBAR HELP SECTION
# ============================================================================

def render_help_sidebar_section():
    """Render quick help section in sidebar with FAQ, links, and shortcuts.

    Includes:
    - FAQ accordion
    - Contact/feedback link (placeholder)
    - Documentation link (placeholder)
    - Keyboard shortcuts list
    """
    with st.sidebar:
        st.divider()
        st.subheader("Need Help?")

        # Quick help accordion
        with st.expander("Frequently Asked Questions"):
            for item in HELP_CONTENT['faq_items']:
                st.markdown(f"**Q: {item['q']}**")
                st.markdown(f"A: {item['a']}")
                st.markdown("---")

        # Keyboard shortcuts
        with st.expander("Keyboard Shortcuts"):
            for shortcut in HELP_CONTENT['keyboard_shortcuts']:
                st.markdown(f"**{shortcut['key']}** - {shortcut['action']}")

        # Links section
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Show Tour", key="sidebar_tour", use_container_width=True):
                st.session_state.show_tour = True
                st.session_state.tour_step = 0
                st.rerun()
        with col2:
            if st.button("Reset Welcome", key="reset_welcome", use_container_width=True):
                st.session_state.hide_welcome = False
                # Remove from config
                try:
                    config_file = 'config.json'
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            cfg = json.load(f)
                        cfg['hide_welcome'] = False
                        with open(config_file, 'w') as f:
                            json.dump(cfg, f, indent=4, sort_keys=True)
                except:
                    pass
                st.rerun()

        # Documentation link placeholder
        st.caption("Documentation: [GitHub Wiki](#) (coming soon)")
        st.caption("Feedback: [Open Issue](#) (coming soon)")


# ============================================================================
# CONTEXTUAL HELP
# ============================================================================

def render_contextual_help(context):
    """Show relevant help based on current tab context.

    Args:
        context: The current context/tab name ('overview', 'isa', 'sipp', 'rebalancing', 'assistant')
    """
    tips = HELP_CONTENT['contextual_tips'].get(context.lower(), [])

    if not tips:
        return

    with st.expander("Tips for this section", expanded=False):
        for tip in tips:
            st.markdown(f"- {tip}")

        # Show relevant help content
        help_key = f"{context.lower()}_help" if context.lower() not in ['isa', 'sipp'] else 'holdings_help'
        if context.lower() == 'overview':
            help_key = 'overview_intro'

        if help_key in HELP_CONTENT:
            st.divider()
            st.markdown(HELP_CONTENT[help_key])


# ============================================================================
# KEYBOARD SHORTCUTS MODAL
# ============================================================================

def render_keyboard_shortcuts_modal():
    """Render a modal showing all keyboard shortcuts.

    Displays:
    - R = Refresh
    - 1-5 = Tab navigation
    - ? = Show help
    - T = Toggle theme
    """
    if not st.session_state.get('show_shortcuts', False):
        return

    with st.container():
        st.markdown("""
        <style>
        .shortcuts-modal {
            background: rgba(30, 41, 59, 0.98);
            border: 2px solid #d4af37;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        .shortcut-row {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(100, 116, 139, 0.3);
        }
        .shortcut-key {
            background: #d4af37;
            color: #0f172a;
            padding: 4px 12px;
            border-radius: 4px;
            font-family: monospace;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("### Keyboard Shortcuts")
        with col2:
            if st.button("Close", key="close_shortcuts"):
                st.session_state.show_shortcuts = False
                st.rerun()

        st.markdown("Use these shortcuts for faster navigation:")

        for shortcut in HELP_CONTENT['keyboard_shortcuts']:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"<span class='shortcut-key'>{shortcut['key']}</span>",
                           unsafe_allow_html=True)
            with col2:
                st.write(shortcut['action'])

        st.caption("Note: Keyboard shortcuts require browser focus on the dashboard.")
        st.divider()


# ============================================================================
# INLINE HELP ICON
# ============================================================================

def render_inline_help_icon(help_key, label=None):
    """Render a small help icon that shows help content in a popover.

    Args:
        help_key: Key to look up in HELP_CONTENT
        label: Optional label to show next to the icon
    """
    content = HELP_CONTENT.get(help_key, "Help content not available.")

    if isinstance(content, str):
        with st.popover("(?)"):
            if label:
                st.markdown(f"**{label}**")
            st.markdown(content)
    elif isinstance(content, list):
        with st.popover("(?)"):
            if label:
                st.markdown(f"**{label}**")
            for item in content[:5]:  # Limit to first 5 items
                if isinstance(item, dict):
                    st.markdown(f"**{item.get('q', '')}**")
                    st.markdown(item.get('a', ''))
                else:
                    st.markdown(f"- {item}")


# ============================================================================
# KEYBOARD SHORTCUT HANDLERS
# ============================================================================

def inject_keyboard_shortcut_handlers():
    """Inject JavaScript for handling keyboard shortcuts.

    Note: Due to Streamlit's architecture, keyboard shortcuts have limited
    functionality. The 'R' key for refresh works, but tab navigation
    would require custom JavaScript integration with Streamlit's tab system.
    """
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(e) {
        // Only handle if not in an input field
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }

        switch(e.key.toLowerCase()) {
            case 'r':
                // Refresh - trigger browser reload
                window.location.reload();
                break;
            case '?':
                // Show shortcuts - would need Streamlit callback
                console.log('Show shortcuts - press the (?) button in sidebar');
                break;
        }
    });
    </script>
    """, unsafe_allow_html=True)


# ============================================================================
# CSS INJECTION FOR HELP SYSTEM
# ============================================================================

def inject_help_css():
    """Inject CSS styles for help system components."""
    st.markdown("""
    <style>
    /* Help icon styling */
    .help-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: rgba(212, 175, 55, 0.2);
        color: #d4af37;
        font-size: 12px;
        font-weight: bold;
        cursor: pointer;
        margin-left: 4px;
        transition: all 0.2s ease;
    }

    .help-icon:hover {
        background: #d4af37;
        color: #0f172a;
    }

    /* Tour container styling */
    .tour-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.98) 0%, rgba(15, 23, 42, 0.98) 100%);
        border: 2px solid #d4af37;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        position: relative;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* Welcome modal styling */
    .welcome-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 2px solid #d4af37;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* FAQ styling */
    .faq-question {
        font-weight: 600;
        color: #d4af37;
        margin-bottom: 0.5rem;
    }

    .faq-answer {
        color: #94a3b8;
        margin-bottom: 1rem;
        padding-left: 1rem;
        border-left: 2px solid rgba(212, 175, 55, 0.3);
    }

    /* Keyboard shortcut key styling */
    .kbd {
        display: inline-block;
        padding: 4px 8px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 600;
        line-height: 1;
        color: #0f172a;
        background-color: #d4af37;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)
