# ============================================================================
# GOAL TRACKING SYSTEM
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import uuid
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Goal type definitions
GOAL_TYPES = {
    'portfolio_value': {
        'name': 'Portfolio Value Target',
        'icon': '💰',
        'description': 'Reach a total portfolio value',
        'unit': '£',
        'example': 'Reach £100,000'
    },
    'monthly_income': {
        'name': 'Monthly Income Target',
        'icon': '💵',
        'description': 'Achieve monthly dividend income',
        'unit': '£/month',
        'example': '£500/month dividends'
    },
    'savings_rate': {
        'name': 'Monthly Savings Rate',
        'icon': '📈',
        'description': 'Invest a set amount each month',
        'unit': '£/month',
        'example': 'Invest £1,000/month'
    },
    'holding_shares': {
        'name': 'Specific Holding Target',
        'icon': '🎯',
        'description': 'Own a specific number of shares',
        'unit': 'shares',
        'example': 'Own 100 shares of VUSA'
    }
}

CONFIG_FILENAME = 'config.json'

def load_goals():
    """Load goals from config.json"""
    if os.path.exists(CONFIG_FILENAME):
        with open(CONFIG_FILENAME, 'r') as f:
            cfg = json.load(f)
            return cfg.get('goals', [])
    return []

def save_goals(goals):
    """Save goals to config.json"""
    if os.path.exists(CONFIG_FILENAME):
        with open(CONFIG_FILENAME, 'r') as f:
            cfg = json.load(f)
    else:
        cfg = {}
    cfg['goals'] = goals
    with open(CONFIG_FILENAME, 'w') as f:
        json.dump(cfg, f, indent=4, sort_keys=True)

def create_goal(name, goal_type, target, deadline, ticker=None):
    """Create a new goal object"""
    return {
        'id': str(uuid.uuid4())[:8],
        'name': name,
        'type': goal_type,
        'target': target,
        'ticker': ticker,  # For holding_shares type
        'deadline': deadline,
        'created_at': datetime.now().isoformat()
    }

def calculate_goal_progress(goal, metrics, holdings_df, monthly_income_df=None, transaction_df=None):
    """
    Calculate current progress toward a goal
    Returns: dict with current_value, target_value, progress_pct, on_track, projected_completion, status_color
    """
    goal_type = goal['type']
    target = goal['target']
    deadline = datetime.fromisoformat(goal['deadline']) if goal['deadline'] else None
    created_at = datetime.fromisoformat(goal['created_at']) if 'created_at' in goal else datetime.now() - timedelta(days=30)

    current_value = 0
    projected_completion = None

    if goal_type == 'portfolio_value':
        # Total portfolio value
        current_value = metrics.get('total_value', 0) if metrics else 0

        # Project completion based on historical growth
        if current_value > 0 and deadline:
            days_elapsed = (datetime.now() - created_at).days
            if days_elapsed > 0:
                # Assume steady growth rate
                daily_growth = current_value * 0.0003  # ~10% annual
                days_to_target = (target - current_value) / daily_growth if daily_growth > 0 else float('inf')
                if days_to_target > 0 and days_to_target < 10000:
                    projected_completion = datetime.now() + timedelta(days=days_to_target)

    elif goal_type == 'monthly_income':
        # Current monthly dividend income (projected)
        if metrics:
            current_value = metrics.get('projected_monthly', 0) or metrics.get('projected_annual', 0) / 12

        # Project when we might hit target based on income growth
        if current_value > 0 and current_value < target and deadline:
            # Assume 15% annual income growth through reinvestment
            monthly_growth_rate = 0.15 / 12
            months_needed = np.log(target / current_value) / np.log(1 + monthly_growth_rate) if current_value > 0 else float('inf')
            if months_needed > 0 and months_needed < 500:
                projected_completion = datetime.now() + timedelta(days=months_needed * 30)

    elif goal_type == 'savings_rate':
        # Calculate average monthly investment from transactions
        if transaction_df is not None and not transaction_df.empty:
            orders = transaction_df[transaction_df['Type'] == 'ORDER'].copy()
            if not orders.empty:
                orders['Timestamp'] = pd.to_datetime(orders['Timestamp'])
                orders['Month'] = orders['Timestamp'].dt.to_period('M')
                monthly_totals = orders.groupby('Month')['Total Amount'].sum().abs()
                if len(monthly_totals) > 0:
                    current_value = monthly_totals.mean()

        # For savings rate, check if they're meeting the target
        if current_value >= target:
            projected_completion = datetime.now()  # Already achieved!

    elif goal_type == 'holding_shares':
        # Specific holding target
        ticker = goal.get('ticker', '')
        if holdings_df is not None and not holdings_df.empty:
            holding = holdings_df[holdings_df['Ticker'].str.contains(ticker, case=False, na=False)]
            if not holding.empty:
                current_value = holding['Shares'].sum()

        # Project when we might hit share target
        if current_value > 0 and current_value < target:
            # Estimate based on current accumulation rate (simple projection)
            days_elapsed = (datetime.now() - created_at).days
            if days_elapsed > 7:
                shares_per_day = current_value / days_elapsed
                days_to_target = (target - current_value) / shares_per_day if shares_per_day > 0 else float('inf')
                if days_to_target > 0 and days_to_target < 10000:
                    projected_completion = datetime.now() + timedelta(days=days_to_target)

    # Calculate progress percentage
    progress_pct = min(100, (current_value / target * 100)) if target > 0 else 0

    # Determine if on track
    on_track = True
    status_color = 'green'

    if deadline:
        days_remaining = (deadline - datetime.now()).days
        total_days = (deadline - created_at).days

        if progress_pct >= 100:
            status_color = 'green'  # Completed!
            on_track = True
        elif days_remaining < 0:
            status_color = 'red'  # Overdue
            on_track = False
        elif total_days > 0:
            expected_progress = ((total_days - days_remaining) / total_days) * 100
            if progress_pct >= expected_progress * 0.9:
                status_color = 'green'  # On track
                on_track = True
            elif progress_pct >= expected_progress * 0.7:
                status_color = 'yellow'  # Slightly behind
                on_track = True
            else:
                status_color = 'red'  # Behind schedule
                on_track = False

    return {
        'current_value': current_value,
        'target_value': target,
        'progress_pct': progress_pct,
        'on_track': on_track,
        'projected_completion': projected_completion,
        'status_color': status_color
    }

def render_goal_card(goal, progress, key_suffix=""):
    """Render a visual goal card with progress bar"""
    goal_info = GOAL_TYPES.get(goal['type'], {})
    icon = goal_info.get('icon', '🎯')

    # Status color mapping
    color_map = {
        'green': '#22c55e',
        'yellow': '#f59e0b',
        'red': '#ef4444'
    }
    status_color = color_map.get(progress['status_color'], '#94a3b8')

    # Progress bar color
    if progress['progress_pct'] >= 100:
        bar_color = '#22c55e'  # Green for completed
    elif progress['on_track']:
        bar_color = '#3b82f6'  # Blue for on track
    else:
        bar_color = '#ef4444'  # Red for behind

    # Format values based on goal type
    if goal['type'] == 'holding_shares':
        current_str = f"{progress['current_value']:.1f}"
        target_str = f"{progress['target_value']:.0f}"
        ticker = goal.get('ticker', '')
        unit_display = f" {ticker} shares"
    elif goal['type'] in ['monthly_income', 'savings_rate']:
        current_str = f"£{progress['current_value']:,.0f}"
        target_str = f"£{progress['target_value']:,.0f}"
        unit_display = "/month"
    else:
        current_str = f"£{progress['current_value']:,.0f}"
        target_str = f"£{progress['target_value']:,.0f}"
        unit_display = ""

    # Time remaining
    deadline = datetime.fromisoformat(goal['deadline']) if goal['deadline'] else None
    if deadline:
        days_remaining = (deadline - datetime.now()).days
        if days_remaining < 0:
            time_str = f"Overdue by {abs(days_remaining)} days"
        elif days_remaining == 0:
            time_str = "Due today!"
        elif days_remaining < 30:
            time_str = f"{days_remaining} days left"
        elif days_remaining < 365:
            time_str = f"{days_remaining // 30} months left"
        else:
            time_str = f"{days_remaining // 365} years left"
    else:
        time_str = "No deadline"

    # Render the card
    with st.container():
        # Card header
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            st.markdown(f"### {icon}")
        with col2:
            st.markdown(f"**{goal['name']}**")
            st.caption(f"{goal_info.get('name', goal['type'])}")
        with col3:
            st.markdown(f"<div style='width:12px;height:12px;border-radius:50%;background-color:{status_color};margin-top:10px;'></div>", unsafe_allow_html=True)

        # Progress bar
        progress_html = f"""
        <div style="background-color:#374151;border-radius:10px;height:24px;width:100%;margin:10px 0;">
            <div style="background-color:{bar_color};border-radius:10px;height:24px;width:{min(100, progress['progress_pct'])}%;display:flex;align-items:center;justify-content:center;">
                <span style="color:white;font-weight:bold;font-size:12px;">{progress['progress_pct']:.1f}%</span>
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)

        # Stats row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current", f"{current_str}{unit_display}")
        with col2:
            st.metric("Target", f"{target_str}{unit_display}")
        with col3:
            st.caption(time_str)

        # Projected completion
        if progress['projected_completion'] and progress['progress_pct'] < 100:
            proj_date = progress['projected_completion'].strftime('%b %Y')
            if deadline and progress['projected_completion'] <= deadline:
                st.success(f"On track to complete by {proj_date}")
            elif deadline:
                st.warning(f"Projected completion: {proj_date} (after deadline)")
            else:
                st.info(f"Projected completion: {proj_date}")
        elif progress['progress_pct'] >= 100:
            st.success("Goal achieved!")

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Edit", key=f"edit_{goal['id']}_{key_suffix}", use_container_width=True):
                st.session_state.editing_goal = goal['id']
        with col2:
            if st.button("Delete", key=f"delete_{goal['id']}_{key_suffix}", use_container_width=True):
                st.session_state.deleting_goal = goal['id']

        st.divider()

def render_goals_dashboard(goals, metrics, holdings_df, monthly_income_df=None, transaction_df=None, plotly_template='plotly_dark'):
    """Render the full goals dashboard with grid of goal cards"""
    st.subheader("🎯 Financial Goals")

    if not goals:
        st.info("No goals set yet. Add your first goal below to start tracking your progress!")
        render_add_goal_form(holdings_df)
        return

    # Calculate progress for all goals
    goals_with_progress = []
    on_track_count = 0
    completed_count = 0

    for goal in goals:
        progress = calculate_goal_progress(goal, metrics, holdings_df, monthly_income_df, transaction_df)
        goals_with_progress.append((goal, progress))
        if progress['progress_pct'] >= 100:
            completed_count += 1
        elif progress['on_track']:
            on_track_count += 1

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Goals", len(goals))
    with col2:
        st.metric("Completed", completed_count)
    with col3:
        st.metric("On Track", on_track_count)
    with col4:
        behind = len(goals) - completed_count - on_track_count
        st.metric("Need Attention", behind)

    # Handle deletion
    if 'deleting_goal' in st.session_state:
        goal_id = st.session_state.deleting_goal
        st.warning(f"Are you sure you want to delete this goal?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, delete", key="confirm_delete", use_container_width=True):
                goals = [g for g in goals if g['id'] != goal_id]
                save_goals(goals)
                del st.session_state.deleting_goal
                st.rerun()
        with col2:
            if st.button("Cancel", key="cancel_delete", use_container_width=True):
                del st.session_state.deleting_goal
                st.rerun()
        return

    # Handle editing
    if 'editing_goal' in st.session_state:
        goal_id = st.session_state.editing_goal
        goal_to_edit = next((g for g in goals if g['id'] == goal_id), None)
        if goal_to_edit:
            render_edit_goal_form(goal_to_edit, goals, holdings_df)
            return

    st.divider()

    # Render goal cards in a grid
    cols = st.columns(2)
    for i, (goal, progress) in enumerate(goals_with_progress):
        with cols[i % 2]:
            render_goal_card(goal, progress, key_suffix=str(i))

    # Add new goal section
    with st.expander("Add New Goal"):
        render_add_goal_form(holdings_df)

    # Projections chart
    if len(goals_with_progress) > 0:
        with st.expander("Goal Projections"):
            render_goal_projections(goals_with_progress, metrics, plotly_template)

def render_add_goal_form(holdings_df=None):
    """Render the form to add a new goal"""
    with st.form("add_goal_form"):
        st.write("**Create a New Goal**")

        # Goal type selector
        goal_type = st.selectbox(
            "Goal Type",
            options=list(GOAL_TYPES.keys()),
            format_func=lambda x: f"{GOAL_TYPES[x]['icon']} {GOAL_TYPES[x]['name']}",
            help="Select the type of financial goal you want to track"
        )

        # Show description
        st.caption(f"{GOAL_TYPES[goal_type]['description']} (e.g., {GOAL_TYPES[goal_type]['example']})")

        # Goal name
        name = st.text_input(
            "Goal Name",
            placeholder=GOAL_TYPES[goal_type]['example'],
            help="Give your goal a memorable name"
        )

        # Ticker selector for holding goals
        ticker = None
        if goal_type == 'holding_shares':
            if holdings_df is not None and not holdings_df.empty:
                tickers = holdings_df['Ticker'].unique().tolist()
                ticker = st.selectbox("Select Holding", options=tickers)
            else:
                ticker = st.text_input("Ticker Symbol", placeholder="e.g., VUSA.L")

        # Target value
        col1, col2 = st.columns(2)
        with col1:
            if goal_type == 'holding_shares':
                target = st.number_input("Target Shares", min_value=1, value=100, step=10)
            elif goal_type in ['monthly_income', 'savings_rate']:
                target = st.number_input("Target (per month)", min_value=1.0, value=500.0, step=50.0)
            else:
                target = st.number_input("Target Value", min_value=1.0, value=50000.0, step=1000.0)

        with col2:
            deadline = st.date_input(
                "Target Date",
                value=datetime.now() + timedelta(days=365),
                min_value=datetime.now().date(),
                help="When do you want to achieve this goal?"
            )

        submitted = st.form_submit_button("Create Goal", use_container_width=True)

        if submitted:
            if not name:
                st.error("Please enter a goal name")
            else:
                new_goal = create_goal(
                    name=name,
                    goal_type=goal_type,
                    target=float(target),
                    deadline=deadline.isoformat() if deadline else None,
                    ticker=ticker
                )
                goals = load_goals()
                goals.append(new_goal)
                save_goals(goals)
                st.success(f"Goal '{name}' created!")
                st.rerun()

def render_edit_goal_form(goal, all_goals, holdings_df=None):
    """Render the form to edit an existing goal"""
    st.subheader(f"Edit Goal: {goal['name']}")

    with st.form("edit_goal_form"):
        # Goal type (read-only)
        goal_type = goal['type']
        st.write(f"**Type:** {GOAL_TYPES[goal_type]['icon']} {GOAL_TYPES[goal_type]['name']}")

        # Goal name
        name = st.text_input("Goal Name", value=goal['name'])

        # Ticker for holding goals
        ticker = goal.get('ticker')
        if goal_type == 'holding_shares':
            if holdings_df is not None and not holdings_df.empty:
                tickers = holdings_df['Ticker'].unique().tolist()
                if ticker and ticker not in tickers:
                    tickers.append(ticker)
                ticker = st.selectbox("Select Holding", options=tickers, index=tickers.index(ticker) if ticker in tickers else 0)
            else:
                ticker = st.text_input("Ticker Symbol", value=ticker or "")

        # Target value
        col1, col2 = st.columns(2)
        with col1:
            if goal_type == 'holding_shares':
                target = st.number_input("Target Shares", min_value=1, value=int(goal['target']), step=10)
            elif goal_type in ['monthly_income', 'savings_rate']:
                target = st.number_input("Target (per month)", min_value=1.0, value=float(goal['target']), step=50.0)
            else:
                target = st.number_input("Target Value", min_value=1.0, value=float(goal['target']), step=1000.0)

        with col2:
            current_deadline = datetime.fromisoformat(goal['deadline']).date() if goal['deadline'] else datetime.now().date() + timedelta(days=365)
            deadline = st.date_input("Target Date", value=current_deadline)

        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Save Changes", use_container_width=True)
        with col2:
            cancelled = st.form_submit_button("Cancel", use_container_width=True)

        if submitted:
            # Update the goal
            for i, g in enumerate(all_goals):
                if g['id'] == goal['id']:
                    all_goals[i]['name'] = name
                    all_goals[i]['target'] = float(target)
                    all_goals[i]['deadline'] = deadline.isoformat() if deadline else None
                    if ticker:
                        all_goals[i]['ticker'] = ticker
                    break
            save_goals(all_goals)
            if 'editing_goal' in st.session_state:
                del st.session_state.editing_goal
            st.success("Goal updated!")
            st.rerun()

        if cancelled:
            if 'editing_goal' in st.session_state:
                del st.session_state.editing_goal
            st.rerun()

def render_goal_projections(goals_with_progress, metrics, plotly_template='plotly_dark'):
    """Render a line chart showing projected progress for all goals"""
    if not goals_with_progress:
        return

    # Create projection data
    today = datetime.now()
    future_dates = pd.date_range(start=today, periods=365, freq='D')

    fig = go.Figure()

    colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']

    for i, (goal, progress) in enumerate(goals_with_progress):
        if progress['progress_pct'] >= 100:
            continue  # Skip completed goals

        current = progress['current_value']
        target = progress['target_value']
        deadline = datetime.fromisoformat(goal['deadline']) if goal['deadline'] else today + timedelta(days=365)

        # Calculate daily growth needed
        days_to_deadline = (deadline - today).days
        if days_to_deadline <= 0:
            continue

        daily_growth_needed = (target - current) / days_to_deadline if days_to_deadline > 0 else 0

        # Project values
        projected_values = [current + daily_growth_needed * d for d in range(min(365, days_to_deadline + 30))]
        projection_dates = future_dates[:len(projected_values)]

        # Add projection line
        fig.add_trace(go.Scatter(
            x=projection_dates,
            y=projected_values,
            mode='lines',
            name=goal['name'],
            line=dict(color=colors[i % len(colors)], width=2)
        ))

        # Add target line
        fig.add_trace(go.Scatter(
            x=[today, deadline],
            y=[target, target],
            mode='lines',
            name=f"{goal['name']} Target",
            line=dict(color=colors[i % len(colors)], width=1, dash='dash'),
            showlegend=False
        ))

        # Add deadline marker
        fig.add_trace(go.Scatter(
            x=[deadline],
            y=[target],
            mode='markers',
            name=f"{goal['name']} Deadline",
            marker=dict(color=colors[i % len(colors)], size=12, symbol='star'),
            showlegend=False
        ))

    fig.update_layout(
        template=plotly_template,
        title="Goal Progress Projections",
        xaxis_title="Date",
        yaxis_title="Value",
        height=400,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True, key="goal_projections_chart")

    st.caption("Lines show required daily progress to meet each goal by its deadline. Stars mark target deadlines.")
