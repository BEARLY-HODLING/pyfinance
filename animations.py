"""
Premium Animations & Micro-Interactions for Freetrade Dashboard
===============================================================

This module provides a comprehensive set of CSS animations and helper functions
for creating smooth, premium micro-interactions throughout the dashboard.

Animation Classes:
- .fade-in, .fade-in-up, .fade-in-down, .fade-in-left, .fade-in-right
- .fade-in-stagger (for lists/grids)
- .hover-lift, .hover-glow
- .pulse, .shimmer, .spin
- .count-up
- .shake-error, .flash-success
- .celebrate, .badge-reveal

Usage:
    from animations import inject_animation_css, render_success_feedback

    # In main():
    inject_animation_css()

    # For success feedback:
    render_success_feedback("Configuration saved!")
"""

import streamlit as st


def inject_animation_css():
    """Inject premium animations and micro-interactions CSS"""
    animation_css = """
    <style>
        /* ===== KEYFRAME ANIMATIONS ===== */

        /* Fade-in animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInLeft {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes fadeInRight {
            from {
                opacity: 0;
                transform: translateX(20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        /* Loading animations */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }

        /* Success/Error animations */
        @keyframes flashSuccess {
            0% { background-color: rgba(34, 197, 94, 0); }
            50% { background-color: rgba(34, 197, 94, 0.3); }
            100% { background-color: rgba(34, 197, 94, 0); }
        }

        @keyframes flashError {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
            20%, 40%, 60%, 80% { transform: translateX(5px); }
        }

        @keyframes checkmark {
            0% { stroke-dashoffset: 100; }
            100% { stroke-dashoffset: 0; }
        }

        /* Celebration animations */
        @keyframes confetti {
            0% { transform: translateY(0) rotate(0deg); opacity: 1; }
            100% { transform: translateY(100vh) rotate(720deg); opacity: 0; }
        }

        @keyframes glow {
            0%, 100% { box-shadow: 0 0 5px rgba(212, 175, 55, 0.3); }
            50% { box-shadow: 0 0 20px rgba(212, 175, 55, 0.6), 0 0 30px rgba(212, 175, 55, 0.4); }
        }

        @keyframes badgeReveal {
            0% { transform: scale(0) rotate(-180deg); opacity: 0; }
            50% { transform: scale(1.2) rotate(10deg); }
            100% { transform: scale(1) rotate(0deg); opacity: 1; }
        }

        /* Number counting animation */
        @keyframes countUp {
            from { opacity: 0.3; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Underline animation */
        @keyframes underlineExpand {
            from { width: 0; }
            to { width: 100%; }
        }

        /* Tab slide animations */
        @keyframes slideLeft {
            from { opacity: 0; transform: translateX(30px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes slideRight {
            from { opacity: 0; transform: translateX(-30px); }
            to { opacity: 1; transform: translateX(0); }
        }

        /* ===== ANIMATION CLASSES ===== */

        /* Fade-in classes */
        .fade-in {
            animation: fadeIn 0.3s ease-out forwards;
        }

        .fade-in-up {
            animation: fadeInUp 0.4s ease-out forwards;
        }

        .fade-in-down {
            animation: fadeInDown 0.4s ease-out forwards;
        }

        .fade-in-left {
            animation: fadeInLeft 0.4s ease-out forwards;
        }

        .fade-in-right {
            animation: fadeInRight 0.4s ease-out forwards;
        }

        /* Staggered fade-in for lists/grids */
        .fade-in-stagger > * {
            opacity: 0;
            animation: fadeInUp 0.4s ease-out forwards;
        }
        .fade-in-stagger > *:nth-child(1) { animation-delay: 0ms; }
        .fade-in-stagger > *:nth-child(2) { animation-delay: 50ms; }
        .fade-in-stagger > *:nth-child(3) { animation-delay: 100ms; }
        .fade-in-stagger > *:nth-child(4) { animation-delay: 150ms; }
        .fade-in-stagger > *:nth-child(5) { animation-delay: 200ms; }
        .fade-in-stagger > *:nth-child(6) { animation-delay: 250ms; }
        .fade-in-stagger > *:nth-child(7) { animation-delay: 300ms; }
        .fade-in-stagger > *:nth-child(8) { animation-delay: 350ms; }

        /* Hover lift effect for cards */
        .hover-lift {
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .hover-lift:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
        }

        /* Hover glow effect */
        .hover-glow {
            transition: box-shadow 0.3s ease;
        }
        .hover-glow:hover {
            box-shadow: 0 0 20px rgba(212, 175, 55, 0.4);
        }

        /* Loading classes */
        .pulse {
            animation: pulse 1.5s ease-in-out infinite;
        }

        .shimmer {
            background: linear-gradient(90deg,
                rgba(255,255,255,0) 0%,
                rgba(255,255,255,0.2) 50%,
                rgba(255,255,255,0) 100%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
        }

        .spin {
            animation: spin 1s linear infinite;
        }

        /* Count-up animation */
        .count-up {
            animation: countUp 0.5s ease-out forwards;
        }

        /* Success/Error feedback */
        .flash-success {
            animation: flashSuccess 0.6s ease-out;
        }

        .shake-error {
            animation: flashError 0.5s ease-out;
        }

        /* Celebration classes */
        .celebrate {
            animation: glow 2s ease-in-out infinite;
        }

        .badge-reveal {
            animation: badgeReveal 0.6s ease-out forwards;
        }

        /* ===== STREAMLIT COMPONENT ANIMATIONS ===== */

        /* Main content fade-in on load */
        .main .block-container {
            animation: fadeIn 0.3s ease-out;
        }

        /* Metrics animation */
        [data-testid="stMetric"] {
            animation: fadeInUp 0.4s ease-out;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
        }

        /* Staggered metrics */
        [data-testid="stHorizontalBlock"] [data-testid="stMetric"]:nth-child(1) { animation-delay: 0ms; }
        [data-testid="stHorizontalBlock"] [data-testid="stMetric"]:nth-child(2) { animation-delay: 50ms; }
        [data-testid="stHorizontalBlock"] [data-testid="stMetric"]:nth-child(3) { animation-delay: 100ms; }
        [data-testid="stHorizontalBlock"] [data-testid="stMetric"]:nth-child(4) { animation-delay: 150ms; }
        [data-testid="stHorizontalBlock"] [data-testid="stMetric"]:nth-child(5) { animation-delay: 200ms; }

        /* Metric value count-up effect */
        [data-testid="stMetricValue"] {
            animation: countUp 0.5s ease-out;
        }

        /* Button hover effects */
        .stButton > button {
            transition: all 0.2s ease !important;
            position: relative;
            overflow: hidden;
        }

        .stButton > button:hover {
            transform: scale(1.02);
        }

        .stButton > button:active {
            transform: scale(0.98);
        }

        /* Button ripple effect */
        .stButton > button::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: width 0.3s ease, height 0.3s ease;
        }

        .stButton > button:active::after {
            width: 200%;
            height: 200%;
        }

        /* Table row hover animation */
        [data-testid="stDataFrame"] tbody tr {
            transition: background-color 0.15s ease;
        }

        [data-testid="stDataFrame"] tbody tr:hover {
            background-color: rgba(212, 175, 55, 0.1) !important;
        }

        /* DataFrame fade-in */
        [data-testid="stDataFrame"] {
            animation: fadeInUp 0.4s ease-out;
        }

        /* Expander animation */
        .streamlit-expanderHeader {
            transition: background-color 0.2s ease;
        }

        .streamlit-expanderContent {
            animation: fadeIn 0.3s ease-out;
        }

        /* Tab content transition */
        .stTabs [data-baseweb="tab-panel"] {
            animation: fadeIn 0.3s ease-out;
        }

        /* Tab indicator animation */
        .stTabs [data-baseweb="tab-highlight"] {
            transition: left 0.3s ease, width 0.3s ease !important;
        }

        /* Active tab animation */
        .stTabs [aria-selected="true"] {
            transition: color 0.2s ease;
        }

        /* Charts animation container */
        .js-plotly-plot {
            animation: fadeIn 0.4s ease-out;
        }

        /* Sidebar animation */
        [data-testid="stSidebar"] {
            animation: fadeInLeft 0.3s ease-out;
        }

        /* Form inputs animation */
        .stTextInput, .stNumberInput, .stSelectbox {
            animation: fadeIn 0.3s ease-out;
        }

        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            box-shadow: 0 0 0 2px rgba(212, 175, 55, 0.3);
        }

        /* Alert animations */
        .stAlert {
            animation: fadeInUp 0.3s ease-out;
        }

        .stSuccess {
            animation: fadeInUp 0.3s ease-out, flashSuccess 0.6s ease-out;
        }

        .stError {
            animation: fadeInUp 0.3s ease-out, flashError 0.5s ease-out;
        }

        /* Spinner styling */
        .stSpinner > div {
            animation: spin 0.8s linear infinite !important;
        }

        /* Progress bar shimmer */
        .stProgress > div > div > div {
            position: relative;
            overflow: hidden;
        }

        .stProgress > div > div > div::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg,
                transparent 0%,
                rgba(255,255,255,0.3) 50%,
                transparent 100%);
            animation: shimmer 2s infinite;
        }

        /* Link underline animation */
        a {
            position: relative;
            text-decoration: none;
        }

        a::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 0;
            height: 1px;
            background: currentColor;
            transition: width 0.3s ease;
        }

        a:hover::after {
            width: 100%;
        }

        /* Card hover effect for containers */
        [data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlockBorderWrapper"] {
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        [data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-2px);
        }

        /* Header animation */
        h1, h2, h3 {
            animation: fadeInDown 0.4s ease-out;
        }

        /* Divider animation */
        hr {
            animation: fadeIn 0.5s ease-out;
        }

        /* Success checkmark animation */
        .checkmark-animated {
            stroke-dasharray: 100;
            stroke-dashoffset: 100;
            animation: checkmark 0.5s ease-out forwards;
            animation-delay: 0.2s;
        }

        /* Portfolio high celebration overlay */
        .celebration-confetti {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 9999;
        }

        .confetti-piece {
            position: absolute;
            width: 10px;
            height: 10px;
            animation: confetti 3s ease-out forwards;
        }

        /* Achievement badge */
        .achievement-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 4px 12px;
            border-radius: 20px;
            background: linear-gradient(135deg, #d4af37, #fbbf24);
            color: #0f172a;
            font-weight: 600;
            font-size: 0.85em;
            animation: badgeReveal 0.6s ease-out forwards;
        }

        /* Skeleton loading state */
        .skeleton {
            background: linear-gradient(90deg,
                rgba(255,255,255,0.1) 25%,
                rgba(255,255,255,0.2) 50%,
                rgba(255,255,255,0.1) 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: 4px;
        }

        /* Number highlight animation for positive/negative */
        .number-positive {
            animation: fadeIn 0.3s ease-out;
            color: var(--success) !important;
        }

        .number-negative {
            animation: fadeIn 0.3s ease-out;
            color: var(--error) !important;
        }

        /* Smooth scroll behavior */
        html {
            scroll-behavior: smooth;
        }

        /* Reduce motion for accessibility */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
    </style>

    <script>
        // Number counting animation helper
        function animateValue(element, start, end, duration) {
            const range = end - start;
            const increment = range / (duration / 16);
            let current = start;
            const timer = setInterval(() => {
                current += increment;
                if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
                    element.textContent = end.toLocaleString();
                    clearInterval(timer);
                } else {
                    element.textContent = Math.round(current).toLocaleString();
                }
            }, 16);
        }

        // Confetti celebration function
        function celebratePortfolioHigh() {
            const colors = ['#d4af37', '#fbbf24', '#22c55e', '#3b82f6', '#f59e0b'];
            const container = document.createElement('div');
            container.className = 'celebration-confetti';
            document.body.appendChild(container);

            for (let i = 0; i < 50; i++) {
                const confetti = document.createElement('div');
                confetti.className = 'confetti-piece';
                confetti.style.left = Math.random() * 100 + '%';
                confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
                confetti.style.animationDelay = Math.random() * 0.5 + 's';
                confetti.style.borderRadius = Math.random() > 0.5 ? '50%' : '0';
                container.appendChild(confetti);
            }

            setTimeout(() => container.remove(), 4000);
        }
    </script>
    """
    st.markdown(animation_css, unsafe_allow_html=True)


def render_animated_metric(label: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Render a metric with count-up animation effect"""
    delta_html = ""
    if delta:
        color = "var(--success)" if delta_color == "normal" else "var(--error)"
        delta_html = f'<div style="font-size: 0.75rem; color: {color};">{delta}</div>'

    st.markdown(f'''
        <div class="fade-in-up" style="margin-bottom: 1rem;">
            <div style="font-size: 0.875rem; color: var(--text-secondary);">{label}</div>
            <div class="count-up" style="font-size: 1.75rem; font-weight: 600;">{value}</div>
            {delta_html}
        </div>
    ''', unsafe_allow_html=True)


def render_success_feedback(message: str):
    """Render a success message with flash animation and animated checkmark"""
    st.markdown(f'''
        <div class="flash-success fade-in-up" style="
            padding: 0.75rem 1rem;
            border-radius: 8px;
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.3);
            color: var(--success);
            margin-bottom: 1rem;
        ">
            <svg class="checkmark-animated" style="width: 20px; height: 20px; margin-right: 8px; vertical-align: middle;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
                <path d="M5 12l5 5L20 7"/>
            </svg>
            {message}
        </div>
    ''', unsafe_allow_html=True)


def render_error_feedback(message: str):
    """Render an error message with shake animation"""
    st.markdown(f'''
        <div class="shake-error fade-in-up" style="
            padding: 0.75rem 1rem;
            border-radius: 8px;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: var(--error);
            margin-bottom: 1rem;
        ">
            {message}
        </div>
    ''', unsafe_allow_html=True)


def render_achievement_badge(text: str):
    """Render an achievement badge with reveal animation"""
    st.markdown(f'''
        <span class="achievement-badge badge-reveal">
            {text}
        </span>
    ''', unsafe_allow_html=True)


def render_celebration_confetti():
    """Trigger confetti celebration for portfolio highs"""
    st.markdown('''
        <script>
            if (typeof celebratePortfolioHigh === 'function') {
                celebratePortfolioHigh();
            }
        </script>
    ''', unsafe_allow_html=True)


def get_plotly_animation_config():
    """Get Plotly animation configuration for smooth chart transitions"""
    return {
        'frame': {'duration': 500, 'redraw': True},
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'mode': 'immediate'
    }


def apply_chart_animation(fig):
    """Apply animation settings to a Plotly figure for smooth transitions"""
    fig.update_layout(
        transition={'duration': 500, 'easing': 'cubic-in-out'},
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'visible': False
        }]
    )
    # Add draw-in effect for line charts with spline smoothing
    for trace in fig.data:
        if hasattr(trace, 'line') and trace.line is not None:
            trace.update(
                line={'shape': 'spline', 'smoothing': 0.8}
            )
    return fig


def render_loading_pulse(text: str = "Loading..."):
    """Render a pulsing loading indicator"""
    st.markdown(f'''
        <div class="pulse fade-in" style="
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-secondary);
        ">
            <div class="spin" style="
                width: 16px;
                height: 16px;
                border: 2px solid var(--text-secondary);
                border-top-color: var(--accent);
                border-radius: 50%;
            "></div>
            {text}
        </div>
    ''', unsafe_allow_html=True)


def render_shimmer_skeleton(width: str = "100%", height: str = "20px"):
    """Render a shimmer skeleton loading placeholder"""
    st.markdown(f'''
        <div class="skeleton shimmer" style="
            width: {width};
            height: {height};
        "></div>
    ''', unsafe_allow_html=True)
