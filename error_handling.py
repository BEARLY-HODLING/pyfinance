"""
Error Handling System for Freetrade Dashboard

A comprehensive error handling system with premium error displays,
helpful suggestions, and user-friendly error recovery.
"""

import streamlit as st
import functools
import traceback
import logging
import uuid

# Configure logging for error tracking
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ERROR TYPES AND CONSTANTS
# ============================================================================

class ErrorType:
    """Error type constants for categorizing errors"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    OFFLINE = "offline"


ERROR_ICONS = {
    ErrorType.ERROR: "🚫",
    ErrorType.WARNING: "⚠️",
    ErrorType.INFO: "ℹ️",
    ErrorType.OFFLINE: "📡"
}

ERROR_COLORS = {
    ErrorType.ERROR: {
        'border': '#ef4444',
        'background': 'rgba(239, 68, 68, 0.1)',
        'header_bg': 'rgba(239, 68, 68, 0.2)',
        'text': '#fca5a5'
    },
    ErrorType.WARNING: {
        'border': '#f59e0b',
        'background': 'rgba(245, 158, 11, 0.1)',
        'header_bg': 'rgba(245, 158, 11, 0.2)',
        'text': '#fcd34d'
    },
    ErrorType.INFO: {
        'border': '#3b82f6',
        'background': 'rgba(59, 130, 246, 0.1)',
        'header_bg': 'rgba(59, 130, 246, 0.2)',
        'text': '#93c5fd'
    },
    ErrorType.OFFLINE: {
        'border': '#6b7280',
        'background': 'rgba(107, 114, 128, 0.1)',
        'header_bg': 'rgba(107, 114, 128, 0.2)',
        'text': '#d1d5db'
    }
}

TYPE_LABELS = {
    ErrorType.ERROR: "Error",
    ErrorType.WARNING: "Warning",
    ErrorType.INFO: "Information",
    ErrorType.OFFLINE: "Connection Issue"
}


# ============================================================================
# CSS INJECTION
# ============================================================================

def inject_error_css():
    """Inject CSS for error handling components"""
    css = """
    <style>
        /* Error Card Base Styles */
        .error-card {
            border-radius: 12px;
            padding: 0;
            margin: 1rem 0;
            overflow: hidden;
            transition: all 0.3s ease;
        }

        .error-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }

        .error-card-header {
            padding: 1rem 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .error-card-icon {
            font-size: 1.5rem;
            line-height: 1;
        }

        .error-card-title {
            font-weight: 600;
            font-size: 1.1rem;
            margin: 0;
        }

        .error-card-body {
            padding: 1rem 1.25rem;
        }

        .error-card-message {
            font-size: 0.95rem;
            line-height: 1.6;
            margin-bottom: 0.75rem;
        }

        .error-card-suggestions {
            margin-top: 1rem;
            padding: 0.75rem;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.2);
        }

        .error-card-suggestions h4 {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            opacity: 0.9;
        }

        .error-card-suggestions ul {
            margin: 0;
            padding-left: 1.25rem;
            font-size: 0.9rem;
        }

        .error-card-suggestions li {
            margin-bottom: 0.25rem;
            opacity: 0.85;
        }

        .error-card-actions {
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        .error-card-details {
            margin-top: 0.75rem;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.8rem;
            padding: 0.75rem;
            border-radius: 6px;
            background: rgba(0, 0, 0, 0.3);
            white-space: pre-wrap;
            word-break: break-all;
            max-height: 150px;
            overflow-y: auto;
        }

        /* Empty State Styles */
        .empty-state {
            text-align: center;
            padding: 3rem 2rem;
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(107, 114, 128, 0.1) 0%, rgba(75, 85, 99, 0.05) 100%);
            border: 2px dashed rgba(107, 114, 128, 0.3);
            margin: 2rem 0;
        }

        .empty-state-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            opacity: 0.6;
        }

        .empty-state-message {
            font-size: 1.2rem;
            font-weight: 500;
            color: #94a3b8;
            margin-bottom: 0.5rem;
        }

        .empty-state-hint {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }

        /* Partial Data Warning */
        .partial-warning {
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #f59e0b;
            background: rgba(245, 158, 11, 0.1);
        }

        .partial-warning-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .partial-warning-items {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }

        .partial-warning-item {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            background: rgba(245, 158, 11, 0.2);
            font-size: 0.85rem;
            font-family: monospace;
        }

        /* Offline Mode */
        .offline-banner {
            background: linear-gradient(135deg, #374151 0%, #1f2937 100%);
            border: 1px solid #4b5563;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            text-align: center;
        }

        .offline-banner-icon {
            font-size: 3rem;
            margin-bottom: 0.75rem;
        }

        .offline-banner-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #f3f4f6;
        }

        .offline-banner-message {
            color: #9ca3af;
            margin-bottom: 1rem;
        }

        .cache-info {
            font-size: 0.85rem;
            color: #6b7280;
            margin-top: 0.5rem;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================================
# MAIN ERROR DISPLAY FUNCTION
# ============================================================================

def render_error_message(
    error_type: str,
    message: str,
    details: str = None,
    suggestions: list = None,
    show_retry: bool = False,
    retry_key: str = None,
    show_report: bool = True
) -> bool:
    """
    Render a premium error display card.

    Args:
        error_type: One of ErrorType constants (error, warning, info, offline)
        message: Main error message to display
        details: Optional technical details (shown in collapsible)
        suggestions: Optional list of suggested actions
        show_retry: Whether to show a retry button
        retry_key: Unique key for the retry button (required if show_retry=True)
        show_report: Whether to show "Report Issue" link

    Returns:
        True if retry button was clicked, False otherwise
    """
    colors = ERROR_COLORS.get(error_type, ERROR_COLORS[ErrorType.ERROR])
    icon = ERROR_ICONS.get(error_type, "❓")
    type_label = TYPE_LABELS.get(error_type, "Notice")

    # Build suggestions HTML
    suggestions_html = ""
    if suggestions:
        items = "".join([f"<li>{s}</li>" for s in suggestions])
        suggestions_html = f"""
        <div class="error-card-suggestions">
            <h4>Suggested Actions:</h4>
            <ul>{items}</ul>
        </div>
        """

    html = f"""
    <div class="error-card" style="
        border: 2px solid {colors['border']};
        background: {colors['background']};
    ">
        <div class="error-card-header" style="background: {colors['header_bg']};">
            <span class="error-card-icon">{icon}</span>
            <span class="error-card-title" style="color: {colors['text']};">{type_label}</span>
        </div>
        <div class="error-card-body">
            <p class="error-card-message">{message}</p>
            {suggestions_html}
        </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)

    # Technical details in expander
    if details:
        with st.expander("🔍 Technical Details"):
            st.code(details, language="text")

    # Action buttons
    retry_clicked = False
    if show_retry or show_report:
        cols = st.columns([1, 1, 3] if show_retry and show_report else [1, 4])
        col_idx = 0

        if show_retry:
            with cols[col_idx]:
                if st.button("🔄 Retry", key=retry_key or f"retry_{uuid.uuid4().hex[:8]}", use_container_width=True):
                    retry_clicked = True
                    st.cache_data.clear()
                    st.rerun()
            col_idx += 1

        if show_report:
            with cols[col_idx]:
                st.markdown(
                    '<a href="https://github.com/your-repo/issues/new" target="_blank" style="text-decoration: none;">'
                    '<small style="color: #64748b;">📝 Report Issue</small></a>',
                    unsafe_allow_html=True
                )

    return retry_clicked


# ============================================================================
# ERROR HANDLERS
# ============================================================================

def handle_api_error(error: Exception, context: str) -> dict:
    """
    Handle yfinance API errors gracefully.

    Args:
        error: The exception that occurred
        context: Description of what was being attempted (e.g., "fetching AAPL price")

    Returns:
        dict with keys: error_type, message, details, suggestions
    """
    error_str = str(error).lower()
    error_name = type(error).__name__

    # Log for debugging
    logger.warning(f"API Error during {context}: {error_name} - {error}")

    # Rate limiting detection
    if any(term in error_str for term in ['rate limit', 'too many requests', '429', 'throttle']):
        return {
            'error_type': ErrorType.WARNING,
            'message': "Yahoo Finance is rate limiting requests. Please wait a moment.",
            'details': f"Context: {context}\nError: {error_name}\n{str(error)}",
            'suggestions': [
                "Wait 30-60 seconds before retrying",
                "Reduce the number of tickers being fetched",
                "Try refreshing the page later"
            ]
        }

    # Network error detection
    if any(term in error_str for term in ['connection', 'timeout', 'network', 'socket', 'ssl', 'dns']):
        return {
            'error_type': ErrorType.OFFLINE,
            'message': "Unable to connect to Yahoo Finance. Please check your internet connection.",
            'details': f"Context: {context}\nError: {error_name}\n{str(error)}",
            'suggestions': [
                "Check your internet connection",
                "Try again in a few seconds",
                "Yahoo Finance servers may be temporarily unavailable"
            ]
        }

    # Invalid ticker
    if any(term in error_str for term in ['not found', 'invalid', 'no data', 'delisted']):
        return {
            'error_type': ErrorType.WARNING,
            'message': "Ticker data could not be found. It may be invalid or delisted.",
            'details': f"Context: {context}\nError: {error_name}\n{str(error)}",
            'suggestions': [
                "Check if the Yahoo Finance ticker symbol is correct",
                "LSE stocks typically need a '.L' suffix (e.g., VUSA.L)",
                "US stocks usually don't need a suffix"
            ]
        }

    # JSON/parsing errors
    if any(term in error_str for term in ['json', 'parse', 'decode', 'key error']):
        return {
            'error_type': ErrorType.WARNING,
            'message': "Received unexpected data format from Yahoo Finance.",
            'details': f"Context: {context}\nError: {error_name}\n{str(error)}",
            'suggestions': [
                "This is usually temporary - try again",
                "Yahoo Finance API format may have changed",
                "Check if the ticker symbol is valid"
            ]
        }

    # Generic API error
    return {
        'error_type': ErrorType.ERROR,
        'message': f"An error occurred while {context}.",
        'details': f"Context: {context}\nError Type: {error_name}\nMessage: {str(error)}\n\nTraceback:\n{traceback.format_exc()}",
        'suggestions': [
            "Try refreshing the page",
            "If the problem persists, wait a few minutes",
            "Check if Yahoo Finance is accessible in your browser"
        ]
    }


def handle_data_error(error: Exception, data_source: str) -> dict:
    """
    Handle CSV parsing and data errors.

    Args:
        error: The exception that occurred
        data_source: Description of the data source (e.g., "ISA CSV file")

    Returns:
        dict with keys: error_type, message, details, suggestions
    """
    error_str = str(error).lower()
    error_name = type(error).__name__

    # Log for debugging
    logger.warning(f"Data Error in {data_source}: {error_name} - {error}")

    # Missing column detection
    if 'key' in error_str or 'column' in error_str or 'keyerror' in error_name.lower():
        return {
            'error_type': ErrorType.ERROR,
            'message': f"The {data_source} is missing required columns or has an unexpected format.",
            'details': f"Data Source: {data_source}\nError: {error_name}\n{str(error)}",
            'suggestions': [
                "Ensure you're using a fresh export from Freetrade",
                "Expected columns: Timestamp, Type, Ticker, Quantity, Total Amount, ISIN, Title",
                "Check if the CSV file was modified or corrupted",
                "Re-download the file from Freetrade"
            ]
        }

    # File not found
    if 'no such file' in error_str or 'filenotfound' in error_name.lower():
        return {
            'error_type': ErrorType.WARNING,
            'message': f"Could not find the {data_source}.",
            'details': f"Data Source: {data_source}\nError: {error_name}\n{str(error)}",
            'suggestions': [
                "Place your Freetrade export files in the same folder as the dashboard",
                "Files should be named: freetrade_ISA_*.csv or freetrade_SIPP_*.csv",
                "Export fresh files from the Freetrade app"
            ]
        }

    # Data type errors
    if 'convert' in error_str or 'parse' in error_str or 'dtype' in error_str or 'numeric' in error_str:
        return {
            'error_type': ErrorType.ERROR,
            'message': f"The {data_source} contains data in an unexpected format.",
            'details': f"Data Source: {data_source}\nError: {error_name}\n{str(error)}",
            'suggestions': [
                "Some values may be malformed or contain unexpected characters",
                "Try downloading a fresh export from Freetrade",
                "Check that monetary values are numeric (no currency symbols in data)"
            ]
        }

    # Empty data
    if 'empty' in error_str or 'no data' in error_str:
        return {
            'error_type': ErrorType.INFO,
            'message': f"The {data_source} appears to be empty or contains no valid data.",
            'details': f"Data Source: {data_source}\nError: {error_name}\n{str(error)}",
            'suggestions': [
                "Export a fresh file from Freetrade with your transaction history",
                "Ensure you have at least one completed order in your account"
            ]
        }

    # Generic data error
    return {
        'error_type': ErrorType.ERROR,
        'message': f"An error occurred while processing {data_source}.",
        'details': f"Data Source: {data_source}\nError Type: {error_name}\nMessage: {str(error)}\n\nTraceback:\n{traceback.format_exc()}",
        'suggestions': [
            "Try re-downloading the file from Freetrade",
            "Ensure the file hasn't been modified in Excel or other programs",
            "If the issue persists, check the file in a text editor for issues"
        ]
    }


# ============================================================================
# EMPTY STATE
# ============================================================================

def render_empty_state(
    message: str,
    icon: str = "📭",
    hint: str = None,
    action_label: str = None,
    action_key: str = None
) -> bool:
    """
    Render a beautiful empty state display.

    Args:
        message: Main message to display
        icon: Large emoji icon to show
        hint: Optional helpful hint text
        action_label: Optional call-to-action button text
        action_key: Key for the action button (required if action_label provided)

    Returns:
        True if action button was clicked, False otherwise
    """
    hint_html = f'<p class="empty-state-hint">{hint}</p>' if hint else ''

    html = f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <p class="empty-state-message">{message}</p>
        {hint_html}
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)

    action_clicked = False
    if action_label and action_key:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(action_label, key=action_key, use_container_width=True):
                action_clicked = True

    return action_clicked


# ============================================================================
# PARTIAL DATA WARNING
# ============================================================================

def render_partial_data_warning(missing_items: list, context: str = "data") -> bool:
    """
    Show warning when some data couldn't be loaded.

    Args:
        missing_items: List of items that failed to load
        context: Description of what type of data is missing

    Returns:
        True if retry was clicked, False otherwise
    """
    if not missing_items:
        return False

    items_html = "".join([f'<span class="partial-warning-item">{item}</span>' for item in missing_items[:10]])
    more_text = f" and {len(missing_items) - 10} more..." if len(missing_items) > 10 else ""

    html = f"""
    <div class="partial-warning">
        <div class="partial-warning-header">
            <span>⚠️</span>
            <span>Some {context} couldn't be loaded</span>
        </div>
        <p style="font-size: 0.9rem; opacity: 0.85;">
            The dashboard is showing partial data. The following items had issues:
        </p>
        <div class="partial-warning-items">
            {items_html}{more_text}
        </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 Retry", key=f"retry_partial_{uuid.uuid4().hex[:8]}", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
            return True

    return False


# ============================================================================
# ERROR BOUNDARY DECORATOR
# ============================================================================

def error_boundary(func):
    """
    Decorator to wrap functions with error handling.
    Displays error message instead of crashing the app.

    Usage:
        @error_boundary
        def my_function():
            # code that might fail
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_info = {
                'function': func.__name__,
                'args': str(args)[:200],
                'kwargs': str(kwargs)[:200],
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }

            # Log the error
            logger.error(f"Error in {func.__name__}: {type(e).__name__} - {str(e)}")

            # Render error message
            render_error_message(
                error_type=ErrorType.ERROR,
                message=f"An unexpected error occurred in {func.__name__}",
                details=f"Function: {error_info['function']}\n"
                        f"Error Type: {error_info['error_type']}\n"
                        f"Message: {error_info['error_message']}\n\n"
                        f"Traceback:\n{error_info['traceback']}",
                suggestions=[
                    "Try refreshing the page",
                    "Clear browser cache and reload",
                    "If the problem persists, report this issue"
                ],
                show_retry=True,
                retry_key=f"error_boundary_{func.__name__}_{uuid.uuid4().hex[:8]}"
            )

            return None

    return wrapper


# ============================================================================
# OFFLINE MODE
# ============================================================================

def render_offline_mode(cached_data_info: dict = None):
    """
    Display when API calls fail and we're showing cached or no data.

    Args:
        cached_data_info: Optional dict with cache info:
            - last_updated: datetime of last successful fetch
            - data_available: bool indicating if cached data exists
    """
    cache_html = ""
    if cached_data_info:
        if cached_data_info.get('data_available'):
            last_updated = cached_data_info.get('last_updated', 'Unknown')
            cache_html = f'<p class="cache-info">📦 Showing cached data from {last_updated}</p>'
        else:
            cache_html = '<p class="cache-info">⚠️ No cached data available</p>'

    html = f"""
    <div class="offline-banner">
        <div class="offline-banner-icon">📡</div>
        <h3 class="offline-banner-title">Unable to Connect</h3>
        <p class="offline-banner-message">
            We're having trouble connecting to market data services.
            This could be due to network issues or the service being temporarily unavailable.
        </p>
        {cache_html}
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Retry Connection", key="retry_offline", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


# ============================================================================
# SAFE API CALL WRAPPER
# ============================================================================

def safe_api_call(func, *args, context: str = "API call", default=None, **kwargs):
    """
    Wrapper for safe API calls with automatic error handling.

    Args:
        func: Function to call
        *args: Arguments for the function
        context: Description of what the call is doing
        default: Default value to return on error
        **kwargs: Keyword arguments for the function

    Returns:
        Function result or default value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_info = handle_api_error(e, context)
        logger.warning(f"Safe API call failed: {context} - {str(e)}")
        return default


# ============================================================================
# DISPLAY API ERROR
# ============================================================================

def display_api_error(error: Exception, context: str, show_retry: bool = True, retry_key: str = None):
    """
    Convenience function to handle and display an API error.

    Args:
        error: The exception that occurred
        context: Description of what was being attempted
        show_retry: Whether to show retry button
        retry_key: Key for retry button
    """
    error_info = handle_api_error(error, context)
    render_error_message(
        error_type=error_info['error_type'],
        message=error_info['message'],
        details=error_info['details'],
        suggestions=error_info['suggestions'],
        show_retry=show_retry,
        retry_key=retry_key
    )


def display_data_error(error: Exception, data_source: str, show_retry: bool = True, retry_key: str = None):
    """
    Convenience function to handle and display a data error.

    Args:
        error: The exception that occurred
        data_source: Description of the data source
        show_retry: Whether to show retry button
        retry_key: Key for retry button
    """
    error_info = handle_data_error(error, data_source)
    render_error_message(
        error_type=error_info['error_type'],
        message=error_info['message'],
        details=error_info['details'],
        suggestions=error_info['suggestions'],
        show_retry=show_retry,
        retry_key=retry_key
    )
