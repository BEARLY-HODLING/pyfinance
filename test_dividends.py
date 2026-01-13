import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_dividend_sources(ticker):
    """Test multiple sources for dividend yield data"""
    print(f"\n{'='*60}")
    print(f"Testing: {ticker}")
    print(f"{'='*60}")
    
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # Get current price
        price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
        currency = info.get('currency', 'N/A')
        
        # Handle pence scaling
        scale = 100 if '.L' in ticker or currency in ['GBX', 'GBp'] or (currency == 'GBP' and price and price > 100) else 1
        if price and scale > 1:
            price /= scale
        
        print(f"\nPrice: £{price:.2f} ({currency})")
        
        # Method 1: Yahoo Finance reported yield
        yahoo_yield = (info.get('trailingAnnualDividendYield') or 0) * 100
        print(f"\n1. Yahoo Reported Yield: {yahoo_yield:.2f}%")
        
        # Method 2: Calculate from dividend history (TTM)
        divs = t.dividends
        if not divs.empty:
            # Get last 12 months - handle timezone-aware dates
            divs.index = divs.index.tz_localize(None)  # Make naive
            one_year_ago = pd.Timestamp.now() - timedelta(days=365)
            recent_divs = divs[divs.index > one_year_ago]
            
            if not recent_divs.empty:
                # Scale dividends if in pence
                if scale > 1:
                    recent_divs = recent_divs / 100
                
                ttm_dividends = recent_divs.sum()
                calculated_yield = (ttm_dividends / price * 100) if price else 0
                
                print(f"\n2. Calculated from History (TTM):")
                print(f"   - Dividend payments: {len(recent_divs)}")
                print(f"   - Total dividends: £{ttm_dividends:.4f}")
                print(f"   - Calculated yield: {calculated_yield:.2f}%")
                print(f"   - Payment dates:")
                for date, amount in recent_divs.items():
                    print(f"     {date.strftime('%Y-%m-%d')}: £{amount:.4f}")
            else:
                print(f"\n2. Calculated from History: No dividends in last 12 months")
        else:
            print(f"\n2. Calculated from History: No dividend data available")
        
        # Method 3: Forward dividend info
        fwd_annual_div = info.get('dividendRate')
        if fwd_annual_div:
            if scale > 1:
                fwd_annual_div /= 100
            fwd_yield = (fwd_annual_div / price * 100) if price else 0
            print(f"\n3. Forward Annual Dividend:")
            print(f"   - Amount: £{fwd_annual_div:.4f}")
            print(f"   - Yield: {fwd_yield:.2f}%")
        else:
            print(f"\n3. Forward Annual Dividend: Not available")
        
        # Additional useful info
        print(f"\n4. Additional Info:")
        print(f"   - Five Year Avg Yield: {info.get('fiveYearAvgDividendYield', 'N/A')}")
        print(f"   - Payout Ratio: {info.get('payoutRatio', 'N/A')}")
        print(f"   - Ex-Dividend Date: {info.get('exDividendDate', 'N/A')}")
        
        # Recommendation
        print(f"\n5. Recommendation:")
        if not divs.empty and len(recent_divs) > 0:
            print(f"   ✓ Use calculated TTM yield: {calculated_yield:.2f}%")
            return calculated_yield
        elif yahoo_yield > 0:
            print(f"   ⚠ Use Yahoo yield (no history): {yahoo_yield:.2f}%")
            return yahoo_yield
        else:
            print(f"   ✗ Use proxy yield - no reliable data")
            return None
            
    except Exception as e:
        print(f"\n❌ Error fetching data: {e}")
        return None

# Test the problematic tickers
test_tickers = [
    'JEPQ.L',  # JPMorgan Equity Premium Income (proxy: 11%)
    'YMAP.L',  # JPMorgan BetaBuilders US Equity (proxy: 19%)
    'FEPG.L',  # Fidelity Enhanced Income (proxy: 23%)
    'JEQP.L',  # Alternative spelling
]

results = {}
for ticker in test_tickers:
    results[ticker] = test_dividend_sources(ticker)

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"\n{'Ticker':<12} {'Calculated':<15} {'Proxy':<10} {'Recommendation'}")
print("-" * 60)

proxy_yields = {
    'JEPQ.L': 11.0,
    'YMAP.L': 19.0,
    'FEPG.L': 23.0,
    'JEQP.L': 11.0,
}

for ticker in test_tickers:
    calc = results[ticker]
    proxy = proxy_yields.get(ticker, 0)
    
    if calc is not None:
        diff = calc - proxy
        if abs(diff) < 1:
            rec = "✓ Close match"
        elif calc < proxy * 0.8:
            rec = "⚠ Much lower - check data"
        else:
            rec = "⚠ Differs - verify"
    else:
        calc = 'N/A'
        rec = "✗ Use proxy"
    
    print(f"{ticker:<12} {str(calc) + '%' if calc != 'N/A' else calc:<15} {proxy}%{'':<6} {rec}")

print(f"\n{'='*60}")
print("Run this script to see which method works best!")
print(f"{'='*60}")