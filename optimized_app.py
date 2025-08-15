import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
import time

# Configure page
st.set_page_config(
    page_title="StockSentry ML",
    page_icon="📈",
    layout="wide"
)

# Cache the model training to avoid re-running
@st.cache_data(ttl=3600)  # Cache for 1 hour
def train_model_cached(ticker, start_date, end_date, api_key):
    """Cache model training to make it faster"""
    from Senetry_ML_Streamlit import StockSentryML
    
    # Use silent mode for cached version too
    stock_sentry = StockSentryML(api_key if api_key else "demo_key", silent_mode=True)
    best_model = stock_sentry.train_and_evaluate(ticker, start_date, end_date)
    predicted_price = stock_sentry.predict_next_day(ticker)
    
    return stock_sentry, best_model, predicted_price

def main():
    st.title("📈 StockSentry ML - Dynamic Interface")
    st.markdown("Fast stock price predictions with smart caching")
    
    # Sidebar with all controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Quick presets
        st.subheader("🚀 Quick Start")
        if st.button("📊 Demo AAPL (Fast)", help="Pre-configured fast demo"):
            st.session_state.ticker = "AAPL"
            st.session_state.start_date = date(2023, 6, 1)
            st.session_state.end_date = date(2023, 6, 15)
            st.session_state.run_analysis = True
        
        st.divider()
        
        # Dynamic inputs
        st.subheader("📝 Custom Analysis")
        
        # Ticker input
        ticker = st.text_input(
            "Stock Ticker",
            value=st.session_state.get("ticker", "AAPL"),
            help="Enter any stock symbol (AAPL, GOOGL, TSLA, MSFT, etc.)",
            key="ticker_input"
        ).upper()
        
        # Speed vs accuracy trade-off
        speed_mode = st.radio(
            "⚡ Analysis Speed",
            ["Fast (2 weeks)", "Balanced (1 month)", "Thorough (3+ months)"],
            index=0,
            help="Choose speed vs accuracy trade-off"
        )
        
        # Dynamic date ranges based on speed mode
        if speed_mode == "Fast (2 weeks)":
            default_start = date.today() - timedelta(days=14)
            default_end = date.today() - timedelta(days=1)
            max_days = 14
        elif speed_mode == "Balanced (1 month)":
            default_start = date.today() - timedelta(days=30)
            default_end = date.today() - timedelta(days=1)
            max_days = 30
        else:
            default_start = date(2023, 1, 1)
            default_end = date(2023, 6, 30)
            max_days = 365
        
        # Date inputs
        start_date = st.date_input(
            "📅 Start Date",
            value=st.session_state.get("start_date", default_start),
            max_value=date.today() - timedelta(days=1),
            key="start_date_input"
        )
        
        end_date = st.date_input(
            "📅 End Date",
            value=st.session_state.get("end_date", default_end),
            min_value=start_date + timedelta(days=1),
            max_value=date.today() - timedelta(days=1),
            key="end_date_input"
        )
        
        # Show estimated time
        days_selected = (end_date - start_date).days
        if days_selected <= 14:
            estimated_time = "⚡ ~30 seconds"
            time_color = "green"
        elif days_selected <= 30:
            estimated_time = "🕒 ~1-2 minutes"
            time_color = "orange"
        else:
            estimated_time = "⏳ ~3-5 minutes"
            time_color = "red"
        
        st.markdown(f"**Estimated time:** :{time_color}[{estimated_time}]")
        st.caption(f"Analyzing {days_selected} days of data")
        
        # API key (optional)
        with st.expander("🔑 Advanced Options"):
            api_key = st.text_input(
                "News API Key (Optional)",
                type="password",
                help="Leave empty for demo mode with random sentiment"
            )
            
            use_cache = st.checkbox(
                "Use cached results",
                value=True,
                help="Faster results by reusing previous calculations"
            )
        
        st.divider()
        
        # Analyze button
        analyze_btn = st.button(
            "🚀 Analyze Stock",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        # Info panel
        st.markdown("### 📊 Current Selection")
        st.info(f"""
        **Ticker:** {ticker}
        **Period:** {start_date} to {end_date}
        **Days:** {days_selected}
        **Mode:** {speed_mode}
        """)
        
        # Quick stats
        if 'last_analysis' in st.session_state:
            st.markdown("### 📈 Last Result")
            result = st.session_state.last_analysis
            st.metric("Prediction", f"${result['predicted_price']:.2f}")
            st.metric("Change", f"{result['change_pct']:+.2f}%")
    
    with col1:
        # Analysis area
        if analyze_btn or st.session_state.get("run_analysis", False):
            # Clear the run_analysis flag
            if "run_analysis" in st.session_state:
                del st.session_state.run_analysis
            
            if not ticker:
                st.error("❌ Please enter a stock ticker")
                return
            
            if start_date >= end_date:
                st.error("❌ Start date must be before end date")
                return
            
            # Warning for long analysis
            if days_selected > 60:
                st.warning(f"⚠️ Analyzing {days_selected} days will take longer. Consider using 'Fast' or 'Balanced' mode.")
                if not st.button("Continue anyway"):
                    return
            
            # Run analysis
            run_analysis(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), api_key, use_cache)
        
        else:
            # Welcome screen
            st.markdown("### 👋 Welcome to StockSentry ML!")
            
            st.markdown("""
            **Dynamic stock prediction with multiple speed options:**
            
            🚀 **Fast Mode** (2 weeks): Get results in ~30 seconds
            ⚖️ **Balanced Mode** (1 month): Good accuracy in ~1-2 minutes  
            🎯 **Thorough Mode** (3+ months): Best accuracy in ~3-5 minutes
            
            **How to use:**
            1. Select a stock ticker (AAPL, GOOGL, TSLA, etc.)
            2. Choose your speed preference
            3. Adjust dates if needed
            4. Click "Analyze Stock"
            
            ✨ **Smart caching** remembers recent analyses for instant results!
            """)
            
            # Sample chart
            st.markdown("### 📊 Sample Analysis")
            import numpy as np
            dates = pd.date_range(start='2023-06-01', end='2023-06-15', freq='D')
            prices = 150 + np.cumsum(np.random.randn(len(dates)) * 0.5)
            sample_data = pd.DataFrame({'Price': prices}, index=dates)
            st.line_chart(sample_data)

def run_analysis(ticker, start_date, end_date, api_key, use_cache):
    """Run the stock analysis with progress tracking"""
    
    # Progress tracking
    progress = st.progress(0)
    status = st.empty()
    start_time = time.time()
    
    try:
        status.info(f"🔄 Analyzing {ticker}...")
        progress.progress(10)
        
        # Use cached function if enabled
        if use_cache:
            stock_sentry, best_model, predicted_price = train_model_cached(
                ticker, start_date, end_date, api_key
            )
        else:
            # Import the silent version
            from Senetry_ML_Streamlit import StockSentryML
            
            # Initialize in silent mode (no prints, no manual input)
            stock_sentry = StockSentryML(api_key if api_key else "demo_key", silent_mode=True)
            
            progress.progress(30)
            status.info("📊 Fetching market data...")
            
            progress.progress(50)
            status.info("🤖 Training ML models...")
            
            best_model = stock_sentry.train_and_evaluate(ticker, start_date, end_date)
            
            progress.progress(80)
            status.info("📈 Making prediction...")
            
            predicted_price = stock_sentry.predict_next_day(ticker)
        
        progress.progress(100)
        elapsed_time = time.time() - start_time
        status.success(f"✅ Analysis complete in {elapsed_time:.1f} seconds!")
        
        # Display results
        if predicted_price and stock_sentry.data is not None:
            display_results(stock_sentry, ticker, predicted_price)
        else:
            st.error("❌ Failed to generate prediction")
            
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.info("💡 Try a different ticker or date range")

def display_results(stock_sentry, ticker, predicted_price):
    """Display analysis results"""
    
    current_price = float(stock_sentry.data.iloc[-1]['Close'])
    change = predicted_price - current_price
    change_pct = (change / current_price) * 100
    
    # Store results for quick reference
    st.session_state.last_analysis = {
        'ticker': ticker,
        'predicted_price': predicted_price,
        'current_price': current_price,
        'change_pct': change_pct
    }
    
    # Results display
    st.markdown("## 🎯 Prediction Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        st.metric(
            "Predicted Price", 
            f"${predicted_price:.2f}", 
            delta=f"${change:+.2f}",
            delta_color="normal"
        )
    
    with col3:
        # Custom color coding for change percentage
        if change_pct > 0:
            st.markdown(f"""
            <div style="text-align: center;">
                <h3>Expected Change</h3>
                <h1 style="color: #28a745;">+{change_pct:.2f}%</h1>
                <p style="color: #28a745;">↑ ${change:+.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        elif change_pct < 0:
            st.markdown(f"""
            <div style="text-align: center;">
                <h3>Expected Change</h3>
                <h1 style="color: #dc3545;">{change_pct:.2f}%</h1>
                <p style="color: #dc3545;">↓ ${change:+.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric("Expected Change", "0.00%", delta="No change")
    
    with col4:
        if change > 0:
            st.markdown("""
            <div style="text-align: center;">
                <h3>Signal</h3>
                <h1 style="color: #28a745;">📈 Bullish</h1>
                <p style="color: #28a745;">Buy Signal</p>
            </div>
            """, unsafe_allow_html=True)
        elif change < 0:
            st.markdown("""
            <div style="text-align: center;">
                <h3>Signal</h3>
                <h1 style="color: #dc3545;">📉 Bearish</h1>
                <p style="color: #dc3545;">Sell Signal</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric("Signal", "➡️ Neutral", delta="Hold")
    
    # Price chart
    st.markdown("### 📈 Price History")
    if len(stock_sentry.data) > 0:
        chart_data = stock_sentry.data.set_index('Date')['Close']
        st.line_chart(chart_data)
        
        # Data summary
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📊 Data Summary**")
            st.write(f"• Training samples: {len(stock_sentry.data)}")
            st.write(f"• Date range: {len(stock_sentry.data)} days")
            
        with col2:
            st.markdown("**💰 Price Statistics**")
            stats = stock_sentry.data['Close'].describe()
            st.write(f"• Average: ${stats['mean']:.2f}")
            st.write(f"• Range: ${stats['min']:.2f} - ${stats['max']:.2f}")
    
    # Raw data
    with st.expander("📋 View Training Data"):
        st.dataframe(stock_sentry.data)

if __name__ == "__main__":
    main()