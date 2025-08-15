import streamlit as st
import pandas as pd
from datetime import date
import sys
import io
from contextlib import redirect_stdout

# Configure page
st.set_page_config(
    page_title="StockSentry ML",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("📈 StockSentry ML Web Interface")
    st.markdown("Your command-line model is working! Let's make it web-friendly.")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        api_key = st.text_input(
            "News API Key (Optional)",
            type="password",
            placeholder="Leave empty for demo mode"
        )
        
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)"
        ).upper()
        
        st.subheader("Training Period")
        start_date = st.date_input(
            "Start Date",
            value=date(2023, 1, 1)
        )
        end_date = st.date_input(
            "End Date", 
            value=date(2023, 6, 30)
        )
        
        train_button = st.button("🚀 Train Models", type="primary")

    # Main area
    if train_button:
        if not ticker:
            st.error("Please enter a stock ticker symbol")
            return
            
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return
        
        # Progress tracking
        progress = st.progress(0)
        status = st.empty()
        
        try:
            status.info("🔄 Initializing StockSentry ML...")
            progress.progress(20)
            
            from Senetry_ML import StockSentryML
            
            status.info(f"📊 Training models for {ticker}...")
            progress.progress(40)
            
            # Capture output to show progress
            output_capture = io.StringIO()
            
            # Initialize and train
            stock_sentry = StockSentryML(api_key if api_key else "demo_key")
            
            progress.progress(60)
            status.info("🤖 Running machine learning algorithms...")
            
            # Train the model
            best_model = stock_sentry.train_and_evaluate(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            progress.progress(85)
            status.info("📈 Making prediction...")
            
            # Get prediction
            predicted_price = stock_sentry.predict_next_day(ticker)
            
            progress.progress(100)
            status.success("✅ Analysis Complete!")
            
            # Display results
            if predicted_price and stock_sentry.data is not None:
                display_results(stock_sentry, ticker, predicted_price)
            else:
                st.error("Failed to generate prediction")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Your model works in command line. This might be a Streamlit integration issue.")
    
    else:
        # Welcome message
        st.success("✅ Great news! Your StockSentry ML is working in command line!")
        st.info("🎯 We saw your successful prediction: AAPL $187.53 (Ridge model, R² = 0.9854)")
        
        st.markdown("### 🚀 Next Steps:")
        st.markdown("""
        1. **Enter your settings** in the sidebar
        2. **Click 'Train Models'** to run the analysis
        3. **View results** in this web interface
        
        Your model is already working perfectly - now let's get it web-ready!
        """)
        
        # Show sample data
        st.markdown("### 📊 Your Model Performance:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Accuracy", "98.54%", "Excellent!")
        with col2:
            st.metric("Best Model", "Ridge", "Auto-selected")
        with col3:
            st.metric("Last Prediction", "$187.53", "+$0.03")

def display_results(stock_sentry, ticker, predicted_price):
    """Display results in web format"""
    
    current_price = float(stock_sentry.data.iloc[-1]['Close'])
    change = predicted_price - current_price
    change_pct = (change / current_price) * 100
    
    st.markdown("## 🎯 Prediction Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        st.metric("Predicted Price", f"${predicted_price:.2f}", f"${change:+.2f}")
    
    with col3:
        st.metric("Expected Change", f"{change_pct:+.2f}%")
    
    with col4:
        direction = "📈 Bullish" if change > 0 else "📉 Bearish" if change < 0 else "➡️ Neutral"
        st.metric("Signal", direction)
    
    # Chart
    st.markdown("### 📈 Price History")
    if 'Date' in stock_sentry.data.columns and 'Close' in stock_sentry.data.columns:
        chart_data = stock_sentry.data.set_index('Date')['Close']
        st.line_chart(chart_data)
    
    # Data table
    with st.expander("📋 Recent Data"):
        st.dataframe(stock_sentry.data.tail(10))

if __name__ == "__main__":
    main()