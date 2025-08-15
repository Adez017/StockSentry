import streamlit as st
import pandas as pd
from datetime import datetime, date
import os

# Configure page
st.set_page_config(
    page_title="StockSentry ML",
    page_icon="📈",
    layout="wide"
)

def main():
    # Header
    st.title("📈 StockSentry ML")
    st.markdown("Advanced Stock Price Prediction with Machine Learning")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input(
            "News API Key (Optional)",
            type="password",
            help="Enter your NewsAPI key for sentiment analysis. Leave empty for demo mode."
        )
        
        # Stock ticker input
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)"
        ).upper()
        
        # Date range
        st.subheader("Training Period")
        start_date = st.date_input(
            "Start Date",
            value=date(2023, 1, 1),
            max_value=date.today()
        )
        end_date = st.date_input(
            "End Date",
            value=date(2023, 6, 30),
            max_value=date.today()
        )
        
        # Train button
        train_button = st.button("🚀 Train Models", type="primary")

    # Main content area
    if train_button:
        if not ticker:
            st.error("Please enter a stock ticker symbol")
            return
            
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return
        
        # Show progress
        with st.spinner("Training models..."):
            try:
                # Import here to avoid initial loading issues
                from Senetry_ML import StockSentryML
                
                # Initialize StockSentry
                stock_sentry = StockSentryML(api_key if api_key else "demo_key")
                
                # Train models
                best_model = stock_sentry.train_and_evaluate(
                    ticker,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                # Get prediction
                predicted_price = stock_sentry.predict_next_day(ticker)
                
                if predicted_price and stock_sentry.data is not None:
                    display_results(stock_sentry, ticker, predicted_price)
                else:
                    st.error("Failed to generate prediction")
                    
            except ImportError as e:
                st.error("Missing dependencies. Please install required packages.")
                st.code("pip install -r requirements.txt")
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    else:
        # Welcome screen
        st.markdown("### Welcome to StockSentry ML! 👋")
        st.markdown("""
        This application uses machine learning to predict stock prices.
        
        **How to get started:**
        1. Enter a stock ticker symbol (e.g., AAPL, GOOGL, TSLA)
        2. Choose your training date range
        3. Click "Train Models" to start the analysis
        """)

def display_results(stock_sentry, ticker, predicted_price):
    """Display the analysis results"""
    
    # Get current price
    current_price = float(stock_sentry.data.iloc[-1]['Close'])
    change = predicted_price - current_price
    change_pct = (change / current_price) * 100
    
    # Results
    st.success("🎯 Analysis Complete!")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        st.metric("Predicted Price", f"${predicted_price:.2f}", f"${change:+.2f}")
    
    with col3:
        direction = "📈 Bullish" if change > 0 else "📉 Bearish" if change < 0 else "➡️ Neutral"
        st.metric("Signal", direction)
    
    # Show change percentage
    st.markdown(f"**Expected Change:** {change_pct:+.2f}%")
    
    # Basic chart using Streamlit's built-in chart
    st.markdown("### Price Chart")
    chart_data = stock_sentry.data.set_index('Date')['Close']
    st.line_chart(chart_data)
    
    # Show data table
    with st.expander("View Raw Data"):
        st.dataframe(stock_sentry.data.tail(10))

if __name__ == "__main__":
    main()