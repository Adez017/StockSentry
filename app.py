import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import os
from dotenv import load_dotenv
from Senetry_ML import StockSentryML
import logging

# Configure page
st.set_page_config(
    page_title="StockSentry ML",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 StockSentry ML</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Stock Price Prediction with Machine Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "News API Key (Optional)",
            type="password",
            help="Enter your NewsAPI key for sentiment analysis. Leave empty for demo mode."
        )
        
        st.divider()
        
        # Stock ticker input
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)"
        ).upper()
        
        # Date range
        st.subheader("📅 Training Period")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=date(2023, 1, 1),
                max_value=date.today()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=date(2023, 6, 30),
                max_value=date.today()
            )
        
        st.divider()
        
        # Model options
        st.subheader("🤖 Model Settings")
        show_advanced = st.checkbox("Show Advanced Metrics", value=False)
        
        # Train button
        train_button = st.button(
            "🚀 Train Models",
            type="primary",
            use_container_width=True
        )

    # Main content area
    if train_button:
        if not ticker:
            st.error("Please enter a stock ticker symbol")
            return
            
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize StockSentry
            status_text.text("Initializing StockSentry ML...")
            progress_bar.progress(10)
            
            stock_sentry = StockSentryML(api_key if api_key else "demo_key")
            
            # Train models
            status_text.text("Training machine learning models...")
            progress_bar.progress(30)
            
            best_model = stock_sentry.train_and_evaluate(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            progress_bar.progress(70)
            status_text.text("Making predictions...")
            
            # Get prediction
            predicted_price = stock_sentry.predict_next_day(ticker)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            if predicted_price and stock_sentry.data is not None:
                display_results(stock_sentry, ticker, predicted_price, show_advanced)
            else:
                st.error("Failed to generate prediction")
                
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.info("Try with different dates or check your internet connection")
    
    else:
        # Welcome screen
        st.markdown("### Welcome to StockSentry ML! 👋")
        st.markdown("""
        This application uses advanced machine learning algorithms to predict stock prices based on:
        - **Historical price data** from Yahoo Finance
        - **News sentiment analysis** (with API key)
        - **Multiple ML models** including Random Forest, Ridge Regression, and more
        
        **How to get started:**
        1. Enter a stock ticker symbol (e.g., AAPL, GOOGL, TSLA)
        2. Choose your training date range
        3. Optionally add your NewsAPI key for sentiment analysis
        4. Click "Train Models" to start the analysis
        """)
        
        # Sample results or demo
        st.markdown("### 📊 Sample Analysis")
        create_sample_charts()

def display_results(stock_sentry, ticker, predicted_price, show_advanced):
    """Display the analysis results"""
    
    # Get current price
    current_price = float(stock_sentry.data.iloc[-1]['Close'])
    change = predicted_price - current_price
    change_pct = (change / current_price) * 100
    
    # Results header
    st.success("🎯 Analysis Complete!")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Predicted Price",
            f"${predicted_price:.2f}",
            delta=f"${change:+.2f}"
        )
    
    with col3:
        st.metric(
            "Expected Change",
            f"{change_pct:+.2f}%",
            delta=f"${change:+.2f}"
        )
    
    with col4:
        direction = "📈 Bullish" if change > 0 else "📉 Bearish" if change < 0 else "➡️ Neutral"
        st.metric(
            "Signal",
            direction,
            delta=None
        )
    
    # Charts
    st.markdown("### 📈 Price Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price chart
        fig_price = px.line(
            stock_sentry.data,
            x='Date',
            y='Close',
            title=f'{ticker} Price Trend',
            labels={'Close': 'Price ($)', 'Date': 'Date'}
        )
        fig_price.update_layout(showlegend=False)
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        # Volume chart
        if 'Volume' in stock_sentry.data.columns:
            fig_volume = px.bar(
                stock_sentry.data,
                x='Date',
                y='Volume',
                title=f'{ticker} Trading Volume',
                labels={'Volume': 'Volume', 'Date': 'Date'}
            )
            fig_volume.update_layout(showlegend=False)
            st.plotly_chart(fig_volume, use_container_width=True)
    
    # Advanced metrics
    if show_advanced:
        st.markdown("### 🔬 Advanced Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Data Summary**")
            st.write(f"Training samples: {len(stock_sentry.data)}")
            st.write(f"Date range: {stock_sentry.data['Date'].iloc[0].strftime('%Y-%m-%d')} to {stock_sentry.data['Date'].iloc[-1].strftime('%Y-%m-%d')}")
            
            # Price statistics
            st.markdown("**💰 Price Statistics**")
            stats = stock_sentry.data['Close'].describe()
            st.write(f"Mean: ${stats['mean']:.2f}")
            st.write(f"Std: ${stats['std']:.2f}")
            st.write(f"Min: ${stats['min']:.2f}")
            st.write(f"Max: ${stats['max']:.2f}")
        
        with col2:
            st.markdown("**🤖 Model Information**")
            if hasattr(stock_sentry, 'models') and stock_sentry.models:
                st.write(f"Available models: {len(stock_sentry.models)}")
                for model_name in stock_sentry.models.keys():
                    st.write(f"• {model_name}")
            
            # API status
            st.markdown("**🔗 API Status**")
            api_status = "✅ NewsAPI Connected" if stock_sentry.news_api_key and stock_sentry.news_api_key != "demo_key" else "⚠️ Demo Mode (No NewsAPI)"
            st.write(api_status)
    
    # Raw data
    with st.expander("📋 View Raw Data"):
        st.dataframe(stock_sentry.data.tail(10))

def create_sample_charts():
    """Create sample charts for the welcome screen"""
    import numpy as np
    
    # Sample data
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
    prices = 150 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    sample_data = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    
    fig = px.line(
        sample_data,
        x='Date',
        y='Price',
        title='Sample Stock Price Analysis',
        labels={'Price': 'Price ($)', 'Date': 'Date'}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the app
    main()