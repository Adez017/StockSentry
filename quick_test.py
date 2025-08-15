import streamlit as st
from datetime import date

st.title("🚀 Quick StockSentry Test")

# Use much smaller date range for testing
ticker = st.text_input("Stock Ticker", value="AAPL")
start_date = st.date_input("Start Date", value=date(2023, 6, 1))
end_date = st.date_input("End Date", value=date(2023, 6, 15))  # Only 2 weeks!

if st.button("Quick Test"):
    st.info(f"Testing with just {(end_date - start_date).days} days of data...")
    
    try:
        from Senetry_ML import StockSentryML
        
        with st.spinner("Running quick test..."):
            stock_sentry = StockSentryML("demo_key")
            
            # Train with small dataset
            best_model = stock_sentry.train_and_evaluate(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            predicted_price = stock_sentry.predict_next_day(ticker)
            
            if predicted_price:
                current_price = float(stock_sentry.data.iloc[-1]['Close'])
                change = predicted_price - current_price
                
                st.success("✅ Quick test successful!")
                st.metric("Current Price", f"${current_price:.2f}")
                st.metric("Predicted Price", f"${predicted_price:.2f}", f"${change:+.2f}")
            else:
                st.error("Prediction failed")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Check if all dependencies are installed correctly")