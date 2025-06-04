import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import tempfile
import base64
import os
import ollama  # Ensure this is installed and configured

# Enable kaleido for image export in plotly
pio.kaleido.scope.default_format = "png"

def fetch_stock_data(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download historical stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end)
    return data

def compute_vwap(data: pd.DataFrame) -> pd.Series:
    """Calculate VWAP for the given stock data."""
    vwap = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap

def add_technical_indicators(fig: go.Figure, data: pd.DataFrame, indicators: list):
    """Add selected technical indicators to the plotly figure."""
    if "VWAP" in indicators:
        data['VWAP'] = compute_vwap(data)

    for indicator in indicators:
        if indicator == "20-Day SMA":
            sma = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
        elif indicator == "20-Day EMA":
            ema = data['Close'].ewm(span=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
        elif indicator == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            fig.add_trace(go.Scatter(x=data.index, y=sma + 2 * std, mode='lines', name='BB Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=sma - 2 * std, mode='lines', name='BB Lower'))
        elif indicator == "VWAP":
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

def plot_candlestick_chart(data: pd.DataFrame, indicators: list, ticker: str) -> go.Figure:
    """Generate a plotly candlestick chart with optional technical indicators."""
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlestick"
        )
    ])

    add_technical_indicators(fig, data, indicators)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        title=f"{ticker} Candlestick Chart with Technical Indicators"
    )
    return fig

def save_figure_as_image(fig: go.Figure) -> str:
    """Save plotly figure as a temporary PNG image and return file path."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.write_image(tmpfile.name)
        return tmpfile.name

def encode_image_to_base64(image_path: str) -> str:
    """Read image from file path and return base64 encoded string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def run_ai_analysis(image_base64: str) -> str:
    """Send image to AI model for technical analysis and return the result."""
    messages = [{
        'role': 'user',
        'content': (
            "You are a Stock Trader specializing in Technical Analysis at a top financial institution. "
            "Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation. "
            "Base your recommendation only on the candlestick chart and the displayed technical indicators. "
            "First, provide the recommendation, then, provide your detailed reasoning."
        ),
        'images': [image_base64]
    }]
    response = ollama.chat(model='llama3.2-vision', messages=messages)
    return response["message"]["content"]

def main():
    # Streamlit app configuration
    st.set_page_config(layout="wide")
    st.title("Sentinel: AI-Powered Technical Stock Analysis Dashboard")
    st.sidebar.header("Configuration")

    # User inputs
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-14"))

    if st.sidebar.button("Fetch Data"):
        with st.spinner("Fetching data..."):
            data = fetch_stock_data(ticker, start_date, end_date)
            if data.empty:
                st.error("No data fetched for the given ticker and date range.")
                if "stock_data" in st.session_state:
                    del st.session_state["stock_data"]
            else:
                st.session_state["stock_data"] = data
                st.success(f"Loaded {len(data)} rows of data for {ticker}")

    if "stock_data" in st.session_state and not st.session_state["stock_data"].empty:
        data = st.session_state["stock_data"]

        # Select technical indicators
        st.sidebar.subheader("Technical Indicators")
        indicators = st.sidebar.multiselect(
            "Select Indicators:",
            ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
            default=["20-Day SMA"]
        )

        fig = plot_candlestick_chart(data, indicators, ticker)
        st.plotly_chart(fig, use_container_width=True)

        # AI-powered analysis
        st.subheader("AI-Powered Analysis")
        if st.button("Run AI Analysis"):
            with st.spinner("Analyzing the chart, please wait..."):
                image_path = None
                try:
                    image_path = save_figure_as_image(fig)
                    image_base64 = encode_image_to_base64(image_path)
                    ai_result = run_ai_analysis(image_base64)
                    st.markdown("**AI Analysis Results:**")
                    st.write(ai_result)
                except Exception as e:
                    st.error(f"Error during AI analysis: {e}")
                finally:
                    if image_path and os.path.exists(image_path):
                        os.remove(image_path)
    else:
        st.info("Please fetch stock data to display the chart.")

if __name__ == "__main__":
    main()
