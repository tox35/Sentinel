# Sentinel: AI-Powered Technical Stock Analysis Dashboard 📈 

## Overview
**Sentinel** is a Streamlit dashboard that integrates stock charting with AI-driven trading insights. It combines real-time stock data with candlestick charts and technical indicators, then uses vision-based LLMs to analyze and recommend trades.

## Features
- ✅ Real-time stock data from Yahoo Finance
- ✅ Candlestick chart with technical indicators
- ✅ Export chart as image
- ✅ LLaMA 3.2-Vision analysis via Ollama
- ✅ Buy/Hold/Sell recommendation

## Tech Stack
- Python
- Streamlit
- yfinance
- plotly
- kaleido
- ollama

## How It Works
1. User inputs ticker/date range
2. Chart generation with selected indicators
3. Chart saved and sent to LLM
4. AI analyzes and gives trading advice

## 🛠️ Usage
```bash
pip install streamlit yfinance plotly kaleido ollama
ollama run llama3.2-vision
streamlit run trading-ai.py
```
