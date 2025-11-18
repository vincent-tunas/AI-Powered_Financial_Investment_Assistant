# AI-Powered Stock Analysis Dashboard 📊

An interactive Streamlit application that helps modern retail investors interpret technical indicators like Simple Moving Averages (SMA), Volume Breakouts, and potential Order Blocks using real-time stock data and LLM-based explanations.

---

## 🧩 Project Overview

Manual stock analysis can be time-consuming and confusing, especially for beginners who must track multiple indicators across many stocks.

This project automates that workflow by:
- Fetching real-time financial data from the Sectors API
- Calculating 5-day and 20-day SMAs and Volume Breakouts
- Highlighting potential bullish/bearish zones (order blocks)
- Using an AI agent to explain signals in simple language (buy / watch / sell)

---

## ✨ Key Features

- 📈 **SMA Analysis**  
  - 5-day SMA for short-term trend  
  - 20-day SMA for medium-term trend  

- 📊 **Volume Breakout Detection**  
  - Compares current volume with 20-day average  
  - Flags strong interest periods that may confirm price moves

- 🧠 **AI Financial Assistant**  
  - Uses LLM reasoning to generate human-readable explanations  
  - Interprets SMA trends and volume behavior (no personal financial advice)

- 🖥️ **Interactive Streamlit Dashboard**  
  - Select stock and date range  
  - Visualize price, SMA, and volume in charts  
  - See signal (buy / watch / sell) with explanation

---

## 🏗 Tech Stack

- **Python**
- **Streamlit**
- **Pandas / NumPy**
- **Matplotlib**
- **Sectors API** (for financial data)
- **Groq / LLM** (for AI explanations)

---
