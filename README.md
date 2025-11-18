# AI-Powered Stock Analysis Dashboard 📊

### 1. Purpose:
 Modern retail investors struggle to interpret multiple technical indicators across various stocks in real-time. Manual analysis of signals like Simple Moving Average (SMA), Volume Breakouts and Order Blocks requires both experience and time.
This project addresses that by automating the analysis and providing clear, AI-assisted buy/sell notifications through an interactive Streamlit web application.

### 2. Objectives:

This application integrates real-time financial data from the sectors API to compute the simple moving Average, Volume Breakouts, detect the potential order blocks (bullish/bearish zones) and provide automated signal interpretation whether to buy/ watch/ sell the stock using LLM reasoning.
This insights are visualized through an interactive streamlit dashboard and notifies the users with the visual cues and explanations for each of the signals

---

## Tech Stack

- **Python**
- **Streamlit**
- **Pandas / NumPy**
- **Matplotlib**
- **Sectors API** (for financial data)
- **Groq / LLM** (for AI explanations)

---
### System design:
    
LLM Usage Strategy:
The application integrates a Large Language Model (LLM) through an API-based prompt-driven approach using Groq Cloud. 

The LLM functions as a financial reasoning engine, converting computed metrics such as the Simple Moving Average (SMA), Volume Breakout, and Order Block signals into human-understandable recommendations (e.g., *Buy*, *Watch*, *Sell*).
The LLM model is use
**Example Output:**
> “SMA is trending above its long-term average and stochastic values show momentum recovery — this may indicate a bullish setup. Consider buying.”

1. Integration Type: API-Based, system-prompt driven model llama-3.3-70b-versatile through Groq. The LLM is embedded into the streamlit application and is called and runned whenever the user request an analysus

2. prompt design: the role of the model is a financial analysis agent that provides analysis based on the Simple moving average, volume breakout and price momentum. The model provides a simple and educational financial explaination. Its output tone and reasoning quality is consistent
3. Context Management: StreamlitChatMessageHistory is used together with RunnableWithMessageHistory in order to maintain the context after each analysis sessions
4. tools and agent: the agent is created using the langchain's agent framework. The tools provided to the agent is the get_stock_daily_transaction tool to retrieve historical stock prices and trade volumes from the Sectors API.
5. RAG: The system uses an on-demand structured data retrieval from sectors.api where the stock time-series data is fetched via the tool function and the LLM interprets the numeric data. This enables the LLM to base its explaination on real, current market data and not hallucinates.
---
The workflow:
The diagram below illustrates how data moves from the user input through the Sectors API, 
indicator analysis module, and LLM reasoning engine before producing insights in the Streamlit UI.

<p align="center">
  <img src="project_report/finance-AI-Workflow.jpg" alt="Data Flow Diagram" width="700"/>
</p>

1. User accesses the Streamlit app.

2. App calls the Sectors API to fetch stock price and volume data.

3. The app computes:
    - SMA (Simple Moving Average)
    - Volume Breakout detection
    - Order Block (Bullish/Bearish zone detection)

4. A Lightweight LLM agent (via API or local reasoning function) interprets the signals and provides a natural-language summary.

5. provide the Results and buy/sell indicators are shown in an interactive dashboard.

