import json
import requests
from datetime import datetime
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

import pandas as pd
import matplotlib.pyplot as plt

# Retrieve key from secrets.toml
SECTORS_API_KEY = st.secrets["SECTORS_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
BASE_URL = "https://api.sectors.app/v1/stocks"
headers = {"Authorization": SECTORS_API_KEY}


####-----Tools-----####

@tool
def get_data(url: str) -> dict:
    """
    this function is used to perform the get request to sectors.api 
    provides error handling such as fail http request and url errors caused by wrong stock name
    """

    headers = {"Authorization": SECTORS_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    except requests.HTTPError as http_error:
        return {
            "error": f"HTTPError {response.status_code} - {response.reason}",
            "url": url,
            "detail": response.text
        }

    except Exception as error:
        return {
            "error": f"Unexpected error: {type(error).__name__} - {error}",
            "url": url
        }


@tool
def get_stock_daily_transaction(stock: str, start_date: str, end_date: str) -> list[dict]:
    """
    this function gets the daily transaction of the stocks in the user's watchlist based on the start date and end date determined by the user

    Parameters:
    - stock: the stock code of the company
    - start_date: the start date in YYYY-MM-DD format
    - end_date: the end date in YYYY-MM-DD format
    - return: daily transaction data of the stocks for a certain interval
    """
    url = f"https://api.sectors.app/v1/daily/{stock}/?start={start_date}&end={end_date}"

    return get_data(url)


def ai_financial_assistant():

    # Defined Tools
    tools = [
        get_stock_daily_transaction
    ]

    # Create the Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                    You are an AI financial analysis assistant integrated into a Streamlit stock analytics application. 
                    You have access to stock data through the Sectors API and can analyze metrics such as 
                    Simple Moving Average (SMA), price trends, and volume breakouts.

                    Your primary goal is to help users understand stock market behavior and provide 
                    clear, data-driven insights without offering personal financial advice.

                    When analyzing data:
                    1. Interpret the stock’s performance using key metrics:
                    - Simple Moving Average (SMA)
                    - Volume breakout patterns
                    - Price momentum (bullish/bearish trends)
                    2. Explain your reasoning in simple and educational terms, suitable for non-expert users.
                    3. Summarize the current signal clearly as one of:
                    - **BUY** → bullish trend, strong momentum, or breakout above SMA
                    - **WATCH** → sideways trend, uncertain signals, or nearing breakout
                    - **SELL** → bearish trend, breakdown below SMA, or volume decline
                    4. Always back your signal with numeric evidence and concise justification.

                    Tone and style:
                    - Be concise, confident, and factual.
                    - Avoid financial jargon when possible.
                    - Use markdown formatting to improve readability (e.g., bold terms, bullet points, short paragraphs).

                    Avoid making guaranteed predictions or giving personalized investment recommendations.
                    Instead, focus on analyzing data, identifying patterns, and providing educational explanations.
                Today's date is {datetime.today().strftime("%Y-%m-%d")}
                """
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Initializing the LLM
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
    )

    # Create the Agent and AgentExecutor
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Add Memory to the AgentExecutor
    def get_session_history(session_id: str):

        return StreamlitChatMessageHistory(key=session_id)
    
    agent_with_chat_memory = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    return agent_with_chat_memory


####-----Step 1: Watchlist-----####
st.title("📈 AI Financial Assistant")

# create a header " Your watchlist" 
# this is where the application ask the user what stocks they want to monitor
st.sidebar.header("Your Watchlist")

# Initialize session state if not exists
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# provide Text input for user to add the stock code of the company they want to monitor into their watchlist
tickers_input = st.sidebar.text_input(
    "Enter stock tickers (comma separated):", 
    placeholder="e.g. ADRO, GOTO, BBCA"
)

# the Add button is used update the watchlist with the stock codes inputted by the user
if st.sidebar.button("➕ Add to Watchlist"):
    #in the ticker, users can input multiple stock codes separated by commas
    if tickers_input:
        new_tickers = [t.strip().upper() for t in tickers_input.split(",")]
        # if the stock code is already in the watchlist, do not add it again, 
        #if not, append it to the watchlist
        for t in new_tickers:
            if t not in st.session_state.watchlist:
                st.session_state.watchlist.append(t)
        st.sidebar.success(f"Added: {', '.join(new_tickers)}")
    else:
        st.sidebar.warning("Please enter at least one ticker.")

# Button to clear the watchlist
if st.sidebar.button("🗑️ Clear Watchlist"):
    st.session_state.watchlist = []
    st.sidebar.info("Watchlist cleared!")

# Display current watchlist
if st.session_state.watchlist:
    st.sidebar.write("✅ Monitoring:", st.session_state.watchlist)
else:
    st.warning("Please add at least one stock ticker to start monitoring.")

#---------Data Visualization of the user's watchlist---------#

#display the header 'daily transaction data visualization'
st.header("📊 Daily Transaction Data Visualization")

#check the user's watchlist, if there are stock codes in the watchlist, provide date input for the user to choose the start date and end date for the data visualization
if st.session_state.watchlist:
    start_date = st.date_input("Start Date", datetime(2024, 1, 1))
    end_date = st.date_input("End Date", datetime.today())

    #create a button to fetch and visualize the data
    if st.button("📈 Fetch & Visualize Data"):
        #for each stock in the watchlist, fetch the daily transaction data from the sectors api between the user given start date and end date
        for stock in st.session_state.watchlist:
            st.subheader(f"Stock: {stock}")

            # Call the tool get_stock_daily_transaction to retrieve the data user requires
            data = get_stock_daily_transaction.invoke({"stock": stock,
                                        "start_date": start_date.strftime("%Y-%m-%d"),
                                        "end_date": end_date.strftime("%Y-%m-%d"),
                                    })


            # if the api returns an error, display the error message and skip to the next stock
            if "error" in data:
                st.error(f"Error fetching data for {stock}: {data['error']}")
                continue

             # Ensure the DataFrame is not empty
            df = pd.DataFrame(data)
            if df.empty:
                st.warning(f"No data found for {stock}.")
                continue

            # Convert the date column into date format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            # Show first few rows of the data
            st.dataframe(df.head())

            # Plot closing price trend
            if all(col in df.columns for col in ['date', 'close']):
                fig, ax1 = plt.subplots(figsize=(8, 4))
                ax1.plot(df['date'], df['close'], label='Close Price')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Close Price')
                ax1.set_title(f'{stock} - Closing Price Trend')
                ax1.legend()
                st.pyplot(fig)

            # Plot trading volume
            if 'volume' in df.columns:
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                ax2.bar(df['date'], df['volume'], color='orange', alpha=0.6)
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Volume')
                ax2.set_title(f'{stock} - Daily Volume')
                st.pyplot(fig2)
#If Watchlist Is Empty
else:
    st.info("Add some tickers in your watchlist to visualize their daily data.")


#-------------------- Simple Moving Average (SMA) and Volume Breakout Implementation --------------------#

st.header("📊 SMA and Volume Breakout Analysis")

#check if there are stock codes in the user's watchlist, if there are, provide date input for the user to choose the start date and end date for the analysis

if st.session_state.watchlist:
    start_date = st.date_input("Start Date", datetime(2024, 1, 1), key="analysis_start_date")
    end_date = st.date_input("End Date", datetime.today(), key="analysis_end_date")

    #create a button for the user to analyze the stocks in their watchlist once he is done choosing the start date and end date.
    #for each of the stocks in the user's watchlist, the app fetches its daily transaction data within the chosen period.
    if st.button("📈 Analyze Stocks", key="analyze_button"):
        for stock in st.session_state.watchlist:
            st.subheader(f"📈 {stock} Analysis")

            # --- Retrieve data from Sectors API ---
            data = get_stock_daily_transaction.invoke({"stock": stock,
                                        "start_date": start_date.strftime("%Y-%m-%d"),
                                        "end_date": end_date.strftime("%Y-%m-%d"),
                                    })


            # error handling for API response to prevent the app from crashing if we get an error fromn the get request
            if "error" in data:
                st.error(f"Error fetching data for {stock}: {data['error']}")
                continue

            # Convert the data from sectors.api into a DataFrame
            df = pd.DataFrame(data)
            if df.empty:
                st.warning(f"No data found for {stock}.")
                continue

            # Normalize the column names, 
            # change the timestamp column into date
            # change the closing column into close
            # change the Vol column into columns
            # this makes the names of the features in the dataframe more universal.
            if "timestamp" in df.columns and "date" not in df.columns:
                df.rename(columns={"timestamp": "date"}, inplace=True)
            if "closing" in df.columns and "close" not in df.columns:
                df.rename(columns={"closing": "close"}, inplace=True)
            if "vol" in df.columns and "volume" not in df.columns:
                df.rename(columns={"vol": "volume"}, inplace=True)

            # Ensure date column exists and sort the dataframe by date
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")

            # --- Compute Simple Moving Averages ---
            #with the 'close' column of the data section of the dataframe, calculate the 5-day and 20-day simple moving averages (SMA)
            if "close" in df.columns:
                df["SMA_5"] = df["close"].rolling(window=5, min_periods=1).mean()
                df["SMA_20"] = df["close"].rolling(window=20, min_periods=1).mean()

            # --- Detect Volume Breakouts ---
            #check the volume data of the dataframe and calculate the 20-day average trading volume
            #if a current day's volume is 50% higher than the 20-day average volume, mark it as a volume breakout
            #A breakout indicates a strong market interest in the stock, which can confirm price movements either going upwards or downwards.
            if "volume" in df.columns:
                df["Avg_Vol_20"] = df["volume"].rolling(window=20, min_periods=1).mean()
                df["Volume_Breakout"] = df["volume"] > 1.5 * df["Avg_Vol_20"]

            # --- Generate Technical Signal ---
            #initialize the signal as "Watch" by default
            #initialize the reason as "No major movement yet." as default
            signal = "📊 Watch"
            reason = "No major movement yet."

            # Determine Buy/Sell signals based on SMA crossover and price position
            #if price is above both SMAs, it indicates an uptrend, hence a Buy signal
            #if price is below both SMAs, it indicates a downtrend, hence a Sell signal

            if "close" in df.columns:
                latest = df.iloc[-1]
                if latest["close"] > latest["SMA_5"] > latest["SMA_20"]:
                    signal = "🟢 Buy"
                    reason = "Price is above both SMAs — uptrend detected."
                elif latest["close"] < latest["SMA_5"] < latest["SMA_20"]:
                    signal = "🔴 Sell"
                    reason = "Price is below both SMAs — downtrend detected."

                # If a volume breakout occurs, it reinforces the confidence of the signal.
                if "Volume_Breakout" in df.columns and latest["Volume_Breakout"]:
                    reason += " Volume breakout confirms strong move."

            # --- Display Technical Analysis ---
            st.markdown(f"**Signal:** {signal}")
            st.caption(reason)

#------------------------- Use AI Agent for Explanation -----------------------------#
            # Display the section title "AI Insight" in your Streamlit Dashboard
            st.markdown("#### 🤖 AI Insight")

            # Initialize the Groq-based LLM financial agent to interpret the financial data and generate the reasonings
            finance_agent = ai_financial_assistant() 

            # Create a System message to determine the AI Agent's behaviour and way of thinking during processing the response to us
            system_message = f"""
            You are an experienced financial analyst. Based on the following indicators,
            explain why stock {stock} shows a **{signal}** signal.
            Consider the relationships between price, SMA trends, and volume activity.

            Latest data:
            - Date: {latest.get('date')}
            - Close: {latest.get('close')}
            - SMA_5: {latest.get('SMA_5')}
            - SMA_20: {latest.get('SMA_20')}
            - Volume: {latest.get('volume')}
            - 20-day Avg Volume: {latest.get('Avg_Vol_20')}
            - Volume Breakout: {latest.get('Volume_Breakout')}
            """

            # Run the explanation using the existing AI agent
            with st.spinner("AI analyzing market reasoning..."):
                ai_response = finance_agent.invoke(
                    {"input": system_message},
                    config={"configurable": {"session_id": f"{stock}_analysis"}}
                )

            # Display the AI’s explanation
            st.success(ai_response["output"])

            # --- Plot the closing Price trend over the chosen periods and the Simple moving Average ---
            # using the column 'date' and 'close' from the dataframe, create the plot
            # Plot the closing price trend over the period of time using a blue line
            # Plot the 5 day-simple moving average results with an orange-dashed line
            # Plot the 20-day simple moving average with a green-dashed line
            # This dashboard visualizes the price trends along with its crossovers with the simple moving average.
            if {"date", "close"} <= set(df.columns):
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(df["date"], df["close"], label="Close", color="blue", linewidth=1.5)
                ax.plot(df["date"], df["SMA_5"], label="SMA 5", color="orange", linestyle="--")
                ax.plot(df["date"], df["SMA_20"], label="SMA 20", color="green", linestyle="--")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.set_title(f"{stock} - Price & SMAs")
                ax.legend()
                st.pyplot(fig)

            # --- Plot the Volume of the stocks daily transactions and the volume Breakouts that occurs ---
            # Plot the daily traing volume as a grey bar chart.
            if {"date", "volume"} <= set(df.columns):
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                ax2.bar(df["date"], df["volume"], color="grey", alpha=0.5, label="Volume")
                #set the colour of the volume breakout days as a red bar.
                if "Volume_Breakout" in df.columns:
                    breakout_points = df[df["Volume_Breakout"]]
                    ax2.bar(breakout_points["date"], breakout_points["volume"], color="red", alpha=0.7, label="Breakout")

                ax2.set_xlabel("Date")
                ax2.set_ylabel("Volume")
                ax2.set_title(f"{stock} - Volume & Breakouts")
                ax2.legend()
                st.pyplot(fig2)

            # --- show the recent stock data for users to inspect. ---
            st.dataframe(df.tail(10))

else:
    st.info("Please add tickers to your watchlist to analyze their trends.")

# ------------------------------------------------------------
#  Project: AI-Powered Financial Analysis Dashboard
#  Author : Vincenntius Patrick Tunas
#  Created: 2025
#  
#  Description:
#    This code is part of a personal project developed by 
#    Vincenntius Patrick Tunas. Any form of reproduction or 
#    redistribution without proper credit is not allowed.
#
#  Copyright © 2025 Vincenntius Patrick Tunas
# ------------------------------------------------------------
