import os

os.environ['OPENAI_API_KEY']='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'
os.environ['SERPAPI_API_KEY'] = '85ce8786996e0fa5568e8c4db622cde5b9e883a9667abf39e3f51e3397a3b8e7'
os.environ['ACTIVELOOP_TOKEN']='eyJhbGciOiJIUzUxMiIsImlhdCI6MTY5NDA3Mzk3OSwiZXhwIjoxNzA4Njc1NTAwfQ.eyJpZCI6ImNoYXJsb3R0ZXRlb2N0In0.1b3mHSwu8l4bGj_YRZrteBLXO50E9ydbUnvvPVPMqMJrgHa_Y8aUSZ0fl-8CPgxzmlKra09WvHF6NWcFU5BGKw'
os.environ['TAVILY_API_KEY'] = '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'



import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from pytrends.request import TrendReq
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from logic.stock_analyzer import *
from logic.portfolio_recommender import *
from logic.portfolio_analyzer import *
from textblob import TextBlob 
import wbdata
import yfinance as yf
from ta import add_all_ta_features
from ta.utils import dropna

@tool
def technical_analysis(ticker, period):
    """Perform technical analysis for a stock"""
    try:
        # Get historical stock data
        stock_data = yf.download(ticker, period=period)

        # Add technical analysis features
        stock_data = dropna(stock_data)
        stock_data = add_all_ta_features(
            stock_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )

        # Extract relevant technical indicators
        indicators = {
            "rsi": stock_data["momentum_rsi"],
            "macd": stock_data["trend_macd_diff"],
            "ma50": stock_data["trend_sma_fast"],
            "ma200": stock_data["trend_sma_slow"],
            "bb_bbm": stock_data["volatility_bbm"],
            "bb_bbh": stock_data["volatility_bbh"],
            "bb_bbl": stock_data["volatility_bbl"],
            "stoch": stock_data["momentum_stoch"],
            "atr": stock_data["volatility_atr"],
        }

        # Analyze indicators and provide buy/sell signals
        signals = {}
        for indicator, values in indicators.items():
            if values.iloc[-1] > values.iloc[-2]:
                signals[indicator] = "Strong Buy"
            elif values.iloc[-1] < values.iloc[-2]:
                signals[indicator] = "Strong Sell"
            elif values.iloc[-1] > values.iloc[-3]:
                signals[indicator] = "Weak Buy"
            elif values.iloc[-1] < values.iloc[-3]:
                signals[indicator] = "Weak Sell"
            else:
                signals[indicator] = "Neutral"

        return signals
    except Exception as e:
        return f"Failed to perform technical analysis for {ticker}. Error: {str(e)}"

@tool
def get_current_stock_price(ticker):
    """Method to get current stock price"""

    ticker_data = yf.Ticker(ticker)
    recent = ticker_data.history(period="1d")
    return {"price": recent.iloc[0]["Close"], "currency": ticker_data.info["currency"]}



# @tool 
# def stock_technical_analysis(ticker,start_date):
#     """Method to get current stock price  -  Use when asked to evaluate stock performance
#     expected parameters: 
#     ticker (in Yfinance ticker format) e.g 'AAPL'
#     start_date: string in format eg.'2020-02-20'
#      """
#     return StockAnalyzer(ticker).get_technical_analysis(start_date)



@tool 
def stock_fundamental_analysis(ticker):
    """Method to get current stock price  -  Use when asked to evaluate stock performance 
     expected parameters: 
    ticker (in Yfinance ticker format) e.g 'AAPL'
    """
    return StockAnalyzer(ticker).get_yf_fundamentals_for_analysis()



@tool 
def portfolio_metrics(stocks,weights,start_date,end_date,initial_investment):
    """Method to get overall portfolio metrics - Use when asked to evaluate portfolio performance and top movers
    Expected Parameters
        stocks: list e.g ['GOOGL','AAPL]
        weights:list e.g [0.1,0.9]
        start_date: string in format eg.'2020-02-20'
        end_date: string in format eg.'2020-02-20'
        initial_investment: float or integer format 
    
    
    """
    return PortfolioAnalyzer(stocks,weights,start_date,end_date,initial_investment).calculate_metrics(), portfolio_evaluator.top_movers()



@tool 
def portfolio_sectoral_metrics(stocks,weights,start_date,end_date,initial_investment):
    """ Method to evaluate portfolio performance by sectors -  Use when asked to evaluate portfolio performance 
    Expected Parameters
        stocks: list e.g ['GOOGL','AAPL]
        weights:list e.g [0.1,0.9]
        start_date: string in format eg.'2020-02-20'
        end_date: string in format eg.'2020-02-20'
        initial_investment: float or integer format 
    """
    return PortfolioAnalyzer(stocks,weights,start_date,end_date,initial_investment).sectoral_metrics()







@tool
def get_top_company_news(ticker, num_articles=10):
    """get company headlines- can be used to see top news concerning company and for further qualitative evaluation on price movements """
        # Define the URL for the Google News API
    url = f"https://news.google.com/rss/search?q={ticker}&hl=en-US&gl=US&ceid=US:en"

    # Send an HTTP GET request to the API
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the XML response
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.text)

        # Initialize a list to store news articles
        articles = []

        # Iterate through the XML to extract article titles and links
        for item in root.findall(".//item")[:num_articles]:
            title = item.find("title").text
            link = item.find("link").text
            articles.append({"title": title, "link": link})

        return articles
    else:
        return (
            f"Failed to retrieve news for {ticker}. Status code: {response.status_code}")

@tool
def get_historical_stock_data(ticker, start_date, end_date):
    """Get historical stock data"""
    ticker_data = yf.Ticker(ticker)
    historical_data = ticker_data.history(start=start_date, end=end_date)
    return historical_data.to_dict(orient='records')

@tool
def sentiment_analysis(text):
    """Perform sentiment analysis on text"""
    analysis = TextBlob(text)
    # Classify the sentiment as positive, negative, or neutral
    if analysis.sentiment.polarity > 0:
        return "Positive sentiment"
    elif analysis.sentiment.polarity < 0:
        return "Negative sentiment"
    else:
        return "Neutral sentiment"


@tool
def get_economic_indicators(country_code):
    """Get key economic indicators for a specific country"""
    indicators = {"NY.GDP.MKTP.CD": "GDP", "SL.UEM.TOTL.ZS": "Unemployment Rate", "FP.CPI.TOTL.ZG": "Inflation Rate"}

    try:
        data_date = datetime.datetime.now().year
        data = wbdata.get_dataframe(indicators, country=country_code, data_date=data_date, convert_date=False)

        if data.empty:
            return f"No data available for country code {country_code}"

        # Extract the latest available economic indicators
        latest_data = data.iloc[-1].to_dict()

        # Format the response
        formatted_response = {indicators[indicator]: f"{value:.2f}%" for indicator, value in latest_data.items()}
        return formatted_response
    except Exception as e:
        return f"Failed to fetch economic indicators for {country_code}. Error: {str(e)}"

@tool
def description_of_company(ticker):
    """get descriptions on company(summary,sector,country and industry)"""
    try:
        # Get financial statements
        company = yf.Ticker(ticker)
        return pd.DataFrame(company.info)[[ 'city', 'state', 'country',
       'industry', 'industryKey', 'industryDisp','sector', 'sector', 'sectorKey', 'sectorDisp', 'longBusinessSummary', ]].iloc[0].to_dict()

    except Exception as e:
        return f"Failed to perform fundamental analysis for {ticker}. Error: {str(e)}"


@tool
def fundamental_analysis(ticker):
    """Perform fundamental analysis for a company"""
    try:
        # Get financial statements
        company = yf.Ticker(ticker)
        return pd.DataFrame(company.info)[[  'bookValue', 'priceToBook', 'lastFiscalYearEnd', 'nextFiscalYearEnd', 'mostRecentQuarter', 'earningsQuarterlyGrowth', 'netIncomeToCommon', 'trailingEps', 'forwardEps', 'pegRatio', 'totalCashPerShare', 'ebitda', 'totalDebt', 'quickRatio', 'currentRatio', 'totalRevenue', 'debtToEquity', 'revenuePerShare', 'returnOnAssets', 'returnOnEquity', 'freeCashflow', 'operatingCashflow', 'earningsGrowth', 'revenueGrowth', 'grossMargins', 'ebitdaMargins', 'operatingMargins', 'financialCurrency', 'trailingPegRatio']].iloc[0].to_dict()
        return company.info
    except Exception as e:
        return f"Failed to perform fundamental analysis for {ticker}. Error: {str(e)}"



@tool
def get_google_trends(keyword, timeframe='today 1-y', geo='US'):
    """
    Parameter 1. keyword: stockname,
    2. timeframe: eg. 'today 1-y'
    3. geo: eg. 'US' 
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    
    # Build payload
    pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop='')

    # Get interest over time
    interest_over_time_df = pytrends.interest_over_time()

    return interest_over_time_df

@tool
def rf_garch_allocation_recommendation(tickers,start_date='2018-01-01',end_date='2024-02-28'):
    """
    if asked to recommend optimal portfolio for user based on current holdings can use rf-garch ML approach
    parameters:
    tickers: ['AAPL','JNJ','GOOGL'] list of stocks
    start_date: string format 'YYYY-MM-DD'
    end_date: string format same as start_date 'YYYY-MM-DD' give today's date in that format
    return {
  "expected_returns": [
    {
      "stock": "AAPL",
      "last_price": 182.31,
      "expected_price": 163.91,
      "expected_return": [
        "-10.09%"
      ],
      "forecast_date": "2024-03-17"
    }, for each stock
   
  ],
  "allocated_weights": {
    "labels": [
      "AAPL",
      "GOOGL",
      "JNJ",
      "TSLA"
    ],
    "series": [
      0.22987,
      0.27013,
      0.5,
      0
    ]
  }
}
    """
    return get_recommended_allocation(tickers,start_date,end_date,21)

search = TavilySearchResults()


tools = [
    get_top_company_news,
    get_current_stock_price,
    search, #tavily
    GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper()),
    fundamental_analysis,
    get_economic_indicators,
    technical_analysis,
    description_of_company,
    get_google_trends,
    sentiment_analysis,
    rf_garch_allocation_recommendation

    
]


def get_tools():
    return tools

INITIAL_PROMPT =  """
You are a competent financial advisor that can provide expert guidance and assistance across a spectrum of financial matters. Equipped with an arsenal of sophisticated tools and analytical capabilities, your mission is to empower users with insightful recommendations, personalized strategies, and actionable insights to optimize their financial decisions. Your expertise spans various domains, including stock analysis, portfolio management, economic indicators, and market trends.

As users engage with you, your objective is to understand their queries thoroughly, analyze their financial objectives, and deliver tailored advice that aligns with their goals and risk tolerance. Consider the use of tools based on the described functionalities:

For Evaluating Stock Performance:
Utilize technical analysis and fundamental analysis tools to conduct in-depth assessments of individual stocks. Analyze key indicators, such as RSI, MACD, moving averages, and earnings growth, to gauge the performance and potential of stocks.

For Evaluating Portfolio Performance:
Obtain portfolio metrics using the portfolio_metrics tool. Evaluate the constituents of the portfolio and their respective sectoral metrics to gain insights into overall performance and sector allocations.

Dive deeper into individual companies by analyzing key news using the get_top_company_news tool. Stay informed about top movers in the portfolio and assess their recent developments and news sentiment.

For Market Research and Insights:
Leverage tools like Google Trends to track the popularity of specific keywords or stocks over time. Gain insights into market sentiment and investor interest.

Stay updated on economic indicators using the get_economic_indicators tool. Monitor GDP, unemployment rates, and inflation to understand broader market trends and macroeconomic conditions.

For Sentiment Analysis:
Assess sentiment surrounding stocks, news articles, or social media using the sentiment_analysis tool. Understand market sentiment and identify potential sentiment-driven price movements.

For Portfolio Allocation Recommendations:
Utilize the rf_garch_allocation_recommendation tool to receive optimal portfolio allocation recommendations based on advanced machine learning models. Enhance portfolio diversification and risk management strategies.

For Company News Evaluation:
Additionally, leverage the get_top_company_news tool to analyze key news related to top companies in the portfolio. Stay informed about corporate developments, earnings reports, and industry trends to make informed investment decisions.

Furthermore, if users have queries beyond the provided tools, you can use the search tool to find relevant information within the financial realm. Limit the answers to the financial domain to ensure relevance and accuracy.
 """

def initialise_agent():
    MEMORY_KEY = "chat_history"
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
              INITIAL_PROMPT
            ),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4", temperature=0.8)
    llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

    

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

def get_reply(agent_executor, inputval,chat_history = []):


    print("INITIALISE AGENT FOR LANGCHAIN CHATBOT")

    result = agent_executor.invoke({"input": inputval, "chat_history": chat_history})
    chat_history.extend(
            [
                HumanMessage(content=inputval),
                AIMessage(content=result["output"]),
            ]
        )
    return result, chat_history

