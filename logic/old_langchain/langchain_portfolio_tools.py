from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
import os
from logic.portfolio_analyzer import PortfolioAnalyzer
from logic.stock_analyzer import StockAnalyzer


class CurrentStockPriceInput(BaseModel):
    """Inputs for get_current_stock_price"""
    ticker: str = Field(description="Ticker symbol of the stock")


class CurrentStockPriceTool(BaseTool):
    name = "get_current_stock_price"
    description = """
        Useful when you want to get the current stock price.
        You should enter the stock ticker symbol recognized by Yahoo Finance.
    """
    args_schema: Type[BaseModel] = CurrentStockPriceInput

    def _run(self, ticker: str):
        try:
            price_response = StockAnalyzer(ticker).get_cur_price()
            return price_response
        except Exception as e:
            return str(e)

    def _arun(self, ticker: str):
        raise NotImplementedError(
            "get_current_stock_price does not support async")


class StockPercentChangeInput(BaseModel):
    """Inputs for get_stock_performance"""
    ticker: str = Field(description="Ticker symbol of the stock")
    days: int = Field(
        description="Timedelta days to get past date from the current date")


class StockPerformanceTool(BaseTool):
    name = "get_stock_performance"
    description = """
        Useful when you want to check the performance of the stock or change in the stock price represented as a percentage.
        You should enter the stock ticker symbol recognized by Yahoo Finance.
    """
    args_schema: Type[BaseModel] = StockPercentChangeInput

    def _run(self, ticker: str, days: int):
        try:
            response = StockAnalyzer(ticker).get_stock_performance(days)
            return response
        except Exception as e:
            return str(e)

    def _arun(self, ticker: str):
        raise NotImplementedError(
            "get_stock_performance does not support async")


class StockNameInput(BaseModel):
    """Inputs for get_stock_fundamentals and get_top_stock_headlines"""
    ticker: str = Field(description="Ticker symbol of the stock")


class PortfolioDictInput(BaseModel):
    """Inputs for get_portfolio_data"""
    portfolio: dict = Field(description="""Input a dictionary data type containing information about the portfolio. For example:
        {"stocks": ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    "weights": [0.25, 0.25, 0.25, 0.25],
    "start_date": '2012-01-01',
    "end_date": '2022-12-31',
    "initial_investment":1000000}
    """)


class PortfolioInfoInput(BaseModel):
    """Inputs for get_portfolio_data"""
    stocks: list = Field(description="List of stocks in the portfolio")
    weights: list = Field(
        description="List of weights of stocks in the portfolio")
    start_date: str = Field(
        description="initial holding date of the portfolio. If not found, use 1 year ago from today in string format in yyyy-mm-dd")
    end_date: str = Field(
        description="today's date in string format yyyy-mm-dd ")
    initial_investment: float = Field(
        description="initial value of investment")


class StockFundamentalsTool(BaseTool):
    name = "get_stock_fundamentals"
    description = """ For stocks NOT ETFs
        Useful when you want to check fundamentals of the company.
        You should enter the stock ticker symbol recognized by Yahoo Finance.
        Output will be a dictionary with relevant information on the company. This can be used to compare stocks.
    """
    args_schema: Type[BaseModel] = StockNameInput

    def _run(self, ticker: str):
        try:
            response = str(StockAnalyzer(ticker).get_yf_fundamentals())
            return response
        except Exception as e:
            return str(e)

    def _arun(self, ticker: str):
        raise NotImplementedError(
            "get_stock_fundamentals does not support async")


class StockNewsTool(BaseTool):
    name = "get_top_stock_headlines"
    description = """
        Useful when you want to check top headlines of the company.
        You should enter the stock ticker symbol recognized by Yahoo Finance.
        Output will be a list of top news on the company. This can be used to evaluate the current sentiment on the stock.
    """
    args_schema: Type[BaseModel] = StockNameInput

    def _run(self, ticker: str):
        try:
            response = str(StockAnalyzer(ticker).get_top_company_news())
            return response
        except Exception as e:
            return str(e)

    def _arun(self, ticker: str):
        raise NotImplementedError(
            "get_top_stock_headlines does not support async")


class PortfolioEvaluationTool(BaseTool):
    name = "get_portfolio_metrics"
    description = """
        Useful when you want to evaluate a portfolio's quantitative performance.
        You should enter the stocks, weights, start_date, end_date, and initial_investment based on user input.
        Output will be a dictionary with relevant information on the performance.
    """
    args_schema: Type[BaseModel] = PortfolioInfoInput

    def _run(self, stocks: list, weights: list, start_date: str, end_date: str, initial_investment: float):
        try:
            response = str(PortfolioAnalyzer(
                stocks, weights, start_date, end_date, initial_investment).calculate_metrics())
            return response
        except Exception as e:
            return str(e)

    def _arun(self, ticker: str):
        raise NotImplementedError(
            "get_portfolio_metrics does not support async")
