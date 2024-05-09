# portfolio_analyzer.py
from asyncio import selector_events
import yfinance as yf
import numpy as np
from datetime import date, timedelta
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import ta
from nltk.sentiment import SentimentIntensityAnalyzer
from logic.newsfeed_analyzer import get_article_summary
import re
import unicodedata
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI


class StockAnalyzer:
    def __init__(self, ticker: str):
        self.asset_name = ticker
        # self.asset = yf.Ticker(self.asset_name) if yf.Ticker(self.asset_name)!=None else None
        self.asset = self._get_asset(ticker)
        self.asset_info = self.asset.info
        self.company_name = self.asset_info['shortName']
        self.etf = 'fundFamily' in self.asset_info.keys()

    def _get_asset(self, ticker: str):
        try:
            res = yf.Ticker(ticker)
            info = res.info  # You might want to use this 'info' somewhere

        except Exception as e:
            raise ValueError(
                f"Failed to fetch data for ticker {ticker}. Error: {str(e)}")

        return res

    def get_cur_price(self):
        if self.etf: 
            return self.asset_info['open']
        else:
            return self.asset_info['currentPrice']

  

    def get_asset_info(self):
        return self.asset_info

    def get_stock_performance(self, days) -> float:
        """Method to get stock price change in percentage"""

        past_date = date.today() - timedelta(days=days)
        ticker_data = yf.Ticker(self.asset_name)
        history = ticker_data.history(start=past_date)
        old_price = history.iloc[0]["Close"]
        current_price = history.iloc[-1]["Close"]
        # return {"percent_change": ((current_price - old_price) / old_price) * 100}
        return ((current_price - old_price) / old_price) * 100

    def preprocess_fundamental_data(self,raw_data):
        processed_data = {}
        for key, value in raw_data.items():
            if value is not None:
                if isinstance(value, (int, float)):
                    # Format numeric values with commas
                    processed_data[key] = "{:,}".format(value)
                else:
                    processed_data[key] = value
        return processed_data
    
    def get_yf_fundamentals_for_analysis(self):
        
        fundamental_data = self.asset_info
        if self.etf:
           market_data = {
            'Total Assets': fundamental_data.get('totalAssets'),
            '52-Week Low': fundamental_data.get('fiftyTwoWeekLow'),
            '52-Week High': fundamental_data.get('fiftyTwoWeekHigh'),
            '50-Day Average': fundamental_data.get('fiftyDayAverage'),
            '200-Day Average': fundamental_data.get('twoHundredDayAverage'),
            'NAV Price': fundamental_data.get('navPrice'),
            'Currency': fundamental_data.get('currency'),
            'Category': fundamental_data.get('category'),
            'YTD Return': fundamental_data.get('ytdReturn'),
            'Fund Family': fundamental_data.get('fundFamily'),
            'Fund Inception Date': fundamental_data.get('fundInceptionDate'),
            'Legal Type': fundamental_data.get('legalType')
        }

        else:
        # Extracting required fundamental data
            market_data = {
            'Average Volume': fundamental_data.get('averageVolume'),
            'Market Cap': fundamental_data.get('marketCap'),
            'Trailing PE': fundamental_data.get('trailingPE'),
            'Forward PE': fundamental_data.get('forwardPE'),
            'Beta': fundamental_data.get('beta'),
            '52-Week Low': fundamental_data.get('fiftyTwoWeekLow'),
            '52-Week High': fundamental_data.get('fiftyTwoWeekHigh'),
            'Dividend Yield': fundamental_data.get('dividendYield'),
            'Dividend Rate': fundamental_data.get('dividendRate'),
            'Profit Margins': fundamental_data.get('profitMargins'),
            'Operating Margins': fundamental_data.get('operatingMargins'),
            'Gross Margins': fundamental_data.get('grossMargins'),
            'EBITDA': fundamental_data.get('ebitda'),
            'Revenue': fundamental_data.get('revenue'),
            'Revenue Per Share': fundamental_data.get('revenuePerShare'),
            'Gross Profits': fundamental_data.get('grossProfits'),
            'Total Cash': fundamental_data.get('totalCash'),
            'Total Debt': fundamental_data.get('totalDebt'),
            'Current Ratio': fundamental_data.get('currentRatio'),
            'Quick Ratio': fundamental_data.get('quickRatio'),
            'Total Revenue': fundamental_data.get('totalRevenue'),
            'Debt to Equity': fundamental_data.get('debtToEquity'),
            'Return on Assets': fundamental_data.get('returnOnAssets'),
            'Return on Equity': fundamental_data.get('returnOnEquity'),
            'Operating Cash Flow': fundamental_data.get('operatingCashflow'),
            'Free Cash Flow': fundamental_data.get('freeCashflow')
            }

            # 'exDividendDate': fundamental_data.get('exDividendDate'),
            # 'earningsTimestamp': fundamental_data.get('earningsTimestamp'),
            # 'earningsTimestampStart': fundamental_data.get('earningsTimestampStart'),
            # 'earningsTimestampEnd': fundamental_data.get('earningsTimestampEnd')
        market_data = {key: value for key, value in market_data.items() if value is not None}

        return self.preprocess_fundamental_data(market_data)


    


    def get_stock_data(self, start_date_str, freq):
        """Method to get stock price change in percentage"""
        # start_date = date.today()-timedelta(days=history)
        # start_date_str = "{}-{}-{}".format(start_date.year,
        #                                    start_date.month, start_date.day)
        end_date = date.today()
        end_date_str = "{}-{}-{}".format(end_date.year,
                                         end_date.month, end_date.day)
        data = yf.download(self.asset_name, start=start_date_str, end=end_date_str)[[
            'Adj Close']]

        if freq == 'M':
            data = data.resample('M').last()
        elif freq == 'Y':
            data = data.resample('Y').last()
        data = data.reset_index()
        data['Date'] = data['Date'].apply(lambda x: x.date())
        data.columns = ['Date', 'Close']
      
        return  [ {'dates':list(data['Date']), \
            'values':[round(y,2) for y in list(data['Close'])], \
            'name': self.asset_name}]

    def get_technical_analysis(self, start_date):
    
        end_date = date.today()
        end_date_str = "{}-{}-{}".format(end_date.year,
                                         end_date.month, end_date.day)
        data = yf.download(self.asset_name, start=start_date, end=end_date_str)

        # Add technical indicators
        data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

        # Drop NaN values
        data.dropna(inplace=True)

        # Get the latest values
        latest_values = data.iloc[-1]

        # Determine buy/sell signals
      
        buy_signal = (
            (latest_values['trend_macd'] > latest_values['trend_macd_signal']) and
            (latest_values['momentum_rsi'] < 30) and
            (latest_values['momentum_stoch_rsi_k'] < 0.2) and
            (latest_values['trend_adx'] > 25) and
            (latest_values['trend_ema_fast'] > latest_values['trend_ema_slow']) and
            (latest_values['trend_sma_fast'] > latest_values['trend_sma_slow']) and
            (latest_values['volume_obv'] > latest_values['volume_obv_mean']) and
            (latest_values['volatility_bbh'] < latest_values['close']) and
            (latest_values['volatility_bbl'] > latest_values['close']) and
            (latest_values['volatility_bbp'] < 0.05) 
        )

        sell_signal = (
            (latest_values['trend_macd'] < latest_values['trend_macd_signal']) and
            (latest_values['momentum_rsi'] > 70) and
            (latest_values['momentum_stoch_rsi_k'] > 0.8) and
            (latest_values['trend_adx'] > 25) and
            (latest_values['trend_ema_fast'] < latest_values['trend_ema_slow']) and
            (latest_values['trend_sma_fast'] < latest_values['trend_sma_slow']) and
          
            (latest_values['volatility_bbh'] > latest_values['close']) and
            (latest_values['volatility_bbl'] < latest_values['close']) and
            (latest_values['volatility_bbp'] > 0.95) 
        )
        if (sell_signal and buy_signal) or (not buy_signal and not buy_signal):
            signal = 'Neutral'
        elif buy_signal and not sell_signal:
            signal = 'Buy'
        else:
            signal = 'Sell'

        # Construct dictionary with indicators and buy/sell signals
        technical_analysis = {
            'MACD': latest_values['trend_macd'],
            'MACD Signal': latest_values['trend_macd_signal'],
            'RSI': latest_values['momentum_rsi'],
            'Stoch RSI K': latest_values['momentum_stoch_rsi_k'],
            'ADX': latest_values['trend_adx'],
            'EMA Fast': latest_values['trend_ema_fast'],
            'EMA Slow': latest_values['trend_ema_slow'],
            'SMA Fast': latest_values['trend_sma_fast'],
            'SMA Slow': latest_values['trend_sma_slow'],
            'Volume OBV': latest_values['volume_obv'],
            # 'Volume OBV Mean': latest_values['volume_obv_mean'],
            'BBH': latest_values['volatility_bbh'],
            'BBL': latest_values['volatility_bbl'],
            'BBP': latest_values['volatility_bbp'],
            'Signal': signal,
            # 'Sell Signal': sell_signal
        }
        print(technical_analysis)

        for key, value in technical_analysis.items():
            # Check if the value is a dictionary, if so, apply rounding recursively
            if isinstance(value, float):
               technical_analysis[key] = round(value, 2)
        return technical_analysis
     
     

    def get_stock_summary(self,start_date='2020-01-01'):
        # article_url = "https://finance.yahoo.com/news/artificial-intelligence-ai-stock-most-074900103.html?guccounter=1&guce_referrer=aHR0cHM6Ly9uZXdzLmdvb2dsZS5jb20v&guce_referrer_sig=AQAAAB705r70NuPA--iv_R50WZUTtjNZvCHeFlWJ-HysOwkDlTbJUbuKEJFQKLOEISWWDIK6pKl_RTtn4kF1vtza39i-4jldi5cnQejXJuXf7f0mq4ZNKG2MnyJOzcsOjO9BaZ3t5omwqhQ1HzkqDMEUgncguxyERSut5KGnNwOHryaW"
        
        # We get the article data from the scraping part
        fundamentals = self.get_yf_fundamentals_for_analysis()
        technicals = self.get_technical_analysis(start_date)
        company_news = self.get_top_company_news(summary=False)

        # Prepare template for prompt
        template = """You are a very good financial advisor.

        Here's the stock {asset_name} you want to evaluate.

        ==================
        fundamentals: {fundamentals}
        technicals: {technicals}
        recent company_news: {company_news}
        ==================

        Write a detailed evaluation of the stock performance and news that may influence performance. Ensure it is useful for investor. Make it sound like you retrieved the information yourself for example
        "Apple (AAPL) has had a strong financial position in the past year. Considering its strong fundamentals such as ..."
        """

        prompt = template.format(asset_name= self.asset_name, fundamentals=fundamentals, technicals=technicals, company_news=company_news)

        messages = [HumanMessage(content=prompt)]

        
        # Load the model
        chat = ChatOpenAI(model_name="gpt-4", temperature=0)
        # Generate summary
        summary = chat(messages)
        return summary.content




    def get_yf_fundamentals(self):
        fundamental_info = {
            "current_price": self.asset_info.get("currentPrice", float('nan')),
            "industry": self.asset_info.get('industry', float('nan')),
            "sector": self.asset_info.get('sector', float('nan')),
            "website": self.asset_info.get('website', float('nan')),
            "full_time_employees": self.asset_info.get('fullTimeEmployees', float('nan')),
            "long_business_summary": self.asset_info.get('longBusinessSummary', float('nan')),
            'Previous Close': self.asset_info.get('previousClose', float('nan')),
            'Day Low': self.asset_info.get('dayLow', float('nan')),
            'Day High': self.asset_info.get('dayHigh', float('nan')),
            '52-Week Low': self.asset_info.get('fiftyTwoWeekLow', float('nan')),
            '52-Week High': self.asset_info.get('fiftyTwoWeekHigh', float('nan')),
            'Trailing P/E Ratio': self.asset_info.get('trailingPE', float('nan')),
            'Forward P/E Ratio': self.asset_info.get('forwardPE', float('nan')),
            'Volume': self.asset_info.get('volume', float('nan')),
            'Market Cap': self.asset_info.get('marketCap', float('nan')),
            'Earnings Per Share (Trailing)': self.asset_info.get('trailingEps', float('nan')),
            'Earnings Per Share (Forward)': self.asset_info.get('forwardEps', float('nan')),
            'Price to Sales (Trailing 12 Months)': self.asset_info.get('priceToSalesTrailing12Months', float('nan')),
            'PEG Ratio': self.asset_info.get('pegRatio', float('nan')),
        }
        return fundamental_info

    def remove_special_characters(input_string):
        return re.sub(r'[^\w\s]', '', input_string)

    def has_special_characters(input_string):
        for char in input_string:
            if unicodedata.category(char) in ['Pc', 'Pd', 'Pe', 'Pf', 'Pi', 'Po', 'Ps']:
                return True
        return False

    # evaluate company news sentiment
    def get_top_company_news(self, num_articles=5, summary=False):
        # Define the URL for the Google News API
        url = f"https://news.google.com/rss/search?q={self.company_name}&hl=en-US&gl=US&ceid=US:en"

        # Send an HTTP GET request to the API
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the XML response
            
            root = ET.fromstring(response.text)
            sid = SentimentIntensityAnalyzer()
            # Initialize a list to store news articles
            articles = []
            if summary: 
                # Iterate through the XML to extract article titles and links
                for item in root.findall(".//item")[:num_articles]:
                    title = item.find("title").text
                    link = item.find("link").text
                    summary = get_article_summary(link)
                    pub_date = item.find("pubDate").text  # Extract publication date
                    sentiment_score = sid.polarity_scores(title)["compound"]
                    # if self.has_special_characters(title):
                    #     continue
                    articles.append({"title": title, "link": link,'date':pub_date,
                                    "sentiment": sentiment_score,"summary":summary})
                    
                

                return articles
            else:
                for item in root.findall(".//item")[:num_articles]:
                    title = item.find("title").text
                    link = item.find("link").text
                    pub_date = item.find("pubDate").text  # Extract publication date
                    sentiment_score = sid.polarity_scores(title)["compound"]
                    # if self.has_special_characters(title):
                    #     continue
                    articles.append({"title": title, "link": link,'date':pub_date,
                                    "sentiment": sentiment_score})

                return articles
                    

        else:
            print(
                f"Failed to retrieve news for {self.company_name}. Status code: {response.status_code}")
            return []



    

    def evaluate_stock_performance(self):
        # Get the current price and previous close price
        current_price = self.get_cur_price()
        previous_close = self.asset_info['previousClose']

        # Calculate the one-day change in percentage
        try:
            one_day_change_percent = self.get_stock_performance(days=1)
        except:
            one_day_change_percent = self.get_stock_performance(days=3)

        # Calculate the one-week change in percentage
        one_week_change_percent = self.get_stock_performance(days=7)

        # Calculate the one-month change in percentage
        one_month_change_percent = self.get_stock_performance(days=30)

        # Calculate volatility (standard deviation of returns) for the one-month period
        # one_month_volatility = self.calculate_volatility(days=30)

        # Create a dictionary to store the performance metrics
        performance_metrics = {
            "one_day_change_percent": one_day_change_percent,
            "one_week_change_percent": one_week_change_percent,
            "one_month_change_percent": one_month_change_percent,
            # "one_month_volatility": one_month_volatility  # Add volatility to the metrics
        }

        return performance_metrics

    # def calculate_volatility(self, days):
    #     # Get historical data for the specified number of days
    #     start_date = date.today()-timedelta(days=days)
    #     start_date_str = "{}-{}-{}".format(start_date.year,
    #                                        start_date.month, start_date.day)
    #     historical_data = self.get_stock_data(start_date_str, 'D')['Close']

    #     # Calculate daily returns
    #     daily_returns = historical_data.pct_change().dropna()

    #     # Calculate volatility (standard deviation of returns)
    #     volatility = np.std(daily_returns)

    #     return volatility


if __name__ == "__main__":
    stock = 'TSLA'
    sa = StockAnalyzer(stock)
    # print(sa.get_asset_info())
    # print(sa.get_cur_price())
    # # print(sa.get_yf_fundamentals())
    # print(sa.get_top_company_news())
    # print(sa.evaluate_stock_performance())
    # print(sa.get_stock_data(300, 'D'))
