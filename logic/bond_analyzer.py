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
import re
import unicodedata
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI

# ['phone', 'longBusinessSummary', 'maxAge', 'priceHint', 'previousClose', 'open', 'dayLow', 'dayHigh', 'regularMarketPreviousClose', 'regularMarketOpen', \
#     'regularMarketDayLow', 'regularMarketDayHigh', 'volume', 'regularMarketVolume', 'averageVolume', 'averageVolume10days', 'averageDailyVolume10Day', 'bid', 'ask', \
#         'bidSize', 'askSize', 'yield', 'totalAssets', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'fiftyDayAverage', \
#     'twoHundredDayAverage', 'trailingAnnualDividendRate', 'trailingAnnualDividendYield', 'navPrice', 'currency', 'category', 'ytdReturn', 'beta3Year',\
#          'fundFamily', 'fundInceptionDate', 'legalType', 'threeYearAverageReturn', 'fiveYearAverageReturn', 'exchange', 'quoteType', 'symbol', 'underlyingSymbol', 'shortName', \
#     'longName', 'firstTradeDateEpochUtc', 'timeZoneFullName', 'timeZoneShortName', 'uuid', 'messageBoardId', 'gmtOffSetMilliseconds', 'trailingPegRatio']

class BondAnalyzer:
    def __init__(self, ticker: str):
        self.asset_name = ticker
        # self.asset = yf.Ticker(self.asset_name) if yf.Ticker(self.asset_name)!=None else None
        self.asset = self._get_asset(ticker)
        self.asset_info = self.asset.info
        self.company_name = self.asset_info['shortName']
        self.bond = ['fundFamily' in self.asset_info.keys()]

    def _get_asset(self, ticker: str):
        try:
            res = yf.Ticker(ticker)
            info = res.info  # You might want to use this 'info' somewhere

        except Exception as e:
            raise ValueError(
                f"Failed to fetch data for ticker {ticker}. Error: {str(e)}")

        return res

    # def get_cur_price(self):
    #     return self.asset_info['currentPrice']

    # def get_past_day_change(self):
    #     return (self.asset_info['currentPrice'] - self.asset_info['previousClose'])/self.asset_info['previousClose']

    def get_asset_info(self):
        return self.asset_info

    def get_bond_performance(self, days) -> float:
        """Method to get bond price change in percentage"""

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
        


    def get_bond_data(self, start_date_str, freq):
        """Method to get bond price change in percentage"""
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
     
     

    def get_bond_summary(self,start_date='2020-01-01'):
        # article_url = "https://finance.yahoo.com/news/artificial-intelligence-ai-bond-most-074900103.html?guccounter=1&guce_referrer=aHR0cHM6Ly9uZXdzLmdvb2dsZS5jb20v&guce_referrer_sig=AQAAAB705r70NuPA--iv_R50WZUTtjNZvCHeFlWJ-HysOwkDlTbJUbuKEJFQKLOEISWWDIK6pKl_RTtn4kF1vtza39i-4jldi5cnQejXJuXf7f0mq4ZNKG2MnyJOzcsOjO9BaZ3t5omwqhQ1HzkqDMEUgncguxyERSut5KGnNwOHryaW"
        
        # We get the article data from the scraping part
        fundamentals = self.get_yf_fundamentals_for_analysis()
        technicals = self.get_technical_analysis(start_date)
        company_news = self.get_top_company_news(summary=False)

        # Prepare template for prompt
        template = """You are a very good financial advisor.

        Here's the bond you want to evaluate.

        ==================
        fundamentals: {fundamentals}
        technicals: {technicals}
        recent company_news: {company_news}
        ==================

        Write a detailed evaluation of the bond performance and news that may influence performance. Ensure it is useful for investor. Make it sound like you retrieved the information yourself for example
        "AAPL has had a strong financial position in the past year. Considering its strong fundamentals such as ..."
        """

        prompt = template.format(fundamentals=fundamentals, technicals=technicals, company_news=company_news)

        messages = [HumanMessage(content=prompt)]

        
        # Load the model
        chat = ChatOpenAI(model_name="gpt-4", temperature=0)
        # Generate summary
        summary = chat(messages)
        return summary.content




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



    

    # def evaluate_bond_performance(self):
    #     # Get the current price and previous close price
    #     current_price = self.get_cur_price()
    #     previous_close = self.asset_info['previousClose']

    #     # Calculate the one-day change in percentage
    #     one_day_change_percent = self.get_past_day_change()

    #     # Calculate the one-week change in percentage
    #     one_week_change_percent = self.get_bond_performance(days=7)

    #     # Calculate the one-month change in percentage
    #     one_month_change_percent = self.get_bond_performance(days=30)

    #     # Calculate volatility (standard deviation of returns) for the one-month period
    #     # one_month_volatility = self.calculate_volatility(days=30)

    #     # Create a dictionary to store the performance metrics
    #     performance_metrics = {
    #         "one_day_change_percent": one_day_change_percent,
    #         "one_week_change_percent": one_week_change_percent,
    #         "one_month_change_percent": one_month_change_percent,
    #         # "one_month_volatility": one_month_volatility  # Add volatility to the metrics
    #     }

    #     return performance_metrics

    

if __name__ == "__main__":
    bond = 'IEF'
    sa = BondAnalyzer(bond)
    print(sa.get_asset_info())
    # print(sa.get_cur_price())
    print(sa.get_bond_summary())
    print(sa.get_technical_analysis(start_date='2022-02-20'))
    print(sa.get_top_company_news())
    # print(sa.evaluate_bond_performance())
    # print(sa.get_bond_data(300, 'D'))
