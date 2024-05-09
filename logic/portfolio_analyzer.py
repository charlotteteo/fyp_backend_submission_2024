import pandas as pd
import numpy as np
import json
import yfinance as yf
from fastapi import  Response
from scipy.stats import skew, kurtosis
from logic.stock_analyzer import StockAnalyzer
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from logic.langchain_chatbot import get_tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

class PortfolioAnalyzer:
    def __init__(self, stocks, weights, start_date, end_date, initial_investment):
        self.stocks = stocks
        self.weights = weights
        self.start_date = start_date
        self.end_date = end_date
        self.initial_investment = initial_investment
        self.data = yf.download(self.stocks, start=self.start_date, end=self.end_date)['Adj Close']
        self.portfolio_metrics = None

    def convert_data(self,data,round_digit=2):
        result = []
        # Iterate over each row
        columns_list = data.columns
        for x in columns_list:
            for index, row in data.iterrows():
                stock_data = {}
                stock_data['index']=index
                stock_data['ticker'] = x
                stock_data['value'] = round(row[x],round_digit)
                result.append(stock_data)

        # Convert list of dictionaries to JSON format
        json_result = json.dumps(result, indent=4)
        return Response(content=json_result, media_type="application/json")
        # return json_result


    def download_data(self,convert_data = True):
        # Download historical stock price data
        data = yf.download(self.stocks, start=self.start_date, end=self.end_date)['Adj Close']
        data.index = [str(x.date()) for x in data.index]
        if convert_data:
            return self.convert_data(data)
        else:
            return data
    




    def calculate_metrics(self):
        # Calculate daily returns
        returns = self.data.pct_change().dropna()

        # Calculate portfolio value over time
        portfolio_value = self.initial_investment * np.cumprod(1 + np.dot(returns, self.weights))

        # Calculate portfolio returns
        portfolio_returns = (portfolio_value[-1] / portfolio_value[0]) - 1

        # Calculate the number of years
        num_years = len(returns) / 252 if len(returns) > 252 else 1  # Adjust 252 to the appropriate number of trading days in a year

        # Calculate annualized portfolio returns
        annualized_portfolio_returns = (1 + portfolio_returns) ** (1 / num_years) - 1

        # Calculate portfolio volatility (standard deviation of daily returns)
        portfolio_volatility = np.std(np.dot(returns, self.weights))

        # Calculate annualized portfolio volatility
        annualized_portfolio_volatility = portfolio_volatility * np.sqrt(252)  # Annualize daily volatility

        # Calculate the Sharpe ratio
        risk_free_rate = 0.0  # Adjust as needed
        sharpe_ratio = (annualized_portfolio_returns - risk_free_rate) / annualized_portfolio_volatility

        # Calculate maximum drawdown
        cumulative_returns = portfolio_value / self.initial_investment
        peak_idx = np.argmax(np.maximum.accumulate(cumulative_returns) - cumulative_returns)
        trough_idx = np.argmax(cumulative_returns[:peak_idx])
        max_drawdown = cumulative_returns[peak_idx] - cumulative_returns[trough_idx]

        # Calculate the Calmar ratio (assuming a risk-free rate of 0%)
        calmar_ratio = portfolio_returns / max_drawdown

        # Calculate the Sortino ratio (assuming a target return of 0%)
        target_return = 0.0
        downside_returns = np.minimum(returns - target_return, 0)
        downside_volatility = np.std(np.dot(downside_returns, self.weights))
        sortino_ratio = (portfolio_returns - target_return) / downside_volatility

        # Calculate Treynor ratio
        beta = np.cov(returns, rowvar=False)[0, 1]
        treynor_ratio = (portfolio_returns - risk_free_rate) / beta

        # Calculate Information ratio
        benchmark_returns = yf.download("SPY", start=self.start_date, end=self.end_date)['Adj Close'].pct_change().dropna()
        active_returns = returns.mean(axis=1) - benchmark_returns
        tracking_error = np.std(active_returns)
        information_ratio = active_returns.mean() / tracking_error

        self.portfolio_metrics = {
            "Initial Investment": self.initial_investment,
            "Final Portfolio Value": portfolio_value[-1],
            "Portfolio Returns": portfolio_returns,
            "Annualized Portfolio Returns": annualized_portfolio_returns,
            "Portfolio Volatility": portfolio_volatility,
            "Annualized Portfolio Volatility": annualized_portfolio_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Maximum Drawdown": max_drawdown,
            "Calmar Ratio": calmar_ratio,
            # "Sortino Ratio": sortino_ratio,
            # "Treynor Ratio": treynor_ratio,
            # "Information Ratio": information_ratio
        }

        # Round the metrics to 2 decimal places
        for key, value in self.portfolio_metrics.items():
            if isinstance(value, (int, float)):
                self.portfolio_metrics[key] = round(value, 2)

        return self.portfolio_metrics


    def get_portfolio_summary(self):
        # article_url = "https://finance.yahoo.com/news/artificial-intelligence-ai-stock-most-074900103.html?guccounter=1&guce_referrer=aHR0cHM6Ly9uZXdzLmdvb2dsZS5jb20v&guce_referrer_sig=AQAAAB705r70NuPA--iv_R50WZUTtjNZvCHeFlWJ-HysOwkDlTbJUbuKEJFQKLOEISWWDIK6pKl_RTtn4kF1vtza39i-4jldi5cnQejXJuXf7f0mq4ZNKG2MnyJOzcsOjO9BaZ3t5omwqhQ1HzkqDMEUgncguxyERSut5KGnNwOHryaW"
        
        # We get the article data from the scraping part
        metrics = self.calculate_metrics()


        company_news = []
        for x in self.stocks:
        # Get top company news
            company_news += StockAnalyzer(x).get_top_company_news(num_articles=3,summary=False)

        # Get sectoral exposure
        sector_exposure = self.sectoral_pie_chart_data()

        # Get sectoral metrics
        sector_metrics = self.sectoral_metrics()

        # Prepare template for prompt
        template = f"""
        You are a financial advisor evaluating a stock portfolio.
        Write a very detailed evaluation of the portfolio performance and news that may influence performance. Raise upcoming news that could affect portfolio. DO Not make ANY recommendations. 

        Format:

        1. Overall Portfolio Performance
        2. Each Sector's Performance
        3. Potential Risk Factors
        4. Trends
    
    Steps:
        1. Assess the allocation of the portfolio to identify potential over-exposure in specific sectors or constituents.
        2. Evaluate Risk Factors
        3. Assess Trend (Downward Trending) & Fundamentals of Constituents and Sectors
       
        Here's the summary of the portfolio:

        ==================
        Fundamentals:
        {metrics}

        Recent Company News:
        {company_news}

        Sectoral Exposure:
        {sector_exposure}

        Sectoral Metrics:
        {sector_metrics}
        ==================

        

        """
        # print([metrics], company_news,sector_exposure,sector_metrics)
        # prompt = template.format(company_news=str(company_news),sector_exposure=str(sector_exposure),sector_metrics=str(sector_metrics))
        template = template[:8100]
        
        messages = [HumanMessage(content=template)]

        # # Load the model
        chat = ChatOpenAI(model_name="gpt-4", temperature=0)
        # Generate summary
        summary = chat(messages)
        return summary.content.split("=")[-1].split("}")[-1]


        
    def get_portfolio_rec(self):
        # article_url = "https://finance.yahoo.com/news/artificial-intelligence-ai-stock-most-074900103.html?guccounter=1&guce_referrer=aHR0cHM6Ly9uZXdzLmdvb2dsZS5jb20v&guce_referrer_sig=AQAAAB705r70NuPA--iv_R50WZUTtjNZvCHeFlWJ-HysOwkDlTbJUbuKEJFQKLOEISWWDIK6pKl_RTtn4kF1vtza39i-4jldi5cnQejXJuXf7f0mq4ZNKG2MnyJOzcsOjO9BaZ3t5omwqhQ1HzkqDMEUgncguxyERSut5KGnNwOHryaW"
        
        # We get the article data from the scraping part
        metrics = self.calculate_metrics()


        company_news = []
        for x in self.stocks:
        # Get top company news
            company_news += StockAnalyzer(x).get_top_company_news(num_articles=3,summary=False)

        # Get sectoral exposure
        sector_exposure = self.sectoral_pie_chart_data()

        # Get sectoral metrics
        sector_metrics = self.sectoral_metrics()

        # Prepare template for prompt
        template = f"""
        You are a financial advisor evaluating a stock portfolio.
        
         Write a very detailed recommendation of the portfolio performance and news. DO NOT need to evaluate company!

        Format:

        1. Recommendations: 
        Changes in Sectoral Exposure?
        Changes in constituents ratio?
        
        2. Potential Concerns & News
        3. Identify Untapped Sectors that are Promising:
        Explore untapped sectors with promising potential for diversification and growth.
        4. Gather Potential Additional Constituents/Sector:
        Identify sectors or individual assets with low correlation to the removed constituent or sector.
        Propose potential additions to enhance diversification and mitigate correlation risks.
       
    
        Here's the summary of the portfolio:

        ==================
        Fundamentals:
        {metrics}

        Recent Company News:
        {company_news}

        Sectoral Exposure:
        {sector_exposure}

        Sectoral Metrics:
        {sector_metrics}
        ==================

      

        """
        template = template[:8100]
        
        messages = [HumanMessage(content=template)]

        # # Load the model
        chat = ChatOpenAI(model_name="gpt-4", temperature=0)
        # Generate summary
        summary = chat(messages)
        return summary.content




    def get_portfolio_ts(self):
        initial_value = {}
        for i,x in enumerate(self.stocks):
            initial_value[x] = [self.initial_investment * self.weights[i]]
        returns = self.data.pct_change().dropna()
        dates_list = [str(x.date()) for x in  returns.cumsum().index]
        constitutents_breakdown = pd.DataFrame((np.array(returns.cumsum()) + 1)* np.array(pd.DataFrame(initial_value)), index = dates_list, columns = returns.cumsum().columns)
        constitutents_breakdown['Portfolio Value'] = constitutents_breakdown[constitutents_breakdown.columns].sum(axis = 1)
        # return self.convert_data(constitutents_breakdown.reset_index())
        return constitutents_breakdown

    def get_correct_format_portfolio_ts(self,convert_data = True):
        constitutents_breakdown = self.get_portfolio_ts()
        # # print(constitutents_breakdown.reset_index().columns)
        # return self.convert_data(constitutents_breakdown)
        ts_list = []
        for x in constitutents_breakdown.columns:
            data_dict = {'dates':list(constitutents_breakdown[x].index), \
            'values':[round(y,2) for y in list(constitutents_breakdown[x].values)], \
            'ticker': x }
            ts_list.append(data_dict)

        return ts_list
    

    def get_allocation_drift_ts(self,convert_data = True):
        # Calculate allocation drift
        constitutents_breakdown_pct = self.get_portfolio_ts()
        constitutents_breakdown_pct.iloc[:, :-1] = constitutents_breakdown_pct.iloc[:, :-1].div(constitutents_breakdown_pct['Portfolio Value'], axis=0)
        constitutents_breakdown_pct = constitutents_breakdown_pct.drop(['Portfolio Value'],axis=1)
        if convert_data:
            ts_list = []
            for x in constitutents_breakdown_pct.columns:
                data_dict = {'dates':list(constitutents_breakdown_pct[x].index), \
                'values':list(constitutents_breakdown_pct[x].values), \
                'ticker': x }
                ts_list.append(data_dict)

            return ts_list
        else:
            return constitutents_breakdown_pct

    def constituent_pie(self):
        consti_pie = pd.DataFrame(self.get_allocation_drift_ts(convert_data=False).iloc[-1])
        ts_list = []
        for x in consti_pie.columns:
            data_dict = {'labels':list(consti_pie[x].index), \
            'series':list(consti_pie[x].values)}
            ts_list.append(data_dict)


        return ts_list




    def calculate_percentage_change(self,data, interval='year'):
        if interval == 'day':
            return data.pct_change()
        elif interval == 'month':
            return data.pct_change(periods=30)
        elif interval == 'year':
            return data.pct_change(periods=365)
        else:
            raise ValueError("Invalid interval. Choose from 'day', 'month', or 'year'.")


    def returns_ts(self,interval='M',convert_data = True):
        portfolio_ts = self.get_portfolio_ts()
        df = portfolio_ts.reset_index()
        df['index'] = pd.to_datetime(df['index'])
        df = df.set_index('index')
        df = df.resample(interval).last().pct_change().dropna()
        ts_list = []
        data_dict = {'dates':[str(x.date()) for x in list(df['Portfolio Value'].index)], \
        'values':[round(x*100,2) for x in list(df['Portfolio Value'].values)],
        }
        ts_list.append(data_dict)
        return ts_list

    def max_drawdown_ts(self):
        #  Calculate daily returns
        portfolio_ts = self.get_portfolio_ts()
        returns = portfolio_ts.reset_index().set_index('index').pct_change().dropna()

        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        peak_idx = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak_idx) / peak_idx
        drawdown_df = pd.DataFrame(drawdown.min(axis=1))

        ts_list = []
        for x in drawdown_df.columns:
            data_dict = {'dates':list(drawdown_df[x].index), \
            'values':list(drawdown_df[x].values)}
            ts_list.append(data_dict)

        return ts_list
    
    def sectoral_exposure(self,convert_data = False):
        stocks = [x for x in self.stocks if not 'fundFamily' in yf.Ticker(x).info.keys() ]
        sector_info = {ticker: yf.Ticker(ticker).info['sector'] for ticker in stocks}
        # Group stocks by sector
        sector_stocks = {}
        for ticker, sector in sector_info.items():
            if sector not in sector_stocks:
                sector_stocks[sector] = []
            sector_stocks[sector].append(ticker)
        allocation_drift_data = self.get_allocation_drift_ts(convert_data=False)
        allocation_drift_data_ts = allocation_drift_data * self.initial_investment

        sector_sum = {}
        for sector, stocks in sector_stocks.items():
            sector_sum[sector] = allocation_drift_data_ts[stocks].sum(axis=1)

        # Convert to DataFrame
        sector_sum_df = pd.DataFrame(sector_sum)
        if convert_data:
            ts_list = []
            for x in sector_sum_df.columns:
                data_dict = {'dates':list(sector_sum_df[x].index), \
                'values':[round(y,2) for y in list(sector_sum_df[x].values)], \
                'ticker': x }
                ts_list.append(data_dict)

            return ts_list
        else:
            return sector_sum_df

    def asset_type_exposure(self,convert_data = False):
        etfs = [x for x in self.stocks if  'fundFamily' in yf.Ticker(x).info.keys() ]
        stocks = [x for x in self.stocks if not 'fundFamily' in yf.Ticker(x).info.keys() ]
        
        # Group stocks by sector
        asset_type = {}
        asset_type['Bonds'] = etfs
        asset_type['Stocks'] = stocks
        allocation_drift_data = self.get_allocation_drift_ts(convert_data=False)
        allocation_drift_data_ts = allocation_drift_data * self.initial_investment

        asset_sum = {}
        for asset_type, consti in asset_type.items():
            asset_sum[asset_type] = allocation_drift_data_ts[consti].sum(axis=1)

        # Convert to DataFrame
        asset_sum_df = pd.DataFrame(asset_sum)
        if convert_data:
            ts_list = []
            for x in asset_sum_df.columns:
                data_dict = {'dates':list(asset_sum_df[x].index), \
                'values':[round(y,2) for y in list(asset_sum_df[x].values)], \
                'ticker': x }
                ts_list.append(data_dict)

            return ts_list
        else:
            return asset_sum_df


    def get_allocation_drift_sectoral_ts(self,convert_data = True):
        # Calculate allocation drift
        constitutents_breakdown_pct = self.sectoral_exposure()
        constitutents_breakdown_pct['Portfolio Value'] = constitutents_breakdown_pct.sum(axis=1) 
        constitutents_breakdown_pct.iloc[:, :-1] = constitutents_breakdown_pct.iloc[:, :-1].div(constitutents_breakdown_pct['Portfolio Value'], axis=0)
        constitutents_breakdown_pct = constitutents_breakdown_pct.drop(['Portfolio Value'],axis=1)
        if convert_data:
            ts_list = []
            for x in constitutents_breakdown_pct.columns:
                data_dict = {'dates':list(constitutents_breakdown_pct[x].index), \
                'values':list(constitutents_breakdown_pct[x].values), \
                'ticker': x }
                ts_list.append(data_dict)

            return ts_list
        else:
            return constitutents_breakdown_pct
    
    def get_allocation_drift_type_ts(self,convert_data = True):
        # Calculate allocation drift
        constitutents_breakdown_pct = self.asset_type_exposure()
        constitutents_breakdown_pct['Portfolio Value'] = constitutents_breakdown_pct.sum(axis=1) 
        constitutents_breakdown_pct.iloc[:, :-1] = constitutents_breakdown_pct.iloc[:, :-1].div(constitutents_breakdown_pct['Portfolio Value'], axis=0)
        constitutents_breakdown_pct = constitutents_breakdown_pct.drop(['Portfolio Value'],axis=1)
        if convert_data:
            ts_list = []
            for x in constitutents_breakdown_pct.columns:
                data_dict = {'dates':list(constitutents_breakdown_pct[x].index), \
                'values':list(constitutents_breakdown_pct[x].values), \
                'ticker': x }
                ts_list.append(data_dict)

            return ts_list
        else:
            return constitutents_breakdown_pct


    def calculate_percentage_change(self,data, interval='year'):
        if interval == 'day':
            return data.pct_change()
        elif interval == 'month':
            return data.pct_change(periods=30)
        elif interval == 'year':
            return data.pct_change(periods=365)
        else:
            raise ValueError("Invalid interval. Choose from 'day', 'month', or 'year'.")


    def top_movers(self,ranking=3):
        df = self.get_portfolio_ts()
        overall_moves_7d = {}
        overall_moves_7d['ticker'] = []
        overall_moves_7d['7d_change'] = []
        for x in df.columns:
            if x == 'Portfolio Value':
                continue

            overall_moves_7d['ticker'] += [x]
            overall_moves_7d['7d_change'] += [round(100*(df[x].iloc[-1] - df[x].iloc[-7])/df[x].iloc[-7],2)]

        
        print("winners",{"tickers": list(pd.DataFrame(overall_moves_7d).sort_values(by="7d_change",ascending=False).iloc[:ranking]['ticker'].values),
"7d_change": list(pd.DataFrame(overall_moves_7d).sort_values(by="7d_change",ascending=False).iloc[:ranking]['7d_change'].values)
        })
        print("losers",{"tickers": list(pd.DataFrame(overall_moves_7d).sort_values(by="7d_change",ascending=True).iloc[:ranking]['ticker'].values),
"7d_change": list(pd.DataFrame(overall_moves_7d).sort_values(by="7d_change",ascending=True).iloc[:ranking]['7d_change'].values)
        })



        return [{"winners":{"tickers": list(pd.DataFrame(overall_moves_7d).sort_values(by="7d_change",ascending=False).iloc[:ranking]['ticker'].values),
"7d_change": list(pd.DataFrame(overall_moves_7d).sort_values(by="7d_change",ascending=False).iloc[:ranking]['7d_change'].values)
        },
        "losers":{"tickers": list(pd.DataFrame(overall_moves_7d).sort_values(by="7d_change",ascending=True).iloc[:ranking]['ticker'].values),
"7d_change": list(pd.DataFrame(overall_moves_7d).sort_values(by="7d_change",ascending=True).iloc[:ranking]['7d_change'].values)
        }}]

    
    def sectoral_metrics(self):
        sector_sum_df = self.sectoral_exposure()
        returns = sector_sum_df.pct_change().dropna()
        market_returns = yf.download('^GSPC', start=self.start_date, end=self.end_date)['Adj Close'].pct_change().dropna()
        market_returns.index = [str(x.date()) for x in market_returns.index]
        # Define risk-free rate (for Sharpe and Treynor ratios)
        risk_free_rate = 0

        # Sector Weight
        sector_weights = sector_sum_df.mean() / sector_sum_df.mean().sum()

        # Sector Return
        sector_returns = returns.mean()

        # Sector Volatility
        sector_volatility = returns.std()


        # Sector Correlation
        sector_correlation = returns.corr()



        # Sector Sharpe Ratio
        trading_days = len(returns)
        sector_sharpe_ratio = (sector_returns - risk_free_rate) / (sector_volatility * np.sqrt(trading_days / 252))


        # Sector Treynor Ratio

        # Sector Information Ratio (assuming benchmark return is market return)
        benchmark_return = market_returns.mean()
        sector_active_returns = sector_returns - benchmark_return
        sector_tracking_error = np.sqrt(np.diagonal(sector_correlation)) * sector_volatility
        sector_information_ratio = sector_active_returns / sector_tracking_error

        # Sector Maximum Drawdown
        sector_max_drawdown = returns.apply(lambda x: (1 + x).cumprod().div((1 + x).cumprod().cummax()).sub(1).min())

        # Sector Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.sqrt((downside_returns ** 2).mean())
        sector_sortino_ratio = (sector_returns - risk_free_rate) / downside_deviation

        # Sector Skewness
        sector_skewness = skew(returns)

        # Sector Kurtosis
        sector_kurtosis = kurtosis(returns)

        # Filter returns data to include only dates present in market returns data
        aligned_returns = returns.join(market_returns, how='inner')


        # # Calculate variance of market returns
        # market_variance = np.var(market_returns)

        # # Calculate sector beta
        # sector_beta_cov = covariance / market_variance
        beta_list = pd.DataFrame((aligned_returns.corr()['Adj Close'] * aligned_returns.std())/aligned_returns.std()['Adj Close'],columns=['beta'])
        beta_list = np.array(beta_list[:-1]).flatten()

        # Output results
        sector_metrics = pd.DataFrame({
            # 'Weight': sector_weights,
            # 'Return': sector_returns,
            # 'Volatility': sector_volatility,
            'Sharpe Ratio': sector_sharpe_ratio,
            'Beta':beta_list,
            'Information Ratio': sector_information_ratio,
            'Max Drawdown': sector_max_drawdown,
            'Sortino Ratio': sector_sortino_ratio,
            'Skewness': sector_skewness,
            'Kurtosis': sector_kurtosis
        })

        for key, value in sector_metrics.items():
            # Check if the value is a dictionary, if so, apply rounding recursively
            if isinstance(value, dict):
               sector_metrics[key] = {k: round(v, 3) for k, v in value.items()}
            else:
                # Round the value to 2 decimal places
                sector_metrics[key] = round(value, 3)
            # if key == 'Return':
            #     sector_metrics[key] = {k: "{}%".format(round(v*100, 2)) for k, v in value.items()}
        return sector_metrics.reset_index().to_dict(orient='records')

    def constituent_table(self):
        df = self.get_portfolio_ts()
        ts_list = []
        for x in df.columns:
            if x == 'Portfolio Value':
                continue
            data_dict = {
                'current_value': round(df[x].iloc[-1],2),
                'initial_investment': round(df[x].iloc[0],2),
                'overall_change': round(100*(df[x].iloc[-1] - df[x].iloc[0])/df[x].iloc[0],2),
                '1d_change': round(100*(df[x].iloc[-1] - df[x].iloc[-2])/df[x].iloc[-2],2),
                '30d_change': round(100*(df[x].iloc[-1] - df[x].iloc[-28])/df[x].iloc[-28],2),
            'ticker': x }
            if  (pd.to_datetime(self.end_date) -  pd.to_datetime(self.start_date)).days>365:
                data_dict['1y_change'] = round(100*(df[x].iloc[-1] - df[x].iloc[-252])/df[x].iloc[-252],2),
            ts_list.append(data_dict)

        return ts_list

    def sectoral_pie_chart_data(self):
        last_sector_row = self.get_allocation_drift_sectoral_ts(convert_data=False).iloc[-1]
        last_row = self.get_allocation_drift_ts(convert_data=False).iloc[-1]
        stocks = [x for x in self.stocks if not 'fundFamily' in yf.Ticker(x).info.keys()]
        sector_info = {ticker: yf.Ticker(ticker).info['sector'] for ticker in stocks}
        sector_stocks = {}
        for ticker, sector in sector_info.items():
            if sector not in sector_stocks:
                sector_stocks[sector] = []
            sector_stocks[sector].append(ticker)
        sector_pie_data = {}
        sector_pie_data['series'] = [round(x,2) for x in list(last_sector_row.values)]
        sector_pie_data['labels'] = list(last_sector_row.index)
        # sector_pie_data_children = {}
        # for sector,ticker  in sector_stocks.items():
            
        #     sector_pie_data_children[sector] = {
        #         'series':[round(last_row.loc[x]/last_sector_row.loc[sector],2) for x in ticker],
        #         'labels':ticker,
        #         'children':{}
        #     }
        # sector_pie_data['children']=sector_pie_data_children
        return [sector_pie_data]

    def type_pie_chart_data(self):
        last_sector_row = self.get_allocation_drift_type_ts(convert_data=False).iloc[-1]
        sector_pie_data = {}
        sector_pie_data['series'] = [round(x,2) for x in list(last_sector_row.values)]
        sector_pie_data['labels'] = list(last_sector_row.index)
        return [sector_pie_data]



    def sectoral_box_plot(self):
        sector_sum_df = self.sectoral_exposure(convert_data = False)
        returns = sector_sum_df.pct_change().dropna()

        ts_list = []
        for x in returns.columns:
            data_dict = {
            'values':[round(y,2) for y in list(returns[x].values)], \
            'ticker': x }
            ts_list.append(data_dict)
        return ts_list

    def sectoral_corr_matrix(self):
        sector_sum_df = self.sectoral_exposure(convert_data = False)
        returns = sector_sum_df.pct_change().dropna()
        correlation_matrix = returns.corr()
        # corrdata = []
        # for i, row in correlation_matrix.iterrows():
        #     for j, val in row.items():
        #         if pd.notna(val):
        #             corrdata.append({'sector1': i, 'sector2': j, 'value': val})

        return {"data":[list(row) for row in list(returns.corr().values)],"labels":list(returns.corr().index)}
        # return correlation_matrix

if __name__ == "__main__":
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'IEF']
    weights = [0.25, 0.25, 0.25, 0.25]
    start_date = '2021-01-01'
    end_date = '2022-12-31'
    initial_investment = 10000

    portfolio_evaluator = PortfolioAnalyzerFinal(stocks, weights, start_date, end_date, initial_investment)
    # print(portfolio_evaluator.download_data())
    # print(portfolio_evaluator.get_allocation_drift_ts())
    # print(portfolio_evaluator.constituent_pie())
    print(portfolio_evaluator.get_allocation_drift_sectoral_ts())
    print(portfolio_evaluator.sectoral_box_plot())
