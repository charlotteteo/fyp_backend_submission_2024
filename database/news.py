from models.news import NewsArticle
from logic.newsfeed_analyzer import *
from logic.stock_analyzer import StockAnalyzer


async def retrieve_newsfeed(query: str,retrieve_newsfeed:int):
    try:
        news_list= StockAnalyzer(query).get_top_company_news(retrieve_newsfeed)
        return news_list
    except:
        return None

async def retrieve_headlines_newsfeed(query: str,retrieve_newsfeed:int):
    try:
        news_list= get_top_finance_news('finance',retrieve_newsfeed)
        return news_list
    except:
        return None
    

async def retrieve_portfolio_relevant_newsfeed(tickers,retrieve_newsfeed:int):
    try:
        
        # tickers = 'microsoft;apple'.split(";")
        # tickers = 'microsoft;apple'.split(";")
        news_list = []
        for x in tickers:
            # news_list += get_top_finance_news(x,5)
            news_list += StockAnalyzer(x).get_top_company_news(5)
        # print(news_list)
        return news_list
    except:
        return None


# async def retrieve_news_summary(link):
#     return get_article_summary(link)



async def retrieve_news_evaluation(link):
    return get_article_evaluation(link)
