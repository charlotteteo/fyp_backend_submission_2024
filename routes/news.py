from fastapi import APIRouter, Body
from beanie.odm.documents import PydanticObjectId
from database.news import *
from models.news import *
from database.portfolio import *
from models.portfolio import *

router = APIRouter(tags=["Newsfeed"], prefix="/newsfeed")


@router.get("/{ticker}/{number}", response_description="Newsfeed retrieved")
async def get_newsfeed(ticker: str,number:int):
    news_list = await retrieve_newsfeed(ticker,number)
    if len(news_list):
        return {
            "status_code": 200,
            "response_type": "success",
            "description": "stock data retrieved successfully",
            "data": news_list,
        }
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "stock analysis data doesn't exist",
    }


@router.get("/headlines_news/{query}/{number}", response_description="Newsfeed retrieved")
async def get_portfolio_related_newsfeed(query: str,number:int):
    news_list = await retrieve_headlines_newsfeed(query,number)
    # news_list = await retrieve_newsfeed('AAPL',number)
    if news_list:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": "stock data retrieved successfully",
            "data": news_list,
        }
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "stock analysis data doesn't exist",
    }


@router.get("/portfolio_news/{stocks}/{number}", response_description="Newsfeed retrieved")
async def get_portfolio_related_newsfeed(stocks: str,number:int):
    stocks = stocks.split(";")
    news_list = await retrieve_portfolio_relevant_newsfeed(stocks,number)
    # news_list = await retrieve_newsfeed('AAPL',number)
    if news_list:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": "stock data retrieved successfully",
            "data": news_list,
        }
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "stock analysis data doesn't exist",
    }





@router.get("/news_evaluation", response_description="Newsfeed retrieved")
async def get_newsfeed_evaluation():
    link = 'https://finance.yahoo.com/news/advanced-micro-devices-nasdaq-amd-110028061.html'
    news_list = await retrieve_news_evaluation(link)
    # news_list = await retrieve_newsfeed('AAPL',number)
    if news_list:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": "stock data retrieved successfully",
            "data": news_list,
        }
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "stock analysis data doesn't exist",
    }



# @router.get("/news_summary/{link}", response_description="Summary retrieved")
# async def get_news_summary(link: str):
#     try:
#         news_list = await retrieve_news_summary(link)
#         if news_list:
#             return {
#                 "status_code": 200,
#                 "response_type": "success",
#                 "description": "News summaries retrieved successfully",
#                 "data": news_list,
#             }
#         else:
#             return {
#         "status_code": 404,
#         "response_type": "error",
#         "description": "stock analysis data doesn't exist",
#     }


#     except Exception as e:
#         return {
#         "status_code": 404,
#         "response_type": "error",
#         "description": "stock analysis data doesn't exist",
#     }

