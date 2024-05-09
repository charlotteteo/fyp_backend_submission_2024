from fastapi import APIRouter, Body
from beanie.odm.documents import PydanticObjectId
from database.stocks import *
from models.stocks import *
from database.portfolio import *
from models.portfolio import *

router = APIRouter(tags=["Analysis"], prefix="/analysis")


@router.get("/stock_time_series/{ticker}/{start_date_str}/{freq}", response_description="Stock Time Series retrieved")
async def get_stock_time_series(ticker: str, start_date_str: str, freq: str):
    stock_analysis = await retrieve_stock_time_series(ticker, start_date_str, freq)
    if len(stock_analysis):
        return stock_analysis
        
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "stock analysis data doesn't exist",
    }

@router.get("/stock_fundamental/{ticker}", response_description="Stock Fundamental retrieved")
async def get_stock_fundamental_analysis(ticker: str):
    stock_analysis: StockAnalysis = await retrieve_stock_fundamental_analysis(ticker)
    if stock_analysis:
        return stock_analysis
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "stock analysis data doesn't exist",
    }




@router.get("/stock_technical/{ticker}/{start_date}", response_description="Stock Fundamental retrieved")
async def get_stock_technical_analysis(ticker: str,start_date: str):
    stock_analysis: StockAnalysis = await retrieve_stock_technical_analysis(ticker,start_date)
    if stock_analysis:
        return stock_analysis
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "stock analysis data doesn't exist",
    }


@router.get("/stock_summary/{ticker}/{start_date}", response_description="Stock Fundamental retrieved")
async def get_stock_summary(ticker: str,start_date: str):
    stock_analysis: StockAnalysis = await retrieve_stock_summary(ticker,start_date)
    if stock_analysis:
        return stock_analysis
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "stock analysis data doesn't exist",
    }






@router.get("/stock/{ticker}", response_description="Stock analysis retrieved")
async def get_stock_analysis(ticker: str):
    stock_analysis: StockAnalysis = await retrieve_stock_analysis(ticker)
    if stock_analysis:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": "stock data retrieved successfully",
            "data": stock_analysis,
        }
    return {
        "status_code": 404,
        "response_type": "error",
        "description": "stock analysis data doesn't exist",
    }


# @router.get("/portfolio/{id}/{p_id}", response_description="Get Portfolio Summary")
# async def get_portfolio_analysis(u_id: PydanticObjectId, p_id: PydanticObjectId):
#     updated_portfolio = await retrieve_portfolio_analysis(u_id, p_id)
#     if updated_portfolio:
#         return {
#             "status_code": 200,
#             "response_type": "success",
#             "description": f"Portfolio with ID: {p_id} metrics",
#             "data": updated_portfolio,
#         }
#     return {
#         "status_code": 404,
#         "response_type": "error",
#         "description": "portfolio not found",
#     }


# @router.get("/portfolio/{id}", response_description="Get Portfolio News")
# async def get_relevant_news(id: PydanticObjectId):
#     portfolios = await retrieve_portfolios(id)
#     # portfolios.
#     if updated_portfolio:
#         return {
#             "status_code": 200,
#             "response_type": "success",
#             "description": f"Portfolio with ID: {p_id} metrics",
#             "data": updated_portfolio,
#         }
#     return {
#         "status_code": 404,
#         "response_type": "error",
#         "description": "portfolio not found",
#     }
