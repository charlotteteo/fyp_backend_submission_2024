from typing import List, Dict, Union
from beanie import Document
from pydantic import BaseModel, EmailStr, Field


class StockAnalysis(BaseModel):
    ticker: str
    company_name: str
    live_price: float
    full_yfinance_data: dict
    company_news: List[Dict[str, Union[str, float]]]
    stock_performance: Dict[str, float]

    class Config:
        schema_extra = {
            "example": {
                "ticker": "Apple",
                "company_name": "live_price",
                "live_price": 180.9,
                "live_volume": 66763827,
                "company_news": [{'title': 'Apple Inc. stock underperforms Wednesday when compared to competitors - MarketWatch'},
                                 {'title': 'Apple: A potential Autumn windfall - Yahoo Finance'},
                                 {'title': 'Apple to Scale Up India Production Fivefold to $40 Billion - Bloomberg'},
                                 {'title': 'UBS Reiterates Apple (AAPL) Neutral Recommendation - Nasdaq'},
                                 {'title': 'Apple Inc Faces Decline in Shares Due to Market Weakness and ... - Best Stocks'}],
                "full_yfinance_data": {'industry': 'Consumer Electronics',
                                       'sector': 'Technology',
                                       'website': 'https://www.apple.com',
                                       'full_time_employees': 164000,
                                       'Previous Close': 171.96,
                                       'Day Low': 169.05,
                                       'Day High': 173.04,
                                       '52-Week Low': 124.17,
                                       '52-Week High': 198.23,
                                       'Trailing P/E Ratio': 28.547739,
                                       'Forward P/E Ratio': 27.400322,
                                       'Volume': 66763827,
                                       'Market Cap': 2664536473600
                                       }

            }

        }

    class Settings:
        name = "stocks_analysis_data"
