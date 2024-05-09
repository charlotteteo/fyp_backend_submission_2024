from datetime import date
from re import M
from typing import List, Optional, Dict
from enum import Enum
from beanie import Document
from pydantic import BaseModel, Field
from beanie.odm.documents import PydanticObjectId
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf


class PortfolioTrades(BaseModel):
    trade_date: str
    instrument_code: str
    qty: int
    price: float
    transaction: str

class Portfolio(Document):
    portfolio_name: str
    portfolio_trades: List[PortfolioTrades]

    class Config:
        orm_mode = True
        json_schema_extra = {
            "example": {
                "portfolio_name": "Technology Portfolio",
                "portfolio_trades":[{
                     "trade_date":'21-01-2021',
                     "instrument_code":'AAPL',
                      "qty":100,
                      "price":90,
                      "transaction":'BUY'
                }]
                  
                }
        }




class UpdatePortfolioModel(BaseModel):
    portfolio_name: Optional[str]
    portfolio_trades: Optional[ List[PortfolioTrades]]
    class Config:
        json_schema_extra = {
            "example": {
                "portfolio_name": "Technology Portfolio",
                "portfolio_trades": {
                    "trade_dates":['21-01-2021','14-05-2021'],
                    "instrument_code":['AAPL','GOOGL'],
                    "qty":[100,50],
                    "price":[90,100],
                    "transaction":['BUY','BUY']
                }}
        }
