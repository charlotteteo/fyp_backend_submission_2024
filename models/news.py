from beanie import Document
from fastapi.security import HTTPBasicCredentials
from pydantic import BaseModel, EmailStr


class NewsArticle(BaseModel):
    title: str
    source: str
    time: str
    description: str
    url: str
    langchain_evaluation: str

    class Config:
        schema_extra = {
            "example": {'title': 'Buy or sell: Sumeet Bagadia recommends three stocks to buy during special trading session on 2nd March 2024 | Mint - Mint',
             'source': 'Livemint', 'time': '2024-03-03T04:15:39Z', 
             'description': 'Buy or sell stocks: Sumeet Bagadia has recommended three stocks to buy during special trading session on Saturday', 
             'url': 'https://www.livemint.com/market/stock-market-news/buy-or-sell-sumeet-bagadia-recommends-three-stocks-to-buy-during-special-trading-session-on-2nd-march-2024-11709341973335.html',
            'langchain_evaluation': 'The article recommended three stocks to buy during special trading session on Saturday with a ', }
        }

    class Setting:
        name = 'newsfeed_data'



