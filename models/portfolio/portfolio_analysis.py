from sqlite3 import Timestamp
from xxlimited import Str
from pydantic import BaseModel, Field
from beanie.odm.documents import PydanticObjectId
from typing import List, Dict


class PortfolioAnalysis(BaseModel):
    qualitative_summary: str
    quantitative_metrics: Dict[str, float]


class PortfolioChatBot(BaseModel):
    prompt: str
    chatbot_reply: str
    timestamp: str
