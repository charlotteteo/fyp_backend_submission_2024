from pydantic import BaseModel
from typing import List, Dict

class PortfolioRecommendation(BaseModel):
    portfolio_name: str
    author: str
    asset_allocation: Dict[str,float]
    articles: List[str]
    overview: str
    description: str
    score: float
    risk: int
    investor_profile: str
    
    

